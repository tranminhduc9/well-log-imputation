"""Frozen SAITS with a supervised, gap-aware residual tree corrector.

The corrector sees fresh masked TRAIN predictions from the fitted SAITS member.
Validation alone chooses trees and correction strength. This is an explicitly
two-stage model; training predictions are not out-of-fold predictions.
"""
from dataclasses import dataclass
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d
from xgboost import XGBRegressor
from src.models.model import AbstractModel
from src.models.geoboost import GeoBoostConfig, _GeoBoostBackend, gap_groups
from src.preprocessing.pipeline import create_missing_mask

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAITSBoostConfig(GeoBoostConfig):
    n_estimators: int = 800
    max_depth: int = 4
    max_train_rows: int = 180000
    mask_repeats: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.mask_repeats < 1:
            raise ValueError('mask_repeats must be positive.')


class _SAITSBoostBackend:
    def __init__(self, config):
        self.config = config
        self.member = None
        self.models = []
        self.weights = np.zeros((config.n_features, 4))
        self.training_history = []
        self.calibration = []

    def _features(self, values, prediction, target):
        context, _ = _GeoBoostBackend(self.config)._residual_features(values, target)
        target_prediction = prediction[..., target, None]
        extras = [prediction, target_prediction - uniform_filter1d(target_prediction, 9, axis=1, mode='nearest'),
                  uniform_filter1d(target_prediction, 33, axis=1, mode='nearest')]
        return np.column_stack((context, np.concatenate(extras, axis=2).reshape(len(context), -1)))

    @staticmethod
    def _groups(values, target):
        return np.stack([gap_groups(np.isfinite(segment[:, target])) for segment in values])

    def fit(self, train_set, val_set=None):
        if self.member is None or val_set is None:
            raise ValueError('SAITSBoost requires fitted SAITS and validation data.')
        intact = np.asarray(train_set['X'], dtype=np.float32)
        values = np.tile(intact, (self.config.mask_repeats, 1, 1))
        masks = np.concatenate([create_missing_mask(intact, random_state=self.config.seed + 50000 + r)
                                for r in range(self.config.mask_repeats)])
        if not masks.any(axis=(0, 1)).all():
            if len(masks) < self.config.n_features:
                raise ValueError('SAITSBoost needs enough training segments to mask every log.')
            # Tiny datasets can randomly omit a channel. Cover every log without
            # changing the normal full-data mask sequence.
            for target in range(self.config.n_features):
                masks[target] = False
                masks[target, :, target] = True
        values[masks] = np.nan
        LOGGER.info('SAITSBoost generating fresh train-mask predictions: %d segments', len(values))
        prediction = self.member.impute({'X': values})
        validation = []
        for data in val_set['validation_scenarios'].values():
            validation.append((data, self.member.impute({'X': data['X']})))
        rng = np.random.default_rng(self.config.seed)
        for target in range(self.config.n_features):
            hidden = masks[..., target]
            take = np.flatnonzero(hidden.reshape(-1))
            if len(take) > self.config.max_train_rows:
                take = rng.choice(take, self.config.max_train_rows, replace=False)
            features = self._features(values, prediction, target)[take]
            truth = np.tile(intact[..., target], (self.config.mask_repeats, 1))
            residual = (truth - prediction[..., target]).reshape(-1)[take]
            weight = np.broadcast_to(1 / hidden.sum(axis=1, keepdims=True).clip(1), hidden.shape).reshape(-1)[take]
            weight /= weight.mean()
            val_x, val_y, val_groups = [], [], []
            for data, base in validation:
                selected = data['indicating_mask'][..., target].astype(bool)
                val_x.append(self._features(data['X'], base, target)[selected.reshape(-1)])
                val_y.append((data['X_intact'][..., target] - base[..., target])[selected])
                val_groups.append(self._groups(data['X'], target)[selected])
            val_x, val_y, val_groups = map(np.concatenate, (val_x, val_y, val_groups))
            if not len(val_y):
                raise ValueError(f'Validation has no masked values for log {target}.')
            model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
                                 device=self.config.device, n_jobs=self.config.n_jobs,
                                 n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
                                 learning_rate=self.config.learning_rate, early_stopping_rounds=self.config.early_stopping_rounds,
                                 min_child_weight=30, reg_lambda=20, subsample=.85,
                                 colsample_bytree=.9, random_state=self.config.seed)
            LOGGER.info('SAITSBoost fitting residual log %d/%d', target+1, self.config.n_features)
            model.fit(features, residual, sample_weight=weight, eval_set=[(val_x, val_y)], verbose=False)
            self.models.append(model)
            self.training_history.extend(dict(log_index=target, iteration=i+1, validation_rmse=score)
                                         for i,score in enumerate(model.evals_result()['validation_0']['rmse']))
            corrections = model.predict(val_x)
            for group in range(4):
                selected = val_groups == group
                if not selected.any():
                    continue
                grid = np.linspace(0, 1, 5)
                loss = ((corrections[selected, None] * grid - val_y[selected, None])**2).mean(axis=0)
                best = int(loss.argmin())
                self.weights[target, group] = grid[best]
                self.calibration.append(dict(log_index=target, gap_group=group, weight=float(grid[best]),
                                             validation_rmse=float(np.sqrt(loss[best]))))

    def predict(self, dataset):
        values = np.asarray(dataset['X'], dtype=np.float32)
        base = self.member.impute({'X': values})
        result = base.copy()
        for target, model in enumerate(self.models):
            active = np.flatnonzero(np.isnan(values[..., target]).any(axis=1))
            if not len(active):
                continue
            groups = self._groups(values[active], target)
            correction = model.predict(self._features(values[active], base[active], target)).reshape(len(active), -1)
            for local, segment in enumerate(active):
                missing = groups[local] >= 0
                result[segment, missing, target] += correction[local, missing] * self.weights[target, groups[local, missing]]
        return {'imputation': result}


class SAITSBoost(AbstractModel):
    name = 'saits_boost'

    def __init__(self, config=None):
        super().__init__(config or SAITSBoostConfig())

    def _build_backend(self):
        return _SAITSBoostBackend(self.config)
