"""GeoBoost: cross-log trees with observation-only local residual correction.

This is a project-specific model, not an implementation of a published model.
The target log is excluded from *all* global features. Local adaptation uses
only observed samples in the input segment. Validation fits convex weights;
neither scenario labels nor intact targets are accepted by prediction.
"""
from dataclasses import dataclass
import itertools
import logging

import numpy as np
from scipy.ndimage import uniform_filter1d
from xgboost import XGBRegressor

from src.models.model import AbstractModel, ModelConfig
from src.models.geo_tcn import interpolation_features

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoBoostConfig(ModelConfig):
    n_estimators: int = 1200
    max_depth: int = 6
    learning_rate: float = 0.035
    max_train_rows: int = 250000
    early_stopping_rounds: int = 60
    n_jobs: int = 4
    device: str = "cpu"
    ridge_alpha: float = 2.0
    residual_learning: bool = True

    def __post_init__(self):
        super().__post_init__()
        if min(self.n_estimators, self.max_depth, self.max_train_rows,
               self.early_stopping_rounds, self.ridge_alpha) <= 0:
            raise ValueError("GeoBoost settings must be positive.")
        if self.n_features < 2:
            raise ValueError("GeoBoost requires at least two logs.")
        if self.seq_len < 2:
            raise ValueError("GeoBoost requires at least two depth samples.")


def gap_groups(observed):
    """Classify each missing run from its length, without evaluation labels."""
    groups = np.full(len(observed), -1, dtype=int)
    missing = ~observed
    boundaries = np.diff(np.r_[False, missing, False].astype(int))
    for start, stop in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        length = stop - start
        group = 3 if not observed.any() else (0 if length == 1 else (1 if length <= 20 else 2))
        groups[start:stop] = group
    return groups


class _GeoBoostBackend:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.residual_models = []
        self.training_history = []
        self.calibration = []
        # Linear, global + residual, local ridge + residual, learned residual.
        self.weights = np.tile(np.array([0., 1., 0., 0.]), (config.n_features, 4, 1))

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Preserve the earlier three-candidate validation pilot on reload.
        if self.weights.shape[-1] == 3:
            self.weights = np.pad(self.weights, ((0, 0), (0, 0), (0, 1)))

    @staticmethod
    def _features(values, target):
        other = np.delete(np.asarray(values, dtype=np.float32), target, axis=2)
        # GeoLink masks one target channel. Filling supports additional NaNs
        # deterministically without ever borrowing intact evaluation values.
        other = np.nan_to_num(other, nan=0., posinf=0., neginf=0.)
        steps = other.shape[1]
        features = [other]
        for offset in (-16, -4, -1, 1, 4, 16):
            features.append(other[:, np.clip(np.arange(steps) + offset, 0, steps - 1)])
        for width in (9, 33):
            features.append(uniform_filter1d(other, width, axis=1, mode="nearest"))
        for stat in (other.mean(axis=1, keepdims=True), other.std(axis=1, keepdims=True)):
            features.append(np.broadcast_to(stat, other.shape))
        return np.concatenate(features, axis=2).reshape(-1, sum(f.shape[2] for f in features))

    def fit(self, train_set, val_set=None):
        values = np.asarray(train_set["X"], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("GeoBoost training requires intact finite training segments.")
        scenarios = {} if val_set is None else val_set.get("validation_scenarios", {"validation": val_set})
        if not scenarios:
            raise ValueError("GeoBoost requires masked validation data for calibration.")
        validation = scenarios.get("Entire-Log", next(iter(scenarios.values())))
        rng = np.random.default_rng(self.config.seed)
        take = rng.choice(values.shape[0] * values.shape[1],
                          min(self.config.max_train_rows, values.shape[0] * values.shape[1]), replace=False)
        self.models, self.training_history, self.calibration = [], [], []
        for target in range(values.shape[2]):
            mask = np.asarray(validation["indicating_mask"])[..., target].reshape(-1).astype(bool)
            kwargs = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                          device=self.config.device, n_jobs=self.config.n_jobs,
                          n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
                          learning_rate=self.config.learning_rate, subsample=0.85,
                          colsample_bytree=0.9, min_child_weight=20, reg_lambda=10,
                          random_state=self.config.seed)
            fit_kwargs = {}
            if mask.any():
                kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds
                fit_kwargs["eval_set"] = [(self._features(validation["X"], target)[mask],
                                          validation["X_intact"][..., target].reshape(-1)[mask])]
            model = XGBRegressor(**kwargs)
            LOGGER.info("GeoBoost fitting log %d/%d (%d training rows)", target + 1, values.shape[2], len(take))
            model.fit(self._features(values, target)[take], values[..., target].reshape(-1)[take],
                      verbose=False, **fit_kwargs)
            self.models.append(model)
            if mask.any():
                self.training_history.extend({"log_index": target, "iteration": i + 1,
                                              "validation_rmse": score}
                                             for i, score in enumerate(model.evals_result()["validation_0"]["rmse"]))
            LOGGER.info("GeoBoost log %d selected %d trees", target + 1,
                        getattr(model, "best_iteration", self.config.n_estimators - 1) + 1)
        if self.config.residual_learning:
            self._fit_residuals(values, scenarios)
        self._calibrate(scenarios)

    def _residual_features(self, values, target):
        inputs, baseline, _ = interpolation_features(values)
        observed = np.isfinite(values[..., target])
        position = np.arange(values.shape[1])[None, :]
        left = np.maximum.accumulate(np.where(observed, position, -1), axis=1)
        right = np.minimum.accumulate(np.where(observed, position, values.shape[1])[:, ::-1], axis=1)[:, ::-1]
        left_idx, right_idx = left.clip(0, values.shape[1]-1), right.clip(0, values.shape[1]-1)
        clean = np.nan_to_num(values, nan=0.)
        left_values = np.take_along_axis(clean, left_idx[..., None], axis=1)
        right_values = np.take_along_axis(clean, right_idx[..., None], axis=1)
        # Other-log excursions relative to the boundary-to-boundary line encode
        # shape changes inside the gap, where the target cannot be observed.
        fraction = ((position-left) / np.maximum(right-left, 1))[..., None]
        boundary_line = left_values + fraction * (right_values-left_values)
        other_delta = np.delete(clean-boundary_line, target, axis=2)
        count = observed.sum(axis=1, keepdims=True).clip(1)
        mean = clean[..., target].sum(axis=1, keepdims=True) / count
        var = (np.where(observed, (clean[..., target]-mean)**2, 0).sum(axis=1, keepdims=True) / count)
        extra = np.concatenate((baseline[..., target, None], left_values, right_values, other_delta,
                                fraction, (position-left)[..., None], (right-position)[..., None],
                                np.broadcast_to(mean[..., None], (*observed.shape, 1)),
                                np.broadcast_to(np.sqrt(var)[..., None], (*observed.shape, 1))), axis=2)
        return np.column_stack((self._features(values, target), extra.reshape(-1, extra.shape[2]))), baseline[..., target]

    def _fit_residuals(self, values, scenarios):
        self.residual_models = []
        rng = np.random.default_rng(self.config.seed + 40000)
        for target in range(self.config.n_features):
            # Two independent pseudo-gaps per training segment. Training rows
            # contain no hidden target values in any feature.
            masked = np.concatenate((values, values)).copy()
            truth = masked[..., target].copy()
            hidden = np.zeros(truth.shape, dtype=bool)
            for segment in range(len(masked)):
                length = int(rng.choice([1, min(20, values.shape[1]-1), min(100, values.shape[1]-1)]))
                start = int(rng.integers(values.shape[1] - length + 1))
                hidden[segment, start:start+length] = True
            masked[..., target][hidden] = np.nan
            features, baseline = self._residual_features(masked, target)
            take = np.flatnonzero(hidden.reshape(-1))
            if len(take) > self.config.max_train_rows:
                take = rng.choice(take, self.config.max_train_rows, replace=False)
            train_x = features[take]
            train_y = (truth-baseline).reshape(-1)[take]
            # Each segment has equal total weight, as in neural MIT.
            sample_weight = np.broadcast_to(1 / hidden.sum(axis=1, keepdims=True), hidden.shape).reshape(-1)[take]
            sample_weight = sample_weight / sample_weight.mean()
            val_x, val_y = [], []
            for dataset in scenarios.values():
                seen = np.isfinite(dataset['X'][..., target]).any(axis=1)
                selected = dataset['indicating_mask'][..., target].astype(bool) & seen[:, None]
                if not selected.any():
                    continue
                features, baseline = self._residual_features(dataset['X'], target)
                val_x.append(features[selected.reshape(-1)])
                val_y.append((dataset['X_intact'][..., target]-baseline)[selected])
            kwargs = dict(objective='reg:squarederror', tree_method='hist', eval_metric='rmse',
                          device=self.config.device, n_jobs=self.config.n_jobs, n_estimators=self.config.n_estimators,
                          max_depth=self.config.max_depth, learning_rate=self.config.learning_rate,
                          min_child_weight=20, reg_lambda=10, subsample=.85, colsample_bytree=.9,
                          random_state=self.config.seed)
            fit_kwargs = {}
            if val_x:
                kwargs['early_stopping_rounds'] = self.config.early_stopping_rounds
                fit_kwargs['eval_set'] = [(np.concatenate(val_x), np.concatenate(val_y))]
            model = XGBRegressor(**kwargs)
            LOGGER.info('GeoBoost residual log %d/%d (%d rows)', target+1, self.config.n_features, len(take))
            model.fit(train_x, train_y, sample_weight=sample_weight, verbose=False, **fit_kwargs)
            self.residual_models.append(model)
            if val_x:
                self.training_history.extend(dict(log_index=target, component='residual', iteration=i+1,
                                                  validation_rmse=score)
                                             for i, score in enumerate(model.evals_result()['validation_0']['rmse']))

    def _candidates(self, values, target):
        """Yield only affected segments; all candidates retain observed values."""
        active = np.flatnonzero(np.isnan(values[..., target]).any(axis=1))
        if not len(active):
            return
        subset = values[active]
        global_prediction = self.models[target].predict(self._features(subset, target)).reshape(len(active), -1)
        learned_prediction = None
        if getattr(self, 'residual_models', []):
            features, baseline = self._residual_features(subset, target)
            learned_prediction = baseline + self.residual_models[target].predict(features).reshape(baseline.shape)
        positions = np.arange(values.shape[1])
        for local_index, segment in enumerate(active):
            y = values[segment, :, target]
            observed = np.isfinite(y)
            global_y = global_prediction[local_index].astype(float)
            groups = gap_groups(observed)
            if not observed.any():
                yield segment, groups, np.column_stack([global_y] * 4)
                continue
            linear = np.interp(positions, positions[observed], y[observed])
            corrected = global_y + np.interp(positions, positions[observed], (y - global_y)[observed])
            other = np.nan_to_num(np.delete(values[segment], target, axis=1), nan=0.)
            design = np.column_stack((other, global_y))
            center = design[observed].mean(axis=0)
            scale = np.maximum(design[observed].std(axis=0), 0.05)
            design = (design - center) / scale
            residual = y[observed] - global_y[observed]
            offset = residual.mean()
            coefficients = np.linalg.solve(design[observed].T @ design[observed] +
                                           self.config.ridge_alpha * np.eye(design.shape[1]),
                                           design[observed].T @ (residual - offset))
            local = global_y + offset + design @ coefficients
            local += np.interp(positions, positions[observed], (y - local)[observed])
            learned = corrected if learned_prediction is None else learned_prediction[local_index]
            yield segment, groups, np.column_stack((linear, corrected, local, learned))

    def _calibrate(self, scenarios):
        grid = np.array([w for w in itertools.product(range(5), repeat=4) if sum(w) == 4]) / 4
        for target in range(self.config.n_features):
            candidates, targets, groups = [], [], []
            for dataset in scenarios.values():
                for segment, group, predictions in self._candidates(np.asarray(dataset["X"]), target):
                    selected = np.asarray(dataset["indicating_mask"])[segment, :, target].astype(bool)
                    candidates.append(predictions[selected])
                    targets.append(dataset["X_intact"][segment, selected, target])
                    groups.append(group[selected])
            if not candidates:
                continue
            predictions, truth, kind = np.concatenate(candidates), np.concatenate(targets), np.concatenate(groups)
            for group in range(4):
                selected = kind == group
                if not selected.any():
                    continue
                errors = predictions[selected] @ grid.T - truth[selected, None]
                mse = np.mean(errors ** 2, axis=0)
                best = int(np.argmin(mse))
                self.weights[target, group] = grid[best]
                self.calibration.append({"log_index": target, "gap_group": group,
                                         "count": int(selected.sum()), "weights": grid[best].tolist(),
                                         "validation_rmse": float(np.sqrt(mse[best]))})
        LOGGER.info("GeoBoost validation-only correction weights: %s", self.weights.tolist())

    def predict(self, dataset):
        values = np.asarray(dataset["X"], dtype=float)
        result = values.copy()
        for target in range(self.config.n_features):
            for segment, groups, candidates in self._candidates(values, target):
                missing = groups >= 0
                result[segment, missing, target] = np.sum(
                    candidates[missing] * self.weights[target, groups[missing]], axis=1)
        return {"imputation": result}


class GeoBoost(AbstractModel):
    name = "geoboost"

    def __init__(self, config=None):
        super().__init__(config or GeoBoostConfig())

    def _build_backend(self):
        return _GeoBoostBackend(self.config)
