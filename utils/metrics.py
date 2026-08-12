"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

import numpy as np
from sklearn import metrics

def cal_r2(class_predictions, targets, masks):
    '''
    Calculate the R-squared Error between ``class_predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    @param class_predictions: The prediction data to be evaluated.
    @param targets: The target data for helping evaluate the predictions.
    @param masks: The masks for filtering the specific values in inputs and target from evaluation.
                  When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.
    '''

    mask = masks.astype(bool)
    return metrics.r2_score(targets[mask], class_predictions[mask])

def cal_cc(class_predictions, targets, masks):
    '''
    Calculate the Pearson correlation coefficient between ``class_predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    @param class_predictions: The prediction data to be evaluated.
    @param targets: The target data for helping evaluate the predictions.
    @param masks: The masks for filtering the specific values in inputs and target from evaluation.
                  When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.
    '''

    mask = masks.astype(bool)
    class_predictions = class_predictions[mask]
    targets = targets[mask]

    # Pearson correlation is undefined for fewer than two values or a constant
    # vector. Treat that degenerate prediction as no correlation instead of
    # propagating NaN through all fold summaries.
    if class_predictions.size < 2:
        return 0.0
    if np.isclose(np.std(class_predictions), 0) or np.isclose(np.std(targets), 0):
        return 0.0

    return float(np.corrcoef(targets, class_predictions)[0, 1])
