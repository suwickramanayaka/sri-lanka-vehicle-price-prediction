"""Model rationale and interpretation shared by reports and the interface."""
MODEL_NOTES = {
    'Random Forest': (
        ['Averages many trees to reduce prediction variance.', 'Captures nonlinear relationships and interactions.'],
        ['Rare vehicles can have large errors.', 'Cannot reliably extrapolate beyond training prices.']),
    'XGBoost': (
        ['Successive trees correct earlier errors.', 'Regularization controls complex nonlinear patterns.'],
        ['Sensitive to parameter choices.', 'Rare categories and market changes can still cause large errors.']),
    'CatBoost': (
        ['Uses categorical fields directly, without external one-hot encoding.', 'Can capture patterns across many vehicle model categories.'],
        ['Rare categories still have limited supporting evidence.', 'Categorical statistics must be learned inside training folds.']),
    'LightGBM': (
        ['Histogram-based boosting is an efficient tabular candidate.', 'Leaf-wise trees capture nonlinear relationships.'],
        ['Leaf growth can overfit small groups without constraints.', 'This implementation uses one-hot categories; unseen labels provide limited information.']),
    'Ridge': (
        ['Regularized linear reference for measuring the benefit of tree models.', 'Simple additive structure and relatively inexpensive inference.'],
        ['Does not automatically learn nonlinear interactions.', 'Extrapolation can produce unrealistic or negative prices.']),
    'Stacking': (
        ['Combines Random Forest and XGBoost predictions using Ridge.', 'Meta-model learns from later out-of-fold predictions.'],
        ['Similar base-model errors can limit improvement.', 'More training work; early rows are excluded from meta-training.']),
}
