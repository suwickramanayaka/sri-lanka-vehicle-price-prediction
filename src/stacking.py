"""Temporal out-of-fold stacking with a regularized linear meta-model."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from src.data import temporal_folds


class TemporalStackingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators, alpha=10.0):
        self.estimators = estimators
        self.alpha = alpha

    def fit(self, X, y, *, dates, groups):
        context = pd.DataFrame({'date': np.asarray(dates), 'group': np.asarray(groups)})
        target = np.asarray(y)
        # Each column holds one base model's predictions for later, unseen fold rows.
        oof = np.full((len(X), len(self.estimators)), np.nan)
        self.fold_audit_ = []
        for tr, va in temporal_folds(context):
            if not len(tr):
                raise ValueError('No training rows remain after temporal purging')
            self.fold_audit_.append({'train': tr.tolist(), 'validation': va.tolist()})
            for j, (_, estimator) in enumerate(self.estimators):
                fitted = clone(estimator).fit(X.iloc[tr], target[tr])
                oof[va, j] = fitted.predict(X.iloc[va])
        # Early rows have no preceding training fold and cannot train the meta-model.
        usable = np.isfinite(oof).all(axis=1)
        self.meta_rows_ = np.flatnonzero(usable)
        self.oof_predictions_ = oof[usable]
        self.meta_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.meta_model_.fit(self.oof_predictions_, target[usable])
        # Refit bases on all supplied training rows only after fitting Ridge on OOF predictions.
        self.estimators_ = [(name, clone(est).fit(X, target)) for name, est in self.estimators]
        return self

    def predict(self, X):
        check_is_fitted(self, 'meta_model_')
        predictions = np.column_stack([est.predict(X) for _, est in self.estimators_])
        return self.meta_model_.predict(predictions)
