"""Fold-fitted preprocessing that preserves native categorical column names."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.data import FEATURES, NUMERIC, CATEGORICAL


class NativeCategories(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.medians_ = X[NUMERIC].median().fillna(0)
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        frame = X[FEATURES].copy()
        frame[NUMERIC] = frame[NUMERIC].fillna(self.medians_).astype(float)
        for c in CATEGORICAL:
            frame[c] = frame[c].fillna('__MISSING__').astype(str)
        return frame


from sklearn.base import RegressorMixin


class NativeCatBoostRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-cloneable adapter; categorical names are supplied only during fit."""
    def __init__(self, iterations=350, depth=6, learning_rate=.05, l2_leaf_reg=5,
                 random_seed=42, thread_count=2):
        self.iterations=iterations
        self.depth=depth
        self.learning_rate=learning_rate
        self.l2_leaf_reg=l2_leaf_reg
        self.random_seed=random_seed
        self.thread_count=thread_count

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_=CatBoostRegressor(**self.get_params(),loss_function='RMSE',
                                     verbose=False,allow_writing_files=False)
        self.model_.fit(X,y,cat_features=CATEGORICAL)
        return self

    def predict(self, X):
        check_is_fitted(self,'model_')
        return self.model_.predict(X)
