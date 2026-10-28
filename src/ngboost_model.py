"""NGBoost adapter for scikit-learn pipeline fitted-state checks."""
from ngboost import NGBRegressor


class PipelineNGBoostRegressor(NGBRegressor):
    def __sklearn_is_fitted__(self):
        # NGBoost stores fitted trees without sklearn's trailing-underscore convention.
        # Expose the actual fitted state rather than allowing an unfitted pipeline through.
        return bool(getattr(self, 'base_models', []))
