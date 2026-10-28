### NGBoost: strengths and limitations

**Strengths.** Learns both location and scale of a price distribution using natural gradients. Provides a nonlinear probabilistic comparison to point-prediction models.

**Limitations.** Normal-distribution assumptions may poorly describe skewed prices and outliers. Uncertainty can be miscalibrated; negative estimates and interval bounds are possible.

**Measured results.** Validation MAE: **9.43 lakhs**; test MAE: **8.58 lakhs**; test RMSE: **16.45 lakhs**; test R²: **0.861**. Training MAE was **8.20 lakhs**. Lower MAE/RMSE is better; higher R² is better. R² is unitless, not percentage accuracy.

**Feature-importance chart.** The strongest fields are **year, brand, engine cc, gear**, in that order. Longer bars mean shuffling that feature increased test error more. This is overall model reliance, not proof of causation, the direction of an effect, or an explanation for a single car. Correlated fields can share importance, and shuffling can produce unrealistic combinations.

**Actual-versus-predicted graph.** Points on the dashed diagonal are correct predictions. Points below it are underestimates; points above it are overestimates. Each point represents one unseen test listing.

**Residual graph.** Residual = actual price − predicted price. Positive values mean underestimation; negative values mean overestimation. A tight cloud around zero is desirable. The mean signed residual is **-0.19 lakhs**; the largest absolute test error is **313.17 lakhs**. These describe the whole test set, not a guaranteed range for the current vehicle.

**What the graphs show here.** Most listings form a dense cluster at the lower end of the price range, broadly following the diagonal. The diagnostics show the direction and size of errors for this model. Points far from the diagonal or zero line identify large errors; no model is guaranteed reliable across all vehicles. The plots alone do not establish why those listings were difficult.

**Large-error interpretation.** RMSE (16.45) is much larger than MAE (8.58), consistent with some large errors having a strong effect on the squared-error metric. Inspect isolated points far from the diagonal/zero line; an average score can hide these failures.
