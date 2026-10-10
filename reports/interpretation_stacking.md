### Stacking: strengths and limitations

**Strengths.** Combines fixed Random Forest and XGBoost pipelines with a scaled Ridge meta-model. Meta-training predictions come from purged, expanding temporal folds; base pipelines are then refitted on all training rows.

**Limitations.** Correlated base errors can limit gains. Training and inference cost more; early rows without out-of-fold predictions are excluded from meta-training. Stacking is not guaranteed to improve on either model. It was added after earlier test results were inspected, so this is an exploratory comparison on an existing holdout.

**Measured results.** Validation MAE: **7.30 lakhs**; test MAE: **6.47 lakhs**; test RMSE: **14.41 lakhs**; test R²: **0.894**. Training MAE was **4.78 lakhs**. Lower MAE/RMSE is better; higher R² is better. R² is unitless, not percentage accuracy.

**Feature-importance chart.** The strongest fields are **year, brand, engine cc, gear**, in that order. Longer bars mean shuffling that feature increased test error more. This is overall model reliance, not proof of causation, the direction of an effect, or an explanation for a single car. Correlated fields can share importance, and shuffling can produce unrealistic combinations.

**Actual-versus-predicted graph.** Points on the dashed diagonal are correct predictions. Points below it are underestimates; points above it are overestimates. Each point represents one unseen test listing.

**Residual graph.** Residual = actual price − predicted price. Positive values mean underestimation; negative values mean overestimation. A tight cloud around zero is desirable. The mean signed residual is **-0.90 lakhs**; the largest absolute test error is **296.06 lakhs**. These describe the whole test set, not a guaranteed range for the current vehicle.

**What the graphs show here.** Most listings form a dense cluster at the lower end of the price range, broadly following the diagonal. Several high-priced listings fall well below the diagonal, indicating substantial underestimation. The residual plot has isolated large positive and negative errors, so neither model is uniformly reliable across all vehicles. The plots alone do not establish why those listings were difficult.

**Large-error interpretation.** RMSE (14.41) is much larger than MAE (6.47), consistent with some large errors having a strong effect on the squared-error metric. Inspect isolated points far from the diagonal/zero line; an average score can hide these failures.
