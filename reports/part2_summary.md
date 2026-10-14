### Model choice and evidence

**Random Forest** is selected because it has the lowest mean validation MAE
(**7.15 lakhs**) among the evaluated configurations. Test MAE is
**5.83 lakhs**, compared with **28.67** for a
training-median prediction (79.7% lower error on this historical split).
This demonstrates improvement over a simple local benchmark, not over external studies.

The training data has **1,061 model labels**, including **758** occurring fewer
than three times. Median training price is **42.5 lakhs** and
95th percentile **120.0 lakhs**. CatBoost tests native categorical
handling; LightGBM tests another efficient nonlinear approach; Ridge provides a
simpler additive reference. RF, XGBoost and stacking retain the Part 1 comparisons.

Selection uses MAE because it measures average monetary error directly. RMSE
reveals the influence of large mistakes; R² is not percentage accuracy. Runtime
and fold variation are supporting evidence, not alternate post-hoc selection rules.
A more complex model is not automatically better. Error by actual-price band and
unseen model label must be considered before trusting an individual estimate.

**Limit:** this test split was already inspected in Part 1. Part 2 results are
exploratory. Newly collected future listings are required for independent confirmation.
The source covers December 2024–February 2025; estimates are historical asking
prices rather than current valuations or verified sale prices.
