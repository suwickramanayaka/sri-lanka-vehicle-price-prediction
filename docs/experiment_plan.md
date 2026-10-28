# Experiment plan

Prepared 14 October 2026, before running the expanded candidate comparison.

Problem: predict historical Sri Lankan advertised vehicle prices from tabular fields.
The study uses a fixed source dataset and temporal split to compare tabular regression approaches.

Training-only exploration shows 1,061 vehicle model labels, 758 occurring fewer
than three times, price skewness 3.80 and a median of 42.5 lakhs versus a maximum
of 790 lakhs. The year/price Spearman association is 0.65. These observations
motivate categorical-aware models, nonlinear candidates and scrutiny of tail errors.
They do not establish causality.

## Fixed evaluation decisions

- Preserve the established temporal split and purged feature groups; no random reshuffle.
- Compare Random Forest, XGBoost, CatBoost, LightGBM, Ridge and RF/XGBoost stacking.
- Include the training-median prediction as a non-deployable reference benchmark.
- Use three bounded configurations per standalone model. Do not expand the search
  after reading the new test results. Stacking retains its fixed base configurations.
- Select the smallest mean temporal validation MAE. Also report fold variation,
  test MAE/RMSE/R², runtime, errors by brand and actual-price band.
- Price-band boundaries: 0–25, 25–50, 50–100, 100–200 and above 200 lakhs.
- Keep extreme prices for evaluation; do not delete hard examples to improve scores.
- CatBoost uses native categorical handling. Other standalone models use
  fold-fitted imputation and one-hot encoding; Ridge additionally scales numeric inputs.
- No target encoding, preprocessing fit, parameter search or meta-model fitting
  may include validation/test target values outside its corresponding training partition.

## Evidence limits

The existing test set was inspected during earlier model and stacking development.
Expanded results are exploratory, even though the new selection uses validation
only. A truly independent confirmation needs newly collected future listings;
repartitioning these known rows would not make them unseen again. No claims of
statistical superiority or present-day valuation accuracy are made.

## NGBoost extension - frozen 27 October 2026

NGBoost is the principal new-method study; existing models remain benchmarks.
No NGBoost or natural-gradient entry was found in the searchable lecture material.
This is a document check, not confirmation of unrecorded classroom discussion.

Before fitting, fix a Normal conditional distribution and log-score objective,
natural gradients, depth-3 trees with minimum leaf size 10, learning rate 0.03,
seed 42, and candidates of 100, 200 and 300 boosting rounds. Select using the
same three purged temporal folds and MAE rule. No test-driven expansion of this
search is permitted. Normal is a transparent first distributional assumption;
its limitations for positive, skewed prices must be reported rather than hidden.
NGBoost receives fold-fitted dense one-hot inputs. Other estimators keep their
existing settings. Refit all artifacts in the updated dependency environment.

Evaluate point MAE/RMSE/R² and runtime. Additionally report test negative
log-likelihood and empirical 90% interval coverage/width without using them to
retune. Intervals are model-based, uncalibrated and not a current-market guarantee.
Use permutation importance and training-background partial dependence for year
and engine capacity. Keep the established test-exposure limitation explicit.
