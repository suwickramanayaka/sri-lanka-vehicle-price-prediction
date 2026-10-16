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
