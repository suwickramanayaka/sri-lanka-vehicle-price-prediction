### NGBoost: probabilistic vehicle-price regression

NGBoost is the principal method studied in this extension. It uses shallow trees
and natural-gradient updates to learn the mean and scale of a conditional Normal
price distribution. Its training objective is negative log-likelihood; configuration
selection uses validation MAE so point predictions remain comparable to the other models.
Unlike Random Forest's averaging or conventional point-prediction boosting, it
updates multiple distribution parameters using a geometry-adjusted gradient.
This is a non-deep-learning method. See [Duan et al. (2020)](https://proceedings.mlr.press/v119/duan20a.html).

**Design.** Fold-fitted median/mode imputation and dense one-hot encoding with rare
category pooling; depth-3 base trees, minimum leaf size 10, learning rate 0.03,
seed 42, and 100/200/300 rounds. Selected settings: `{"model__n_estimators": 300}`.
Normal is an explicit simplifying assumption, not evidence that car prices are Normal.

**Measured performance.** Validation MAE 9.43 lakhs; test MAE
8.58 lakhs; RMSE 16.45 lakhs; R² 0.861.
The validation-selected recommendation is Random Forest. The research focus
and the best point-prediction model need not be the same.

**Global explanation.** The four strongest permutation-importance fields are
year, brand, engine cc, gear. Importance is the increase in test MAE after shuffling; it describes
model reliance rather than causation or a direction of effect.

**Partial-dependence interpretation.** Manufacture year: the average model prediction changes from 27.28 to 94.33 lakhs across 1986–2018. Engine capacity (cc): the average model prediction changes from 48.66 to 84.69 lakhs across 650–2000.
Newer vehicles often have higher asking prices, and engine capacity can track
vehicle class. The plotted response should be compared with these expectations;
non-monotonic sections may reflect model interactions, sampling and sparse data.
The curves average over 256 training vehicles and cover the training 5th–95th
percentiles. Varying one feature can create implausible brand/model/year/engine
combinations. They are not causal depreciation curves or individual explanations.

**Distribution check.** Test negative log-likelihood is 3.741
(in lakh units). Nominal 90% intervals contain 89.9%
of observed test prices, with mean width 33.61 lakhs.
2.4% have negative lower bounds.
Coverage must be considered alongside width: a wide range can cover many prices
without being useful. These intervals have not been independently calibrated and
are not displayed as guaranteed price ranges in the prediction form. Negative
bounds expose a limitation of the Normal assumption for positive prices.

**Evidence limits.** The test set was examined before this extension; results are
exploratory. No test-driven retuning was performed. New, later listings are needed
for independent evaluation and calibration. Better uncertainty or point accuracy
than existing models is not assumed.
