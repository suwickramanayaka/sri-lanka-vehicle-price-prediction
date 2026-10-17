"""Build technical report and reproducible notebook from saved evidence."""
from pathlib import Path
import json
import sys
import pandas as pd
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'reports').mkdir(exist_ok=True)
a=json.loads((ROOT/'reports/data/data_audit.json').read_text())
m=json.loads((ROOT/'models/metadata.json').read_text())
d=json.loads((ROOT/'models/demo.json').read_text())
c=pd.read_csv(ROOT/'reports/metrics/model_comparison.csv')
e=json.loads((ROOT/'reports/data/eda_summary.json').read_text())

def table(df):
    labels={'model':'Model','feature':'Feature','unique_values':'Distinct values','categories_under_3_rows':'Categories with fewer than 3 rows','rows_in_rare_categories':'Rows in rare categories','cv_mae':'Validation MAE (lakhs)','cv_std':'Fold SD (lakhs)','test_mae':'Test MAE (lakhs)','test_rmse':'Test RMSE (lakhs)','test_r2':'Test R²','training_seconds':'Training (s)','prediction_ms_per_row':'Prediction (ms/row)','actual_price_band_lakhs':'Actual price band (lakhs)','count':'Listings','mae_lakhs':'MAE (lakhs)','rmse_lakhs':'RMSE (lakhs)','category':'Model-label coverage'}
    return '| '+' | '.join(labels.get(c,c) for c in df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

report=f'''# Sri Lankan Vehicle Listing Price Prediction

## Technical Report

**Author:** Sithum Wickramanayaka

## Abstract
This study evaluates six regression approaches for predicting historical advertised
vehicle prices in Sri Lanka. The dataset contains {a['raw_rows']:,} listings collected
between December 2024 and February 2025. Training-only exploration informs model
selection, while temporal validation and feature-group purging reduce leakage.
The comparison includes Random Forest, XGBoost, CatBoost, LightGBM, Ridge and a
Random Forest–XGBoost stack with a Ridge meta-model. Random Forest achieves the
lowest validation MAE of {c.iloc[0]['cv_mae']:.2f} lakhs and a test MAE of
{c.iloc[0]['test_mae']:.2f} lakhs. A deployed Streamlit application supports
single-vehicle inference and interactive inspection of the results. Because the
test set was examined during earlier development, the expanded findings are exploratory
and require confirmation using newly collected listings.

## Contents

1. Objective and local relevance
2. Dataset and provenance
3. Exploratory data analysis
4. Data preparation and leakage controls
5. Model selection and preprocessing
6. Evaluation and model justification
7. Error analysis
8. Comparison with existing approaches
9. Single-sample inference and user interface
10. Deployment and repository
11. Limitations and future evaluation
12. Conclusion
13. References

Appendix A provides model-specific results and diagnostic figures.


## 1. Objective and local relevance
Predict historical advertised vehicle prices from tabular listing characteristics.
Buyers and sellers can explore how vehicle characteristics relate to asking prices,
while users can compare models and inspect their limitations.
It does not estimate verified transaction prices or current 2026 market values.

## 2. Dataset and provenance
[Prasad Nirmal, SriLankan Vehicle(Car) Price Dataset, version 3](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset).
Publisher metadata identifies Ikman.lk and Riyasewana.com as sources and lists MIT
licensing. Original CSV and metadata are retained in `data/raw`.
SHA-256: `{m['raw_sha256']}`.
There are {a['raw_rows']:,} source rows, spanning {a['date_min']} to {a['date_max']}.
Price is in LKR lakhs, confirmed by the project owner; 1 lakh is LKR 100,000.

## 3. Exploratory data analysis
Exploration in `scripts/analyze_data.py` uses only the training partition.
The training price median is {e['price_median']:.1f} lakhs, 95th percentile
{e['price_p95']:.1f}, maximum {e['price_max']:.1f}, and skewness {e['price_skew']:.2f}.
The long upper tail motivates reporting RMSE and price-band errors alongside MAE.
Extreme observations remain in evaluation rather than being removed to improve scores.

{table(pd.read_csv(ROOT/'reports/data/categorical_profile.csv'))}

Many model labels are rare, motivating native categorical handling and regularized
one-hot alternatives. Year/price Spearman association is
{e['numeric_spearman']['year']['price']:.2f}; engine capacity/price association is
{e['numeric_spearman']['engine_cc']['price']:.2f}. These are marginal associations,
not causal effects or sufficient reasons to remove an interacting feature.

![Training-only exploration](figures/training_eda.png)

## 4. Data preparation and leakage controls
{a['duplicates_removed']} exact duplicate rows were removed after category trimming,
and {a['invalid_rows']} rows failed target/date/year checks. No literal source nulls
were found, but the publisher already preprocessed the table; missing original
observations and prior imputation cannot be reconstructed.
Mileage equals `(2025 - year) * 11000` throughout the source and is excluded.
The exported index, date, target and group identifiers are not predictors.

The latest record is reserved for a single-example demo, excluding its feature
signature from development. Test dates start on {a['test_cutoff']}. Matching feature
groups are removed from earlier training data ({a['purged_train_rows']} rows).
There are {a['train_rows']:,} training, {a['test_rows']:,} test and one demo row.
Feature signatures are a conservative proxy; without original listing IDs they
cannot establish true vehicle identity. Three expanding temporal folds are used
for parameter search, with matching validation groups purged from each training fold.

## 5. Model selection and preprocessing

- **Random Forest:** bagging reference for nonlinear interactions and variance reduction.
- **XGBoost:** regularized sequential boosting for nonlinear interactions.
- **CatBoost:** native categorical processing is relevant to rare vehicle model labels.
- **LightGBM:** an efficient histogram-based boosting alternative with bounded leaf growth.
- **Ridge:** additive linear comparison to test whether tree complexity adds value.
- **Stacking:** fixed RF and XGBoost pipelines combined through scaled Ridge regression.
- **Training median:** no-feature reference benchmark, not a selectable app model.

All preprocessing is fitted within folds. RF, XGBoost, LightGBM and Ridge use
numeric median and categorical mode imputation plus one-hot encoding with rare
category pooling. Ridge scales numeric inputs; tree models do not require scaling.
CatBoost receives string categories with a missing marker and fold-fitted numeric
median imputation. No globally fitted target encoder is used.

RF, XGBoost, CatBoost and LightGBM each compare three explicitly constrained
configurations. Ridge compares alpha 1, 10 and 100. Actual settings and every search
result are in `experiments.csv`; bounded searches are not exhaustive optimization.
CatBoost's adapter supplies categorical names during fit for sklearn compatibility.
LightGBM uses ordinary GBDT with row sampling, not a replication of GOSS.
Stacking uses inner purged temporal out-of-fold predictions to train its Ridge
meta-model (alpha 10), then refits bases on the training partition. Outer temporal
folds evaluate the complete stack. Its bases are fixed, not selected with outer
validation labels. Earliest rows without out-of-fold predictions do not train Ridge.

## 6. Evaluation and model justification
Mean validation MAE is the selection rule fixed before the expanded run. MAE
expresses average monetary error directly; RMSE emphasizes large mistakes and R²
is unitless, not percentage accuracy. Fold standard deviation is descriptive,
not a confidence interval. Timing includes parameter search and refitting on this
machine; hardware-specific timings are not published-paper speed comparisons.

### Predictive performance

{table(c[['model','cv_mae','cv_std','test_mae','test_rmse','test_r2']])}

### Computational performance

{table(c[['model','training_seconds','prediction_ms_per_row']])}

{(ROOT/'reports/interpretations/model_selection.md').read_text()}

## 7. Error analysis
The band table below uses actual target prices for retrospective analysis only;
those bands are not available when making a new prediction. Counts matter:
small subgroups cannot establish reliable group-level performance.

{table(pd.read_csv(ROOT/'reports/errors/errors_by_price_band.csv'))}

{table(pd.read_csv(ROOT/'reports/errors/errors_by_category_coverage.csv'))}

Per-brand counts and MAE for every model are saved in `errors_by_brand_all_models.csv`.
Permutation importance measures how much test MAE changes when an input is shuffled.
It explains global reliance, not causation or the reason for a single prediction.
Correlated inputs can share importance; shuffling may create implausible vehicles.

## 8. Comparison with existing approaches
{(ROOT/'docs/related_work.md').read_text().split('## References',1)[0].split(chr(10),1)[1].strip()}

## 9. Single-sample inference and user interface
Reserved source row {d['row_id']}: actual {d['actual']:.3f} lakhs, selected-model
prediction {d['prediction']:.3f}, absolute error {abs(d['actual']-d['prediction']):.3f}.
All exported estimators are reloaded and checked against their in-memory prediction.
The Streamlit interface validates fields, switches models, shows estimates in LKR,
and presents model-specific importance, diagnostics, comparison and optional data exploration.

## 10. Deployment and repository

The application is deployed on **Streamlit Community Cloud** and uses the GitHub
repository as its deployment source. The entry point is `app/app.py`. Saved model
pipelines and their metadata support inference without retraining when a user
changes the selected model.

- **Deployed application:** [Sri Lankan Vehicle Price Prediction on Streamlit](https://sri-lanka-vehicle-price-prediction.streamlit.app/)
- **Source repository:** [Sri Lankan Vehicle Price Prediction on GitHub](https://github.com/suwickramanayaka/sri-lanka-vehicle-price-prediction)

Users enter vehicle characteristics, choose one of the six trained models and
request an estimated listing price in LKR. The interface also provides model
comparison metrics, permutation importance, prediction diagnostics and an optional
explorer for the raw and processed datasets. Model switching updates the relevant
results and explanations. Estimates describe the historical listing data used in
this study and should not be interpreted as verified transaction prices.

The repository contains the application, preprocessing and training code, saved
model artifacts, evaluation evidence and the analysis notebook. The README
provides installation and reproduction instructions. The interface specification
is available in `docs/interface.md`.

The deployment verification on 15 October 2026 confirmed that the application
loaded all six model options and comparison charts. Live inference checks for
CatBoost, LightGBM and Ridge returned price estimates. Local verification also
passed all 28 automated tests, including split isolation, model loading,
model switching and stacking checks. These checks establish implementation
functionality rather than independent predictive validity.

Access follows the application's and repository's sharing settings. Where either
resource remains private, the reviewer requires authorized access.


## 11. Limitations and future evaluation
The test set was already examined during earlier development. New models are exploratory additions,
not independently confirmed improvements. Resampling previously examined rows cannot provide an independent test. Future
evaluation should use later listings with verified units, observed mileage, vehicle
identifiers and, where available, sale prices. The model choice should be frozen
before evaluation, with duplicate vehicle identities excluded across partitions.
Data covers only about two months, is publisher-preprocessed, and may not reflect
later import policies, exchange rates or supply changes. No causal, fairness,
statistical-superiority or current-market accuracy claim is made.
No deep learning, computer vision or reinforcement learning is used.

## 12. Conclusion
The study demonstrates an end-to-end workflow for historical vehicle listing-price
regression, from data preparation and temporal evaluation to a deployed inference
interface. Random Forest is the preferred model under the predefined validation-MAE
criterion. XGBoost achieves a slightly lower test RMSE, while stacking does not
improve validation MAE over Random Forest. The expanded comparison therefore
supports evidence-based selection rather than assuming that additional model
complexity produces better predictions.

Performance varies substantially across price bands and previously unseen vehicle
model labels. Reliable evaluation on new listings, with verified vehicle identities
and source measurements, is the principal next step before broader practical use.

## 13. References
{(ROOT/'docs/related_work.md').read_text().split('## References',1)[1].strip()}

## Appendix A. Model-specific strengths, limitations and results
{(ROOT/'reports/interpretations/results_interpretation.md').read_text().replace('## Interpreting the model comparison', '### Overview').replace('](../figures/', '](figures/')}
'''
(ROOT/'reports/final_report.md').write_text(report.rstrip()+'\n')
if '--report-only' in sys.argv:
    print('Technical report updated')
    raise SystemExit(0)
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
author_details = '# Sri Lankan Vehicle Listing Price Prediction\n# Author: Sithum Wickramanayaka\n# Degree: MSc in Artificial Intelligence\n# University: University of Moratuwa\n# Student ID: 269214N\n# Batch: 19\n# Module: IT5514 - Applied Machine Learning\n'
nb.cells=[
 nbf.v4.new_markdown_cell('# Sri Lankan Vehicle Listing Price Prediction\nTraining-only exploration, expanded tabular model comparison, interpretation and inference. Values are in lakhs unless labelled LKR. The existing test split is exploratory.'),
 nbf.v4.new_code_cell(author_details),
 nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nfrom src.data import load_data, split_data, FEATURES\ndf, audit = load_data()\ntrain, test, demo, cutoff, purged = split_data(df)\ndisplay(pd.Series(audit))\ndisplay(train.head())"),
 nbf.v4.new_markdown_cell('## Training-only exploration and candidate rationale'),
 nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'reports/data/numeric_profile.csv'))\ndisplay(pd.read_csv(ROOT/'reports/data/categorical_profile.csv'))\ndisplay(Image(filename=str(ROOT/'reports/figures/training_eda.png')))\ndisplay(Markdown((ROOT/'docs/experiment_plan.md').read_text()))"),
 nbf.v4.new_markdown_cell('## Reproduce model fitting\nSet RETRAIN=True to rerun the frozen search. No test-driven tuning.'),
 nbf.v4.new_code_cell("RETRAIN=False\nif RETRAIN:\n    from scripts.analyze_data import main as analyze\n    from src.train import main as train_models\n    from scripts.summarize_results import main as summarize\n    analyze()\n    train_models()\n    summarize()\ndisplay(pd.read_csv(ROOT/'reports/metrics/model_comparison.csv'))\ndisplay(pd.read_csv(ROOT/'reports/metrics/experiments.csv'))\ndisplay(pd.read_csv(ROOT/'reports/metrics/validation_folds.csv'))\ndisplay(pd.Series(json.loads((ROOT/'reports/metrics/baseline.json').read_text())))"),
 nbf.v4.new_markdown_cell('## Justification and subgroup errors'),
 nbf.v4.new_code_cell("display(Markdown((ROOT/'reports/interpretations/model_selection.md').read_text()))\ndisplay(pd.read_csv(ROOT/'reports/errors/errors_by_price_band.csv'))\ndisplay(pd.read_csv(ROOT/'reports/errors/errors_by_category_coverage.csv'))\ndisplay(Markdown((ROOT/'docs/related_work.md').read_text()))"),
 nbf.v4.new_markdown_cell('## Model-specific diagnostics and interpretations'),
 nbf.v4.new_code_cell("metadata=json.loads((ROOT/'models/metadata.json').read_text())\nfor name in metadata['models']:\n    slug=name.lower().replace(' ','_')\n    display(Markdown((ROOT/f'reports/interpretations/interpretation_{slug}.md').read_text()))\n    display(Image(filename=str(ROOT/f'reports/figures/diagnostics_{slug}.png')))"),
 nbf.v4.new_markdown_cell('## Reload all models and infer the reserved sample'),
 nbf.v4.new_code_cell("from src.predict import predict, load_artifacts\nexample=json.loads((ROOT/'models/demo.json').read_text())\ndisplay(example['input'])\nfor name in metadata['models']:\n    display(predict(example['input'],*load_artifacts(name)))\nprint('Actual price (lakhs):',example['actual'])"),
 nbf.v4.new_markdown_cell('## Application and reproducibility\nRun `streamlit run app/app.py`. See README for the deployed application, reproduction commands and access information.')]
nbf.write(nb,ROOT/'notebooks/vehicle_price_analysis.ipynb')
print('Technical report and notebook created')
