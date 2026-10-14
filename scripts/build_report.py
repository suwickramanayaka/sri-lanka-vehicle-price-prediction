"""Build Part 2 report and reproducible notebook from saved evidence."""
from pathlib import Path
import json
import pandas as pd
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'reports/data_audit.json').read_text())
m=json.loads((ROOT/'models/metadata.json').read_text())
d=json.loads((ROOT/'models/demo.json').read_text())
c=pd.read_csv(ROOT/'reports/model_comparison.csv')
e=json.loads((ROOT/'reports/eda_summary.json').read_text())

def table(df):
    return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

report=f'''# Part 2 — Sri Lankan Vehicle Listing Price Prediction

## Objective and local relevance
Predict historical advertised vehicle prices from tabular listing characteristics.
Buyers and sellers can explore how vehicle characteristics relate to asking prices,
while reviewers can compare models and inspect their limitations. This extends
Part 1 into a broader model-selection study, as permitted by the guideline.
It does not estimate verified transaction prices or current 2026 market values.

## Source and provenance
[Prasad Nirmal, SriLankan Vehicle(Car) Price Dataset, version 3](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset).
Publisher metadata identifies Ikman.lk and Riyasewana.com as sources and lists MIT
licensing. Original CSV and metadata are retained in `data/raw`.
SHA-256: `{m['raw_sha256']}`.
There are {a['raw_rows']:,} source rows, spanning {a['date_min']} to {a['date_max']}.
Price is in LKR lakhs, confirmed by the project owner; 1 lakh is LKR 100,000.

## Data analysis before model selection
Exploration in `scripts/analyze_data.py` uses only the training partition.
The training price median is {e['price_median']:.1f} lakhs, 95th percentile
{e['price_p95']:.1f}, maximum {e['price_max']:.1f}, and skewness {e['price_skew']:.2f}.
The long upper tail motivates reporting RMSE and price-band errors alongside MAE.
Extreme observations remain in evaluation rather than being removed to improve scores.

{table(pd.read_csv(ROOT/'reports/categorical_profile.csv'))}

Many model labels are rare, motivating native categorical handling and regularized
one-hot alternatives. Year/price Spearman association is
{e['numeric_spearman']['year']['price']:.2f}; engine capacity/price association is
{e['numeric_spearman']['engine_cc']['price']:.2f}. These are marginal associations,
not causal effects or sufficient reasons to remove an interacting feature.

![Training-only exploration](figures/training_eda.png)

## Cleaning and leakage controls
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

## Candidate selection and preprocessing
- **Random Forest:** bagging reference for nonlinear interactions and variance reduction.
- **XGBoost:** regularized sequential boosting retained from Part 1.
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

## Evaluation and justification
Mean validation MAE is the selection rule fixed before the expanded run. MAE
expresses average monetary error directly; RMSE emphasizes large mistakes and R²
is unitless, not percentage accuracy. Fold standard deviation is descriptive,
not a confidence interval. Timing includes parameter search and refitting on this
machine; hardware-specific timings are not published-paper speed comparisons.

{table(c[['model','cv_mae','cv_std','test_mae','test_rmse','test_r2','training_seconds','prediction_ms_per_row']])}

{(ROOT/'reports/part2_summary.md').read_text()}

## Error analysis
The band table below uses actual target prices for retrospective analysis only;
those bands are not available when making a new prediction. Counts matter:
small subgroups cannot establish reliable group-level performance.

{table(pd.read_csv(ROOT/'reports/errors_by_price_band.csv'))}

{table(pd.read_csv(ROOT/'reports/errors_by_category_coverage.csv'))}

Per-brand counts and MAE for every model are saved in `errors_by_brand_all_models.csv`.
Permutation importance measures how much test MAE changes when an input is shuffled.
It explains global reliance, not causation or the reason for a single prediction.
Correlated inputs can share importance; shuffling may create implausible vehicles.

## Comparison with existing approaches
{(ROOT/'docs/related_work.md').read_text()}

## Single-sample inference and interface
Reserved source row {d['row_id']}: actual {d['actual']:.3f} lakhs, selected-model
prediction {d['prediction']:.3f}, absolute error {abs(d['actual']-d['prediction']):.3f}.
All exported estimators are reloaded and checked against their in-memory prediction.
The Streamlit interface validates fields, switches models, shows estimates in LKR,
and presents model-specific importance, diagnostics, comparison and optional data exploration.

[Private live app](https://sri-lanka-vehicle-price-prediction.streamlit.app/) ·
[Private repository](https://github.com/suwickramanayaka/sri-lanka-vehicle-price-prediction)

Reviewer access must be granted separately for private app/repository links.
The interface specification is in `docs/interface.md`. Reproduction commands are in README.

## Limitations and next evaluation
The test set was already examined in Part 1. New models are exploratory additions,
not independently confirmed improvements. Do not reshuffle familiar rows and call
them a new unseen set. Collect later listings with verified units, real mileage,
vehicle IDs and sale prices where possible; freeze the candidate choice, deduplicate
by vehicle identity, then evaluate the frozen model on that new period.
Data covers only about two months, is publisher-preprocessed, and may not reflect
later import policies, exchange rates or supply changes. No causal, fairness,
statistical-superiority or current-market accuracy claim is made.
No deep learning, computer vision or reinforcement learning is used.

## Submission status
Implementation, report and notebook are prepared. Moodle submission is not performed.
Confirm the lecturer's actual deadline (the guideline says tentatively 18 October),
provide reviewer access, review and explain the work, and submit individually.

## Model-specific strengths, limitations and results
{(ROOT/'reports/results_interpretation.md').read_text()}
'''
(ROOT/'reports/final_report.md').write_text(report.rstrip()+'\n')
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[
 nbf.v4.new_markdown_cell('# Part 2 — Sri Lankan Vehicle Listing Price Prediction\nTraining-only exploration, expanded tabular model comparison, interpretation and inference. Values are in lakhs unless labelled LKR. The existing test split is exploratory.'),
 nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nfrom src.data import load_data, split_data, FEATURES\ndf, audit = load_data()\ntrain, test, demo, cutoff, purged = split_data(df)\ndisplay(pd.Series(audit))\ndisplay(train.head())"),
 nbf.v4.new_markdown_cell('## Training-only exploration and candidate rationale'),
 nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'reports/numeric_profile.csv'))\ndisplay(pd.read_csv(ROOT/'reports/categorical_profile.csv'))\ndisplay(Image(filename=str(ROOT/'reports/figures/training_eda.png')))\ndisplay(Markdown((ROOT/'docs/part2_experiment_plan.md').read_text()))"),
 nbf.v4.new_markdown_cell('## Reproduce model fitting\nSet RETRAIN=True to rerun the frozen search. No test-driven tuning.'),
 nbf.v4.new_code_cell("RETRAIN=False\nif RETRAIN:\n    from scripts.analyze_data import main as analyze\n    from src.train import main as train_models\n    from scripts.part2_results import main as summarize\n    analyze()\n    train_models()\n    summarize()\ndisplay(pd.read_csv(ROOT/'reports/model_comparison.csv'))\ndisplay(pd.read_csv(ROOT/'reports/experiments.csv'))\ndisplay(pd.read_csv(ROOT/'reports/validation_folds.csv'))\ndisplay(pd.Series(json.loads((ROOT/'reports/baseline.json').read_text())))"),
 nbf.v4.new_markdown_cell('## Justification and subgroup errors'),
 nbf.v4.new_code_cell("display(Markdown((ROOT/'reports/part2_summary.md').read_text()))\ndisplay(pd.read_csv(ROOT/'reports/errors_by_price_band.csv'))\ndisplay(pd.read_csv(ROOT/'reports/errors_by_category_coverage.csv'))\ndisplay(Markdown((ROOT/'docs/related_work.md').read_text()))"),
 nbf.v4.new_markdown_cell('## Model-specific diagnostics and interpretations'),
 nbf.v4.new_code_cell("metadata=json.loads((ROOT/'models/metadata.json').read_text())\nfor name in metadata['models']:\n    slug=name.lower().replace(' ','_')\n    display(Markdown((ROOT/f'reports/interpretation_{slug}.md').read_text()))\n    display(Image(filename=str(ROOT/f'reports/figures/diagnostics_{slug}.png')))"),
 nbf.v4.new_markdown_cell('## Reload all models and infer the reserved sample'),
 nbf.v4.new_code_cell("from src.predict import predict, load_artifacts\nexample=json.loads((ROOT/'models/demo.json').read_text())\ndisplay(example['input'])\nfor name in metadata['models']:\n    display(predict(example['input'],*load_artifacts(name)))\nprint('Actual price (lakhs):',example['actual'])"),
 nbf.v4.new_markdown_cell('## Interface and submission\nRun `streamlit run app/app.py`. See README for the deployed private app, reproduction commands and reviewer access. Moodle submission remains an individual step.')]
nbf.write(nb,ROOT/'notebooks/vehicle_price_analysis.ipynb')
print('Part 2 report and notebook created')
