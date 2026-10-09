"""Create the report and notebook from real saved experiment results."""
from pathlib import Path
import json
import pandas as pd
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'reports/data_audit.json').read_text())
m=json.loads((ROOT/'models/metadata.json').read_text())
d=json.loads((ROOT/'models/demo.json').read_text())
c=pd.read_csv(ROOT/'reports/model_comparison.csv')
def table(df):
    return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
scoretable=table(c[['model','cv_mae','cv_std','test_mae','test_rmse','test_r2']])
report=f'''# Sri Lankan Vehicle Listing Price Prediction

## Objective
Estimate advertised vehicle prices from tabular listing characteristics and compare
Random Forest with XGBoost for the class activity. Intended users are students
and reviewers exploring historical asking-price relationships. This is a coursework
prototype, not a present-day vehicle valuation service.

## Source and provenance
Prasad Nirmal, *SriLankan Vehicle(Car) Price Dataset*, Kaggle version 3:
https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset

Publisher metadata lists MIT licensing and describes listings scraped from Ikman.lk
and Riyasewana.com. The downloaded source has {a['raw_rows']:,} rows and 17 columns,
including an exported index. Listings span {a['date_min']} to {a['date_max']}.
Metadata and the original CSV are preserved. SHA-256: `{m['raw_sha256']}`.

## Data analysis and limitations
- {a['duplicates_removed']} exact duplicates were removed after trimming categories and ignoring the exported index.
- {a['invalid_rows']} records failed target/date/year validity checks.
- The downloaded table contains no literal nulls, but the publisher already preprocessed it.
- Every mileage value equals `(2025 - YOM) * 11000`. It is excluded as a deterministic proxy for year, not interpreted as measured mileage.
- The project owner confirmed that Price is in LKR lakhs. One lakh equals LKR 100,000. Training metrics remain in lakhs; the app converts monetary values to LKR.
- Model categories are numerous and often rare. Rare-category handling is learned separately within each fold.
- Listing prices are not verified transaction prices. The original listing IDs and full cleaning history are unavailable.

## Evaluation design
The latest record is reserved for a single-sample demonstration; all records with
its feature signature are excluded from development. The final test period starts
{a['test_cutoff']}. Matching feature signatures are purged from earlier training data,
removing {a['purged_train_rows']} potential overlaps. This conservative proxy can also
exclude distinct cars with identical features; it cannot establish actual identities.

There are {a['train_rows']:,} training records, {a['test_rows']:,} test records and one
demonstration record. Three expanding temporal folds are used inside training.
Validation groups are also excluded from each fold's training partition. The chosen
model minimizes mean validation MAE; test results are computed afterward and are
not used to select or retune it. Fold standard deviations describe variation over
three time windows, not confidence intervals.

## Preprocessing and parameter engineering
Numerical median imputation and categorical most-frequent imputation are fitted
within the pipeline. One-hot encoding pools categories with fewer than three
training observations and handles unknown categories. Both selected
tree models do not require normalization or scaling. Explicit model configurations
and every search result are saved in `experiments.csv` and `src/train.py`.

Random Forest varies tree depth and minimum leaf size while using 180 trees and
80% feature sampling. XGBoost varies depth and number of boosting rounds, with a
0.05 learning rate, 85% row/column sampling and explicit L1/L2 regularization.
Only the two models required for the activity are trained and compared.

## Results
All MAE/RMSE values below are in LKR lakhs (multiply by 100,000 for LKR).

{scoretable}

The validation-selected model is **{m['winner']}**. Its held-out MAE is
**{m['test_metrics']['mae']:.3f}**, RMSE **{m['test_metrics']['rmse']:.3f}**, and
R² **{m['test_metrics']['r2']:.3f}**. R² is not percentage accuracy. A difference in model scores
is descriptive; this experiment does not prove statistical superiority.

![Evaluation plots](figures/evaluation.png)

XGBoost has a lower test RMSE but a higher test MAE than
Random Forest: it performs better under the metric that penalizes large errors more
strongly, while Random Forest has smaller average absolute errors. The chosen metric
was MAE before evaluation.

## Model strengths and limitations
- **Random Forest:** models nonlinear interactions and offers flexible regularization; can struggle with rare vehicles and cannot reliably extrapolate beyond training prices.
- **XGBoost:** sequentially corrects errors and supports explicit regularization; sensitive to tuning, outliers and distribution changes.

Consult `errors_by_brand.csv` for subgroup counts and MAE. Do not draw strong
conclusions from brands with only a few test observations. `test_predictions.csv`
records individual errors. The residual plot helps identify whether expensive cars
or other regions of the target range have systematically larger errors.

## Interpretation
Manufacture year has the largest permutation importance, followed by brand,
transmission and engine capacity. This suggests that vehicle age and basic market
segment characteristics drive much of the fitted model’s discrimination. It does
not show that changing a feature would cause a corresponding price change.

Permutation importance measures the test MAE increase when a feature is shuffled.
It is descriptive analysis after evaluation; it was not used for further tuning.
Correlated features may share importance, and shuffling can create implausible
combinations. These are global associations, not causal effects or explanations
of a particular car's price.

![Feature importance](figures/importance.png)

## Comparison with an existing approach
Basnayaka (2023), *Prediction of Market Price for Used Cars in Sri Lanka Based on
Machine Learning Approaches*, IIT repository:
https://dlib.iit.ac.lk/xmlui/handle/123456789/1634

The abstract describes comparing linear regression, Random Forest, Extra Trees,
gradient boosting, LightGBM and XGBoost, plus averaging predictions. This project
compares a related subset and adds an explicit documented audit of deterministic
mileage and lakh-based price scaling, temporal evaluation, and feature-signature
purging. The repository abstract's reported “accuracy” is not a directly comparable
MAE/R² benchmark, and an identical dataset/split has not been verified. This is a
methodological comparison, not a claim to outperform that study. A broader literature
review is still appropriate before the final assignment submission.

## Single-sample inference
Reserved source row: {d['row_id']}. Actual price (LKR lakhs): {d['actual']:.3f}; prediction:
{d['prediction']:.3f}; absolute error: {abs(d['actual']-d['prediction']):.3f}.
The exported pipeline was reloaded and its prediction matched the in-memory model.
Run `python -m src.predict` to repeat this independently.

## Front end and reproducibility
The Streamlit interface accepts the actual model fields, validates inputs, displays
historical estimates in LKR, supports the held-out example and shows model
comparisons. Interface requirements are documented in `docs/interface.md`.
The fitted preprocessing and estimator are exported together in `models/model.pkl`.
Exact installed versions are saved in `requirements-lock.txt`.

## Remaining submission work
The lakh scale is based on the project owner’s confirmation.
Review the analysis and write the student's own interpretation; extend the literature
review if required. Confirm Moodle deadlines. Publish the repository and submit its
link or notebook individually. This project has not been published or submitted.
'''
report += "\n\n## Interactive model selection\nThe app saves and loads both trained pipelines separately. Select a model in\nthe Prediction model dropdown, then submit the same vehicle details to compare\nestimates. Random Forest remains the recommended default based on validation MAE.\nSwitching clears the previous result and displays the selected model's test MAE.\nDownloaded predictions include the model name. The importance chart switches to the selected model's global permutation importance.\n"
interpretation = ROOT/'reports/results_interpretation.md'
if interpretation.exists():
    report += '\n\n' + interpretation.read_text()
(ROOT/'reports/final_report.md').write_text(report)
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[
 nbf.v4.new_markdown_cell('# Sri Lankan Vehicle Listing Price Prediction\n\nEnd-to-end activity and final-assignment foundation. Training and saved metric values are in LKR lakhs; app predictions are converted to LKR.'),
 nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nfrom src.data import load_data, split_data, FEATURES\ndf, audit = load_data()\ndisplay(pd.Series(audit))\ndisplay(df.head())"),
 nbf.v4.new_markdown_cell('## Source audit\nThe publisher preprocessed the data. All mileage values follow an exact year-based formula and are excluded. The project owner confirmed the source prices are in LKR lakhs. No source nulls does not prove complete original observations.'),
 nbf.v4.new_code_cell("train, test, demo, cutoff, purged = split_data(df)\nprint('Training:', len(train), 'Test:', len(test), 'Demo:', len(demo), 'Test starts:', cutoff)\ndisplay(train[FEATURES + ['price']].describe(include='all'))\ndisplay(train.groupby('brand').agg(count=('price','size'), median_price=('price','median')).sort_values('count', ascending=False).head(15))"),
 nbf.v4.new_markdown_cell('## Train and compare\nThe pipeline fits preprocessing within three purged temporal validation folds. Only explicitly configured Random Forest and XGBoost models are evaluated. Select by validation MAE, not test results. Set RETRAIN=True to repeat training; otherwise use saved results.'),
 nbf.v4.new_code_cell("RETRAIN = False\nif RETRAIN or not (ROOT / 'models/model.pkl').exists():\n    from src.train import main\n    main()\ncomparison = pd.read_csv(ROOT / 'reports/model_comparison.csv')\ndisplay(comparison)\ndisplay(pd.read_csv(ROOT / 'reports/experiments.csv'))"),
 nbf.v4.new_code_cell("display(Image(filename=str(ROOT / 'reports/figures/evaluation.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/importance.png')))\ndisplay(pd.read_csv(ROOT / 'reports/errors_by_brand.csv').head(15))"),
 nbf.v4.new_markdown_cell('## Reload and predict a single unseen record\nThe demo record and all its feature-signature matches were excluded from development.'),
 nbf.v4.new_code_cell("from src.predict import predict\nexample = json.loads((ROOT / 'models/demo.json').read_text())\ndisplay(example['input'])\ndisplay(predict(example['input']))\nprint('Actual price (LKR lakhs):', example['actual'])"),
 nbf.v4.new_markdown_cell('## Interpretation and limitations'),
 nbf.v4.new_code_cell("display(Markdown((ROOT / 'reports/final_report.md').read_text()))"),
 nbf.v4.new_markdown_cell('## Run the interface\nFrom the project root: `streamlit run app/app.py`. See README.md for installation, verification and submission mapping.')]
nbf.write(nb,ROOT/'notebooks/vehicle_price_analysis.ipynb')
print('Report and notebook created')
