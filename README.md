# Sri Lankan Vehicle Listing Price Explorer

An end-to-end regression application for historical Sri Lankan vehicle listing prices.
Compares Random Forest, XGBoost, CatBoost, LightGBM, NGBoost, Ridge and temporal stacking. Includes temporal validation,
model export, an unseen example, an analysis notebook and a Streamlit interface.

## Live application

[Open the Vehicle Listing Price Explorer](https://sri-lanka-vehicle-price-prediction.streamlit.app/)

Hosted on Streamlit Community Cloud. The app is private: sign in with an account
that has been granted access. Repository access and app access are managed separately.

## Presentation and report

- [Project presentation (PowerPoint)](deliverables/vehicle-price-prediction-final.pptx)
- [Demo walkthrough (MP4, 4 min 49 sec)](deliverables/vehicle-price-prediction-demo.mp4) — narrated application screenshots covering prediction, explanations and data exploration.
- [Final technical report (PDF)](deliverables/vehicle-price-prediction-ngboost-report.pdf)
- [Report source (Markdown)](reports/final_report.md)

## Interface preview

Focused screenshots show the current application. Click an image to open it at
full size. The application offers seven models. The comparison and NGBoost screenshots show the latest extension; the original overview and fit-chart captures illustrate the earlier six-model interface.

### Price estimation

Enter vehicle details, choose a trained model, and estimate the listing price in LKR.
The result identifies the model and vehicle used for the prediction.

[![Price estimate for a ZOTYE Z100 using Random Forest](docs/screenshots/price-estimate.jpg)](docs/screenshots/price-estimate.jpg)

<details>
<summary>View the vehicle details form</summary>

[![Vehicle input form with brand, model, year, engine capacity and other characteristics](docs/screenshots/vehicle-details.jpg)](docs/screenshots/vehicle-details.jpg)

</details>

### Model comparison

All models use the same evaluation splits. Random Forest has the lowest validation
and test MAE; XGBoost has the lowest test RMSE and highest test R².

**Average prediction error - lower is better**

[![Validation and test MAE for all seven models](docs/screenshots/model-comparison-mae.jpg)](docs/screenshots/model-comparison-mae.jpg)

**Large errors and model fit**

[![Test RMSE and R-squared for the original six models](docs/screenshots/model-comparison-fit.jpg)](docs/screenshots/model-comparison-fit.jpg)

The scores are also provided as text for easy reading on smaller screens.

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Random Forest | 7.148 | 5.830 | 14.440 | 0.893 |
| Stacking | 7.299 | 6.475 | 14.411 | 0.894 |
| XGBoost | 7.562 | 6.821 | 14.241 | 0.896 |
| CatBoost | 8.740 | 8.414 | 15.606 | 0.875 |
| LightGBM | 9.410 | 7.590 | 15.283 | 0.880 |
| NGBoost | 9.435 | 8.581 | 16.446 | 0.861 |
| Ridge | 11.124 | 9.994 | 20.694 | 0.781 |

MAE and RMSE are in lakhs; R² is unitless. Selection uses validation MAE.
The previously examined test split makes this comparison exploratory.

## Start the application

From this project directory:

```bash
source .venv/bin/activate
streamlit run app/app.py
```

On macOS, XGBoost also requires `brew install libomp`.

For a fresh installation (Python 3.12 recommended):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
streamlit run app/app.py
```

`requirements-lock.txt` records the tested environment. `requirements.txt` gives
portable version ranges if exact platform-specific versions cannot be installed.
Re-train the pickle after changing dependency versions.

## Reproduce the analysis

```bash
python scripts/analyze_data.py
python -m src.train
python scripts/summarize_results.py
python scripts/interpret_results.py
python scripts/explain_ngboost.py
python scripts/build_report.py
python -m src.predict
python -m pytest -q
```

Training writes the comparison, all search candidates, test predictions,
per-brand errors, plots, split IDs, data audit and trusted pickle. Training is
CPU-only. It selects the model by validation MAE before evaluating held-out data.
`notebooks/vehicle_price_analysis.ipynb` presents the full workflow and results;
its training cell can rebuild missing artifacts or explicitly retrain them.

## Data and critical limitations

Source: [Prasad Nirmal, SriLankan Vehicle(Car) Price Dataset, version 3](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset).
Publisher metadata identifies Ikman.lk and Riyasewana.com as listing sources and
lists the MIT license. Original API metadata is preserved in
`data/raw/kaggle_metadata.json`; original CSV is in `data/raw/car_price_dataset.csv`.
Data was downloaded on 9 October 2026. The publisher last updated it in February 2025.

- The CSV contains 9,788 advertisements, not verified completed transactions.
- Dates run from 3 December 2024 through 5 February 2025.
- **Price units are LKR lakhs**, confirmed by the project owner. One lakh equals
  LKR 100,000. Training targets and saved CSV metrics stay in lakhs; the app
  converts predictions to LKR and displays MAE, RMSE and importance in lakhs for readability. The prediction JSON
  includes `prediction` in lakhs and `prediction_lkr` in rupees.
- **All mileage values equal `(2025 - YOM) * 11000`.** Mileage is excluded from
  predictors and the interface; no claim about actual mileage effects is made.
- The publisher already preprocessed the data; original imputation and price
  cleaning cannot be reconstructed. No raw nulls does not prove complete observations.
- Feature signatures are used to reduce possible re-listing overlap. Without
  original listing IDs, duplicate identities cannot be guaranteed; identical
  signatures can also refer to different cars.
- The app describes the historical source market, not current 2026 prices.
- Global importance is associative, not causal or an individual explanation.

## Evaluation design

Remove invalid targets/dates/years, trim categorical whitespace, remove exact
duplicates excluding the source index and omit the deterministic mileage field.
Keep the latest record as a single-sample demo and exclude its full feature group
from development. Reserve the latest 20% of unique dates for testing; remove
matching test feature signatures from earlier training records. Use three expanding
time folds inside training, also purging matching validation signatures.

Imputation and one-hot encoding are fitted inside every fold. Rare categories
are pooled; unknown categories are handled. Tree models do not require numeric scaling; Ridge scales numeric inputs. CatBoost uses native categorical features instead of one-hot encoding. Compare explicit, bounded model
configurations; choose minimum validation MAE. Preserve test scores for all models
without choosing the winner from test performance. Do not retune on these scores.

## Project structure

| Directory | Purpose |
|---|---|
| `app/` | Streamlit interface |
| `src/` | Data preparation, training, stacking and prediction |
| `data/` | Source data, provenance and processed records |
| `models/` | Saved estimators, model registry and example input |
| `reports/` | [Organized evaluation evidence](reports/README.md), technical report and figures |
| `notebooks/` | Executed analysis and reproducible workflow |
| `docs/` | Experiment design, interface documentation and references |
| `scripts/` | Exploration, results summaries and report generation |
| `tests/` | Pipeline, isolation and inference checks |
| `deliverables/` | Presentation and supporting materials |

## NGBoost study

[![NGBoost estimate in the application](docs/screenshots/ngboost-estimate.jpg)](docs/screenshots/ngboost-estimate.jpg)

[![NGBoost partial-dependence explanations](docs/screenshots/ngboost-explanation.jpg)](docs/screenshots/ngboost-explanation.jpg)

NGBoost extends the comparison with natural-gradient probabilistic regression.
It is the focal new-method study; the default prediction model is still selected
by validation MAE. The existing six models remain benchmarks.

The fixed search compares 100, 200 and 300 boosting rounds with a Normal
distribution, log-score objective and depth-3 trees. The application provides
permutation importance and an optional partial-dependence explanation. Distribution
coverage and width are reported as diagnostics, not guaranteed price ranges.
See [methodology and measured findings](reports/interpretations/ngboost_study.md).

## Model switching
All seven fitted estimators are saved separately and registered in `models/metadata.json`.
Use the **Prediction model** dropdown, then click **Estimate listing price**.
The default is selected by validation MAE. Switching clears the previous
estimate and updates the selected model’s test MAE. Downloaded results include the
model name. The feature importance chart updates to the selected model’s own permutation importance. CLI example:
`python -m src.predict --model XGBoost`.

The app includes model-specific strengths, limitations, score interpretation and
actual-versus-predicted/residual graphs. Regenerate these after retraining with
`python scripts/interpret_results.py`. Full text: `reports/interpretations/results_interpretation.md`.

Optional dataset explorer: open the third analysis tab and enable the explorer. Inspect raw or cleaned records, filter by brand/split, view distributions and download the filtered view. This is read-only and does not affect training. Processed records are shown before pipeline encoding.


## Stacking ensemble

`Stacking` combines two predictions—one from Random Forest and one from XGBoost—
using a `StandardScaler` and Ridge regression meta-model (`alpha=10`). It does not
use the original vehicle features directly in the meta-model.

Within each training set, three expanding temporal folds generate predictions
for later rows. Matching vehicle feature groups are purged from each earlier
training fold, and each base pipeline fits its own preprocessing. Only these
out-of-fold predictions train Ridge; the earliest rows have no such predictions
and are excluded from meta-training. The base pipelines are then refitted on the
full training set for inference.

Three outer purged temporal folds evaluate this entire procedure. Stacking uses
fixed base configurations from `src/train.py`, rather than configurations selected
using outer validation labels. The standalone RF and XGBoost models are tuned
separately, so their exported predictions need not equal the stack's internal
base predictions. All models are compared on the same existing test split.
Since that test split was inspected before adding stacking, the new comparison
is exploratory; a fresh future holdout is needed for independent confirmation.
Stacking may improve results but is not guaranteed to do so.

Run `python -m src.predict --model Stacking` or select **Stacking** in the app.
The comparison table, feature importance and diagnostic charts include it.


## Documentation

- [Experiment plan](docs/experiment_plan.md): training-only findings, candidate rationale and fixed evaluation rules.
- [Final report](reports/final_report.md): analysis, comparison, model-choice justification, sources and limitations.
- [Executed notebook](notebooks/vehicle_price_analysis.ipynb): reproducible exploration, results and inference.
- [Related approaches](docs/related_work.md): local vehicle-price study and primary CatBoost/LightGBM sources.

CatBoost uses native categories; LightGBM uses one-hot inputs; Ridge supplies a
regularized linear reference. A training-median benchmark is reported separately
and is not offered as a prediction model in the app. The optional **Model selection**
expander in the comparison tab presents training EDA, selection reasoning,
validation fold scores and error breakdowns without crowding the prediction form.

The expanded comparison reuses an already examined test set, so it is exploratory.
It does not establish current-market accuracy or superiority over published studies.

### Export the technical report

Install the optional report dependencies and render the current Markdown report:

```bash
pip install -r requirements-report.txt
python scripts/export_report_pdf.py
```

The revised PDF is saved separately from the earlier manually edited report. The presentation and its local speaker script are preserved.
