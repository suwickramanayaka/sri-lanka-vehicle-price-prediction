# Application interface

The Streamlit application provides vehicle-price estimation and model analysis in a
wide, two-column layout. The prediction form and results remain visible together;
additional analysis and data exploration are organized into tabs.

## Prediction workflow

Users enter brand, model, manufacture year, engine capacity, transmission, fuel,
town and condition. Equipment and leasing fields are available in an expandable
section. The shared `src.predict.predict` function validates inputs and performs
inference, keeping the application and command-line interface consistent.

Saved pipelines are registered in `models/metadata.json` and cached for inference.
The application does not train models. Estimates are converted from lakhs to LKR
using 1 lakh = LKR 100,000. A reserved example supports demonstration, and results
can be downloaded as JSON with the model name and submitted vehicle details.
Mileage is excluded because the source values are calculated from vehicle age.

## Model selection and analysis

The model selector offers Random Forest, XGBoost, CatBoost, LightGBM, NGBoost, Ridge and
Stacking. The default is chosen by validation MAE. Switching models clears the
previous estimate and updates performance metrics, permutation importance and
prediction diagnostics. Explanations distinguish average error from uncertainty
for an individual vehicle.

The comparison tab presents predictive performance, training time and model
selection evidence. Optional expanded analysis includes training-only exploration,
validation folds and error breakdowns. The dataset explorer supports raw and
processed records, filtering and downloads without changing training data.

## Interpretation

Prices describe advertisements from December 2024 through February 2025, rather
than verified transactions or current market valuations. Feature importance shows
global model reliance, not causation. The comparison reuses a previously examined
test split and is exploratory; independent confirmation requires new listings.
