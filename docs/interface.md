# Interface requirements

Build a Streamlit interface for a Sri Lankan vehicle listing-price regression model.
Load the trusted local models/model.pkl pipeline and models/metadata.json. Call
src.predict.predict so validation and prediction stay consistent with the CLI.
Use the training metadata for brand/model dropdowns and remaining categories.
Accept brand, model, year, engine_cc, gear, fuel, town, leasing, condition,
air_condition, power_steering, power_mirror and power_window.

Show estimates in LKR, converting the confirmed lakh-scale outputs by 100,000.
Explain that listings cover December 2024–February 2025 and do not establish
current market prices. Show the conversion clearly; do not invent a confidence interval.
Exclude mileage because it is exactly (2025-year)*11000 in every source record.
Include a held-out example, input validation, model comparison, global feature
importance, limitations and JSON download. Keep the selected model cached.
Show which submitted vehicle the result belongs to. Do not train inside the app.
Use a restrained teal palette, clear labels and a two-column layout.



## Model switching
Both fitted pipelines are saved separately and registered in `models/metadata.json`.
Use the **Prediction model** dropdown, then click **Estimate listing price**.
Random Forest remains the validation-recommended default. Switching clears the previous
estimate and updates the selected model’s test MAE. Downloaded results include the
model name. The feature importance chart updates to the selected model’s own permutation importance. CLI example:
`python -m src.predict --model XGBoost`.
