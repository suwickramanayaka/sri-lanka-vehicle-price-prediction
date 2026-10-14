# Part 2 requirement coverage

| Guideline requirement | Implemented evidence |
|---|---|
| Local Sri Lankan tabular problem | Vehicle listing-price regression; preserved source CSV and Kaggle provenance |
| Proper data analysis before model choice | `scripts/analyze_data.py`, training-only plots, numeric/categorical profiles and experiment plan |
| Choose suitable models beyond RF/XGBoost | Native-category CatBoost, LightGBM, Ridge, RF, XGBoost and temporal stacking; median reference |
| Compare with existing approaches | `docs/related_work.md`: Sri Lankan vehicle study, CatBoost and LightGBM primary sources; controlled local benchmarks |
| Justify effectiveness | Validation-selected model, improvement versus median, fold scores, errors by price band/category coverage and measured runtime |
| Interpret outputs | Per-model strengths/limitations, metrics, residuals, permutation importance and subgroup counts |
| Front-end system | Existing Streamlit UI extended to six models and optional Part 2 analysis; specification in `docs/interface.md` |
| No deep learning, CV or RL | Tabular regression only; tree, linear and stacking models |
| Individual submission on Moodle | Not performed; owner must review, give lecturer access and submit |

The implementation extends the Part 1 case as explicitly allowed by the guideline.
The previous test results have been seen, so Part 2 is documented as exploratory.
A new future-data confirmation is a research limitation, not a claimed completed experiment.
The guideline gives 18 October as a tentative deadline; verify Moodle's actual date.
