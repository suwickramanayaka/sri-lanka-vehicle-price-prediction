# Related approaches and comparison

Sources checked on 14 October 2026. This is a methodological comparison, not a
replication of published scores. Our controlled numerical comparisons use our
own identical training/validation/test partitions.

| Approach | Source evidence | Comparison with this project | Limit of the comparison |
|---|---|---|---|
| Sri Lankan used-car regression and averaging | Basnayaka (2023) compares linear regression, Random Forest, Extra Trees, gradient boosting, LightGBM and XGBoost, then averages predictions. | We evaluate related linear and tree families; retain RF/XGBoost stacking instead of simple averaging; audit synthetic mileage and use purged temporal evaluation. | The repository abstract reports “accuracy”; its metric and split are not sufficiently specified for numerical equivalence to our MAE/R². We do not claim to reproduce or outperform it. |
| CatBoost | Prokhorenkova et al. (2018) present ordered boosting and categorical-feature processing. | Native categorical input is relevant to our many rare vehicle model labels; encoding is fitted within training folds. | General benchmark results do not guarantee benefits for Sri Lankan vehicle listings. Our configuration uses the library's boosting default; we do not claim to reproduce the paper's experiments. |
| LightGBM | Ke et al. (2017) describe efficient gradient-boosted trees, including GOSS and feature bundling. | Histogram-based LightGBM is evaluated as an efficiency-oriented nonlinear candidate, with restricted leaves and minimum leaf counts. | Our ordinary GBDT configuration uses row subsampling, not a GOSS replication. Runtime is measured locally; published speedups are not transferred to this dataset. |
| Median and regularized linear references | Implemented here on the same splits. | Median establishes a no-feature error level; Ridge tests whether tree complexity improves on an additive relationship. | These are local benchmarks, not reproductions of the Sri Lankan study's linear model. |

| NGBoost | Duan et al. (2020) use natural gradients to learn conditional distribution parameters. | Normal NGBoost is the focal probabilistic regression study; point metrics, likelihood and interval coverage are evaluated on the same splits. | Distributional assumptions and existing test exposure limit interpretation; paper benchmark scores are not directly comparable. |

## References

1. Basnayaka, D. (2023). [Prediction of Market Price for Used Cars in Sri Lanka Based on Machine Learning Approaches](https://dlib.iit.ac.lk/xmlui/handle/123456789/1634). IIT repository abstract.
2. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516). NeurIPS.
3. Ke, G. et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). NeurIPS.
4. [CatBoost categorical-feature documentation](https://catboost.ai/docs/en/features/categorical-features). Supports using native categorical input rather than external one-hot encoding.
5. [LightGBM parameter-tuning documentation](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters-Tuning.html). Supports constraining leaves and minimum observations to control complexity.

6. Duan, T. et al. (2020). [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a.html). ICML, PMLR 119:2690–2700.
7. [NGBoost usage documentation](https://stanfordmlgroup.github.io/ngboost/useage/). Distributional regression, scoring rules and prediction APIs.
