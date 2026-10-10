"""Describe saved models on the fixed test split; no fitting or model selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import FEATURES,load_data,split_data
from src.predict import load_artifacts

scores=pd.read_csv(ROOT/'reports/model_comparison.csv').set_index('model')
_,test,_,_,_=split_data(load_data()[0])
texts={}
for name in scores.index:
    model,meta=load_artifacts(name)
    predicted=model.predict(test[FEATURES])
    residual=test.price.to_numpy()-predicted
    slug=name.lower().replace(' ','_')
    pd.DataFrame({'actual':test.price.to_numpy(),'predicted':predicted,'residual':residual}).to_csv(ROOT/f'reports/diagnostics_{slug}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    hi=max(test.price.max(),predicted.max())
    axes[0].scatter(test.price,predicted,s=10,alpha=.35,color='#147d92')
    axes[0].plot([0,hi],[0,hi],'--',color='#b64a3c',label='Perfect prediction')
    axes[0].set(xlabel='Actual advertised price (lakhs)',ylabel='Predicted price (lakhs)',title='Actual versus predicted');axes[0].legend()
    axes[1].scatter(predicted,residual,s=10,alpha=.35,color='#147d92')
    axes[1].axhline(0,linestyle='--',color='#b64a3c')
    axes[1].set(xlabel='Predicted price (lakhs)',ylabel='Actual − predicted (lakhs)',title='Prediction errors (residuals)')
    fig.suptitle(f'{name} · fixed held-out test listings');fig.tight_layout()
    fig.savefig(ROOT/f'reports/figures/diagnostics_{slug}.png',dpi=150);plt.close(fig)
    r=scores.loc[name]
    importance=pd.read_csv(ROOT/'reports'/meta['models'][name]['importance_file'])
    top=', '.join(importance.feature.head(4).str.replace('_',' '))
    if name=='Random Forest':
        strengths='Combines many trees to capture nonlinear relationships between vehicle details. Averaging reduces dependence on one tree. Its recommendation status is determined by the comparison table.'
        limits='The training MAE is substantially smaller than the test MAE, suggesting some overfitting or a difference between the training and later test listings. Averaging tree outputs also limits extrapolation to prices outside the training range. Sparse examples of rare or expensive cars may be unreliable.'
    elif name=='XGBoost':
        strengths='Builds trees sequentially to correct earlier errors and provides explicit regularization. In this experiment it achieved slightly lower test RMSE and slightly higher R², so it performed better under squared-error measures.'
        limits='It needs careful choices of depth, learning rate, number of trees and regularization. Its validation and test MAE were higher than Random Forest’s here, despite its better RMSE. Neither its smaller training–test gap nor its slightly higher R² proves it will generalize better to new markets.'
    else:
        strengths='Combines fixed Random Forest and XGBoost pipelines with a scaled Ridge meta-model. Meta-training predictions come from purged, expanding temporal folds; base pipelines are then refitted on all training rows.'
        limits='Correlated base errors can limit gains. Training and inference cost more; early rows without out-of-fold predictions are excluded from meta-training. Stacking is not guaranteed to improve on either model. It was added after earlier test results were inspected, so this is an exploratory comparison on an existing holdout.'
    text=f'''### {name}: strengths and limitations

**Strengths.** {strengths}

**Limitations.** {limits}

**Measured results.** Validation MAE: **{r.cv_mae:.2f} lakhs**; test MAE: **{r.test_mae:.2f} lakhs**; test RMSE: **{r.test_rmse:.2f} lakhs**; test R²: **{r.test_r2:.3f}**. Training MAE was **{r.train_mae:.2f} lakhs**. Lower MAE/RMSE is better; higher R² is better. R² is unitless, not percentage accuracy.

**Feature-importance chart.** The strongest fields are **{top}**, in that order. Longer bars mean shuffling that feature increased test error more. This is overall model reliance, not proof of causation, the direction of an effect, or an explanation for a single car. Correlated fields can share importance, and shuffling can produce unrealistic combinations.

**Actual-versus-predicted graph.** Points on the dashed diagonal are correct predictions. Points below it are underestimates; points above it are overestimates. Each point represents one unseen test listing.

**Residual graph.** Residual = actual price − predicted price. Positive values mean underestimation; negative values mean overestimation. A tight cloud around zero is desirable. The mean signed residual is **{residual.mean():.2f} lakhs**; the largest absolute test error is **{abs(residual).max():.2f} lakhs**. These describe the whole test set, not a guaranteed range for the current vehicle.

**What the graphs show here.** Most listings form a dense cluster at the lower end of the price range, broadly following the diagonal. Several high-priced listings fall well below the diagonal, indicating substantial underestimation. The residual plot has isolated large positive and negative errors, so neither model is uniformly reliable across all vehicles. The plots alone do not establish why those listings were difficult.

**Large-error interpretation.** RMSE ({r.test_rmse:.2f}) is much larger than MAE ({r.test_mae:.2f}), consistent with some large errors having a strong effect on the squared-error metric. Inspect isolated points far from the diagonal/zero line; an average score can hide these failures.
'''
    (ROOT/f'reports/interpretation_{slug}.md').write_text(text)
    texts[name]=text
intro=f"""## Interpreting the model comparison

{meta['winner']} is recommended by validation MAE. All three models use the same {len(test):,} held-out listings. MAE measures average error; RMSE emphasizes large errors; R² is unitless. See the comparison table for measured results, not claims of statistical superiority.

Stacking uses fixed base configurations and a Ridge meta-model trained on inner temporal out-of-fold predictions. Outer temporal folds evaluate the full stacking procedure. The test set is unchanged and was already examined before stacking was added; results are exploratory. All models describe historical asking prices, not completed sales or current valuations.
"""
(ROOT/'reports/results_interpretation.md').write_text(intro+'\n'+ '\n'.join(text + f'\n![{name} diagnostic graphs](figures/diagnostics_{name.lower().replace(chr(32), chr(95))}.png)\n' for name,text in texts.items()))
print('Saved interpretations and all model-specific diagnostic graphs')
