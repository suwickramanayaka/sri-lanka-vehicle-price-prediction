"""Explain the saved NGBoost model without tuning against the test partition."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import FEATURES,load_data,split_data
from src.predict import load_artifacts


def main():
    train,test,_,_,_=split_data(load_data()[0])
    model,meta=load_artifacts('NGBoost')
    estimator=model.named_steps['model']
    encoded=model.named_steps['preprocess'].transform(test[FEATURES])
    distribution=estimator.pred_dist(encoded)
    lower,upper=distribution.interval(.9)
    # This is a retrospective calibration check, not a guarantee for a future car.
    scores={'distribution':'Normal','nominal_coverage':.9,
            'test_nll':float(-distribution.logpdf(test.price.to_numpy()).mean()),
            'empirical_coverage':float(np.mean((test.price>=lower)&(test.price<=upper))),
            'mean_interval_width_lakhs':float(np.mean(upper-lower)),
            'negative_lower_bound_fraction':float(np.mean(lower<0)),
            'test_rows':len(test),'interval_status':'uncalibrated; previously examined historical test split'}
    (ROOT/'reports/metrics/ngboost_distribution.json').write_text(json.dumps(scores,indent=2)+'\n')
    pd.DataFrame({'row_id':test.row_id,'actual':test.price,'predicted':model.predict(test[FEATURES]),
                  'lower_90':lower,'upper_90':upper}).to_csv(ROOT/'reports/predictions/ngboost_intervals.csv',index=False)
    background=train[FEATURES].sample(min(256,len(train)),random_state=42)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rows=[];changes=[]
    for ax,feature,label in zip(axes,['year','engine_cc'],['Manufacture year','Engine capacity (cc)']):
        grid=np.unique(np.round(np.linspace(*train[feature].quantile([.05,.95]),15)))
        means=[]
        for value in grid:
            varied=background.copy();varied[feature]=value
            means.append(float(model.predict(varied).mean()))
            rows.append({'feature':feature,'value':value,'mean_prediction_lakhs':means[-1],'background_rows':len(background)})
        changes.append(f'{label}: the average model prediction changes from {means[0]:.2f} to {means[-1]:.2f} lakhs across {grid[0]:g}–{grid[-1]:g}.')
        ax.plot(grid,means,color='#087f8c',marker='o',markersize=3)
        ax.set(xlabel=label,ylabel='Average predicted price (lakhs)',title='NGBoost partial dependence')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/ngboost_partial_dependence.png',dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'reports/importance/ngboost_partial_dependence.csv',index=False)
    comparison=pd.read_csv(ROOT/'reports/metrics/model_comparison.csv').set_index('model')
    r=comparison.loc['NGBoost'];importance=pd.read_csv(ROOT/'reports/importance/feature_importance_ngboost.csv')
    top=', '.join(importance.feature.head(4).str.replace('_',' '))
    text=f'''### NGBoost: probabilistic vehicle-price regression

NGBoost is the principal method studied in this extension. It uses shallow trees
and natural-gradient updates to learn the mean and scale of a conditional Normal
price distribution. Its training objective is negative log-likelihood; configuration
selection uses validation MAE so point predictions remain comparable to the other models.
Unlike Random Forest's averaging or conventional point-prediction boosting, it
updates multiple distribution parameters using a geometry-adjusted gradient.
This is a non-deep-learning method. See [Duan et al. (2020)](https://proceedings.mlr.press/v119/duan20a.html).

**Design.** Fold-fitted median/mode imputation and dense one-hot encoding with rare
category pooling; depth-3 base trees, minimum leaf size 10, learning rate 0.03,
seed 42, and 100/200/300 rounds. Selected settings: `{r.parameters}`.
Normal is an explicit simplifying assumption, not evidence that car prices are Normal.

**Measured performance.** Validation MAE {r.cv_mae:.2f} lakhs; test MAE
{r.test_mae:.2f} lakhs; RMSE {r.test_rmse:.2f} lakhs; R² {r.test_r2:.3f}.
The validation-selected recommendation is {meta['winner']}. The research focus
and the best point-prediction model need not be the same.

**Global explanation.** The four strongest permutation-importance fields are
{top}. Importance is the increase in test MAE after shuffling; it describes
model reliance rather than causation or a direction of effect.

**Partial-dependence interpretation.** {' '.join(changes)}
Newer vehicles often have higher asking prices, and engine capacity can track
vehicle class. The plotted response should be compared with these expectations;
non-monotonic sections may reflect model interactions, sampling and sparse data.
The curves average over 256 training vehicles and cover the training 5th–95th
percentiles. Varying one feature can create implausible brand/model/year/engine
combinations. They are not causal depreciation curves or individual explanations.

**Distribution check.** Test negative log-likelihood is {scores['test_nll']:.3f}
(in lakh units). Nominal 90% intervals contain {100*scores['empirical_coverage']:.1f}%
of observed test prices, with mean width {scores['mean_interval_width_lakhs']:.2f} lakhs.
{100*scores['negative_lower_bound_fraction']:.1f}% have negative lower bounds.
Coverage must be considered alongside width: a wide range can cover many prices
without being useful. These intervals have not been independently calibrated and
are not displayed as guaranteed price ranges in the prediction form. Negative
bounds expose a limitation of the Normal assumption for positive prices.

**Evidence limits.** The test set was examined before this extension; results are
exploratory. No test-driven retuning was performed. New, later listings are needed
for independent evaluation and calibration. Better uncertainty or point accuracy
than existing models is not assumed.
'''
    (ROOT/'reports/interpretations/ngboost_study.md').write_text(text)
    print(text)

if __name__=='__main__':main()
