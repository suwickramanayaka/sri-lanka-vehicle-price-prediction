"""Generate Part 2 evidence from fitted artifacts without retuning."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data import load_data,split_data,FEATURES
from src.predict import load_artifacts


def main():
    train,test,_,_,_=split_data(load_data()[0])
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    scores=pd.read_csv(ROOT/'reports/model_comparison.csv')
    baseline=json.loads((ROOT/'reports/baseline.json').read_text())
    eda=json.loads((ROOT/'reports/eda_summary.json').read_text())
    bands=[];brands=[];coverage=[]
    for name in meta['models']:
        model,_=load_artifacts(name)
        pred=model.predict(test[FEATURES])
        frame=test[['brand','model','price']].copy()
        frame['error']=np.abs(test.price.to_numpy()-pred)
        frame['squared_error']=(test.price.to_numpy()-pred)**2
        frame['band']=pd.cut(frame.price,[0,25,50,100,200,np.inf],labels=['0–25','25–50','50–100','100–200','>200'])
        for band,part in frame.groupby('band',observed=True):
            bands.append({'model':name,'actual_price_band_lakhs':str(band),'count':len(part),'mae_lakhs':part.error.mean(),'rmse_lakhs':np.sqrt(part.squared_error.mean())})
        for brand,part in frame.groupby('brand'):
            brands.append({'model':name,'brand':brand,'count':len(part),'mae_lakhs':part.error.mean()})
        seen=frame['model'].isin(train.model)
        for label,mask in [('Seen model label',seen),('Unseen model label',~seen)]:
            coverage.append({'model':name,'category':label,'count':int(mask.sum()),'mae_lakhs':frame.loc[mask,'error'].mean()})
    pd.DataFrame(bands).to_csv(ROOT/'reports/errors_by_price_band.csv',index=False)
    pd.DataFrame(brands).to_csv(ROOT/'reports/errors_by_brand_all_models.csv',index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'reports/errors_by_category_coverage.csv',index=False)
    winner=scores.set_index('model').loc[meta['winner']]
    reduction=100*(baseline['test_mae']-winner.test_mae)/baseline['test_mae']
    summary=f'''### Model choice and evidence

**{meta['winner']}** is selected because it has the lowest mean validation MAE
(**{winner.cv_mae:.2f} lakhs**) among the evaluated configurations. Test MAE is
**{winner.test_mae:.2f} lakhs**, compared with **{baseline['test_mae']:.2f}** for a
training-median prediction ({reduction:.1f}% lower error on this historical split).
This demonstrates improvement over a simple local benchmark, not over external studies.

The training data has **1,061 model labels**, including **758** occurring fewer
than three times. Median training price is **{eda['price_median']:.1f} lakhs** and
95th percentile **{eda['price_p95']:.1f} lakhs**. CatBoost tests native categorical
handling; LightGBM tests another efficient nonlinear approach; Ridge provides a
simpler additive reference. RF, XGBoost and stacking retain the Part 1 comparisons.

Selection uses MAE because it measures average monetary error directly. RMSE
reveals the influence of large mistakes; R² is not percentage accuracy. Runtime
and fold variation are supporting evidence, not alternate post-hoc selection rules.
A more complex model is not automatically better. Error by actual-price band and
unseen model label must be considered before trusting an individual estimate.

**Limit:** this test split was already inspected in Part 1. Part 2 results are
exploratory. Newly collected future listings are required for independent confirmation.
The source covers December 2024–February 2025; estimates are historical asking
prices rather than current valuations or verified sale prices.
'''
    (ROOT/'reports/part2_summary.md').write_text(summary)
    print(summary)

if __name__=='__main__':main()
