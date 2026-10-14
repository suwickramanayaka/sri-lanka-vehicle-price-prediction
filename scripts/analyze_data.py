"""Training-only exploration used to justify Part 2 model candidates."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import load_data, split_data, CATEGORICAL, NUMERIC


def main():
    df, audit = load_data()
    train, _, _, _, _ = split_data(df)
    out = ROOT/'reports'
    cards = []
    for c in CATEGORICAL:
        counts = train[c].value_counts(dropna=False)
        cards.append({'feature': c, 'unique_values': len(counts),
                      'categories_under_3_rows': int((counts < 3).sum()),
                      'rows_in_rare_categories': int(counts[counts < 3].sum())})
    pd.DataFrame(cards).to_csv(out/'categorical_profile.csv', index=False)
    train[NUMERIC+['price']].describe(percentiles=[.01,.5,.9,.95,.99]).to_csv(out/'numeric_profile.csv')
    train.groupby('brand').agg(count=('price','size'),median_lakhs=('price','median')).sort_values('count',ascending=False).to_csv(out/'training_brands.csv')
    summary = {'training_rows':len(train), 'price_skew':float(train.price.skew()),
               'price_median':float(train.price.median()), 'price_p95':float(train.price.quantile(.95)),
               'price_max':float(train.price.max()), 'numeric_spearman':train[NUMERIC+['price']].corr(method='spearman').to_dict(),
               'scope':'Training partition only; no test labels used for candidate selection.'}
    (out/'eda_summary.json').write_text(json.dumps(summary,indent=2))
    fig, ax = plt.subplots(2,2,figsize=(12,8))
    ax[0,0].hist(train.price,bins=50,color='#147d92');ax[0,0].set(title='Training price distribution',xlabel='Price (lakhs)',ylabel='Listings')
    ax[0,1].scatter(train.year,train.price,s=6,alpha=.25);ax[0,1].set(title='Year and advertised price',xlabel='Manufacture year',ylabel='Price (lakhs)')
    train.brand.value_counts().head(12).sort_values().plot.barh(ax=ax[1,0],color='#147d92');ax[1,0].set(title='Most represented brands',xlabel='Listings',ylabel='Brand')
    train.groupby('date').size().plot(ax=ax[1,1],color='#147d92');ax[1,1].set(title='Training listing coverage',xlabel='Listing date',ylabel='Listings')
    fig.tight_layout();fig.savefig(out/'figures/training_eda.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
