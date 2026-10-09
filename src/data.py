from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['year', 'engine_cc']
CATEGORICAL = ['brand', 'model', 'gear', 'fuel', 'town', 'leasing', 'condition',
               'air_condition', 'power_steering', 'power_mirror', 'power_window']
FEATURES = NUMERIC + CATEGORICAL
RENAME = {'Brand':'brand','Model':'model','YOM':'year','Engine (cc)':'engine_cc',
          'Gear':'gear','Fuel Type':'fuel','Town':'town','Leasing':'leasing',
          'Condition':'condition','AIR CONDITION':'air_condition',
          'POWER STEERING':'power_steering','POWER MIRROR':'power_mirror',
          'POWER WINDOW':'power_window','Price':'price','Date':'date'}

def load_data():
    raw = pd.read_csv(ROOT/'data/raw/car_price_dataset.csv')
    df = raw.rename(columns=RENAME).copy()
    df['row_id'] = np.arange(len(df))
    for c in CATEGORICAL:
        df[c] = df[c].astype('string').str.strip().str.upper().replace('', pd.NA)
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    for c in NUMERIC + ['price']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    valid = df['price'].gt(0) & df['date'].notna() & df['year'].between(1900, 2025)
    df = df.loc[valid].copy()
    df.loc[~df.engine_cc.between(0, 10000), 'engine_cc'] = np.nan
    before = len(df)
    df = df.drop_duplicates(FEATURES + ['date', 'price'])
    # Conservative proxy for re-listed cars; identity cannot be verified without listing IDs.
    df['group'] = pd.util.hash_pandas_object(df[FEATURES].fillna('MISSING'), index=False).astype(str)
    df = df.sort_values(['date','row_id']).reset_index(drop=True)
    audit = {'raw_rows':len(raw), 'clean_rows':len(df), 'invalid_rows':int((~valid).sum()),
             'duplicates_removed':before-len(df), 'missing_raw':raw.isna().sum().to_dict(),
             'mileage_formula_match_fraction':float((raw['Millage(KM)'] == (2025-raw.YOM)*11000).mean()),
             'date_min':str(df.date.min().date()), 'date_max':str(df.date.max().date()),
             'price_unit':'LKR lakhs (user-confirmed)'}
    return df, audit

def split_data(df):
    demo = df.tail(1).copy()
    pool = df.loc[~df.group.isin(demo.group)].copy()
    dates = sorted(pool.date.unique())
    cutoff = dates[int(len(dates)*0.8)]
    test = pool.loc[pool.date >= cutoff].copy()
    candidates = pool.loc[pool.date < cutoff].copy()
    train = candidates.loc[~candidates.group.isin(test.group)].copy()
    assert not set(train.group) & set(test.group)
    return train.reset_index(drop=True), test.reset_index(drop=True), demo, str(pd.Timestamp(cutoff).date()), len(candidates)-len(train)

def temporal_folds(train, n_splits=3):
    from sklearn.model_selection import TimeSeriesSplit
    dates = np.array(sorted(train.date.unique()))
    folds=[]
    for a,b in TimeSeriesSplit(n_splits=n_splits).split(dates):
        va=np.flatnonzero(train.date.isin(dates[b]))
        tr=np.flatnonzero(train.date.isin(dates[a]) & ~train.group.isin(train.iloc[va].group))
        folds.append((tr,va))
    return folds

def validate_sample(sample, metadata):
    missing=set(FEATURES)-set(sample)
    if missing: raise ValueError('Missing fields: '+', '.join(sorted(missing)))
    out={c:sample[c] for c in FEATURES}
    for c in NUMERIC:
        try: out[c]=float(out[c])
        except (ValueError, TypeError): raise ValueError(f'{c} must be numeric')
        if not np.isfinite(out[c]): raise ValueError(f'{c} must be finite')
    if out['year'] != int(out['year']) or not 1900 <= out['year'] <= 2025:
        raise ValueError('Manufacture year must be a whole year between 1900 and 2025')
    if not 0 <= out['engine_cc'] <= 10000: raise ValueError('Engine capacity must be between 0 and 10,000 cc')
    for c in CATEGORICAL:
        if not isinstance(out[c], str) or not out[c].strip(): raise ValueError(f'{c} must not be blank')
        out[c]=out[c].strip().upper()
    warnings=[]
    for c in CATEGORICAL:
        if out[c] not in metadata['categories'][c]: warnings.append(f'Unseen {c}: {out[c]}')
    for c in NUMERIC:
        lo,hi=metadata['ranges'][c]
        if not lo<=out[c]<=hi: warnings.append(f'{c} is outside the training range ({lo:g}–{hi:g})')
    if out['model'] not in metadata['brand_models'].get(out['brand'],[]):
        warnings.append('This brand/model combination was not seen during training')
    return pd.DataFrame([out], columns=FEATURES), warnings
