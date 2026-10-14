import json
import numpy as np
import pandas as pd
import pytest
from src.data import ROOT, FEATURES, load_data, split_data, temporal_folds, validate_sample
from src.predict import load_artifacts, predict

@pytest.fixture(scope='module')
def artifacts(): return load_artifacts()

@pytest.fixture
def sample(): return json.loads((ROOT/'models/demo.json').read_text())['input']

def test_time_and_group_isolation():
    df,_=load_data();train,test,demo,_,_=split_data(df)
    assert train.date.max()<test.date.min()
    assert not set(train.group)&set(test.group)
    assert not set(demo.group)&set(pd.concat([train,test]).group)
    for a,b in temporal_folds(train):
        assert train.iloc[a].date.max()<train.iloc[b].date.min()
        assert not set(train.iloc[a].group)&set(train.iloc[b].group)

def test_known_sample_reload(artifacts,sample):
    model,meta=artifacts
    expected=json.loads((ROOT/'models/demo.json').read_text())['prediction']
    assert predict(sample,model,meta)['prediction']==pytest.approx(expected)
    assert 'Millage(KM)' not in FEATURES

@pytest.mark.parametrize('field,value',[('year',2026),('year',2010.5),('engine_cc',-10),('engine_cc',float('nan')),('brand','')])
def test_invalid_inputs(artifacts,sample,field,value):
    sample[field]=value
    with pytest.raises(ValueError): predict(sample,*artifacts)

def test_missing_input(artifacts,sample):
    del sample['fuel']
    with pytest.raises(ValueError,match='Missing fields'): predict(sample,*artifacts)

def test_unseen_category_is_warned(artifacts,sample):
    sample['town']='UNSEEN TOWN'
    result=predict(sample,*artifacts)
    assert np.isfinite(result['prediction'])
    assert any('Unseen town' in w for w in result['warnings'])

def test_missing_numeric_imputed_by_pipeline(artifacts,sample):
    frame=pd.DataFrame([sample]);frame['engine_cc']=np.nan
    assert np.isfinite(artifacts[0].predict(frame)[0])

def test_app_prediction_and_demo():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app/app.py')).run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label=='Estimate listing price').click().run(timeout=30)
    assert not app.exception
    assert any(m.label=='Estimated listing price' for m in app.metric)
    next(b for b in app.button if b.label=='Load held-out example').click().run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label=='Estimate listing price').click().run(timeout=30)
    assert not app.exception

@pytest.mark.parametrize('name',['Random Forest','XGBoost','CatBoost','LightGBM','Ridge','Stacking'])
def test_each_saved_model(name,sample):
    model,meta=load_artifacts(name)
    result=predict(sample,model,meta)
    assert result['model']==name
    assert result['prediction']==pytest.approx(meta['models'][name]['demo_prediction'])
    assert meta['test_metrics']==meta['models'][name]['test_metrics']

def test_unknown_model_rejected():
    with pytest.raises(ValueError,match='Unknown model'):
        load_artifacts('../unknown')

def test_switch_model_clears_and_recalculates():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app/app.py')).run(timeout=30)
    next(b for b in app.button if b.label=='Estimate listing price').click().run(timeout=30)
    assert not app.exception
    assert next(s for s in app.selectbox if s.label=='Prediction model').options == ['Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Ridge', 'Stacking']
    original=next(m.value for m in app.metric if m.label=='Estimated listing price')
    next(s for s in app.selectbox if s.label=='Prediction model').select('XGBoost').run(timeout=30)
    assert not app.exception
    assert not any(m.label=='Estimated listing price' for m in app.metric)
    next(b for b in app.button if b.label=='Estimate listing price').click().run(timeout=30)
    assert not app.exception
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    value=next(m.value for m in app.metric if m.label=='Estimated listing price')
    assert value==f"LKR {meta['models']['XGBoost']['demo_prediction'] * 100000:,.0f}"
    assert value!=original
    assert app.session_state['result']['model']=='XGBoost'
    mae=next(m.value for m in app.metric if m.label=='Average prediction error (MAE)')
    assert mae==f"{meta['models']['XGBoost']['test_metrics']['mae']:.2f} lakhs"


def test_lakh_to_lkr_conversion(artifacts,sample):
    result=predict(sample,*artifacts)
    assert result['unit']=='LKR lakhs'
    assert result['currency']=='LKR'
    assert result['prediction_lkr']==pytest.approx(result['prediction'] * 100000)

def test_chart_follows_model_selection(monkeypatch):
    from streamlit.testing.v1 import AppTest
    reads=[]
    original=pd.read_csv
    def track(path,*args,**kwargs):
        reads.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',track)
    app=AppTest.from_file(str(ROOT/'app/app.py')).run(timeout=30)
    for name in ['XGBoost','Random Forest','CatBoost','LightGBM','Ridge','Stacking']:
        reads.clear()
        next(s for s in app.selectbox if s.label=='Prediction model').select(name).run(timeout=30)
        assert not app.exception
        suffix='feature_importance_'+name.lower().replace(' ','_')+'.csv'
        assert any(p.endswith(suffix) for p in reads)
        assert any('Factors associated with predictions — '+name in m.value for m in app.markdown)
        assert any(name+': strengths and limitations' in m.value for m in app.markdown)
        assert any('Residual graph' in m.value for m in app.markdown)
    rf=original(ROOT/'reports/feature_importance_random_forest.csv').set_index('feature')
    xgb=original(ROOT/'reports/feature_importance_xgboost.csv').set_index('feature')
    assert set(rf.index)==set(FEATURES)==set(xgb.index)
    assert not np.allclose(rf.loc[FEATURES,'mae_increase'],xgb.loc[FEATURES,'mae_increase'])

def test_dataset_explorer_is_optional_and_filters():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app/app.py')).run(timeout=30)
    assert not app.exception
    assert not any(r.label=='Dataset view' for r in app.radio)
    next(c for c in app.checkbox if c.label=='Open dataset explorer').check().run(timeout=30)
    assert not app.exception
    assert next(m.value for m in app.metric if m.label=='Cleaned records')=='9,770'
    next(m for m in app.multiselect if m.label=='Filter data split').select('Demo').run(timeout=30)
    assert not app.exception
    tables=[d.value for d in app.dataframe if 'split' in d.value.columns]
    assert len(tables)==1 and len(tables[0])==1
    assert tables[0]['split'].iloc[0]=='Demo'
    assert 'Millage(KM)' not in tables[0].columns
    next(r for r in app.radio if r.label=='Dataset view').set_value('Raw').run(timeout=30)
    assert not app.exception
    raw=[d.value for d in app.dataframe if 'Millage(KM)' in d.value.columns][0]
    assert len(raw)==9788
    next(m for m in app.multiselect if m.label=='Filter brands').select('AUDI').run(timeout=30)
    assert not app.exception
    raw=[d.value for d in app.dataframe if 'Millage(KM)' in d.value.columns][0]
    assert set(raw.Brand)=={'AUDI'}


def test_stacking_temporal_meta_training_and_roundtrip():
    import pickle
    from sklearn.tree import DecisionTreeRegressor
    from src.stacking import TemporalStackingRegressor
    X = pd.DataFrame({'value': np.arange(48, dtype=float)})
    dates = pd.Series(pd.date_range('2024-01-01', periods=48))
    groups = pd.Series([str(i % 35) for i in range(48)])
    y = X.value.to_numpy() * 2 + 1
    model = TemporalStackingRegressor([
        ('a', DecisionTreeRegressor(max_depth=2, random_state=42)),
        ('b', DecisionTreeRegressor(max_depth=3, random_state=42)),
    ]).fit(X, y, dates=dates, groups=groups)
    validation_rows = []
    for fold in model.fold_audit_:
        tr, va = fold['train'], fold['validation']
        assert dates.iloc[tr].max() < dates.iloc[va].min()
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])
        validation_rows.extend(va)
        # Each meta feature must come from a model that never fitted this target row.
        expected = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
        positions = np.searchsorted(model.meta_rows_, va)
        np.testing.assert_allclose(model.oof_predictions_[positions, 0], expected)
    assert list(model.meta_rows_) == sorted(validation_rows)
    assert 0 not in model.meta_rows_
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X), model.predict(X))


def test_native_preprocessing_uses_training_statistics():
    from sklearn.base import clone
    from src.preprocessing import NativeCategories, NativeCatBoostRegressor
    train=load_data()[0].head(12)[FEATURES].copy()
    transform=NativeCategories().fit(train)
    unseen=train.iloc[:1].copy()
    unseen['engine_cc']=np.nan
    unseen['town']='NEW TOWN'
    result=transform.transform(unseen)
    assert result.engine_cc.iloc[0]==train.engine_cc.median()
    assert result.town.iloc[0]=='NEW TOWN'
    assert list(result.columns)==FEATURES
    assert clone(NativeCatBoostRegressor()).get_params()==NativeCatBoostRegressor().get_params()


@pytest.mark.parametrize('name',['CatBoost','LightGBM','Ridge'])
def test_part2_model_handles_unseen_categories(name,sample):
    sample['town']='UNSEEN TOWN'
    result=predict(sample,*load_artifacts(name))
    assert np.isfinite(result['prediction'])
    assert result['model']==name
    assert any('Unseen town' in w for w in result['warnings'])


def test_part2_selection_and_error_evidence():
    comparison=pd.read_csv(ROOT/'reports/model_comparison.csv')
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    assert meta['winner']==comparison.loc[comparison.cv_mae.idxmin(),'model']
    folds=pd.read_csv(ROOT/'reports/validation_folds.csv')
    assert set(folds.model)==set(meta['models'])
    for name in meta['models']:
        actual=folds.loc[folds.model==name,'mae']
        assert len(actual)==3
        assert actual.mean()==pytest.approx(comparison.set_index('model').loc[name,'cv_mae'])
    bands=pd.read_csv(ROOT/'reports/errors_by_price_band.csv')
    test=split_data(load_data()[0])[1]
    assert (bands.groupby('model')['count'].sum()==len(test)).all()
