"""Run with: python -m src.train. All scores remain in original LKR lakhs."""
import json, pickle, hashlib, platform, time
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.stacking import TemporalStackingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from src.data import ROOT, FEATURES, NUMERIC, CATEGORICAL, load_data, split_data, temporal_folds

SEED=42

def pipeline(model):
    num=[('impute',SimpleImputer(strategy='median'))]
    pre=ColumnTransformer([('numeric',Pipeline(num),NUMERIC),('category',Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse_output=True))]),CATEGORICAL)])
    return Pipeline([('preprocess',pre),('model',model)])

def metrics(y,p):
    return {'mae':float(mean_absolute_error(y,p)), 'rmse':float(np.sqrt(mean_squared_error(y,p))), 'r2':float(r2_score(y,p))}

def main():
    for p in ['models','reports/figures','data/processed']: (ROOT/p).mkdir(parents=True,exist_ok=True)
    df,audit=load_data(); train,test,demo,cutoff,purged=split_data(df)
    folds=temporal_folds(train)
    audit.update(train_rows=len(train),test_rows=len(test),demo_rows=len(demo),test_cutoff=cutoff,
                 purged_train_rows=purged,cv_fold_sizes=[{'train':len(a),'validation':len(b)} for a,b in folds])
    (ROOT/'reports/data_audit.json').write_text(json.dumps(audit,indent=2))
    parts=[]
    for name,part in [('train',train),('test',test),('demo',demo)]:
        q=part[['row_id','date','group']].copy();q['split']=name;parts.append(q)
    pd.concat(parts).to_csv(ROOT/'data/processed/splits.csv',index=False)
    train.describe(include='all').to_csv(ROOT/'reports/training_summary.csv')
    specs={
      'Random Forest':(RandomForestRegressor(n_estimators=180,max_depth=18,min_samples_leaf=2,max_features=0.8,n_jobs=2,random_state=SEED),
          [{'model__max_depth':[12],'model__min_samples_leaf':[2]}, {'model__max_depth':[22],'model__min_samples_leaf':[1]}, {'model__max_depth':[18],'model__min_samples_leaf':[3]}]),
      'XGBoost':(XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,min_child_weight=3,reg_lambda=5,reg_alpha=0.1,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=SEED),
          [{'model__max_depth':[3],'model__n_estimators':[250]}, {'model__max_depth':[5],'model__n_estimators':[350]}, {'model__max_depth':[7],'model__n_estimators':[250]}]),
    }
    fitted={}; rows=[]; experiments=[]
    for name,(model,grid) in specs.items():
        print('Training',name,flush=True); start=time.perf_counter()
        search=GridSearchCV(pipeline(model),grid,scoring='neg_mean_absolute_error',cv=folds,n_jobs=1,refit=True,error_score='raise')
        search.fit(train[FEATURES],train.price)
        fitted[name]=search.best_estimator_
        rows.append({'model':name,'cv_mae':-float(search.best_score_),'cv_std':float(search.cv_results_['std_test_score'][search.best_index_]),'training_seconds':time.perf_counter()-start,'parameters':json.dumps(search.best_params_)})
        for i,p in enumerate(search.cv_results_['params']):
            experiments.append({'model':name,'parameters':json.dumps(p),'cv_mae':-float(search.cv_results_['mean_test_score'][i]),'cv_std':float(search.cv_results_['std_test_score'][i])})
    # Fixed base configurations avoid using outer validation labels to tune the stack.
    stack_bases = [(name, pipeline(model)) for name, (model, _) in specs.items()]
    print('Training Stacking (nested temporal folds)', flush=True)
    start = time.perf_counter()
    stack_scores = []
    for tr, va in folds:
        part = train.iloc[tr]
        stack = TemporalStackingRegressor(stack_bases).fit(
            part[FEATURES], part.price, dates=part.date, groups=part.group)
        stack_scores.append(mean_absolute_error(train.price.iloc[va], stack.predict(train[FEATURES].iloc[va])))
    fitted['Stacking'] = TemporalStackingRegressor(stack_bases).fit(
        train[FEATURES], train.price, dates=train.date, groups=train.group)
    stack_parameters = {'meta_model': 'StandardScaler + Ridge', 'alpha': 10.0,
                        'base_parameters': {name: model.get_params() for name, (model, _) in specs.items()},
                        'validation': '3 outer purged temporal folds; 3 inner folds for meta-training'}
    stack_row = {'model': 'Stacking', 'cv_mae': float(np.mean(stack_scores)),
                 'cv_std': float(np.std(stack_scores)), 'training_seconds': time.perf_counter()-start,
                 'parameters': json.dumps(stack_parameters)}
    rows.append(stack_row)
    experiments.append({k: stack_row[k] for k in ['model','parameters','cv_mae','cv_std']})
    # Freeze winner by validation MAE before computing any held-out test results.
    winner=min(rows,key=lambda r:r['cv_mae'])['model']; best=fitted[winner]
    print('Validation-selected winner:',winner,flush=True)
    for r in rows:
        m=fitted[r['model']];start=time.perf_counter();pred=m.predict(test[FEATURES])
        r.update({'test_'+k:v for k,v in metrics(test.price,pred).items()})
        r['prediction_ms_per_row']=(time.perf_counter()-start)*1000/len(test)
        r['train_mae']=float(mean_absolute_error(train.price,m.predict(train[FEATURES])))
    comparison=pd.DataFrame(rows).sort_values('cv_mae');comparison.to_csv(ROOT/'reports/model_comparison.csv',index=False)
    pd.DataFrame(experiments).to_csv(ROOT/'reports/experiments.csv',index=False)
    pred=best.predict(test[FEATURES]); errors=test[['row_id','brand','model','year','date','price']].copy()
    errors['prediction']=pred;errors['absolute_error']=abs(errors.price-pred)
    errors.to_csv(ROOT/'reports/test_predictions.csv',index=False)
    errors.groupby('brand').agg(count=('absolute_error','size'),mae=('absolute_error','mean')).sort_values('mae',ascending=False).to_csv(ROOT/'reports/errors_by_brand.csv')
    importances = {}
    for name, estimator in fitted.items():
        importance = permutation_importance(estimator, test[FEATURES], test.price,
            n_repeats=3, random_state=SEED, scoring='neg_mean_absolute_error', n_jobs=1)
        frame = pd.DataFrame({'feature': FEATURES, 'mae_increase': importance.importances_mean,
            'std': importance.importances_std}).sort_values('mae_increase', ascending=False)
        filename = 'feature_importance_' + name.lower().replace(' ', '_') + '.csv'
        frame.to_csv(ROOT/'reports'/filename, index=False)
        importances[name] = (frame, filename)
    imp = importances[winner][0]
    imp.to_csv(ROOT/'reports/feature_importance.csv', index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].hist(train.price,bins=50,color='#147d92');axes[0].set(title='Training listing prices',xlabel='Price (LKR lakhs)',ylabel='Listings')
    axes[1].scatter(test.price,pred,s=8,alpha=.4);v=max(test.price.max(),pred.max());axes[1].plot([0,v],[0,v],'k--');axes[1].set(title='Held-out predictions',xlabel='Actual (LKR lakhs)',ylabel='Predicted (LKR lakhs)')
    axes[2].scatter(pred,test.price-pred,s=8,alpha=.4);axes[2].axhline(0,color='black');axes[2].set(title='Residuals',xlabel='Predicted (LKR lakhs)',ylabel='Actual − predicted')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/evaluation.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));s=imp.head(10).iloc[::-1];ax.barh(s.feature,s.mae_increase,color='#147d92');ax.set(xlabel='Increase in test MAE after shuffling (LKR lakhs)',title='Permutation importance: selected model');fig.tight_layout();fig.savefig(ROOT/'reports/figures/importance.png',dpi=150);plt.close(fig)
    metadata={'winner':winner,'seed':SEED,'features':FEATURES,'price_unit':'LKR lakhs',
        'date_min':audit['date_min'],'date_max':audit['date_max'],'cutoff':cutoff,
        'categories':{c:sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL},
        'brand_models':{b:sorted(p.model.unique().tolist()) for b,p in train.groupby('brand')},
        'ranges':{c:[float(train[c].min()),float(train[c].max())] for c in NUMERIC},
        'versions':{p:version(p) for p in ['scikit-learn','xgboost','pandas','numpy','streamlit']},
        'python':platform.python_version(),'raw_sha256':hashlib.sha256((ROOT/'data/raw/car_price_dataset.csv').read_bytes()).hexdigest(),
        'test_metrics':metrics(test.price,pred),'selected_parameters':(stack_parameters if winner == 'Stacking' else best.named_steps['model'].get_params())}
    metadata['models'] = {}
    for name, estimator in fitted.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        with (ROOT/'models'/filename).open('wb') as f:
            pickle.dump(estimator, f)
        with (ROOT/'models'/filename).open('rb') as f:
            restored = pickle.load(f)
        np.testing.assert_allclose(restored.predict(demo[FEATURES]), estimator.predict(demo[FEATURES]))
        row = next(r for r in rows if r['model'] == name)
        metadata['models'][name] = {
            'file': filename,
            'importance_file': importances[name][1],
            'test_metrics': {k: row['test_' + k] for k in ['mae', 'rmse', 'r2']},
            'demo_prediction': float(restored.predict(demo[FEATURES])[0]),
        }
    with (ROOT/'models/model.pkl').open('wb') as f: pickle.dump(best,f)
    (ROOT/'models/metadata.json').write_text(json.dumps(metadata,indent=2))
    with (ROOT/'models/model.pkl').open('rb') as f: reloaded=pickle.load(f)
    np.testing.assert_allclose(reloaded.predict(demo[FEATURES]),best.predict(demo[FEATURES]))
    record={'input':demo[FEATURES].iloc[0].to_dict(),'actual':float(demo.price.iloc[0]),'prediction':float(reloaded.predict(demo[FEATURES])[0]),'row_id':int(demo.row_id.iloc[0])}
    (ROOT/'models/demo.json').write_text(json.dumps(record,indent=2))
    print(comparison[['model','cv_mae','test_mae','test_rmse','test_r2']].to_string(index=False),flush=True)
    print('Artifacts saved.',flush=True)

if __name__=='__main__': main()
