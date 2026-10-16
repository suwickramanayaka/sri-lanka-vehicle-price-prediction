from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import altair as alt
import streamlit as st
from src.data import ROOT, CATEGORICAL
from src.predict import load_artifacts, predict

st.set_page_config(page_title='Lanka Auto | Price explorer', page_icon='🚘', layout='wide')
st.markdown("""<style>
.block-container {max-width:none!important;padding:3.5rem 2rem 2rem!important;}
h1 {font-size:2rem!important;letter-spacing:-.045em;line-height:1.2!important;}
h2 {font-size:1.25rem!important;letter-spacing:-.02em;}
h3 {font-size:1.05rem!important;}
[data-testid="stVerticalBlock"] {gap:.8rem;}
[data-testid="stMetric"] {padding:1rem 1.1rem;border:1px solid rgba(128,145,158,.22);border-radius:12px;background:rgba(128,145,158,.045);}
[data-testid="stMetricValue"] {font-size:1.65rem!important;letter-spacing:-.035em;}
[data-testid="stMetricLabel"] {opacity:.75;}
[data-testid="stForm"] {border:0;padding:0;}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button {border-radius:8px;}
[data-testid="stBaseButton-primaryFormSubmit"] {background:#087f8c;border-color:#087f8c;color:white;}
[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#096c77;border-color:#096c77;color:white;}
[data-baseweb="tab-list"] {gap:1.5rem;}
[data-baseweb="tab"] {padding-left:0;padding-right:0;}
[data-testid="stCaptionContainer"] p {font-size:.8rem;line-height:1.5;}
.brand-label {font-size:.72rem;letter-spacing:.16em;font-weight:650;color:#4aa9b0;margin-bottom:.35rem;}
.context-line {font-size:.82rem;opacity:.65;margin:0 0 1rem;}
.empty-result {border:1px dashed rgba(128,145,158,.35);border-radius:12px;padding:1.5rem;margin-bottom:.5rem;}
.empty-result strong {font-size:1.1rem;}
.empty-result p {font-size:.9rem;opacity:.65;margin:.5rem 0 0;}
@media(max-width:768px){.block-container{padding:3.5rem 1rem 1rem!important;}h1{font-size:1.6rem!important;}}
</style>""", unsafe_allow_html=True)

if not (ROOT/'models/model.pkl').exists():
    st.info('Train the project first: python -m src.train');st.stop()
# The registry modification time is part of the cache key, refreshing replaced artifacts.
@st.cache_resource
def artifacts(name, revision): return load_artifacts(name)
registry_path=ROOT/'models/metadata.json'
registry=json.loads(registry_path.read_text())
choices=list(registry.get('models',{registry['winner']:{}}))
def clear_estimate():
    # Prevent a previous model's estimate from appearing under the new selection.
    st.session_state.pop('result',None)
    st.session_state.pop('result_input',None)

heading,model_control=st.columns([2.5,1],gap='large')
with heading:
    st.markdown('<div class="brand-label">LANKA AUTO / PRICE EXPLORER</div>',unsafe_allow_html=True)
    st.title('Vehicle price intelligence')
    st.caption('Estimate a listing price. Compare the models. Understand the evidence.')
with model_control:
    selected_model=st.selectbox('Prediction model',choices,index=choices.index(registry['winner']),key='prediction_model',on_change=clear_estimate)
    st.caption('Recommended: '+registry['winner']+' · Lowest validation MAE')
model,meta=artifacts(selected_model,registry_path.stat().st_mtime_ns)
demo=json.loads((ROOT/'models/demo.json').read_text())
scores=pd.read_csv(ROOT/'reports/model_comparison.csv')
st.markdown('<div class="context-line">Historical listings · Dec 2024 – Feb 2025 &nbsp; / &nbsp; Estimates in LKR · Errors in lakhs &nbsp; / &nbsp; 1 lakh = LKR 100,000</div>',unsafe_allow_html=True)

left,right=st.columns([1.15,1],gap='large')
labels={'brand':'Brand','model':'Model','gear':'Transmission','fuel':'Fuel type','town':'Town','leasing':'Leasing status','condition':'Condition','air_condition':'Air conditioning','power_steering':'Power steering','power_mirror':'Power mirrors','power_window':'Power windows'}
with left:
    with st.container(border=True):
        title,example=st.columns([1,1])
        with title: st.subheader('01 / Vehicle details')
        with example:
            if st.button('Load held-out example',use_container_width=True):
                for key,val in demo['input'].items():
                    st.session_state['field_'+key]=int(val) if key in ['year','engine_cc'] else val
                clear_estimate()
        sample={}
        brand_columns=st.columns(2)
        for i,key in enumerate(['brand','model']):
            options=meta['categories'][key] if key=='brand' else meta['brand_models'].get(sample['brand'],meta['categories']['model'])
            default=demo['input'][key] if demo['input'][key] in options else options[0]
            if st.session_state.get('field_'+key) not in options: st.session_state['field_'+key]=default
            with brand_columns[i]: sample[key]=st.selectbox(labels[key],options,key='field_'+key)
        with st.form('vehicle'):
            a,b=st.columns(2)
            for key,value in [('year',int(demo['input']['year'])),('engine_cc',int(demo['input']['engine_cc']))]:
                if 'field_'+key not in st.session_state: st.session_state['field_'+key]=value
            with a: sample['year']=st.number_input('Year of manufacture',min_value=1956,max_value=2024,step=1,key='field_year')
            with b: sample['engine_cc']=st.number_input('Engine capacity (cc)',min_value=0,max_value=10000,step=50,key='field_engine_cc')
            for row_keys in [('gear','fuel'),('town','condition')]:
                columns=st.columns(2)
                for column,key in zip(columns,row_keys):
                    options=meta['categories'][key]
                    if 'field_'+key not in st.session_state:
                        st.session_state['field_'+key]=demo['input'][key] if demo['input'][key] in options else options[0]
                    with column: sample[key]=st.selectbox(labels[key],options,key='field_'+key)
            with st.expander('Equipment & leasing',expanded=False):
                columns=st.columns(2)
                for i,key in enumerate(['leasing','air_condition','power_steering','power_mirror','power_window']):
                    options=meta['categories'][key]
                    if 'field_'+key not in st.session_state:
                        st.session_state['field_'+key]=demo['input'][key] if demo['input'][key] in options else options[0]
                    with columns[i%2]: sample[key]=st.selectbox(labels[key],options,key='field_'+key)
            submitted=st.form_submit_button('Estimate listing price',type='primary',use_container_width=True)
        st.caption('Mileage excluded: source values are calculated from vehicle age. Equipment and leasing fields start with the example values; review them before predicting.')
        if submitted:
            try:
                st.session_state.result=predict(sample,model,meta)
                st.session_state.result_input=sample.copy()
            except ValueError as e: st.error(str(e))
with right:
    with st.container(border=True):
        st.subheader('02 / Price estimate')
        if 'result' in st.session_state:
            result=st.session_state.result
            st.metric('Estimated listing price',f"LKR {result['prediction_lkr']:,.0f}")
            used=st.session_state.result_input
            st.write(f"{used['brand']} {used['model']} · {int(used['year'])} · {used['engine_cc']:g} cc")
            st.caption(f"{result['model']} · Last submitted details. Estimate again after editing.")
            for warning in result['warnings']: st.warning(warning)
            st.download_button('Download estimate',json.dumps({'input':used,**result},indent=2),'vehicle_estimate.json','application/json')
        else:
            st.markdown('<div class="empty-result"><strong>Your estimate appears here</strong><p>Enter vehicle details, or load the example, then select Estimate listing price.</p></div>',unsafe_allow_html=True)
        st.caption('An estimate of historical asking prices, not a current market valuation.')
    error,fit=st.columns(2)
    with error: st.metric('Average prediction error (MAE)',f"{meta['test_metrics']['mae']:.2f} lakhs")
    with fit: st.metric('Test R²',f"{meta['test_metrics']['r2']:.3f}")
    st.caption('Error across unseen listings, not a price range for this car. R² is unitless, not accuracy %.')
    with st.container(border=True):
        st.write('Factors associated with predictions — '+selected_model)
        importance_file=meta['models'][selected_model]['importance_file']
        importance=pd.read_csv(ROOT/'reports'/importance_file).head(5).set_index('feature')['mae_increase']
        importance.index=importance.index.map(lambda key:labels.get(key,{'year':'Manufacture year','engine_cc':'Engine capacity'}.get(key,key)))
        st.bar_chart(importance.rename('MAE increase (lakhs)'),horizontal=True,height=210,color='#1696a3')
        st.caption('Longer bars → greater model reliance. Global importance, not a personal or causal explanation.')

st.divider()
st.subheader('03 / Model analysis')
performance,comparison_tab,dataset_tab=st.tabs(['Performance & interpretation','Compare models & data','Dataset explorer'])
with performance:
    row = scores.set_index('model').loc[selected_model]
    st.caption(('RECOMMENDED' if selected_model == registry['winner'] else 'ALTERNATIVE') + ' · Recommendation uses validation MAE.')
    from src.model_notes import MODEL_NOTES
    strengths, limitations = MODEL_NOTES[selected_model]
    if selected_model == 'Stacking':
        st.caption('Vehicle details → Random Forest + XGBoost → Ridge → final estimate. The base configurations are fixed separately from standalone tuning.')
    mae_card, rmse_card, r2_card = st.columns(3)
    with mae_card:
        st.metric('Average error · MAE', f'{row.test_mae:.2f} lakhs')
        st.caption('Lower is better · Average price difference on unseen listings.')
    with rmse_card:
        st.metric('Large-error measure · RMSE', f'{row.test_rmse:.2f} lakhs')
        st.caption('Lower is better · Gives more weight to large mistakes.')
    with r2_card:
        st.metric('Model fit · R²', f'{row.test_r2:.3f}')
        st.caption('Higher is better · Unitless, not percentage accuracy.')
    good, caution = st.columns(2)
    with good:
        with st.container(border=True):
            st.markdown('#### Strengths')
            st.markdown('\n'.join('- '+item for item in strengths))
    with caution:
        with st.container(border=True):
            st.markdown('#### Limitations')
            st.markdown('\n'.join('- '+item for item in limitations))
    st.caption(f'Training MAE: {row.train_mae:.2f} lakhs · Validation MAE: {row.cv_mae:.2f} lakhs · Test MAE: {row.test_mae:.2f} lakhs. Selection uses validation; the existing test split is exploratory.')
    slug=selected_model.lower().replace(' ','_')
    interpretation_path=ROOT/f'reports/interpretation_{slug}.md'
    diagnostics_path=ROOT/f'reports/diagnostics_{slug}.csv'
    st.markdown('#### Prediction diagnostics')
    st.caption(selected_model+' · Unseen test listings · Hover over a point to inspect its values')
    if diagnostics_path.exists():
        diagnostics=pd.read_csv(diagnostics_path)
        max_price=float(diagnostics[['actual','predicted']].max().max())
        min_price=min(0.0,float(diagnostics[['actual','predicted']].min().min()))
        points=alt.Chart(diagnostics).mark_circle(size=28,opacity=.45,color='#38b2bd').encode(
            tooltip=[alt.Tooltip('actual:Q',title='Actual (lakhs)',format='.2f'),
                     alt.Tooltip('predicted:Q',title='Predicted (lakhs)',format='.2f'),
                     alt.Tooltip('residual:Q',title='Error (lakhs)',format='.2f')])
        actual_guide,residual_guide=st.columns(2,gap='medium')
        with actual_guide:
            with st.container(border=True):
                st.markdown('**Actual vs predicted**')
                st.caption('Closer to the dashed line means a better prediction.')
                scatter=points.encode(x=alt.X('actual:Q',title='Actual price (lakhs)',scale=alt.Scale(domain=[min_price,max_price])),y=alt.Y('predicted:Q',title='Predicted price (lakhs)',scale=alt.Scale(domain=[min_price,max_price])))
                reference=alt.Chart(pd.DataFrame({'actual':[0,max_price],'predicted':[0,max_price]})).mark_line(color='#e8aa62',strokeDash=[5,5]).encode(x='actual:Q',y='predicted:Q')
                st.altair_chart((scatter+reference).properties(height=320).configure_view(strokeOpacity=0),use_container_width=True)
                st.caption('BELOW LINE · Underestimated   /   ABOVE LINE · Overestimated')
        with residual_guide:
            with st.container(border=True):
                st.markdown('**Prediction errors**')
                st.caption('Closer to zero means a smaller price difference.')
                scatter=points.encode(x=alt.X('predicted:Q',title='Predicted price (lakhs)'),y=alt.Y('residual:Q',title='Actual − predicted (lakhs)'))
                reference=alt.Chart(pd.DataFrame({'zero':[0]})).mark_rule(color='#e8aa62',strokeDash=[5,5]).encode(y='zero:Q')
                st.altair_chart((scatter+reference).properties(height=320).configure_view(strokeOpacity=0),use_container_width=True)
                st.caption('POSITIVE · Underestimated   /   NEGATIVE · Overestimated')
        st.caption('Takeaway: some expensive listings are substantially underestimated. Large errors in either direction mean the average score is not a guarantee for every car.')
    else:
        st.info('Run scripts/interpret_results.py to generate prediction diagnostics.')
    with st.expander('How to read feature importance'):
        st.write('Longer bars mean the selected model’s error increased more when that field was shuffled. The chart describes overall model reliance; it does not explain one car or prove cause and effect.')
        st.caption('Charts change when you switch models, not when you edit vehicle details.')
    if interpretation_path.exists():
        with st.expander('Full analysis and technical details', expanded=False):
            st.markdown(interpretation_path.read_text())
with comparison_tab:
    scores=pd.read_csv(ROOT/'reports/model_comparison.csv')
    st.markdown('#### Model comparison')
    st.caption('Same temporal splits · Models ordered by validation MAE · Lower MAE is the selection criterion')
    metric_specs=[('cv_mae','Validation MAE','Lower is better · Model selection','lakhs',False),
                  ('test_mae','Test MAE','Lower is better · Average error','lakhs',False),
                  ('test_rmse','Test RMSE','Lower is better · Larger errors weigh more','lakhs',False),
                  ('test_r2','Test R²','Higher is better · Not percentage accuracy','R²',True)]
    for start in [0,2]:
        panels=st.columns(2)
        for panel,(key,title,guide,unit,higher) in zip(panels,metric_specs[start:start+2]):
            with panel:
                with st.container(border=True):
                    st.markdown('**'+title+'**')
                    st.caption(guide)
                    values=scores[['model',key]].rename(columns={key:'score'})
                    top=values.loc[values.score.idxmax() if higher else values.score.idxmin()]
                    domain=[min(0,float(values.score.min())),max(1,float(values.score.max()))] if higher else [0,float(values.score.max())*1.22]
                    base=alt.Chart(values).encode(
                        y=alt.Y('model:N',title=None,sort=scores.model.tolist()),
                        x=alt.X('score:Q',title=unit,scale=alt.Scale(domain=domain)),
                        tooltip=['model:N',alt.Tooltip('score:Q',title=title,format='.3f')])
                    bars=base.mark_bar(cornerRadiusEnd=4,size=28).encode(
                        color=alt.Color('model:N',scale=alt.Scale(domain=choices,range=['#38b2bd','#e8aa62','#68b984','#ed829b','#8aa5ec','#a78bfa']),legend=None))
                    labels_chart=base.mark_text(align='left',dx=6,color='#cbd5e1').encode(text=alt.Text('score:Q',format='.3f'))
                    st.altair_chart((bars+labels_chart).properties(height=max(175,40*len(scores))).configure_view(strokeOpacity=0),use_container_width=True)
                    st.caption('Best on this metric: '+top['model'])
    st.info(f"{registry['winner']} is recommended by validation MAE. Compare test MAE for average errors and RMSE for larger mistakes. Small differences do not establish statistical superiority. This comparison uses an already examined test split; results are exploratory, not a new untouched confirmation test.")
    st.markdown('#### Detailed scores')
    currency_scores = scores[['model','cv_mae','test_mae','test_rmse','test_r2']].copy()
    currency_scores = currency_scores.rename(columns={'cv_mae':'Validation MAE (lakhs)','test_mae':'Test MAE (lakhs)','test_rmse':'Test RMSE (lakhs)','test_r2':'Test R²'})
    st.dataframe(currency_scores.round(3),hide_index=True)
    st.caption('MAE: average absolute price error. RMSE: gives larger errors more weight. Both are in lakhs and lower is better. R² is unitless: higher is better; 1 is perfect, 0 matches a constant test-mean prediction, and negative values are worse. R² is not percentage accuracy.')
    st.write('The model was selected by time-ordered validation MAE. Test listings are later than training listings, with matching feature signatures excluded across splits. Without original listing IDs, repeated vehicles cannot be identified perfectly.')
    st.write('The dataset was already preprocessed by its publisher. The lakh scale was confirmed by the project owner; original missing-value treatments cannot be independently reconstructed. Rare models, extreme prices and market changes may produce larger errors.')
    st.markdown('[Dataset: Prasad Nirmal, Kaggle](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) · Publisher metadata lists MIT licensing.')

    with st.expander('Model selection · Evidence and rationale'):
        st.markdown((ROOT/'reports/model_selection.md').read_text())
        st.image(str(ROOT/'reports/figures/training_eda.png'))
        st.caption('Training-only exploration used to choose model families.')
        st.markdown('**Validation stability** · Fold scores are descriptive, not confidence intervals.')
        st.dataframe(pd.read_csv(ROOT/'reports/validation_folds.csv'),hide_index=True)
        st.markdown('**Error by price band** · Bands use actual prices, for retrospective evaluation only.')
        st.dataframe(pd.read_csv(ROOT/'reports/errors_by_price_band.csv'),hide_index=True)

with dataset_tab:
    from src.dataset_explorer import render_dataset_explorer
    render_dataset_explorer()
