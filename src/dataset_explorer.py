"""Optional, read-only inspection of source and cleaned modelling records."""
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from src.data import ROOT, FEATURES, load_data, split_data

@st.cache_data
def dataset_views(revision):
    raw = pd.read_csv(ROOT/'data/raw/car_price_dataset.csv')
    clean, audit = load_data()
    train, test, demo, _, _ = split_data(clean)
    assignment = {}
    for label, frame in [('Training', train), ('Test', test), ('Demo', demo)]:
        assignment.update(dict.fromkeys(frame.row_id, label))
    processed = clean[['row_id'] + FEATURES + ['date', 'price']].copy()
    processed['split'] = processed.row_id.map(assignment).fillna('Excluded from split')
    return raw, processed, audit

def render_dataset_explorer():
    st.markdown('#### Dataset explorer')
    st.caption('Optional, read-only view. Exploring or filtering records does not change training, predictions or the saved models.')
    if not st.checkbox('Open dataset explorer', key='open_dataset_explorer'):
        return
    raw, processed, audit = dataset_views((ROOT/'data/raw/car_price_dataset.csv').stat().st_mtime_ns)
    a,b,c = st.columns(3)
    a.metric('Downloaded records', f'{len(raw):,}')
    b.metric('Cleaned records', f'{len(processed):,}')
    c.metric('Duplicates removed', str(audit['duplicates_removed']))
    with st.expander('What changed during processing?'):
        st.markdown('''- Trimmed and standardized category labels; renamed columns and parsed dates/numbers.
- Removed exact duplicates and validated dates, years and positive prices.
- Excluded the source index and calculated mileage from model inputs.
- Added a source row ID and split label for inspection. Dates and split labels are not model inputs.
- Prices stay in **LKR lakhs** in both views: multiply by 100,000 for rupees.''')
        st.caption('Processed means cleaned records before encoding. Imputation and one-hot encoding happen inside the fitted model pipeline. “Raw” is the downloaded CSV, which the publisher had already preprocessed—not original marketplace observations. Excluded split records were removed to prevent matching vehicle signatures appearing across splits.')
    source = st.radio('Dataset view', ['Processed', 'Raw'], horizontal=True, key='dataset_view')
    frame = processed if source == 'Processed' else raw
    brand_col, price_col = ('brand', 'price') if source == 'Processed' else ('Brand', 'Price')
    filters = st.columns(2)
    brands = sorted(frame[brand_col].dropna().unique().tolist())
    with filters[0]:
        selected = st.multiselect('Filter brands', brands, key='dataset_brands_'+source)
    filtered = frame.loc[frame[brand_col].isin(selected)].copy() if selected else frame.copy()
    if source == 'Processed':
        with filters[1]:
            splits = st.multiselect('Filter data split', ['Training','Test','Demo','Excluded from split'], key='dataset_splits')
        if splits:
            filtered = filtered.loc[filtered['split'].isin(splits)]
    else:
        with filters[1]: st.caption('Raw view preserves all downloaded columns, including calculated mileage and the exported index.')
    st.caption(f'{len(filtered):,} matching records · All prices in lakhs · Table and charts follow these filters')
    st.dataframe(filtered, height=340, use_container_width=True, hide_index=True)
    st.download_button('Download filtered '+source.lower()+' CSV', filtered.to_csv(index=False).encode('utf-8'),
        file_name='vehicle_'+source.lower()+'_filtered.csv', mime='text/csv', key='download_dataset_'+source)
    if filtered.empty:
        st.info('No records match these filters. Clear a brand or split filter to continue.')
        return
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('**Price distribution**')
            numeric = pd.to_numeric(filtered[price_col], errors='coerce').dropna()
            if len(numeric):
                counts, edges = np.histogram(numeric, bins=30)
                histogram = pd.DataFrame({'from':edges[:-1],'to':edges[1:],'listings':counts})
                chart = alt.Chart(histogram).mark_bar(color='#38b2bd').encode(
                    x=alt.X('from:Q',bin='binned',title='Price (lakhs)'), x2='to:Q',
                    y=alt.Y('listings:Q',title='Listings'),
                    tooltip=[alt.Tooltip('from:Q',format='.2f'),alt.Tooltip('to:Q',format='.2f'),'listings:Q'])
                st.altair_chart(chart.properties(height=240),use_container_width=True)
            st.caption('Counts of listings in each price range. High-priced outliers remain visible.')
    with right:
        with st.container(border=True):
            st.markdown('**Most represented brands**')
            counts=filtered[brand_col].value_counts().head(12).rename_axis('brand').reset_index(name='listings')
            chart=alt.Chart(counts).mark_bar(color='#38b2bd').encode(
                x=alt.X('listings:Q',title='Listings'),y=alt.Y('brand:N',sort='-x',title=None),tooltip=['brand:N','listings:Q'])
            st.altair_chart(chart.properties(height=240),use_container_width=True)
            st.caption('Top 12 brands in the filtered records. Representation is not a measure of model accuracy.')
    with st.expander('Column types and missing values'):
        profile=pd.DataFrame({'column':filtered.columns,'type':[str(t) for t in filtered.dtypes],
            'missing':filtered.isna().sum().values,'unique_values':filtered.nunique().values})
        st.dataframe(profile,hide_index=True,use_container_width=True)
        st.caption('No missing values in this file does not establish that the original collected data was complete.')
