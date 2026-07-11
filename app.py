# app.py
"""
Interactive Streamlit app for the Walmart Demand Forecasting system.

Run:
    streamlit run app.py

Requires a trained pipeline first:
    python main.py train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path('data/processed/walmart_clean.csv')
META_PATH = Path('models/metadata.json')
RESTOCK_PATH = Path('results/inventory/store_dept_inventory_detailed.csv')
CMP_PATH = Path('results/model_comparison.csv')
CV_PATH = Path('results/cv_results.csv')

BLUE, AQUA, GRID = '#2a78d6', '#1baf7a', '#e1e0d9'

st.set_page_config(page_title='Walmart Demand Forecasting',
                   page_icon='📈', layout='wide')


# ──────────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner='Loading sales data...')
def load_sales():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'],
                     usecols=['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday'])
    return df


@st.cache_resource(show_spinner='Loading trained pipeline...')
def load_pipeline():
    from src.persistence import load_artifacts
    bundle, meta = load_artifacts()
    return bundle, meta


@st.cache_data(show_spinner='Running recursive forecast (one-time, cached)...')
def run_forecast(weeks: int):
    from src.forecaster import RecursiveForecaster
    bundle, _ = load_pipeline()
    raw = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    fc = RecursiveForecaster(bundle['model'], bundle['processor'], bundle['clustering'])
    return fc.forecast(raw, horizon_weeks=weeks)


def _meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding='utf-8'))
    return {}


def _line_layout(fig, title=''):
    fig.update_layout(
        title=title, template='none',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, tickprefix='$', tickformat='~s'),
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right'),
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.title('📈 Demand Forecasting')
page = st.sidebar.radio('Page', ['Overview', 'Store-Dept Explorer',
                                 'Future Forecast', 'Restock Priorities', 'Model Info'])
st.sidebar.markdown('---')
meta = _meta()
if meta:
    st.sidebar.caption(f"Model: **{meta.get('model_name', '?')}**  \n"
                       f"Trained: {meta.get('trained_at', '?')}  \n"
                       f"WMAE: {meta.get('metrics', {}).get('WMAE', float('nan')):,.0f}")
else:
    st.sidebar.warning('No trained model found. Run `python main.py train` first.')

if not DATA_PATH.exists():
    st.error(f'Data file not found: `{DATA_PATH}`. Run `python scripts/merge_walmart.py` first.')
    st.stop()

sales = load_sales()


# ──────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────

if page == 'Overview':
    st.title('Network Overview')

    weekly = sales.groupby('Date')['Weekly_Sales'].sum().reset_index()
    m = meta.get('metrics', {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Stores', f"{sales['Store'].nunique()}")
    c2.metric('Departments', f"{sales['Dept'].nunique()}")
    c3.metric('History', f"{sales['Date'].nunique()} weeks")
    c4.metric('Test WMAE', f"${m.get('WMAE', float('nan')):,.0f}" if m else '—')

    fig = go.Figure(go.Scatter(x=weekly['Date'], y=weekly['Weekly_Sales'],
                               mode='lines', line=dict(color=BLUE, width=2),
                               name='Weekly sales'))
    holidays = sales.loc[sales['IsHoliday'] == True, 'Date'].unique()
    for d in holidays:
        fig.add_vline(x=pd.Timestamp(d), line_width=1, line_dash='dot', line_color='#fab219')
    st.plotly_chart(_line_layout(fig, 'Network weekly sales (dotted lines = holiday weeks)'),
                    use_container_width=True)

    st.info('Full analytics: open **results/dashboard.html** · '
            'Model diagnostics: **results/model_report.html**')

elif page == 'Store-Dept Explorer':
    st.title('Store-Dept Explorer')

    c1, c2 = st.columns(2)
    store = c1.selectbox('Store', sorted(sales['Store'].unique()))
    depts = sorted(sales.loc[sales['Store'] == store, 'Dept'].unique())
    dept = c2.selectbox('Department', depts)

    series = sales[(sales['Store'] == store) & (sales['Dept'] == dept)].sort_values('Date')

    c1, c2, c3 = st.columns(3)
    c1.metric('Avg weekly sales', f"${series['Weekly_Sales'].mean():,.0f}")
    c2.metric('Volatility (CV)',
              f"{series['Weekly_Sales'].std() / max(series['Weekly_Sales'].mean(), 1e-6):.2f}")
    c3.metric('History', f'{len(series)} weeks')

    fig = go.Figure(go.Scatter(x=series['Date'], y=series['Weekly_Sales'],
                               mode='lines', line=dict(color=BLUE, width=2), name='Actual'))
    st.plotly_chart(_line_layout(fig, f'Store {store} / Dept {dept} - weekly sales'),
                    use_container_width=True)

elif page == 'Future Forecast':
    st.title('Future Forecast (recursive)')
    st.caption('Predicts week t+1, feeds it back as a lag, repeats. '
               'Accuracy degrades with horizon - that is the honest trade-off.')

    weeks = st.slider('Horizon (weeks beyond end of data)', 1, 8, 4)
    if st.button('Run forecast', type='primary') or f'fc_{weeks}' in st.session_state:
        st.session_state[f'fc_{weeks}'] = True
        try:
            fc = run_forecast(weeks)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        totals = fc.groupby(['Date', 'IsHoliday'])['Forecast_Weekly_Sales'].sum().reset_index()
        hist = sales.groupby('Date')['Weekly_Sales'].sum().reset_index().tail(26)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Weekly_Sales'],
                                 mode='lines', name='Actual (history)',
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=totals['Date'], y=totals['Forecast_Weekly_Sales'],
                                 mode='lines+markers', name='Forecast',
                                 line=dict(color=AQUA, width=2, dash='dash')))
        st.plotly_chart(_line_layout(fig, f'Network total: last 26 weeks + {weeks}-week forecast'),
                        use_container_width=True)

        st.dataframe(
            totals.assign(Forecast=totals['Forecast_Weekly_Sales'].map('${:,.0f}'.format))
                  [['Date', 'IsHoliday', 'Forecast']],
            use_container_width=True, hide_index=True)

        csv = fc.to_csv(index=False).encode()
        st.download_button('Download per-series forecast CSV', csv,
                           f'forecast_{weeks}w.csv', 'text/csv')

elif page == 'Restock Priorities':
    st.title('Restock Priorities')
    if not RESTOCK_PATH.exists():
        st.warning('No inventory report yet - run `python main.py train` first.')
    else:
        df = pd.read_csv(RESTOCK_PATH)
        c1, c2 = st.columns(2)
        abc = c1.multiselect('ABC class', ['A', 'B', 'C'], default=['A', 'B', 'C'])
        min_sr = c2.slider('Min stockout rate (%)', 0, 100, 0)
        view = df[df['ABC'].isin(abc) & (df['Stockout_Rate'] >= min_sr)]
        st.caption(f'{len(view)} of {len(df)} series shown · values are $ (dataset has no unit prices)')
        st.dataframe(view[['Store', 'Dept', 'ABC', 'XYZ', 'Avg_Weekly_Sales',
                           'Stockout_Rate', 'Service_Level', 'Priority_Score',
                           'Restock_Quantity']].round(1),
                     use_container_width=True, hide_index=True)

elif page == 'Model Info':
    st.title('Model Info & Experiments')

    if meta:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('Trained pipeline')
            st.json({k: v for k, v in meta.items() if k != 'features'})
        with c2:
            st.subheader('Features used')
            st.write(', '.join(meta.get('features', [])) or '—')

    if CMP_PATH.exists():
        st.subheader('Model comparison (latest compare run)')
        st.dataframe(pd.read_csv(CMP_PATH).round(2), use_container_width=True, hide_index=True)

    if CV_PATH.exists():
        st.subheader('Rolling-origin cross-validation')
        cv = pd.read_csv(CV_PATH)
        st.dataframe(cv.round(2), use_container_width=True, hide_index=True)
        st.caption(f"WMAE: {cv['WMAE'].mean():,.0f} ± {cv['WMAE'].std():,.0f} across {len(cv)} folds")
