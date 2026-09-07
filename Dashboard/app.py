"""
CTA Trend-Following Signal Dashboard — Streamlit port of Romain's "TR mapping old"
Dash app (Hardminer architecture: Parquet data, Streamlit dashboard, GitHub repo).
"""

import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Database'

st.set_page_config(page_title='CTA Trend Signals', layout='wide')

# ── House palette / instrument config ───────────────────────────────────────────

MKT_COLOR = {
    'KC': '#3D3D3D',
    'CT': '#909090',
    'SB': '#64B5F6',
    'CC': '#BF6B1A',
    'OJ': '#E65100',
}

PRICE_DECIMALS = {'KC': 2, 'CT': 2, 'SB': 2, 'CC': 0, 'OJ': 2}

INSTRUMENT_LABELS = {
    'KC': 'Coffee', 'CT': 'Cotton', 'SB': 'Sugar', 'CC': 'Cocoa', 'OJ': 'Orange Juice',
}
SHORTS = ['KC', 'CT', 'SB', 'CC', 'OJ']

ST_COLS_PREFIX = ('Mom_5', 'Mom_10', 'Mom_15', 'Mom_20', 'Mom_25')  # not used directly — full lists loaded from indicators columns

# ── Custom CSS (light theme forced, house style — no emojis, per established dashboard style) ──

# Force light theme regardless of the viewer's OS/browser preference — overrides
# Streamlit Cloud's auto dark-mode and the config.toml default in one place.
st.markdown("""
<style>
:root, .stApp { color-scheme: light !important; }
.stApp { background-color: #ffffff !important; }
.kpi-card {
    background: #f7f8fa; border-radius: 8px; padding: 14px 18px;
    border: 1px solid #e0e0e0; text-align: center;
}
.kpi-label { color: #666; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-value { color: #111; font-size: 1.6rem; font-weight: 600; margin-top: 4px; }
.kpi-sub { color: #888; font-size: 0.75rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = 'plotly_white'

# ── Data loading (cached) ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_all():
    price_df = pd.read_parquet(DATA_DIR / 'price_history.parquet')
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    fut_df = pd.read_parquet(DATA_DIR / 'futures_price.parquet')
    fut_df['Date'] = pd.to_datetime(fut_df['Date'])

    ind_df = pd.read_parquet(DATA_DIR / 'indicators.parquet')
    ind_df['Date'] = pd.to_datetime(ind_df['Date'])

    sim_df = pd.read_parquet(DATA_DIR / 'sim_history.parquet')
    sim_df['Run_Date'] = pd.to_datetime(sim_df['Run_Date'])
    sim_df['Horizon_Date'] = pd.to_datetime(sim_df['Horizon_Date'])

    return price_df, fut_df, ind_df, sim_df


price_all, fut_all, ind_all, sim_all = load_all()

if ind_all.empty:
    st.error('No indicator data found in Database/indicators.parquet. Run Code/tr_mapping_fetch.py first.')
    st.stop()

# Signal column buckets — derive from indicators.parquet columns directly (robust
# to the exact list living in Code/tr_mapping_fetch.py; avoids duplicating the
# 144-item lists here and risking drift).
NON_SIGNAL_COLS = {'Commodity', 'Date', 'CLOSE', 'ST_Avg', 'MT_Avg', 'LT_Avg', 'All_Avg', 'WAll_Avg'}
_all_cols = [c for c in ind_all.columns if c not in NON_SIGNAL_COLS]


def _bucket(col: str) -> str:
    """Classify a signal column into ST/MT/LT using the numeric period(s) in its name —
    mirrors the exact ST/MT/LT split defined in Code/tr_mapping_fetch.py (ST_COLS/MT_COLS/LT_COLS)."""
    import re
    nums = [int(n) for n in re.findall(r'\d+', col)]
    if not nums:
        return 'ST'
    n = max(nums)
    if n <= 25:
        return 'ST'
    if n <= 125:
        return 'MT'
    return 'LT'


ST_COLS = [c for c in _all_cols if _bucket(c) == 'ST']
MT_COLS = [c for c in _all_cols if _bucket(c) == 'MT']
LT_COLS = [c for c in _all_cols if _bucket(c) == 'LT']

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_instrument_data(short: str):
    # price_history.parquet / futures_price.parquet use column 'Close' (Title Case);
    # rename to 'CLOSE' here so downstream chart/KPI code has one consistent name.
    price = (price_all[price_all['Commodity'] == short].sort_values('Date')
             .set_index('Date')[['Close']].rename(columns={'Close': 'CLOSE'}))
    fut   = (fut_all[fut_all['Commodity'] == short].sort_values('Date')
             .set_index('Date')[['Close']].rename(columns={'Close': 'CLOSE'}))
    ind   = ind_all[ind_all['Commodity'] == short].sort_values('Date').set_index('Date')
    sim   = sim_all[sim_all['Commodity'] == short].sort_values(['Run_Date', 'Horizon_Day'])
    return price, fut, ind, sim


def fmt_price(short: str, val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'n/a'
    d = PRICE_DECIMALS.get(short, 2)
    return f'{val:,.{d}f}'


def kpi_card(label: str, value: str, sub: str = '') -> str:
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def date_range_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Sticky-controls-row equivalent: radio for quick ranges + custom picker."""
    c1, c2 = st.columns([2, 2])
    with c1:
        choice = st.radio(
            'Date range', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom'],
            index=2, horizontal=True, key=f'{key_prefix}_range',
        )
    max_date = df.index.max()
    if choice == 'Custom':
        with c2:
            min_date = df.index.min()
            start, end = st.date_input(
                'Custom range', value=(max_date - pd.Timedelta(days=365), max_date),
                min_value=min_date, max_value=max_date, key=f'{key_prefix}_custom',
            )
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    years_map = {'1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10}
    if choice == 'All':
        return df
    start = max_date - pd.DateOffset(years=years_map[choice])
    return df.loc[df.index >= start]


def add_tuesday_lines(fig: go.Figure, idx: pd.DatetimeIndex, row=None, col=None):
    tuesdays = idx[idx.dayofweek == 1]
    for d in tuesdays[::4]:  # every 4th Tuesday to avoid clutter, matches original spacing intent
        fig.add_vline(x=d, line_width=0.5, line_dash='dot', line_color='rgba(150,150,150,0.3)',
                       row=row, col=col)

# ── Chart builders ────────────────────────────────────────────────────────────

def chart_price(short: str, price: pd.DataFrame, fut: pd.DataFrame, show_tuesdays: bool):
    fig = go.Figure()
    color = MKT_COLOR.get(short, '#1f77b4')
    fig.add_trace(go.Scatter(x=price.index, y=price['CLOSE'], name=f'{short} GSCI Index',
                             line=dict(color=color, width=1.6)))
    if not fut.empty:
        fig.add_trace(go.Scatter(x=fut.index, y=fut['CLOSE'], name=f'{short} Futures (front-month)',
                                 line=dict(color=color, width=1.0, dash='dot'), yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Futures'))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.08), yaxis_title='GSCI Index Level',
        title=f'{INSTRUMENT_LABELS.get(short, short)} — Price',
    )
    if show_tuesdays:
        add_tuesday_lines(fig, price.index)
    return fig


def chart_signals(ind: pd.DataFrame, show_cols: list[str], composite: str):
    fig = go.Figure()
    for col in show_cols:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=col, line=dict(width=0.8), opacity=0.35))
    fig.add_trace(go.Scatter(x=ind.index, y=ind[composite], name=composite,
                             line=dict(color='#FFD54F', width=2.4)))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.1), yaxis=dict(range=[-1.05, 1.05]),
        title=f'Signals — {composite}',
    )
    return fig


def chart_projection(sim_latest: pd.DataFrame, price_actual: pd.DataFrame, signal_col: str, short: str):
    """sim_latest: rows for the most recent Run_Date, 10 horizon days, one row per scenario col set."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                        subplot_titles=(f'{signal_col} projection', 'Price scenarios'))
    scen_colors = {'up': '#4CAF50', 'down': '#EF5350', 'unch': '#9E9E9E'}
    for scen in ['up', 'down', 'unch']:
        col = f'{signal_col}_{scen}' if signal_col != 'All' else f'All_{scen}'
        if col not in sim_latest.columns:
            col = f'{signal_col}_{scen}'
        fig.add_trace(go.Scatter(x=sim_latest['Horizon_Date'], y=sim_latest[col], name=f'Signal ({scen})',
                                 line=dict(color=scen_colors[scen], width=2)), row=1, col=1)
        pcol = f'price_{scen}'
        if pcol in sim_latest.columns:
            fig.add_trace(go.Scatter(x=sim_latest['Horizon_Date'], y=sim_latest[pcol], name=f'Price ({scen})',
                                     line=dict(color=scen_colors[scen], width=1.4, dash='dash')), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)', row=1, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation='h', y=1.08))
    return fig


def chart_weekly_change(ind: pd.DataFrame, view: str):
    """Tuesday-to-Tuesday change decomposition into weighted ST/MT/LT stacked bars."""
    tues = ind[ind.index.dayofweek == 1].copy()
    if len(tues) < 2:
        return go.Figure()
    tues = tues.sort_index()

    n_st, n_mt, n_lt = len(ST_COLS), len(MT_COLS), len(LT_COLS)
    n_all = n_st + n_mt + n_lt

    if view == 'WAll':
        w_st, w_mt, w_lt = 0.20, 0.45, 0.35
        total_col = 'WAll_Avg'
    else:
        w_st, w_mt, w_lt = n_st / n_all, n_mt / n_all, n_lt / n_all
        total_col = 'All_Avg'

    d_st  = tues['ST_Avg'].diff() * w_st
    d_mt  = tues['MT_Avg'].diff() * w_mt
    d_lt  = tues['LT_Avg'].diff() * w_lt
    d_tot = tues[total_col].diff()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tues.index, y=d_st, name='ST contribution', marker_color='#64B5F6'))
    fig.add_trace(go.Bar(x=tues.index, y=d_mt, name='MT contribution', marker_color='#FFB74D'))
    fig.add_trace(go.Bar(x=tues.index, y=d_lt, name='LT contribution', marker_color='#BA68C8'))
    fig.add_trace(go.Scatter(x=tues.index, y=d_tot, name=f'Total Δ{total_col}',
                             line=dict(color='#FFD54F', width=2), mode='lines+markers'))
    fig.update_layout(barmode='relative', template=PLOTLY_TEMPLATE, height=420,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=1.1),
                      title=f'Weekly Change Decomposition ({view})')
    return fig


def overview_row(short: str) -> dict:
    price, fut, ind, sim = get_instrument_data(short)
    if ind.empty:
        return {'Commodity': short, 'Label': INSTRUMENT_LABELS.get(short, short)}
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else last
    fut_last = fut['CLOSE'].iloc[-1] if not fut.empty else np.nan
    return {
        'Commodity': short,
        'Label': INSTRUMENT_LABELS.get(short, short),
        'Futures Price': fmt_price(short, fut_last),
        'ST_Avg': round(last['ST_Avg'], 3),
        'MT_Avg': round(last['MT_Avg'], 3),
        'LT_Avg': round(last['LT_Avg'], 3),
        'All_Avg': round(last['All_Avg'], 3),
        'WAll_Avg': round(last['WAll_Avg'], 3),
        'Δ WAll_Avg (1d)': round(last['WAll_Avg'] - prev['WAll_Avg'], 3),
        'As of': last.name.date().isoformat(),
    }

# ── Sidebar navigation ──────────────────────────────────────────────────────────

st.sidebar.title('CTA Trend Signals')
tab_names = ['Overview', 'All Projections', 'All Signals'] + [
    f'{s} — {INSTRUMENT_LABELS[s]}' for s in SHORTS
]
tabs = st.tabs(tab_names)

# ── Overview tab ──────────────────────────────────────────────────────────────

with tabs[0]:
    st.subheader('Overview — All Instruments')
    rows = [overview_row(s) for s in SHORTS]
    overview_df = pd.DataFrame(rows)

    kpi_cols = st.columns(len(SHORTS))
    for i, s in enumerate(SHORTS):
        r = rows[i]
        wall = r.get('WAll_Avg', np.nan)
        delta = r.get('Δ WAll_Avg (1d)', np.nan)
        with kpi_cols[i]:
            st.markdown(kpi_card(
                f"{s} — {INSTRUMENT_LABELS[s]}",
                f"{wall:+.3f}" if pd.notna(wall) else 'n/a',
                f"Δ1d {delta:+.3f}" if pd.notna(delta) else '',
            ), unsafe_allow_html=True)

    st.markdown('')
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

# ── All Projections tab ──────────────────────────────────────────────────────

with tabs[1]:
    st.subheader('All Projections — Latest Simulation Run')
    signal_choice = st.radio('Signal', ['WAll', 'All', 'ST', 'MT', 'LT'], horizontal=True, key='allproj_signal')
    for s in SHORTS:
        price, fut, ind, sim = get_instrument_data(s)
        if sim.empty:
            continue
        latest_run = sim['Run_Date'].max()
        sim_latest = sim[sim['Run_Date'] == latest_run].sort_values('Horizon_Day')
        st.markdown(f"**{s} — {INSTRUMENT_LABELS[s]}** (run date: {latest_run.date()})")
        st.plotly_chart(chart_projection(sim_latest, price, signal_choice, s), use_container_width=True)

# ── All Signals tab ──────────────────────────────────────────────────────────

with tabs[2]:
    st.subheader('All Signals — Composite Comparison')
    composite_choice = st.radio('Composite', ['WAll_Avg', 'All_Avg', 'ST_Avg', 'MT_Avg', 'LT_Avg'],
                                horizontal=True, key='allsig_composite')
    fig = go.Figure()
    for s in SHORTS:
        _, _, ind, _ = get_instrument_data(s)
        if ind.empty:
            continue
        fig.add_trace(go.Scatter(x=ind.index, y=ind[composite_choice], name=s,
                                 line=dict(color=MKT_COLOR.get(s, None), width=1.6)))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(template=PLOTLY_TEMPLATE, height=520, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', y=1.08), yaxis=dict(range=[-1.05, 1.05]))
    st.plotly_chart(fig, use_container_width=True)

# ── Per-instrument tabs ────────────────────────────────────────────────────────

for i, short in enumerate(SHORTS):
    with tabs[3 + i]:
        price, fut, ind, sim = get_instrument_data(short)
        if ind.empty:
            st.warning(f'No data for {short}.')
            continue

        last = ind.iloc[-1]
        fut_last = fut['CLOSE'].iloc[-1] if not fut.empty else np.nan
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.markdown(kpi_card('Futures Price', fmt_price(short, fut_last)), unsafe_allow_html=True)
        kc2.markdown(kpi_card('ST_Avg', f"{last['ST_Avg']:+.3f}"), unsafe_allow_html=True)
        kc3.markdown(kpi_card('MT_Avg', f"{last['MT_Avg']:+.3f}"), unsafe_allow_html=True)
        kc4.markdown(kpi_card('LT_Avg', f"{last['LT_Avg']:+.3f}"), unsafe_allow_html=True)
        kc5.markdown(kpi_card('WAll_Avg', f"{last['WAll_Avg']:+.3f}"), unsafe_allow_html=True)
        st.markdown('')

        sub_charts, sub_weekly, sub_proj = st.tabs(['Charts', 'Weekly Change', 'Projection'])

        with sub_charts:
            show_tues = st.checkbox('Show Tuesday lines', value=False, key=f'{short}_tues')
            price_ranged = date_range_filter(price, f'{short}_price')
            st.plotly_chart(chart_price(short, price_ranged, fut, show_tues), use_container_width=True)

            composite = st.radio('Composite', ['WAll_Avg', 'All_Avg', 'ST_Avg', 'MT_Avg', 'LT_Avg'],
                                 horizontal=True, key=f'{short}_composite')
            col_map = {'ST_Avg': ST_COLS, 'MT_Avg': MT_COLS, 'LT_Avg': LT_COLS,
                      'All_Avg': ST_COLS + MT_COLS + LT_COLS, 'WAll_Avg': ST_COLS + MT_COLS + LT_COLS}
            show_underlying = st.checkbox('Show underlying signals', value=False, key=f'{short}_underlying')
            ind_ranged = date_range_filter(ind, f'{short}_ind')
            underlying = col_map[composite] if show_underlying else []
            st.plotly_chart(chart_signals(ind_ranged, underlying, composite), use_container_width=True)

        with sub_weekly:
            view = st.radio('View', ['WAll', 'All'], horizontal=True, key=f'{short}_weekview')
            ind_ranged_w = date_range_filter(ind, f'{short}_weekly')
            st.plotly_chart(chart_weekly_change(ind_ranged_w, view), use_container_width=True)

        with sub_proj:
            if sim.empty:
                st.info('No simulation history for this instrument.')
            else:
                proj_signal = st.radio('Signal', ['WAll', 'All', 'ST', 'MT', 'LT'],
                                       horizontal=True, key=f'{short}_projsignal')
                run_dates = sorted(sim['Run_Date'].unique(), reverse=True)
                run_choice = st.selectbox('Run date', run_dates, format_func=lambda d: pd.Timestamp(d).date().isoformat(),
                                          key=f'{short}_rundate')
                sim_sel = sim[sim['Run_Date'] == run_choice].sort_values('Horizon_Day')
                st.plotly_chart(chart_projection(sim_sel, price, proj_signal, short), use_container_width=True)

                proj_table = sim_sel[['Horizon_Date', 'Horizon_Day',
                                      f'{proj_signal}_up', f'{proj_signal}_down', f'{proj_signal}_unch',
                                      'price_up', 'price_down', 'price_unch', 'Actual_Close']].copy()
                proj_table['Horizon_Date'] = proj_table['Horizon_Date'].dt.date
                st.dataframe(proj_table, use_container_width=True, hide_index=True)
