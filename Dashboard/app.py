"""
CTA Trend-Following Signal Dashboard — Streamlit port of Romain's "TR mapping old"
Dash app (Hardminer architecture: Parquet data, Streamlit dashboard, GitHub repo).
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Database'

# Reuse the exact same 144-indicator calculate_indicators() from Code/ for the
# Monte Carlo feature instead of duplicating that logic here — avoids any risk
# of the dashboard's indicator math drifting out of sync with the ingest script.
# Guarded: Code/tr_mapping_fetch.py hard-imports pandas_ta, which pulls in numba
# — numba has no published wheel for some Python versions Streamlit Cloud may
# run, which broke the whole app's deploy when this was an unguarded import.
# requirements.txt intentionally does NOT list pandas_ta for that reason; the
# Monte Carlo feature degrades to "unavailable" instead of crashing the app
# when the import fails.
sys.path.insert(0, str(BASE_DIR / 'Code'))
try:
    from tr_mapping_fetch import calculate_indicators
    MC_AVAILABLE = True
except ImportError:
    calculate_indicators = None
    MC_AVAILABLE = False

st.set_page_config(page_title='CTA Trend Signals', layout='wide')

# ── House palette / instrument config ───────────────────────────────────────────

MKT_COLOR = {
    'KC':  '#3D3D3D',
    'CT':  '#909090',
    'SB':  '#64B5F6',
    'CC':  '#BF6B1A',
    'OJ':  '#E65100',
    'LCC': '#8D6E63',
    'LSU': '#26A69A',
    'RC':  '#6D4C41',
}

PRICE_DECIMALS = {'KC': 2, 'CT': 2, 'SB': 2, 'CC': 0, 'OJ': 2, 'LCC': 0, 'LSU': 1, 'RC': 0}

INSTRUMENT_LABELS = {
    'KC': 'Coffee', 'CT': 'Cotton', 'SB': 'Sugar', 'CC': 'Cocoa', 'OJ': 'Orange Juice',
    'LCC': 'London Cocoa', 'LSU': 'London Sugar', 'RC': 'Robusta Coffee',
}
SHORTS = ['KC', 'CT', 'SB', 'CC', 'OJ', 'LCC', 'LSU', 'RC']

# Rollex (this desk's own continuous roll-adjusted futures price) covers
# KC/CT/SB/CC/LCC/LSU/RC — no OJ. GSCI_SHORTS is the opposite gap: LCC/LSU/RC
# have no S&P GSCI single-commodity sub-index at all, so they exist as
# Rollex-only instruments. See Code/tr_mapping_fetch.py's module docstring.
ROLLEX_SHORTS = {'KC', 'CT', 'SB', 'CC', 'LCC', 'LSU', 'RC'}
GSCI_SHORTS = {'KC', 'CT', 'SB', 'CC', 'OJ'}


def effective_source(short: str, source_choice: str) -> str:
    """Falls forward/back to whichever source the instrument actually has
    data for, rather than showing an empty/broken tab: OJ has no Rollex (falls
    back to GSCI), LCC/LSU/RC have no GSCI (fall forward to Rollex)."""
    if source_choice == 'Rollex' and short not in ROLLEX_SHORTS:
        return 'GSCI'
    if source_choice == 'GSCI' and short not in GSCI_SHORTS:
        return 'Rollex'
    return source_choice

ST_COLS_PREFIX = ('Mom_5', 'Mom_10', 'Mom_15', 'Mom_20', 'Mom_25')  # not used directly — full lists loaded from indicators columns

# ── Custom CSS (light theme forced, house style — no emojis, per established dashboard style) ──

# Force light theme regardless of the viewer's OS/browser preference — overrides
# Streamlit Cloud's auto dark-mode and the config.toml default in one place.
st.markdown("""
<style>
:root, .stApp { color-scheme: light !important; }
.stApp { background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = 'plotly_white'

# ── Data loading (cached) ────────────────────────────────────────────────────────

_DATA_FILES = ['price_history.parquet', 'futures_price.parquet', 'indicators.parquet', 'sim_history.parquet']


def _data_signature() -> tuple:
    """File mtimes, passed into load_all() so st.cache_data actually invalidates
    when the parquet files change on disk. @st.cache_data's key is normally
    just the function's own bytecode + arguments — a redeploy that updates the
    *data* files (via git pull) without touching load_all()'s source would
    otherwise keep serving the stale cached DataFrames for up to the TTL (or
    until the process restarts), which is exactly what caused a KeyError on a
    'Source' column that only existed in the newly-pushed parquet."""
    return tuple((DATA_DIR / f).stat().st_mtime if (DATA_DIR / f).exists() else 0 for f in _DATA_FILES)


@st.cache_data(ttl=3600)
def load_all(_signature: tuple):
    price_df = pd.read_parquet(DATA_DIR / 'price_history.parquet')
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    fut_df = pd.read_parquet(DATA_DIR / 'futures_price.parquet')
    fut_df['Date'] = pd.to_datetime(fut_df['Date'])

    ind_df = pd.read_parquet(DATA_DIR / 'indicators.parquet')
    ind_df['Date'] = pd.to_datetime(ind_df['Date'])

    sim_df = pd.read_parquet(DATA_DIR / 'sim_history.parquet')
    sim_df['Run_Date'] = pd.to_datetime(sim_df['Run_Date'])
    sim_df['Horizon_Date'] = pd.to_datetime(sim_df['Horizon_Date'])

    # Defensive backward-compat: older cached/on-disk data without the Source
    # column (pre-Rollex) is treated as GSCI rather than crashing downstream.
    for df in (price_df, ind_df, sim_df):
        if 'Source' not in df.columns:
            df.insert(1, 'Source', 'GSCI')

    return price_df, fut_df, ind_df, sim_df


price_all, fut_all, ind_all, sim_all = load_all(_data_signature())

if ind_all.empty:
    st.error('No indicator data found in Database/indicators.parquet. Run Code/tr_mapping_fetch.py first.')
    st.stop()

# Signal column buckets — derive from indicators.parquet columns directly (robust
# to the exact list living in Code/tr_mapping_fetch.py; avoids duplicating the
# 144-item lists here and risking drift).
NON_SIGNAL_COLS = {'Commodity', 'Source', 'Date', 'CLOSE', 'ST_Avg', 'MT_Avg', 'LT_Avg', 'All_Avg', 'WAll_Avg'}
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

def get_instrument_data(short: str, source: str = 'GSCI'):
    """source: 'GSCI' or 'Rollex' — filters price_history/indicators/sim_history
    by Source in addition to Commodity. futures_price.parquet has no Source
    column (it's the raw front-month display price, only meaningful in GSCI
    mode — Rollex mode uses its own price series as the display price too)."""
    # price_history.parquet / futures_price.parquet use column 'Close' (Title Case);
    # rename to 'CLOSE' here so downstream chart/KPI code has one consistent name.
    price = (price_all[(price_all['Commodity'] == short) & (price_all['Source'] == source)]
             .sort_values('Date').set_index('Date')[['Close']].rename(columns={'Close': 'CLOSE'}))
    fut   = (fut_all[fut_all['Commodity'] == short].sort_values('Date')
             .set_index('Date')[['Close']].rename(columns={'Close': 'CLOSE'}))
    ind   = (ind_all[(ind_all['Commodity'] == short) & (ind_all['Source'] == source)]
             .sort_values('Date').set_index('Date'))
    sim   = (sim_all[(sim_all['Commodity'] == short) & (sim_all['Source'] == source)]
             .sort_values(['Run_Date', 'Horizon_Day']))
    return price, fut, ind, sim


def fmt_price(short: str, val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'n/a'
    d = PRICE_DECIMALS.get(short, 2)
    return f'{val:,.{d}f}'


def kpi_strip(items: list) -> str:
    """Super-compact one-line KPI row — replaces 5 separate bordered cards with
    a single thin strip (label + colored value pairs inline). items: list of
    (label, value_float, display_str)."""
    def _color(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '#888'
        return '#1b8a3d' if v > 0 else '#c62828' if v < 0 else '#555'

    cells = ''.join(
        f'<span style="margin-right:20px;white-space:nowrap;">'
        f'<span style="font-size:0.66rem;color:#888;text-transform:uppercase;letter-spacing:0.03em;">{label}</span> '
        f'<span style="font-size:0.88rem;font-weight:700;color:{_color(v)};">{disp}</span></span>'
        for label, v, disp in items
    )
    return (f'<div style="padding:6px 12px;background:#f7f8fa;border:1px solid #e0e0e0;'
           f'border-radius:6px;margin-bottom:8px;">{cells}</div>')


def section_header(title: str, subtitle: str = '') -> str:
    sub_html = (f'<div style="font-size:0.85rem;color:#777;margin-top:2px;">{subtitle}</div>'
                if subtitle else '')
    return f"""<div style="border-bottom:2px solid #1f77b4;padding-bottom:6px;margin:6px 0 16px 0;">
        <div style="font-size:1.35rem;font-weight:700;color:#111;">{title}</div>
        {sub_html}
    </div>"""


def _cell_style(val, numeric: bool, signed: bool) -> str:
    base = "padding:4px 9px;border-bottom:1px solid #eceff1;font-size:0.78rem;"
    base += "text-align:right;" if numeric else "text-align:left;"
    if signed and isinstance(val, (int, float)) and not pd.isna(val):
        if val > 0:
            base += "color:#1b8a3d;font-weight:600;"
        elif val < 0:
            base += "color:#c62828;font-weight:600;"
        else:
            base += "color:#444;"
    else:
        base += "color:#222;"
    return base


def _fmt_header(c: str) -> str:
    return c.replace('_', ' ')


def _fmt_cell(val, is_numeric: bool, is_signed: bool, decimals: int, num_fmt: str) -> str:
    """One consistent decimal count for every numeric cell — signed columns use
    num_fmt (with +/- sign), plain numeric columns use `decimals` places with a
    thousands separator, everything else (dates/strings) is left as-is. NaN
    always renders as an em-dash, never the literal 'nan'."""
    if is_numeric and pd.isna(val):
        return '—'
    if is_signed:
        return num_fmt.format(val)
    if is_numeric:
        return f'{val:,.{decimals}f}'
    return str(val)


def html_table(df: pd.DataFrame, signed_cols: tuple = (), num_fmt: str = '{:+.3f}', decimals: int = 2) -> str:
    """Renders a DataFrame as a fully inline-styled HTML table (no external CSS
    classes) — bordered header, zebra striping, right-aligned numerics, and
    green/red coloring on columns listed in signed_cols. Every numeric column
    (signed or not) is rounded to a single consistent decimal count so nothing
    shows raw float noise like 303.0118775986856."""
    thead = "".join(
        f'<th style="padding:5px 9px;background:#1f2937;color:#fff;font-size:0.68rem;'
        f'text-transform:uppercase;letter-spacing:0.02em;text-align:{"right" if c in signed_cols or pd.api.types.is_numeric_dtype(df[c]) else "left"};">{_fmt_header(c)}</th>'
        for c in df.columns
    )
    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg = '#ffffff' if i % 2 == 0 else '#f7f9fb'
        cells = []
        for c in df.columns:
            val = row[c]
            is_numeric = isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool)
            is_signed = c in signed_cols and is_numeric
            display = _fmt_cell(val, is_numeric, is_signed, decimals, num_fmt)
            cells.append(f'<td style="{_cell_style(val, is_numeric, is_signed)}">{display}</td>')
        rows_html.append(f'<tr style="background:{bg};">{"".join(cells)}</tr>')
    return f"""
    <div style="overflow-x:auto;border:1px solid #dfe3e8;border-radius:8px;">
    <table style="width:100%;border-collapse:collapse;font-family:inherit;">
        <thead><tr>{thead}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    """


def _proj_price_cell(v, ref, decimals: int) -> str:
    try:
        p = float(v)
        if pd.isna(p):
            raise ValueError
    except Exception:
        return '<td style="padding:3px 8px;font-family:monospace;font-size:0.72rem;text-align:right;color:#aaa;border-bottom:1px solid #f0f0f0;">—</td>'
    diff = p - ref if ref is not None else 0
    color = '#1b8a3d' if diff > 0 else '#c62828' if diff < 0 else '#444'
    return (f'<td style="padding:3px 8px;font-family:monospace;font-size:0.72rem;text-align:right;'
            f'color:{color};font-weight:700;border-bottom:1px solid #f0f0f0;">{p:,.{decimals}f}</td>')


def _proj_sig_cell(v, bold: bool = False) -> str:
    try:
        val = float(v)
        if pd.isna(val):
            raise ValueError
    except Exception:
        return '<td style="padding:3px 8px;font-family:monospace;font-size:0.72rem;text-align:right;color:#aaa;border-bottom:1px solid #f0f0f0;">—</td>'
    color = '#1b8a3d' if val > 0 else '#c62828' if val < 0 else '#555'
    weight = '700' if bold else '500'
    return (f'<td style="padding:3px 8px;font-family:monospace;font-size:0.72rem;text-align:right;'
            f'color:{color};font-weight:{weight};border-bottom:1px solid #f0f0f0;">{val * 100:+.0f}</td>')


def projection_table_html(sim_sel: pd.DataFrame, short: str, futures_name: str = None) -> str:
    """UP(day10->1) / UNCH / DOWN(1->10) projection table — mirrors the original
    Dash _projection_table() layout, always showing all 5 signals (ST/MT/LT/All/
    WAll) regardless of which one the chart is filtered to. Actual_Close is an
    addition not present in the original (backfilled outcome for past horizons)."""
    if sim_sel.empty:
        return '<div style="color:#888;font-size:0.82rem;">No simulation data.</div>'

    decimals = PRICE_DECIMALS.get(short, 2)
    s = sim_sel.set_index('Horizon_Day').sort_index()
    head_price = float(s['price_unch'].iloc[0])
    label = futures_name or short

    _TH = ('padding:4px 8px;font-size:0.68rem;color:#888;border-bottom:2px solid #dee2e6;'
          'text-align:right;background:#f8f9fa;')
    thead = (
        f'<tr><th style="{_TH}text-align:left;"></th>'
        f'<th style="{_TH}text-align:left;">{label} <span style="color:#1976D2;">Last: {head_price:,.{decimals}f}</span></th>'
        f'<th style="{_TH}">ST</th><th style="{_TH}">MT</th><th style="{_TH}">LT</th>'
        f'<th style="{_TH}">All</th><th style="{_TH}color:#7B1FA2;">WAll</th></tr>'
    )

    rows = ['<tr>'
           '<td style="padding:4px 8px;font-weight:700;color:#1b8a3d;font-size:0.75rem;">UP</td>'
           + '<td style="border-bottom:none;"></td>' * 6 + '</tr>']
    for day in range(len(s), 0, -1):
        if day not in s.index:
            continue
        row = s.loc[day]
        rows.append(
            f'<tr><td style="padding:3px 8px;color:#aaa;font-size:0.68rem;">Day {day}</td>'
            + _proj_price_cell(row.get('price_up'), head_price, decimals)
            + _proj_sig_cell(row.get('ST_up')) + _proj_sig_cell(row.get('MT_up')) + _proj_sig_cell(row.get('LT_up'))
            + _proj_sig_cell(row.get('All_up')) + _proj_sig_cell(row.get('WAll_up'), bold=True) + '</tr>'
        )

    if 1 in s.index:
        unch = s.loc[1]
        rows.append(
            '<tr><td style="padding:4px 8px;font-weight:700;color:#1976D2;font-size:0.75rem;'
            'border-top:2px solid #1976D2;border-bottom:2px solid #1976D2;">UNCH</td>'
            + _proj_price_cell(head_price, None, decimals)
            + _proj_sig_cell(unch.get('ST_unch')) + _proj_sig_cell(unch.get('MT_unch')) + _proj_sig_cell(unch.get('LT_unch'))
            + _proj_sig_cell(unch.get('All_unch')) + _proj_sig_cell(unch.get('WAll_unch'), bold=True) + '</tr>'
        )

    for day in range(1, len(s) + 1):
        if day not in s.index:
            continue
        row = s.loc[day]
        rows.append(
            f'<tr><td style="padding:3px 8px;color:#aaa;font-size:0.68rem;">Day {day}</td>'
            + _proj_price_cell(row.get('price_down'), head_price, decimals)
            + _proj_sig_cell(row.get('ST_down')) + _proj_sig_cell(row.get('MT_down')) + _proj_sig_cell(row.get('LT_down'))
            + _proj_sig_cell(row.get('All_down')) + _proj_sig_cell(row.get('WAll_down'), bold=True) + '</tr>'
        )
    rows.append('<tr>'
               '<td style="padding:4px 8px;font-weight:700;color:#c62828;font-size:0.75rem;border-top:2px solid #c62828;">DOWN</td>'
               + '<td style="border-bottom:none;"></td>' * 6 + '</tr>')

    return f"""
    <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:inherit;">
        <thead>{thead}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    <div style="color:#aaa;font-size:0.65rem;margin-top:4px;">Signal columns scaled ×100 (range −100 to +100)</div>
    </div>
    """


def get_date_range(df: pd.DataFrame, key_prefix: str) -> tuple:
    """Compact date-range control: radio for quick ranges + two independent
    From/To calendar widgets for Custom. Returns (start_ts, end_ts) rather than
    a filtered frame so the same window can be applied to more than one
    DataFrame (e.g. price + futures) without desyncing them.

    Two separate single-date st.date_input widgets (not one range-mode widget)
    — a range-mode date_input returns a single date instead of a 2-tuple until
    both ends are picked, which crashed the unpack here before; two single-date
    pickers sidestep that entirely since each always returns exactly one date."""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        choice = st.radio(
            'Date range', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom'],
            index=2, horizontal=True, key=f'{key_prefix}_range',
        )
    max_date = pd.Timestamp(df.index.max())
    min_date = pd.Timestamp(df.index.min())
    if choice == 'Custom':
        with c2:
            start = st.date_input(
                'From', value=(max_date - pd.Timedelta(days=365)).date(),
                min_value=min_date, max_value=max_date, key=f'{key_prefix}_from',
            )
        with c3:
            end = st.date_input(
                'To', value=max_date.date(),
                min_value=min_date, max_value=max_date, key=f'{key_prefix}_to',
            )
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start > end:
            start, end = end, start
        return start, end
    if choice == 'All':
        return min_date, max_date
    years_map = {'1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10}
    return max_date - pd.DateOffset(years=years_map[choice]), max_date


def apply_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)]


def date_range_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Single-DataFrame convenience wrapper around get_date_range()+apply_range()."""
    start, end = get_date_range(df, key_prefix)
    return apply_range(df, start, end)


def add_tuesday_lines(fig: go.Figure, idx: pd.DatetimeIndex, row=None, col=None):
    tuesdays = idx[idx.dayofweek == 1]
    for d in tuesdays[::4]:  # every 4th Tuesday to avoid clutter, matches original spacing intent
        fig.add_vline(x=d, line_width=0.5, line_dash='dot', line_color='rgba(150,150,150,0.3)',
                       row=row, col=col)

# ── Chart builders ────────────────────────────────────────────────────────────

def chart_price(short: str, price: pd.DataFrame, fut: pd.DataFrame, show_tuesdays: bool, source: str = 'GSCI'):
    """GSCI mode: front-month futures is the primary line (falls back to the
    GSCI index if futures history is empty), GSCI shown as a thin secondary
    overlay since it's the actual signal-computation basis — matches the
    original Dash chart_price(). Rollex mode: a single line — rollex_px is
    already a continuous, roll-adjusted price that IS both the signal basis
    and a directly presentable price, so no separate futures/secondary-axis
    overlay is needed."""
    fig = go.Figure()
    color = MKT_COLOR.get(short, '#1f77b4')

    if source == 'Rollex':
        fig.add_trace(go.Scatter(x=price.index, y=price['CLOSE'], name=f'{short} Rollex (roll-adjusted)',
                                 line=dict(color=color, width=1.8)))
        primary_index, primary_name = price.index, 'Rollex Price'
    else:
        primary, primary_name = (fut, 'Futures (front-month)') if not fut.empty else (price, 'GSCI Index')
        fig.add_trace(go.Scatter(x=primary.index, y=primary['CLOSE'], name=f'{short} {primary_name}',
                                 line=dict(color=color, width=1.8)))
        if not fut.empty:
            fig.add_trace(go.Scatter(x=price.index, y=price['CLOSE'], name=f'{short} GSCI Index (signal basis)',
                                     line=dict(color=color, width=1.0, dash='dot'), yaxis='y2'))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='GSCI'))
        primary_index = primary.index

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.08), yaxis_title=primary_name,
        title=f'{INSTRUMENT_LABELS.get(short, short)} — Price ({source})',
    )
    if show_tuesdays:
        add_tuesday_lines(fig, primary_index)
    return fig


def chart_signals(ind: pd.DataFrame, show_cols: list[str], composite: str):
    """Single selected composite + optional raw underlying signals (an addition
    beyond the original, which only ever showed the 5 composites — see
    chart_signals_all() for that). Scaled x100 to match the original's display
    convention (range -100..+100) for visual consistency across the dashboard."""
    fig = go.Figure()
    for col in show_cols:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col] * 100, name=col, line=dict(width=0.8), opacity=0.35))
    fig.add_trace(go.Scatter(x=ind.index, y=ind[composite] * 100, name=composite,
                             line=dict(color='#FFD54F', width=2.4)))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.1), yaxis=dict(range=[-105, 105], dtick=20, tickformat='.0f'),
        title=f'Signals — {composite}',
    )
    return fig


def chart_signals_all(ind: pd.DataFrame, short: str):
    """The instrument's own ST/MT/LT/All/WAll composites together — matches the
    original Dash chart_signals() used in the All Signals tab (one chart per
    instrument, all 5 composites, x100 scale)."""
    color = MKT_COLOR.get(short, '#1f77b4')
    traces = {
        'ST':   ('ST_Avg',   '#aaaaaa', 1.2, 'dot'),
        'MT':   ('MT_Avg',   '#888888', 1.2, 'dash'),
        'LT':   ('LT_Avg',   '#555555', 1.5, 'solid'),
        'All':  ('All_Avg',  color,     2.5, 'solid'),
        'WAll': ('WAll_Avg', '#7B1FA2', 2.0, 'dashdot'),
    }
    fig = go.Figure()
    for name, (col, clr, width, dash) in traces.items():
        if col not in ind.columns:
            continue
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col] * 100, name=name,
                                 line=dict(color=clr, width=width, dash=dash)))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=280, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation='h', y=1.12), yaxis=dict(range=[-105, 105], dtick=20, tickformat='.0f'),
    )
    return fig


# ── Monte Carlo signal bands (on-demand, dashboard-side) ─────────────────────
#
# The 3-scenario UP/DOWN/UNCH fan (build_simulation() in Code/) is deterministic
# — the same vol% move compounded every day for 10 days, an extreme stress path
# rather than a likely-range estimate. A real Monte Carlo needs N random paths,
# each with the FULL 144-indicator set recomputed on it — calculate_indicators()
# benchmarks at ~2.3s/call, so N paths costs ~N*2.3s. Running this for every
# instrument/source in the daily ingest batch (12 combos) was rejected as too
# expensive (N=50 -> ~23 extra minutes on the scheduled job); instead this runs
# lazily in the dashboard, only when a viewer explicitly asks for it, cached so
# the first viewer per (instrument, source, day, N) pays the cost and everyone
# else after that gets it instantly until the data changes.

@st.cache_data(ttl=86400, show_spinner=False)
def get_monte_carlo_bands(short: str, source: str, last_date_str: str, n_paths: int, seed: int = 42) -> pd.DataFrame:
    """Bootstrap N random 10-day price paths from the instrument's own recent
    daily-return distribution (not a Normal-distribution assumption — real
    historical returns preserve fat tails/skew), recompute the full indicator
    set on each, and return per-horizon-day percentile bands (p10/p25/p50/p75/
    p90) for ST/MT/LT/All/WAll. `last_date_str` is part of the cache key purely
    so the cache invalidates once new data lands — it isn't otherwise used."""
    if not MC_AVAILABLE:
        return pd.DataFrame()
    price, _, _, _ = get_instrument_data(short, source)
    if price.empty or len(price) < 300:
        return pd.DataFrame()

    horizon = 10
    base = price.tail(1500).copy()
    hist_returns = base['CLOSE'].pct_change().dropna().tail(500).to_numpy()
    if len(hist_returns) < 50:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    last_price = float(base['CLOSE'].iloc[-1])
    last_date = base.index.max()
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    composites = ['ST_Avg', 'MT_Avg', 'LT_Avg', 'All_Avg', 'WAll_Avg']
    samples = {c: np.empty((n_paths, horizon)) for c in composites}

    for i in range(n_paths):
        draws = rng.choice(hist_returns, size=horizon, replace=True)
        sim_prices = last_price * np.cumprod(1 + draws)
        ext = pd.DataFrame({'CLOSE': sim_prices}, index=future_dates)
        combined = pd.concat([base[['CLOSE']], ext])
        ind_sim = calculate_indicators(combined)
        tail = ind_sim.tail(horizon)
        for c in composites:
            samples[c][i, :] = tail[c].to_numpy()

    bands = pd.DataFrame({'Horizon_Date': future_dates, 'Horizon_Day': range(1, horizon + 1)})
    for c in composites:
        prefix = c.replace('_Avg', '')
        for pct, label in [(10, 'p10'), (25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
            bands[f'{prefix}_{label}'] = np.percentile(samples[c], pct, axis=0)
    return bands


def chart_projection(sim_sel: pd.DataFrame, price_actual: pd.DataFrame, signal_col: str, short: str,
                     ind: pd.DataFrame = None, mc_bands: pd.DataFrame = None):
    """Historical trailing signal (last 7 actual days) feeding into a 3-scenario
    10-day fan, with price labels at each node — matches the original Dash
    chart_projection() (single chart, not a 2-row subplot; price shown as text
    labels rather than a separate price panel)."""
    if sim_sel.empty:
        return go.Figure()

    def _safe_round(v, scale: float = 100) -> float:
        """round() raises ValueError on NaN — a legitimate possibility here
        (a scenario column can be NaN for an edge-case sim row). None renders
        as a gap in the Plotly line instead of crashing the whole chart."""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(fv) else round(fv * scale)

    color = MKT_COLOR.get(short, '#1f77b4')
    sig_col = f'{signal_col}_Avg'
    hist_sig = (ind[[sig_col]].tail(7) if ind is not None and not ind.empty and sig_col in ind.columns
               else pd.DataFrame())
    anchor_date = hist_sig.index[-1] if not hist_sig.empty else sim_sel['Horizon_Date'].min()
    last_sig_val = _safe_round(hist_sig[sig_col].iloc[-1], scale=100) if not hist_sig.empty else 0
    if last_sig_val is None:
        last_sig_val = 0

    fig = go.Figure()

    # Monte Carlo percentile bands — drawn first so the deterministic scenario
    # lines and the Actual line render on top of them.
    if mc_bands is not None and not mc_bands.empty and signal_col in ('ST', 'MT', 'LT', 'All', 'WAll'):
        mc = mc_bands
        p10, p25, p50, p75, p90 = (mc[f'{signal_col}_{p}'] * 100 for p in ('p10', 'p25', 'p50', 'p75', 'p90'))
        x_mc = mc['Horizon_Date']
        fig.add_trace(go.Scatter(x=x_mc, y=p90, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p10, fill='tonexty', fillcolor='rgba(120,120,120,0.15)',
                                 line=dict(width=0), name='MC 10-90%', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p75, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p25, fill='tonexty', fillcolor='rgba(90,90,90,0.28)',
                                 line=dict(width=0), name='MC 25-75%', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p50, name='MC median', mode='lines',
                                 line=dict(color='#616161', width=1.4, dash='dot')))

    if not hist_sig.empty:
        fig.add_trace(go.Scatter(x=hist_sig.index, y=hist_sig[sig_col] * 100, name='Actual',
                                 line=dict(color=color, width=2)))

    scen_style = {
        'up':   ('↑ UP',   '#1b8a3d', 'dash',    'top center'),
        'down': ('↓ DOWN', '#c62828', 'dot',     'bottom center'),
        'unch': ('UNCH',   '#9E9E9E', 'dashdot', 'middle right'),
    }
    for scen, (name, clr, dash, tpos) in scen_style.items():
        col = f'{signal_col}_{scen}'
        if col not in sim_sel.columns:
            continue
        pcol = f'price_{scen}'
        prices = sim_sel[pcol].tolist() if pcol in sim_sel.columns else []
        y_vals = [last_sig_val] + [_safe_round(v, scale=100) for v in sim_sel[col]]
        x_vals = [anchor_date] + list(sim_sel['Horizon_Date'])
        p_labels = [''] + [('' if p is None or (isinstance(p, float) and np.isnan(p)) else f'{p:,.2f}') for p in prices]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, name=name, mode='lines+markers+text',
            line=dict(color=clr, width=1.8, dash=dash), marker=dict(size=6, color=clr),
            text=p_labels, textposition=tpos, textfont=dict(size=9, color=clr),
        ))

    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=400, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=1.08), yaxis=dict(range=[-105, 105], dtick=20, tickformat='.0f'),
        title=f'{signal_col} — Signal Projection',
    )
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
    fig.update_layout(barmode='relative', template=PLOTLY_TEMPLATE, height=300,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=1.1),
                      title=f'Weekly Change Decomposition ({view})')
    return fig


def chart_weekly_change_total(ind: pd.DataFrame, view: str):
    """Simple green/red bar of the total weekly change only — a separate,
    simpler companion chart to chart_weekly_change() in the original Dash app
    (chart_weekly_change_total()), missing from the first Streamlit port."""
    tues = ind[ind.index.dayofweek == 1].copy().sort_index()
    if len(tues) < 2:
        return go.Figure()
    total_col = 'WAll_Avg' if view == 'WAll' else 'All_Avg'
    if total_col not in tues.columns:
        return go.Figure()
    d_tot = tues[total_col].diff().iloc[1:] * 100
    colors = ['#1b8a3d' if v >= 0 else '#c62828' for v in d_tot]
    fig = go.Figure(go.Bar(x=tues.index[1:], y=d_tot, marker_color=colors, name=f'Δ{total_col}'))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(template=PLOTLY_TEMPLATE, height=200, margin=dict(l=10, r=10, t=20, b=10),
                      showlegend=False, yaxis=dict(dtick=5, tickformat='+.0f'),
                      title=f'Total Weekly Change ({view})')
    return fig


def _safe_val(series: pd.Series, offset: int = 0) -> float:
    """series.iloc[-1-offset], or NaN if there aren't enough rows — mirrors the
    original _safe_val() used for the Overview change columns."""
    idx = len(series) - 1 - offset
    return float(series.iloc[idx]) if idx >= 0 else np.nan


def overview_row(short: str, source: str) -> dict:
    eff = effective_source(short, source)
    price, fut, ind, sim = get_instrument_data(short, eff)
    if ind.empty:
        return {'Commodity': short, 'Label': INSTRUMENT_LABELS.get(short, short)}
    last = ind.iloc[-1]
    # Rollex's own price doubles as the display price (see chart_price()); GSCI
    # mode uses the separate raw futures_price table.
    if eff == 'Rollex':
        fut_last = price['CLOSE'].iloc[-1] if not price.empty else np.nan
    else:
        fut_last = fut['CLOSE'].iloc[-1] if not fut.empty else np.nan

    all_v = _safe_val(ind['All_Avg'])
    chg1d  = (_safe_val(ind['WAll_Avg']) - _safe_val(ind['WAll_Avg'], 1)) if 'WAll_Avg' in ind.columns else np.nan
    chg5d  = all_v - _safe_val(ind['All_Avg'], 5)
    chg10d = all_v - _safe_val(ind['All_Avg'], 10)

    tues = ind.index[(ind.index.dayofweek == 1) & (ind.index < ind.index[-1])]
    t1_val = float(ind.loc[tues[-1], 'All_Avg']) if len(tues) >= 1 else np.nan
    t2_val = float(ind.loc[tues[-2], 'All_Avg']) if len(tues) >= 2 else np.nan
    chg_t1 = all_v - t1_val
    chg_t2 = all_v - t2_val

    return {
        'Commodity': short,
        'Label': INSTRUMENT_LABELS.get(short, short),
        'Futures Price': fmt_price(short, fut_last),
        'ST_Avg': round(last['ST_Avg'], 3),
        'MT_Avg': round(last['MT_Avg'], 3),
        'LT_Avg': round(last['LT_Avg'], 3),
        'All_Avg': round(last['All_Avg'], 3),
        'WAll_Avg': round(last['WAll_Avg'], 3),
        'Δ 1d': round(chg1d, 3),
        'Δ 5d': round(chg5d, 3),
        'Δ 10d': round(chg10d, 3),
        'Δ Tue': round(chg_t1, 3),
        'Δ 2nd Tue': round(chg_t2, 3),
        'As of': last.name.date().isoformat(),
    }

# ── Sidebar navigation ──────────────────────────────────────────────────────────

st.sidebar.markdown(
    """<div style="padding:4px 0 12px 0;">
        <div style="font-size:1.15rem;font-weight:700;color:#111;">CTA Trend Signals</div>
        <div style="font-size:0.78rem;color:#777;margin-top:2px;">
            Trend-following signal monitor
        </div>
    </div>""",
    unsafe_allow_html=True,
)

source_choice = st.sidebar.radio(
    'Signal Source', ['GSCI', 'Rollex'], index=0, key='source_choice',
    help=('GSCI: S&P GSCI single-commodity sub-index (Romain\'s original methodology, all 5 '
         'instruments, history from ~2006).\n\n'
         'Rollex: this desk\'s own continuous roll-adjusted futures price — KC/CT/SB/CC only '
         '(no OJ), history from ~2010. Same indicator math, different underlying price series, '
         'so signal values differ from GSCI.'),
)
if source_choice == 'Rollex':
    st.sidebar.caption('Rollex has no OJ coverage — the OJ tab falls back to GSCI.')
else:
    st.sidebar.caption('GSCI has no LCC/LSU/RC coverage — those tabs fall forward to Rollex.')

last_update = ind_all.loc[ind_all['Source'] == 'GSCI', 'Date'].max()
st.sidebar.caption(f"Data as of {pd.Timestamp(last_update).date().isoformat()}")

tab_names = ['Overview', 'All Projections', 'All Signals'] + [
    f'{s} — {INSTRUMENT_LABELS[s]}' for s in SHORTS
]
tabs = st.tabs(tab_names)

# ── Overview tab ──────────────────────────────────────────────────────────────

with tabs[0]:
    st.markdown(section_header(f'Overview — All Instruments ({source_choice})',
                               'Latest composite trend signals — Rollex rows fall back to GSCI for OJ'
                               if source_choice == 'Rollex' else
                               'Latest composite trend signals across the GSCI sub-index universe'),
               unsafe_allow_html=True)
    rows = [overview_row(s, source_choice) for s in SHORTS]
    overview_df = pd.DataFrame(rows)

    # KPI cards folded into the table below (they showed the same WAll_Avg/Δ
    # numbers redundantly) — one compact table instead of cards + table.
    st.markdown(
        html_table(overview_df, signed_cols=(
            'ST_Avg', 'MT_Avg', 'LT_Avg', 'All_Avg', 'WAll_Avg',
            'Δ 1d', 'Δ 5d', 'Δ 10d', 'Δ Tue', 'Δ 2nd Tue',
        )),
        unsafe_allow_html=True,
    )

# ── All Projections tab ──────────────────────────────────────────────────────

with tabs[1]:
    st.markdown(section_header('All Projections — Latest Simulation Run',
                               '10-business-day forward scenarios (up / down / unchanged) per instrument'),
               unsafe_allow_html=True)
    signal_choice = st.radio('Signal', ['WAll', 'All', 'ST', 'MT', 'LT'], horizontal=True, key='allproj_signal')
    for s in SHORTS:
        eff = effective_source(s, source_choice)
        price, fut, ind, sim = get_instrument_data(s, eff)
        if sim.empty:
            continue
        latest_run = sim['Run_Date'].max()
        sim_latest = sim[sim['Run_Date'] == latest_run].sort_values('Horizon_Day')
        fallback_note = f'  *(showing {eff} — no {source_choice} coverage)*' if eff != source_choice else ''
        st.markdown(f"**{s} — {INSTRUMENT_LABELS[s]}** (run date: {latest_run.date()}){fallback_note}")
        col_chart, col_table = st.columns([7, 5])
        with col_chart:
            st.plotly_chart(chart_projection(sim_latest, price, signal_choice, s, ind=ind),
                            width='stretch', key=f'allproj_chart_{s}')
        with col_table:
            st.markdown(
                projection_table_html(sim_latest, s),
                unsafe_allow_html=True,
            )

# ── All Signals tab ──────────────────────────────────────────────────────────

with tabs[2]:
    st.markdown(section_header('All Signals — CTA Signals per Instrument',
                               'Each instrument\'s own ST/MT/LT/All/WAll composites, stacked'),
               unsafe_allow_html=True)
    for s in SHORTS:
        eff = effective_source(s, source_choice)
        _, _, ind, _ = get_instrument_data(s, eff)
        if ind.empty:
            continue
        ind_ranged = date_range_filter(ind, f'allsig_{s}')
        fallback_note = (f'<span style="color:#c62828;margin-left:6px;font-size:0.72rem;">'
                         f'(showing {eff} — no {source_choice} coverage)</span>') if eff != source_choice else ''
        st.markdown(
            f'<div style="border-bottom:2px solid {MKT_COLOR.get(s, "#333")};padding-bottom:3px;margin:6px 0;">'
            f'<span style="font-weight:700;color:{MKT_COLOR.get(s, "#333")};">{s}</span>'
            f'<span style="color:#888;margin-left:6px;">{INSTRUMENT_LABELS[s]}</span>{fallback_note}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(chart_signals_all(ind_ranged, s), width='stretch', key=f'allsig_chart_{s}')

# ── Per-instrument tabs ────────────────────────────────────────────────────────

for i, short in enumerate(SHORTS):
    with tabs[3 + i]:
        eff = effective_source(short, source_choice)
        price, fut, ind, sim = get_instrument_data(short, eff)
        if ind.empty:
            st.warning(f'No data for {short}.')
            continue

        last = ind.iloc[-1]
        if eff == 'Rollex':
            subtitle = 'Rollex — this desk\'s own continuous roll-adjusted futures price'
        else:
            fut_last = fut['CLOSE'].iloc[-1] if not fut.empty else np.nan
            subtitle = f'Futures: {fmt_price(short, fut_last)}  |  GSCI-based trend signal'
        if eff != source_choice:
            subtitle += f'  (no {source_choice} coverage for {short} — showing {eff})'
        st.markdown(section_header(f'{short} — {INSTRUMENT_LABELS[short]}', subtitle),
                   unsafe_allow_html=True)

        # KPI row matches the original's 5 composites (ST/MT/LT/All/WAll, x100
        # int-style display); futures price moved to the header above instead
        # of a 6th card. Rendered as one compact strip instead of 5 bordered
        # cards to cut down the vertical space it takes.
        st.markdown(kpi_strip([
            ('ST',   last['ST_Avg'],   f"{last['ST_Avg'] * 100:+.0f}"),
            ('MT',   last['MT_Avg'],   f"{last['MT_Avg'] * 100:+.0f}"),
            ('LT',   last['LT_Avg'],   f"{last['LT_Avg'] * 100:+.0f}"),
            ('All',  last['All_Avg'],  f"{last['All_Avg'] * 100:+.0f}"),
            ('WAll', last['WAll_Avg'], f"{last['WAll_Avg'] * 100:+.0f}"),
        ]), unsafe_allow_html=True)

        sub_charts, sub_weekly, sub_proj = st.tabs(['Charts', 'Weekly Change', 'Projection'])

        with sub_charts:
            show_tues = st.checkbox('Show Tuesday lines', value=False, key=f'{short}_tues')
            # Same window applied to both price (GSCI) and fut (futures) so the
            # secondary-axis GSCI line doesn't fall out of sync with the
            # date-filtered primary line.
            rng_start, rng_end = get_date_range(price, f'{short}_price')
            price_ranged = apply_range(price, rng_start, rng_end)
            fut_ranged = apply_range(fut, rng_start, rng_end)
            st.plotly_chart(chart_price(short, price_ranged, fut_ranged, show_tues, source=eff),
                            width='stretch', key=f'{short}_pricechart')

            composite = st.radio('Composite', ['WAll_Avg', 'All_Avg', 'ST_Avg', 'MT_Avg', 'LT_Avg'],
                                 horizontal=True, key=f'{short}_composite')
            col_map = {'ST_Avg': ST_COLS, 'MT_Avg': MT_COLS, 'LT_Avg': LT_COLS,
                      'All_Avg': ST_COLS + MT_COLS + LT_COLS, 'WAll_Avg': ST_COLS + MT_COLS + LT_COLS}
            show_underlying = st.checkbox('Show underlying signals', value=False, key=f'{short}_underlying')
            ind_ranged = date_range_filter(ind, f'{short}_ind')
            underlying = col_map[composite] if show_underlying else []
            st.plotly_chart(chart_signals(ind_ranged, underlying, composite),
                            width='stretch', key=f'{short}_sigchart')

        with sub_weekly:
            view = st.radio('View', ['WAll', 'All'], horizontal=True, key=f'{short}_weekview')
            ind_ranged_w = date_range_filter(ind, f'{short}_weekly')
            st.plotly_chart(chart_weekly_change(ind_ranged_w, view),
                            width='stretch', key=f'{short}_weeklychart')
            st.plotly_chart(chart_weekly_change_total(ind_ranged_w, view),
                            width='stretch', key=f'{short}_weeklytotalchart')

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

                show_mc = st.checkbox(
                    'Show Monte Carlo bands', value=False, key=f'{short}_mc_toggle',
                    disabled=not MC_AVAILABLE,
                    help=('Temporarily unavailable in this deployment (missing pandas_ta dependency).'
                         if not MC_AVAILABLE else
                         'Bootstraps N random 10-day price paths from recent daily returns and '
                         'recomputes the full indicator set on each — gives a probabilistic p10-p90 '
                         '/ p25-p75 range instead of the 3 deterministic UP/DOWN/UNCH scenarios. Slow '
                         'on first run (recomputes 144 indicators x N times); cached after that.'),
                )
                mc_bands = None
                if show_mc and run_choice == run_dates[0]:
                    n_paths = st.select_slider('Paths (N)', options=[30, 50, 100], value=50, key=f'{short}_mc_n')
                    with st.spinner(f'Running {n_paths} Monte Carlo paths for {short}/{eff} '
                                    f'(~{n_paths * 2.3:.0f}s on first run)…'):
                        last_date_str = ind.index.max().isoformat()
                        mc_bands = get_monte_carlo_bands(short, eff, last_date_str, n_paths)
                    if mc_bands.empty:
                        st.warning('Not enough history to run Monte Carlo for this instrument.')
                elif show_mc:
                    st.caption('Monte Carlo bands are only available for the latest run date.')

                col_chart, col_table = st.columns([7, 5])
                with col_chart:
                    st.plotly_chart(chart_projection(sim_sel, price, proj_signal, short, ind=ind, mc_bands=mc_bands),
                                    width='stretch', key=f'{short}_projchart')
                with col_table:
                    st.markdown(projection_table_html(sim_sel, short), unsafe_allow_html=True)

                st.caption(
                    'Actual Close (backfilled outcome for past horizon dates) — not part of the '
                    'original Dash table, shown here for a quick projection-vs-actual check.'
                )
                actual_tbl = sim_sel[['Horizon_Date', 'Horizon_Day', 'Actual_Close']].copy()
                actual_tbl['Horizon_Date'] = actual_tbl['Horizon_Date'].dt.date
                st.markdown(
                    html_table(actual_tbl, decimals=PRICE_DECIMALS.get(short, 2)),
                    unsafe_allow_html=True,
                )
