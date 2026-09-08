"""
CTA Trend-Following Signal Dashboard — Streamlit port of Romain's "TR mapping old"
Dash app (Hardminer architecture: Parquet data, Streamlit dashboard, GitHub repo).
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Database'

# Only need MC_N_PATHS from Code/ now (for display text) — Monte Carlo bands
# themselves are precomputed by Code/tr_mapping_fetch.py's main() (each daily
# ingest run) and stored to Database/mc_bands.parquet; the dashboard just
# reads that file (get_monte_carlo_bands() below), rather than importing and
# running compute_monte_carlo_bands() live. Live-per-instrument-switch
# computation (~2-4s each, N=200 paths) was the actual cause of the dashboard
# feeling slow to load/click through — this removes that entirely.
sys.path.insert(0, str(BASE_DIR / 'Code'))
from tr_mapping_fetch import MC_N_PATHS

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

INSTRUMENT_LABELS = {
    'KC': 'Coffee', 'CT': 'Cotton', 'SB': 'Sugar', 'CC': 'Cocoa', 'OJ': 'Orange Juice',
    'LCC': 'London Cocoa', 'LSU': 'London Sugar', 'RC': 'Robusta Coffee',
}
# Paired by commodity family (NY/US contract next to its London counterpart)
# rather than alphabetically: KC/RC both coffee, CC/LCC both cocoa, SB/LSU
# both sugar, then the two standalones.
SHORTS = ['KC', 'RC', 'CC', 'LCC', 'SB', 'LSU', 'CT', 'OJ']

# Rollex-only dashboard now — GSCI was dropped for KC/CT/SB/CC (saves data +
# compute). The one exception is OJ: Rollex has no OJ coverage at all, so OJ
# alone still runs on its historical GSCI sub-index. No user-facing source
# toggle any more — each instrument has exactly one source, picked here.
def instrument_source(short: str) -> str:
    return 'GSCI' if short == 'OJ' else 'Rollex'

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

_DATA_FILES = ['price_history.parquet', 'futures_price.parquet', 'indicators.parquet',
              'sim_history.parquet', 'active_labels.parquet', 'mc_bands.parquet']


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

    label_path = DATA_DIR / 'active_labels.parquet'
    if label_path.exists():
        label_df = pd.read_parquet(label_path)
        label_df['Date'] = pd.to_datetime(label_df['Date'])
    else:
        label_df = pd.DataFrame(columns=['Commodity', 'Date', 'Active_Label'])

    # Precomputed by Code/tr_mapping_fetch.py's main() each ingest run — the
    # dashboard only ever reads this, never recomputes it (see the comment
    # by the MC_N_PATHS import above).
    mcbands_path = DATA_DIR / 'mc_bands.parquet'
    if mcbands_path.exists():
        mcbands_df = pd.read_parquet(mcbands_path)
        mcbands_df['Horizon_Date'] = pd.to_datetime(mcbands_df['Horizon_Date'])
    else:
        mcbands_df = pd.DataFrame(columns=['Commodity', 'Source', 'Horizon_Date', 'Horizon_Day'])

    # Defensive backward-compat: older cached/on-disk data without the Source
    # column (pre-Rollex) is treated as GSCI rather than crashing downstream.
    for df in (price_df, ind_df, sim_df):
        if 'Source' not in df.columns:
            df.insert(1, 'Source', 'GSCI')

    return price_df, fut_df, ind_df, sim_df, label_df, mcbands_df


price_all, fut_all, ind_all, sim_all, label_all, mcbands_all = load_all(_data_signature())

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
    mode — Rollex mode uses its own price series as the display price too).
    labels: Rollex's active-contract label (e.g. "Dec'26") indexed by Date,
    single 'Active_Label' column — empty for GSCI-sourced instruments."""
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
    if source == 'Rollex' and not label_all.empty:
        labels = (label_all[label_all['Commodity'] == short].sort_values('Date')
                 .set_index('Date')[['Active_Label']])
    else:
        labels = pd.DataFrame(columns=['Active_Label'])
    return price, fut, ind, sim, labels


def latest_active_label(labels: pd.DataFrame) -> str | None:
    if labels is None or labels.empty:
        return None
    v = labels['Active_Label'].iloc[-1]
    return None if pd.isna(v) else str(v)


def fmt_price(short: str, val: float) -> str:
    # Fixed 1 decimal everywhere — was per-instrument (PRICE_DECIMALS: 0/1/2),
    # now a single consistent precision across the whole dashboard.
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'n/a'
    return f'{val:,.1f}'


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


def html_table(df: pd.DataFrame, signed_cols: tuple = (), num_fmt: str = '{:+.0f}', decimals: int = 0) -> str:
    """Renders a DataFrame as a fully inline-styled HTML table (no external CSS
    classes) — bordered header, zebra striping, right-aligned numerics, and
    green/red coloring on columns listed in signed_cols. Every numeric column
    (signed or not) is rounded to a single consistent decimal count — whole
    numbers by default (0 decimals); price columns are pre-formatted strings
    (fmt_price, 1 decimal) upstream so they pass through unrounded here.
    Table is NOT stretched to full width — columns size to their own content
    (auto table layout), wrapped in a scrollable, inline-block container."""
    thead = "".join(
        f'<th style="padding:4px 8px;background:#1f2937;color:#fff;font-size:0.66rem;'
        f'text-transform:uppercase;letter-spacing:0.02em;white-space:nowrap;'
        f'text-align:{"right" if c in signed_cols or pd.api.types.is_numeric_dtype(df[c]) else "left"};">{_fmt_header(c)}</th>'
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
            cells.append(f'<td style="{_cell_style(val, is_numeric, is_signed)}white-space:nowrap;">{display}</td>')
        rows_html.append(f'<tr style="background:{bg};">{"".join(cells)}</tr>')
    return f"""
    <div style="overflow-x:auto;border:1px solid #dfe3e8;border-radius:8px;display:inline-block;max-width:100%;">
    <table style="width:auto;border-collapse:collapse;font-family:inherit;table-layout:auto;">
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
        return '<td style="padding:3px 6px;font-family:monospace;font-size:0.72rem;text-align:right;color:#aaa;border-bottom:1px solid #f0f0f0;white-space:nowrap;">—</td>'
    color = '#1b8a3d' if val > 0 else '#c62828' if val < 0 else '#555'
    weight = '700' if bold else '500'
    return (f'<td style="padding:3px 6px;font-family:monospace;font-size:0.72rem;text-align:right;'
            f'color:{color};font-weight:{weight};border-bottom:1px solid #f0f0f0;white-space:nowrap;">{val * 100:+.0f}</td>')


def projection_table_html(sim_sel: pd.DataFrame, short: str, futures_name: str = None,
                          active_label: str = None) -> str:
    """UP(day10->1) / UNCH / DOWN(1->10) projection table — mirrors the original
    Dash _projection_table() layout, always showing all 5 signals (ST/MT/LT/All/
    WAll) regardless of which one the chart is filtered to. Actual_Close is an
    addition not present in the original (backfilled outcome for past horizons).
    active_label (Rollex only): the currently-active contract, e.g. "Dec'26" —
    shown next to the instrument label when the price is a Rollex futures price."""
    if sim_sel.empty:
        return '<div style="color:#888;font-size:0.82rem;">No simulation data.</div>'

    decimals = 1  # price column stays 1 decimal — everything else in this table is 0
    s = sim_sel.set_index('Horizon_Day').sort_index()
    head_price = float(s['price_unch'].iloc[0])
    label = futures_name or short
    label_suffix = f' · {active_label}' if active_label else ''

    _TH = ('padding:3px 6px;font-size:0.66rem;color:#888;border-bottom:2px solid #dee2e6;'
          'text-align:right;background:#f8f9fa;white-space:nowrap;')
    thead = (
        f'<tr><th style="{_TH}text-align:left;"></th>'
        f'<th style="{_TH}text-align:left;">{label}{label_suffix} <span style="color:#1976D2;">Last: {head_price:,.{decimals}f}</span></th>'
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
    <div style="overflow-x:auto;display:inline-block;max-width:100%;">
    <table style="border-collapse:collapse;width:auto;font-family:inherit;">
        <thead>{thead}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    <div style="color:#aaa;font-size:0.65rem;margin-top:4px;">Signal columns scaled ×100 (range −100 to +100), whole numbers</div>
    </div>
    """


def get_date_range(df: pd.DataFrame, key_prefix: str) -> tuple:
    """Sidebar date-range control: radio for quick ranges + two independent
    From/To calendar widgets for Custom, stacked vertically (sidebar is narrow
    — no st.columns). One global control instead of a separate picker on every
    Charts/Weekly Change/All Signals view; returns (start_ts, end_ts) rather
    than a filtered frame so it can be applied to several DataFrames (price,
    futures, indicators, ...) without desyncing them.

    Two separate single-date st.date_input widgets (not one range-mode widget)
    — a range-mode date_input returns a single date instead of a 2-tuple until
    both ends are picked, which crashed the unpack here before; two single-date
    pickers sidestep that entirely since each always returns exactly one date."""
    choice = st.sidebar.radio(
        'Date range', ['3M', '1Y', '3Y', '5Y', '10Y', 'All', 'Custom'],
        index=0, horizontal=True, key=f'{key_prefix}_range',
    )
    max_date = pd.Timestamp(df.index.max())
    min_date = pd.Timestamp(df.index.min())
    if choice == 'Custom':
        start = st.sidebar.date_input(
            'From', value=(max_date - pd.Timedelta(days=365)).date(),
            min_value=min_date, max_value=max_date, key=f'{key_prefix}_from',
        )
        end = st.sidebar.date_input(
            'To', value=max_date.date(),
            min_value=min_date, max_value=max_date, key=f'{key_prefix}_to',
        )
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start > end:
            start, end = end, start
        return start, end
    if choice == 'All':
        return min_date, max_date
    if choice == '3M':
        return max_date - pd.DateOffset(months=3), max_date
    years_map = {'1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10}
    return max_date - pd.DateOffset(years=years_map[choice]), max_date


def apply_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)]




def add_tuesday_lines(fig: go.Figure, idx: pd.DatetimeIndex, row=None, col=None):
    tuesdays = idx[idx.dayofweek == 1]
    for d in tuesdays[::4]:  # every 4th Tuesday to avoid clutter, matches original spacing intent
        fig.add_vline(x=d, line_width=0.5, line_dash='dot', line_color='rgba(150,150,150,0.3)',
                       row=row, col=col)

# ── Chart builders ────────────────────────────────────────────────────────────

def chart_price(short: str, price: pd.DataFrame, fut: pd.DataFrame, show_tuesdays: bool, source: str = 'GSCI',
                active_label: str = None):
    """GSCI mode: front-month futures is the primary line (falls back to the
    GSCI index if futures history is empty), GSCI shown as a thin secondary
    overlay since it's the actual signal-computation basis — matches the
    original Dash chart_price(). Rollex mode: a single line — rollex_px is
    already a continuous, roll-adjusted price that IS both the signal basis
    and a directly presentable price, so no separate futures/secondary-axis
    overlay is needed. active_label (Rollex only): current active contract,
    e.g. "Dec'26", appended to the legend/title so the chart shows which
    contract the price series is currently rolled onto."""
    fig = go.Figure()
    color = MKT_COLOR.get(short, '#1f77b4')
    label_suffix = f" · {active_label}" if active_label else ''

    if source == 'Rollex':
        fig.add_trace(go.Scatter(x=price.index, y=price['CLOSE'], name=f'{short} Rollex (roll-adjusted){label_suffix}',
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
        title=f'{INSTRUMENT_LABELS.get(short, short)} — Price ({source}{label_suffix})',
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),  # no weekend gaps
    )
    if show_tuesdays:
        add_tuesday_lines(fig, primary_index)
    return fig


def chart_price_split(short: str, price: pd.DataFrame, fut: pd.DataFrame, sim_sel: pd.DataFrame,
                      source: str = 'GSCI', active_label: str = None):
    """Split-panel version of chart_price() — same column proportions
    (0.72/0.28, 0.02 spacing) as chart_projection_split(), so this chart's
    left panel lines up vertically with the signal chart's left panel right
    below it on the page (a plain single-panel price chart above the split
    signal chart was visibly misaligned once the projection fan's right
    panel was added). Right panel shows the same deterministic UP/DOWN/UNCH
    price scenarios (sim_sel's price_up/down/unch, the ones feeding the
    signal projection below) continuing from the last actual price point —
    GSCI mode's thin secondary-axis GSCI overlay is dropped here for
    simplicity (only OJ uses GSCI; see plain chart_price() for that)."""
    color = MKT_COLOR.get(short, '#1f77b4')
    label_suffix = f" · {active_label}" if active_label else ''

    if source == 'Rollex':
        primary, primary_name = price, f'{short} Rollex (roll-adjusted){label_suffix}'
    else:
        primary, primary_name = ((fut, f'{short} Futures (front-month)') if not fut.empty
                                 else (price, f'{short} GSCI Index'))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.72, 0.28], horizontal_spacing=0.02,
        subplot_titles=('History (chosen date range)', 'Projection (10d)'),
    )
    fig.add_trace(go.Scatter(x=primary.index, y=primary['CLOSE'], name=primary_name,
                             line=dict(color=color, width=1.6), showlegend=True,
                             hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>'), row=1, col=1)
    if not primary.empty:
        fig.update_xaxes(range=[primary.index.min(), primary.index.max()],
                         rangebreaks=[dict(bounds=['sat', 'mon'])], row=1, col=1)

    # ── Right panel: deterministic UP/DOWN/UNCH price scenarios, continuing ──
    # from the last actual price point — same scenarios feeding the signal
    # projection chart below, just in price terms instead of signal terms.
    if sim_sel is not None and not sim_sel.empty and not primary.empty:
        last_price = float(primary['CLOSE'].iloc[-1])
        anchor_date = primary.index[-1]
        scen_style = {
            'up':   ('↑ UP',   '#1b8a3d', 'dash'),
            'down': ('↓ DOWN', '#c62828', 'dot'),
            'unch': ('UNCH',   '#9E9E9E', 'dashdot'),
        }
        for scen, (name, clr, dash) in scen_style.items():
            pcol = f'price_{scen}'
            if pcol not in sim_sel.columns:
                continue
            x_vals = [anchor_date] + list(sim_sel['Horizon_Date'])
            y_vals = [last_price] + sim_sel[pcol].tolist()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, name=name, mode='lines+markers',
                line=dict(color=clr, width=1.6, dash=dash), marker=dict(size=5, color=clr),
                hovertemplate=f'{name}<br>' + '%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>',
            ), row=1, col=2)
        right_end = sim_sel['Horizon_Date'].max()
        fig.update_xaxes(range=[anchor_date, right_end], rangebreaks=[dict(bounds=['sat', 'mon'])], row=1, col=2)
    else:
        fig.update_xaxes(visible=False, row=1, col=2)

    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.12, font=dict(size=10)),
        title=f'{INSTRUMENT_LABELS.get(short, short)} — Price ({source}{label_suffix})',
    )
    for ann in fig.layout.annotations:  # subplot_titles come back as small grey captions
        ann.font = dict(size=10, color='#999')
    return fig


def _indicator_family(col: str) -> str:
    """'Mom_5' -> 'Mom', 'MA_cross_(5, 10)' -> 'MA', '3MA_cross_(...)' -> '3MA',
    etc. — the leading family name shared by every one of PARAMS' 10 indicator
    families in Code/tr_mapping_fetch.py."""
    return col.split('_', 1)[0]


def chart_signals(ind: pd.DataFrame, composite: str, underlying_cols: list[str], detail: str = 'family'):
    """Single selected composite plus an optional underlying-signal overlay.
    detail:
      'none'   — composite line only.
      'family' — underlying signals bucketed/averaged by indicator family
                 (Mom/MA/EMA/HMA/3MA/BB/DC/LRS/TRIX/KAMA — up to 10 lines)
                 instead of every raw signal at once, so the chart stays
                 readable instead of showing "sab shit together".
      'raw'    — every underlying raw signal column (original behavior,
                 up to 144 faint lines) for anyone who wants the full detail.
    Scaled x100 to match the dashboard's -100..+100 display convention."""
    fig = go.Figure()
    if detail == 'family':
        fam_map: dict[str, list[str]] = {}
        for col in underlying_cols:
            fam_map.setdefault(_indicator_family(col), []).append(col)
        for fam, cols in fam_map.items():
            fam_avg = ind[cols].mean(axis=1) * 100
            fig.add_trace(go.Scatter(x=ind.index, y=fam_avg, name=fam, line=dict(width=1.1), opacity=0.55))
    elif detail == 'raw':
        for col in underlying_cols:
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col] * 100, name=col, line=dict(width=0.8), opacity=0.35))
    fig.add_trace(go.Scatter(x=ind.index, y=ind[composite] * 100, name=composite,
                             line=dict(color='#1a237e', width=2.4)))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.1), yaxis=dict(range=[-105, 105], dtick=20, tickformat='.0f'),
        title=f'Signals — {composite}',
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
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
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
    )
    return fig


# ── Monte Carlo signal bands (precomputed — see Code/tr_mapping_fetch.py) ──────
#
# The 3-scenario UP/DOWN/UNCH fan (build_simulation() in Code/) is deterministic
# — the same vol% move compounded every day for 10 days, an extreme stress path
# rather than a likely-range estimate. This gives a real probabilistic range
# instead: N random paths, each with the full indicator set recomputed, reduced
# to percentile bands. Used to be computed live here on first view per
# instrument/session (~2-4s each) — that per-instrument-switch delay was the
# actual cause of the dashboard feeling slow. Now precomputed once per
# instrument/source in Code/tr_mapping_fetch.py's main() (each daily ingest
# run, MC_N_PATHS=100 paths) and stored to Database/mc_bands.parquet —
# get_monte_carlo_bands() below is just an instant read of that.

def get_monte_carlo_bands(short: str, source: str) -> pd.DataFrame:
    sub = mcbands_all[(mcbands_all['Commodity'] == short) & (mcbands_all['Source'] == source)]
    if sub.empty:
        return pd.DataFrame()
    return sub.drop(columns=['Commodity', 'Source']).sort_values('Horizon_Day').reset_index(drop=True)


def _safe_round(v, scale: float = 100) -> float:
    """round() raises ValueError on NaN — a legitimate possibility here (a
    scenario/MC column can be NaN for an edge-case row). None renders as a
    gap in the Plotly line instead of crashing the whole chart."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(fv) else round(fv * scale)


def chart_projection(sim_sel: pd.DataFrame, price_actual: pd.DataFrame, signal_col: str, short: str,
                     ind: pd.DataFrame = None, mc_bands: pd.DataFrame = None, hist_df: pd.DataFrame = None):
    """Historical trailing signal feeding into a 3-scenario 10-day fan, with
    price labels at each node — matches the original Dash chart_projection()
    (single chart, not a 2-row subplot; price shown as text labels rather than
    a separate price panel). By default shows the last 7 actual days of
    history (ind.tail(7)); pass hist_df (e.g. the sidebar's full date-range
    selection) to show a longer trailing history feeding into the same
    projection fan instead. See chart_projection_split() for the 'Full
    History + Projection' sub-tab's split-panel version of this (long history
    got too cluttered squeezed next to a 10-day fan in one continuous axis)."""
    if sim_sel.empty:
        return go.Figure()

    color = MKT_COLOR.get(short, '#1f77b4')
    sig_col = f'{signal_col}_Avg'
    if hist_df is not None and not hist_df.empty and sig_col in hist_df.columns:
        hist_sig = hist_df[[sig_col]]
    else:
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
        # Prepend the anchor point (same value on every percentile, converging
        # to a single point) so the band starts exactly where "Actual" ends
        # instead of a visible gap between the two.
        x_mc = [anchor_date] + list(mc['Horizon_Date'])
        p10, p25, p50, p75, p90 = (
            [last_sig_val] + (mc[f'{signal_col}_{p}'] * 100).tolist()
            for p in ('p10', 'p25', 'p50', 'p75', 'p90')
        )
        # mode='lines' is required on every one of these — Plotly defaults a
        # Scatter trace to 'lines+markers' when it has under ~20 points (our
        # 10-day horizon always does), so without it each band-boundary trace
        # sprouts a stray, differently-auto-colored marker dot per point.
        fig.add_trace(go.Scatter(x=x_mc, y=p90, mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p10, mode='lines', fill='tonexty', fillcolor='rgba(140,140,140,0.14)',
                                 line=dict(width=0), name='MC 10–90%', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p75, mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(100,100,100,0.26)',
                                 line=dict(width=0), name='MC 25–75%', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_mc, y=p50, name='MC median', mode='lines',
                                 line=dict(color='#757575', width=1.3, dash='dot'), hoverinfo='skip'))

    if not hist_sig.empty:
        fig.add_trace(go.Scatter(x=hist_sig.index, y=hist_sig[sig_col] * 100, name='Actual',
                                 line=dict(color=color, width=2)))

    scen_style = {
        'up':   ('↑ UP',   '#1b8a3d', 'dash', 'top center'),
        'down': ('↓ DOWN', '#c62828', 'dot',  'bottom center'),
        'unch': ('UNCH',   '#9E9E9E', 'dot',  'middle right'),
    }
    for scen, (name, clr, dash, tpos) in scen_style.items():
        col = f'{signal_col}_{scen}'
        if col not in sim_sel.columns:
            continue
        y_vals = [last_sig_val] + [_safe_round(v, scale=100) for v in sim_sel[col]]
        x_vals = [anchor_date] + list(sim_sel['Horizon_Date'])
        if scen == 'unch':
            # No price labels on UNCH — just a plain dotted line, no markers/text.
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=name, mode='lines',
                                     line=dict(color=clr, width=1.8, dash=dash)))
            continue
        pcol = f'price_{scen}'
        prices = sim_sel[pcol].tolist() if pcol in sim_sel.columns else []
        p_labels = [''] + [('' if p is None or (isinstance(p, float) and np.isnan(p)) else f'{p:,.1f}') for p in prices]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, name=name, mode='lines+markers+text',
            line=dict(color=clr, width=1.8, dash=dash), marker=dict(size=6, color=clr),
            text=p_labels, textposition=tpos, textfont=dict(size=9, color=clr),
        ))

    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=90, b=10),
        title=dict(text=f'{signal_col} — Signal Projection', x=0, xanchor='left', y=0.99, yanchor='top'),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                   font=dict(size=10), tracegroupgap=4),
        yaxis=dict(range=[-105, 105], dtick=20, tickformat='.0f'),
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
    )
    return fig


def chart_projection_split(sim_sel: pd.DataFrame, signal_col: str, short: str, ind_full: pd.DataFrame,
                           mc_bands: pd.DataFrame = None):
    """Split-panel version of chart_projection() for the 'Full History +
    Projection' sub-tab: the full chosen date-range history on the left
    (compressed — however many years, ending at the latest point), and JUST
    the 10-day projection fan on the right starting from that same latest
    point — no recent-history line redrawn on the right (that duplicated the
    left panel's tail and made the two panels look disconnected instead of
    like one continuing series). Shares one y-axis. Squeezing a multi-year
    history and a 10-day fan onto one continuous time axis crushed the fan
    into an unreadable sliver — this keeps the fan a fixed, legible size no
    matter the history length instead."""
    if sim_sel.empty or ind_full is None or ind_full.empty:
        return go.Figure()
    sig_col = f'{signal_col}_Avg'
    if sig_col not in ind_full.columns:
        return go.Figure()

    color = MKT_COLOR.get(short, '#1f77b4')
    hist_full = ind_full[[sig_col]]
    anchor_date = hist_full.index[-1]
    last_sig_val = _safe_round(hist_full[sig_col].iloc[-1], scale=100)
    if last_sig_val is None:
        last_sig_val = 0

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.72, 0.28], horizontal_spacing=0.02,
        subplot_titles=('History (chosen date range)', 'Projection (10d)'),
    )

    # ── Left panel: full compressed history, ending at the latest point ────
    fig.add_trace(go.Scatter(x=hist_full.index, y=hist_full[sig_col] * 100, name='Actual',
                             line=dict(color=color, width=1.4), showlegend=True), row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)', row=1, col=1)

    # ── Right panel: JUST the MC bands + UP/DOWN/UNCH fan, starting from the ──
    # same anchor point the left panel ends on (no redrawn history line) so
    # the fan reads as a direct continuation of the left panel.
    if mc_bands is not None and not mc_bands.empty and signal_col in ('ST', 'MT', 'LT', 'All', 'WAll'):
        mc = mc_bands
        x_mc = [anchor_date] + list(mc['Horizon_Date'])
        p10, p25, p50, p75, p90 = (
            [last_sig_val] + (mc[f'{signal_col}_{p}'] * 100).tolist()
            for p in ('p10', 'p25', 'p50', 'p75', 'p90')
        )
        fig.add_trace(go.Scatter(x=x_mc, y=p90, mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_mc, y=p10, mode='lines', fill='tonexty', fillcolor='rgba(140,140,140,0.14)',
                                 line=dict(width=0), name='MC 10–90%', hoverinfo='skip'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_mc, y=p75, mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_mc, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(100,100,100,0.26)',
                                 line=dict(width=0), name='MC 25–75%', hoverinfo='skip'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_mc, y=p50, name='MC median', mode='lines',
                                 line=dict(color='#757575', width=1.3, dash='dot'), hoverinfo='skip'), row=1, col=2)

    # Single anchor-point marker (same color/style as "Actual") so the fan's
    # start is visually tied to the left panel's last point, without
    # redrawing any of the recent-history line itself.
    fig.add_trace(go.Scatter(x=[anchor_date], y=[last_sig_val], mode='markers', marker=dict(size=6, color=color),
                             name='Actual', showlegend=False, hoverinfo='skip'), row=1, col=2)

    scen_style = {
        'up':   ('↑ UP',   '#1b8a3d', 'dash', 'top center'),
        'down': ('↓ DOWN', '#c62828', 'dot',  'bottom center'),
        'unch': ('UNCH',   '#9E9E9E', 'dot',  'middle right'),
    }
    for scen, (name, clr, dash, tpos) in scen_style.items():
        col = f'{signal_col}_{scen}'
        if col not in sim_sel.columns:
            continue
        y_vals = [last_sig_val] + [_safe_round(v, scale=100) for v in sim_sel[col]]
        x_vals = [anchor_date] + list(sim_sel['Horizon_Date'])
        if scen == 'unch':
            # No price labels on UNCH — just a plain dotted line, no markers/text.
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=name, mode='lines',
                                     line=dict(color=clr, width=1.8, dash=dash)), row=1, col=2)
            continue
        pcol = f'price_{scen}'
        prices = sim_sel[pcol].tolist() if pcol in sim_sel.columns else []
        p_labels = [''] + [('' if p is None or (isinstance(p, float) and np.isnan(p)) else f'{p:,.1f}') for p in prices]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, name=name, mode='lines+markers+text',
            line=dict(color=clr, width=1.8, dash=dash), marker=dict(size=6, color=clr),
            text=p_labels, textposition=tpos, textfont=dict(size=9, color=clr),
        ), row=1, col=2)

    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)', row=1, col=2)

    # Plotly auto-ranges each subplot's x-axis with ~5% padding beyond its
    # data by default — on the left panel (years of history) that padding is
    # itself several weeks wide, showing up as a visible empty gap right
    # before the panel boundary. Pinning both axes' ranges to their actual
    # data span (no padding) removes it, so "Actual" runs right up to each
    # panel's edge and the two panels read as continuous.
    right_end = sim_sel['Horizon_Date'].max()
    if mc_bands is not None and not mc_bands.empty and 'Horizon_Date' in mc_bands.columns:
        right_end = max(right_end, mc_bands['Horizon_Date'].max())
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])], range=[hist_full.index.min(), hist_full.index.max()],
                     row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])], range=[anchor_date, right_end],
                     row=1, col=2)
    fig.update_yaxes(range=[-105, 105], dtick=20, tickformat='.0f', row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=90, b=10),
        title=dict(text=f'{signal_col} — Full History + Projection', x=0, xanchor='left', y=0.99, yanchor='top'),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                   font=dict(size=10), tracegroupgap=4),
    )
    for ann in fig.layout.annotations:  # subplot_titles come back as small grey captions
        ann.font = dict(size=10, color='#999')
    return fig


def _weekly_tues_series(ind: pd.DataFrame) -> pd.DataFrame:
    """Tuesdays only, plus a partial 'week-to-date' row for the latest date if
    it's past the most recent Tuesday — matches the original Dash tool's
    _weekly_tues_series(): the current, still-incomplete week gets its own
    partial bar instead of waiting for next Tuesday to show anything."""
    tues = ind[ind.index.dayofweek == 1].copy()
    if not tues.empty and not ind.empty and ind.index[-1] > tues.index[-1]:
        tues = pd.concat([tues, ind.iloc[[-1]]])
    elif tues.empty and not ind.empty:
        tues = ind.iloc[[-1]].copy()
    return tues


def chart_weekly_change(ind: pd.DataFrame, view: str):
    """Tuesday-to-Tuesday change decomposition into weighted ST/MT/LT stacked bars."""
    tues = _weekly_tues_series(ind)
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
    fig.update_layout(
        barmode='relative', template=PLOTLY_TEMPLATE, height=430, margin=dict(l=10, r=10, t=90, b=10),
        title=dict(text=f'Weekly Change Decomposition ({view})', x=0, xanchor='left', y=0.99, yanchor='top'),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, font=dict(size=10)),
    )
    return fig


def chart_weekly_change_total(ind: pd.DataFrame, view: str):
    """Simple green/red bar of the total weekly change only — a separate,
    simpler companion chart to chart_weekly_change() in the original Dash app
    (chart_weekly_change_total()), missing from the first Streamlit port."""
    tues = _weekly_tues_series(ind).sort_index()
    if len(tues) < 2:
        return go.Figure()
    total_col = 'WAll_Avg' if view == 'WAll' else 'All_Avg'
    if total_col not in tues.columns:
        return go.Figure()
    d_tot = tues[total_col].diff().iloc[1:] * 100
    colors = ['#1b8a3d' if v >= 0 else '#c62828' for v in d_tot]
    fig = go.Figure(go.Bar(x=tues.index[1:], y=d_tot, marker_color=colors, name=f'Δ{total_col}'))
    fig.add_hline(y=0, line_width=1, line_color='rgba(200,200,200,0.4)')
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=False, yaxis=dict(dtick=5, tickformat='+.0f'),
                      title=f'Total Weekly Change ({view})')
    return fig


def _safe_val(series: pd.Series, offset: int = 0) -> float:
    """series.iloc[-1-offset], or NaN if there aren't enough rows — mirrors the
    original _safe_val() used for the Overview change columns."""
    idx = len(series) - 1 - offset
    return float(series.iloc[idx]) if idx >= 0 else np.nan


def overview_row(short: str) -> dict:
    eff = instrument_source(short)
    price, fut, ind, sim, labels = get_instrument_data(short, eff)
    if ind.empty:
        return {'Commodity': short, 'Label': INSTRUMENT_LABELS.get(short, short)}
    last = ind.iloc[-1]
    # Rollex's own price doubles as the display price (see chart_price()); GSCI
    # mode uses the separate raw futures_price table.
    label = latest_active_label(labels)
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

    # x100 scale, 1 decimal — matches the KPI strip/projection table/charts
    # elsewhere in the dashboard (was raw -1..1 with 3 decimals, the one place
    # that didn't match everything else's display convention).
    fut_price_display = fmt_price(short, fut_last) + (f' ({label})' if label else '')
    return {
        'Commodity': short,
        'Label': INSTRUMENT_LABELS.get(short, short),
        'Futures Price': fut_price_display,
        'ST_Avg': round(last['ST_Avg'] * 100, 1),
        'MT_Avg': round(last['MT_Avg'] * 100, 1),
        'LT_Avg': round(last['LT_Avg'] * 100, 1),
        'All_Avg': round(last['All_Avg'] * 100, 1),
        'WAll_Avg': round(last['WAll_Avg'] * 100, 1),
        'Δ 1d': round(chg1d * 100, 1),
        'Δ 5d': round(chg5d * 100, 1),
        'Δ 10d': round(chg10d * 100, 1),
        'Δ Tue': round(chg_t1 * 100, 1),
        'Δ 2nd Tue': round(chg_t2 * 100, 1),
        'As of': last.name.date().isoformat(),
    }

# ── Sidebar navigation ──────────────────────────────────────────────────────────
#
# No Run Date picker and no Monte Carlo path-count control any more — the app
# always runs/displays the latest available date+run only (saves computing
# every historical run date on every page load), and MC bands are precomputed
# daily at MC_N_PATHS (100), not user-selectable or live-computed. No Signal
# Source toggle either — the dashboard is Rollex-only now except OJ (no
# Rollex coverage), which always uses GSCI automatically (instrument_source())
# rather than a user choice.

st.sidebar.markdown(
    """<div style="padding:4px 0 12px 0;">
        <div style="font-size:1.15rem;font-weight:700;color:#111;">CTA Trend Signals</div>
        <div style="font-size:0.78rem;color:#777;margin-top:2px;">
            Trend-following signal monitor
        </div>
    </div>""",
    unsafe_allow_html=True,
)

selected_instrument = st.sidebar.radio('Instrument', SHORTS, key='instrument_picker')

_eff_for_picker = instrument_source(selected_instrument)
_price_for_picker, _, _ind_for_picker, _sim_for_picker, _ = get_instrument_data(selected_instrument, _eff_for_picker)

# Date range — one global (sidebar) control instead of a separate picker on
# each of the Charts/Weekly Change/All Signals sub-views (every instrument's
# usable range is roughly the same anyway). Bounds come from whichever
# instrument is currently selected.
if not _ind_for_picker.empty:
    sidebar_rng_start, sidebar_rng_end = get_date_range(_ind_for_picker, 'sidebar_daterange')
else:
    sidebar_rng_start, sidebar_rng_end = None, None


def apply_sidebar_range(df: pd.DataFrame) -> pd.DataFrame:
    if sidebar_rng_start is None or df.empty:
        return df
    return apply_range(df, sidebar_rng_start, sidebar_rng_end)


# ── Latest data by instrument — pinned to the BOTTOM of the sidebar ────────────
# Computed fresh from ind_all every run (the same frame everything else reads,
# loaded via the mtime-cache-busted load_all()) rather than a separately
# cached value, so it can't silently go stale on its own.
st.sidebar.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-size:0.66rem;color:#888;text-transform:uppercase;letter-spacing:0.03em;'
    'border-top:1px solid #e0e0e0;padding-top:8px;margin-top:4px;margin-bottom:4px;">Latest Data by Instrument</div>',
    unsafe_allow_html=True,
)
_latest_by_inst = ind_all.groupby(['Commodity', 'Source'])['Date'].max()
_latest_rows = []
for _s in SHORTS:
    _parts = []
    for _src in ('GSCI', 'Rollex'):
        if (_s, _src) in _latest_by_inst.index:
            _parts.append(f'{_src} {pd.Timestamp(_latest_by_inst[(_s, _src)]).date().isoformat()}')
    _latest_rows.append(
        f'<div style="display:flex;justify-content:space-between;gap:8px;font-size:0.68rem;'
        f'color:#555;padding:1px 0;"><b style="color:#111;">{_s}</b>'
        f'<span style="text-align:right;">{" | ".join(_parts) if _parts else "—"}</span></div>'
    )
st.sidebar.markdown(''.join(_latest_rows), unsafe_allow_html=True)

tab_names = [f'Instrument ({INSTRUMENT_LABELS[selected_instrument]})', 'Overview', 'All Signals']
tabs = st.tabs(tab_names)

# ── Overview tab ──────────────────────────────────────────────────────────────

with tabs[1]:
    st.markdown(section_header('Overview — All Instruments',
                               'Latest composite trend signals — Rollex, except OJ (GSCI, no Rollex coverage)'),
               unsafe_allow_html=True)
    rows = [overview_row(s) for s in SHORTS]
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

# ── All Signals tab ──────────────────────────────────────────────────────────

with tabs[2]:
    st.markdown(section_header('All Signals — CTA Signals per Instrument',
                               'Each instrument\'s own ST/MT/LT/All/WAll composites, stacked'),
               unsafe_allow_html=True)
    for s in SHORTS:
        eff = instrument_source(s)
        _, _, ind, _, _ = get_instrument_data(s, eff)
        if ind.empty:
            continue
        ind_ranged = apply_sidebar_range(ind)
        oj_note = (f'<span style="color:#888;margin-left:6px;font-size:0.72rem;font-style:italic;">'
                  f'(GSCI — no Rollex coverage)</span>') if s == 'OJ' else ''
        st.markdown(
            f'<div style="border-bottom:2px solid {MKT_COLOR.get(s, "#333")};padding-bottom:3px;margin:6px 0;">'
            f'<span style="font-weight:700;color:{MKT_COLOR.get(s, "#333")};">{s}</span>'
            f'<span style="color:#888;margin-left:6px;">{INSTRUMENT_LABELS[s]}</span>{oj_note}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(chart_signals_all(ind_ranged, s), width='stretch', key=f'allsig_chart_{s}')

# ── Instrument tab (driven by the sidebar slicer, not a tab per instrument) ────

for short in [selected_instrument]:  # loops exactly once — keeps the body's indentation as-is
    with tabs[0]:
        eff = instrument_source(short)
        price, fut, ind, sim, labels = get_instrument_data(short, eff)
        if ind.empty:
            st.warning(f'No data for {short}.')
            continue
        active_label = latest_active_label(labels)

        if short == 'OJ':
            st.caption('OJ has no Rollex coverage — running on GSCI (the one exception in an otherwise Rollex-only dashboard).')

        # Run date is always the latest available — no picker any more (saves
        # computing every historical run date on every page load). Monte Carlo
        # bands are precomputed daily (MC_N_PATHS=100, 20-day returns lookback
        # — see Code/tr_mapping_fetch.py's main()/compute_monte_carlo_bands())
        # and just read here — instant, no per-switch compute delay.
        run_dates = sorted(sim['Run_Date'].unique(), reverse=True) if not sim.empty else []
        run_choice = run_dates[0] if run_dates else None
        mc_bands = get_monte_carlo_bands(short, eff)

        sub_full, sub_signals, sub_weekly = st.tabs(
            ['Full History + Projection', 'Signals & Composites', 'Weekly Change'])

        col_map = {'ST_Avg': ST_COLS, 'MT_Avg': MT_COLS, 'LT_Avg': LT_COLS,
                  'All_Avg': ST_COLS + MT_COLS + LT_COLS, 'WAll_Avg': ST_COLS + MT_COLS + LT_COLS}

        with sub_full:
            # sim_sel_full computed up front (not just inside the "else" below)
            # so the price chart's right panel can show the same UP/DOWN/UNCH
            # price scenarios too — keeps it visually aligned with (and a price
            # counterpart to) the signal projection chart beneath it.
            sim_sel_full = (sim[sim['Run_Date'] == run_choice].sort_values('Horizon_Day')
                            if not sim.empty and run_choice is not None else pd.DataFrame())

            price_ranged = apply_sidebar_range(price)
            fut_ranged = apply_sidebar_range(fut)
            st.plotly_chart(
                chart_price_split(short, price_ranged, fut_ranged, sim_sel_full, source=eff,
                                  active_label=active_label),
                width='stretch', key=f'{short}_pricechart',
            )

            if sim.empty:
                st.info('No simulation history for this instrument.')
            else:
                full_signal = st.radio('Signal', ['WAll', 'All', 'ST', 'MT', 'LT'],
                                       horizontal=True, key=f'{short}_fullsignal')
                ind_ranged_full = apply_sidebar_range(ind)
                if mc_bands.empty:
                    st.caption('Not enough history to run Monte Carlo for this instrument.')

                # Left: full signal history (per the sidebar date range), up
                # to the latest point. Right: the 10-day Monte Carlo
                # projection fan continuing from that same point — split into
                # two panels so a multi-year history doesn't crush the 10-day
                # fan into an unreadable sliver.
                st.plotly_chart(
                    chart_projection_split(sim_sel_full, full_signal, short, ind_ranged_full, mc_bands=mc_bands),
                    width='stretch', key=f'{short}_fullprojchart',
                )

                with st.expander('Single continuous view (history + fan on one axis)', expanded=False):
                    st.plotly_chart(
                        chart_projection(sim_sel_full, price, full_signal, short, ind=ind,
                                         mc_bands=mc_bands, hist_df=ind_ranged_full),
                        width='stretch', key=f'{short}_fullprojchart_single',
                    )

                st.markdown(projection_table_html(sim_sel_full, short, active_label=active_label), unsafe_allow_html=True)

        with sub_signals:
            composite = st.radio('Composite', ['WAll_Avg', 'All_Avg', 'ST_Avg', 'MT_Avg', 'LT_Avg'],
                                 horizontal=True, key=f'{short}_composite')
            # Bucketed/aggregated slicer instead of a single "show all 144 raw
            # signals at once" checkbox — 'By Family' averages each indicator
            # family (Mom/MA/EMA/HMA/3MA/BB/DC/LRS/TRIX/KAMA) into one line
            # each, so the underlying detail stays readable.
            detail_choice = st.radio(
                'Underlying detail', ['None', 'By Family (avg)', 'All raw signals'],
                index=1, horizontal=True, key=f'{short}_detail',
            )
            detail_map = {'None': 'none', 'By Family (avg)': 'family', 'All raw signals': 'raw'}
            ind_ranged = apply_sidebar_range(ind)
            st.plotly_chart(chart_signals(ind_ranged, composite, col_map[composite], detail=detail_map[detail_choice]),
                            width='stretch', key=f'{short}_sigchart')

            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            st.caption('All 5 composites together, for comparison.')
            st.plotly_chart(chart_signals_all(ind_ranged, short), width='stretch', key=f'{short}_sigchart_all')

        with sub_weekly:
            view = st.radio('View', ['WAll', 'All'], horizontal=True, key=f'{short}_weekview')

            # Week = Tuesday-to-Tuesday (matches this desk's COT reporting
            # cadence, not calendar Mon-Fri) — same convention as the
            # original tool. Called out explicitly since it's easy to assume
            # a normal Mon-Fri week otherwise.
            _tues_only = ind[ind.index.dayofweek == 1]
            _wtd_note = ''
            if not ind.empty and not _tues_only.empty and ind.index[-1] > _tues_only.index[-1]:
                _wtd_note = (f'  Latest bar is **week-to-date** (partial): '
                            f'{_tues_only.index[-1].date()} → {ind.index[-1].date()}.')
            st.caption(
                "Week = **Tuesday-to-Tuesday** (this desk's COT reporting cadence), not calendar Mon-Fri."
                + _wtd_note
            )

            ind_ranged_w = apply_sidebar_range(ind)
            st.plotly_chart(chart_weekly_change(ind_ranged_w, view),
                            width='stretch', key=f'{short}_weeklychart')
            st.plotly_chart(chart_weekly_change_total(ind_ranged_w, view),
                            width='stretch', key=f'{short}_weeklytotalchart')
