"""
tr_mapping_fetch.py — Fetch LSEG/Rollex data, compute CTA trend-following indicators,
build simulation scenarios and maintain simulation history (Rollex-only, except OJ on GSCI).

Data policy: only get_history is used (daily closes).  Run each morning to pick up
the previous day's close.  No intraday fetching.

Storage: parquet files under Database/ (Hardminer architecture — was DuckDB in the
original "TR mapping old" version this was converted from):
    Database/price_history.parquet   Commodity, Source, Date, Close   (signal basis)
    Database/futures_price.parquet   Commodity, Date, Close   (front-month futures — display,
                                      GSCI-mode only; Rollex mode displays its own price)
    Database/indicators.parquet      Commodity, Source, Date, <172 signal/composite columns>
    Database/sim_history.parquet     Commodity, Source, Run_Date, Horizon_Date, Horizon_Day,
                                      ST/MT/LT/All/WAll _down/_up/_unch, price_down/up/unch,
                                      Actual_Close
    Database/active_labels.parquet   Commodity, Date, Active_Label   (Rollex's active-contract
                                      label, e.g. "Dec'26" — Rollex-sourced instruments only)

No pandas_ta anywhere in this file — it pulls in numba unconditionally, and
numba has no published wheel for some Python versions Streamlit Cloud may run
(confirmed via PyPI's file listing), which broke the dashboard's deploy the
moment it needed pandas_ta. calculate_indicators()'s 5 previously-pandas_ta-
dependent pieces (BBands, HMA, linear-regression-slope, TRIX, KAMA) are
hand-rolled here from pandas_ta's own source formulas instead — validated
against the real pandas_ta output (max abs diff 0.0, all lengths, full
multi-year history) before replacing it. calculate_indicators_multi() is a
vectorized sibling that computes N Monte Carlo paths' composites together
(exact match against calculate_indicators() on identical input) — see
Dashboard/app.py for the live, on-demand Monte Carlo feature this enables
(N=100 in ~2s, N=500 in ~11s — fast enough to run in the dashboard itself,
no ingest-side batch step needed).

Signal source is Rollex-only now, with one exception: OJ has no Rollex
coverage at all, so OJ alone still runs on GSCI (its historical S&P GSCI
single-commodity sub-index, .SPGSOJP). GSCI was originally wired up for all 5
of Romain's original instruments (KC/CT/SB/CC/OJ), but has since been dropped
for KC/CT/SB/CC to save data storage and compute — Rollex (this desk's own
continuous roll-adjusted futures price, rollex_px) is the sole source for
KC/CT/SB/CC/LCC/LSU/RC, read directly from the sibling LSEG-Rollex project's
own parquet output (cross-repo read, no re-fetch — Rollex maintains its own
data; also carries an active_label column, e.g. "Dec'26", read alongside
rollex_px and stored in Database/active_labels.parquet for display). Same
144-indicator math applies unchanged regardless of source — it's return/
crossing-based, so it's source-agnostic.

Usage:
    python tr_mapping_fetch.py                    # incremental update (run next morning)
    python tr_mapping_fetch.py --full             # full refresh from 20 years ago
    python tr_mapping_fetch.py --reset-sim        # wipe and rebuild sim history
"""

import argparse
import pathlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Database'
DATA_DIR.mkdir(exist_ok=True)

PRICE_FILE    = DATA_DIR / 'price_history.parquet'
FUTPX_FILE    = DATA_DIR / 'futures_price.parquet'
IND_FILE      = DATA_DIR / 'indicators.parquet'
SIM_FILE      = DATA_DIR / 'sim_history.parquet'
LABEL_FILE    = DATA_DIR / 'active_labels.parquet'  # Rollex active-contract label (e.g. "Dec'26"), Commodity/Date/Active_Label
MC_N_PATHS    = 200  # fixed — vectorized across all N paths at once (calculate_indicators_multi) -> ~4s/instrument/source

# Rollex is a sibling LSEG-* project (own repo, own automator) — CTA only ever
# READS its parquet output, never writes to it. BASE_DIR is CTA's own root, so
# BASE_DIR.parent is the shared LSEG container folder.
ROLLEX_DB_DIR = BASE_DIR.parent / 'Rollex' / 'Database'
ROLLEX_SHORTS = {'KC', 'CT', 'SB', 'CC', 'LCC', 'LSU', 'RC'}  # Rollex has no OJ coverage

# ── LSEG availability flag ─────────────────────────────────────────────────────

try:
    import lseg.data as ld
    LSEG_AVAILABLE = True
except ImportError:
    LSEG_AVAILABLE = False
    print('[warning] lseg.data not available — will use cached parquet only.')

# ── Instruments ────────────────────────────────────────────────────────────────

@dataclass
class Instrument:
    short:       str          # 'KC' — table key, display
    gsci_ric:    str | None   # '.SPGSKCP' — GSCI index for historical price fetch; None = no GSCI sub-index exists
    futures_ric: str | None   # 'KCv1' — front-month futures for display price history; None = not fetched
    decimals:    int          # price decimal places
    label:       str          # 'Coffee'

INSTRUMENTS = [
    # KC/CT/SB/CC: GSCI dropped — Rollex is the sole source now (saves data
    # storage + compute). gsci_ric/futures_ric are None so the GSCI half of
    # main()'s pipeline is skipped entirely for these.
    Instrument('KC', None, None, 0, 'Coffee'),
    Instrument('CT', None, None, 1, 'Cotton'),
    Instrument('SB', None, None, 1, 'Sugar'),
    Instrument('CC', None, None, 0, 'Cocoa'),
    # OJ: the one exception — Rollex has no OJ coverage at all, so OJ keeps
    # running on its historical GSCI sub-index (.SPGSOJP) as its only source.
    Instrument('OJ', '.SPGSOJP', 'OJv1', 1, 'Orange Juice'),
    # Rollex-only — no S&P GSCI single-commodity sub-index ever existed for
    # these London-listed ICE contracts.
    Instrument('LCC', None, None, 0, 'London Cocoa'),
    Instrument('LSU', None, None, 1, 'London Sugar'),
    Instrument('RC',  None, None, 0, 'Robusta Coffee'),
]

# ── Indicator parameter sets ───────────────────────────────────────────────────

PARAMS = {
    'Mom':  [5, 10, 15, 20, 25, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500],
    'MA':   [(5, 10), (10, 15), (5, 15), (15, 20), (10, 20), (25, 40), (25, 50), (25, 60),
             (40, 75), (50, 100), (75, 150), (100, 150), (50, 200), (100, 200), (150, 250), (150, 300),
             (200, 300), (200, 400), (250, 400)],
    'EMA':  [(5, 10), (10, 15), (5, 15), (15, 20), (10, 20), (25, 40), (25, 50), (25, 60),
             (40, 75), (50, 100), (75, 150), (100, 150), (50, 200), (100, 200), (150, 250), (150, 300),
             (200, 300), (200, 400), (250, 400)],
    'HMA':  [(5, 10), (10, 15), (5, 15), (15, 20), (10, 20), (25, 40), (25, 50), (25, 60),
             (40, 75), (50, 100), (75, 150), (100, 150), (50, 200), (100, 200), (150, 250), (150, 300),
             (200, 300), (200, 400), (250, 400)],
    '3MA':  [(5, 10, 20), (10, 15, 20), (15, 20, 25), (10, 20, 50), (25, 50, 75), (60, 125, 150),
             (20, 50, 100), (75, 150, 250), (150, 250, 400), (50, 100, 200), (100, 200, 400),
             (100, 200, 300), (150, 300, 400), (200, 300, 400)],
    'BB':   [10, 15, 20, 25, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400],
    'DC':   [10, 15, 20, 25, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400],
    'LRS':  [10, 15, 20, 25, 40, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400],
    'TRIX': [5, 10, 15, 20, 25, 40, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400],
    'KAMA': [10, 15, 20, 25, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400],
}

# ── Signal column lists ────────────────────────────────────────────────────────

ST_COLS = [
    'Mom_5', 'Mom_10', 'Mom_15', 'Mom_20', 'Mom_25',
    'MA_cross_(5, 10)', 'MA_cross_(10, 15)', 'MA_cross_(5, 15)', 'MA_cross_(15, 20)', 'MA_cross_(10, 20)', 'MA_cross_(25, 40)',
    'EMA_cross_(5, 10)', 'EMA_cross_(10, 15)', 'EMA_cross_(5, 15)', 'EMA_cross_(15, 20)', 'EMA_cross_(10, 20)', 'EMA_cross_(25, 40)',
    'HMA_cross_(5, 10)', 'HMA_cross_(10, 15)', 'HMA_cross_(5, 15)', 'HMA_cross_(15, 20)', 'HMA_cross_(10, 20)', 'HMA_cross_(25, 40)',
    '3MA_cross_(5, 10, 20)', '3MA_cross_(10, 15, 20)', '3MA_cross_(15, 20, 25)', '3MA_cross_(10, 20, 50)',
    'BB_10', 'BB_15', 'BB_20', 'BB_25',
    'DC_10', 'DC_15', 'DC_20', 'DC_25',
    'LRS_10', 'LRS_15', 'LRS_20', 'LRS_25',
    'TRIX_5', 'TRIX_10', 'TRIX_15', 'TRIX_20', 'TRIX_25',
    'KAMA_10', 'KAMA_15', 'KAMA_20', 'KAMA_25',
]

MT_COLS = [
    'Mom_40', 'Mom_50', 'Mom_60', 'Mom_75', 'Mom_100', 'Mom_125',
    'MA_cross_(25, 50)', 'MA_cross_(25, 60)', 'MA_cross_(40, 75)', 'MA_cross_(50, 100)', 'MA_cross_(75, 150)', 'MA_cross_(100, 150)', 'MA_cross_(50, 200)',
    'EMA_cross_(25, 50)', 'EMA_cross_(25, 60)', 'EMA_cross_(40, 75)', 'EMA_cross_(50, 100)', 'EMA_cross_(75, 150)', 'EMA_cross_(100, 150)', 'EMA_cross_(50, 200)',
    'HMA_cross_(25, 50)', 'HMA_cross_(25, 60)', 'HMA_cross_(40, 75)', 'HMA_cross_(50, 100)', 'HMA_cross_(75, 150)', 'HMA_cross_(100, 150)', 'HMA_cross_(50, 200)',
    '3MA_cross_(25, 50, 75)', '3MA_cross_(20, 50, 100)', '3MA_cross_(50, 100, 200)', '3MA_cross_(60, 125, 150)', '3MA_cross_(75, 150, 250)',
    'BB_40', 'BB_50', 'BB_60', 'BB_80', 'BB_100',
    'DC_40', 'DC_50', 'DC_60', 'DC_80', 'DC_100',
    'LRS_40', 'LRS_50', 'LRS_75', 'LRS_100', 'LRS_125',
    'TRIX_40', 'TRIX_50', 'TRIX_75', 'TRIX_100', 'TRIX_125',
    'KAMA_40', 'KAMA_50', 'KAMA_60', 'KAMA_80', 'KAMA_100',
]

LT_COLS = [
    'Mom_150', 'Mom_200', 'Mom_250', 'Mom_300', 'Mom_350', 'Mom_400', 'Mom_500',
    'MA_cross_(100, 200)', 'MA_cross_(150, 250)', 'MA_cross_(150, 300)',
    'MA_cross_(200, 300)', 'MA_cross_(200, 400)', 'MA_cross_(250, 400)',
    'EMA_cross_(100, 200)', 'EMA_cross_(150, 250)', 'EMA_cross_(150, 300)',
    'EMA_cross_(200, 300)', 'EMA_cross_(200, 400)', 'EMA_cross_(250, 400)',
    'HMA_cross_(100, 200)', 'HMA_cross_(150, 250)', 'HMA_cross_(150, 300)',
    'HMA_cross_(200, 300)', 'HMA_cross_(200, 400)', 'HMA_cross_(250, 400)',
    '3MA_cross_(100, 200, 300)', '3MA_cross_(100, 200, 400)', '3MA_cross_(150, 250, 400)',
    '3MA_cross_(150, 300, 400)', '3MA_cross_(200, 300, 400)',
    'BB_150', 'BB_200', 'BB_250', 'BB_300', 'BB_350', 'BB_400',
    'DC_150', 'DC_200', 'DC_250', 'DC_300', 'DC_350', 'DC_400',
    'LRS_150', 'LRS_200', 'LRS_250', 'LRS_300', 'LRS_350', 'LRS_400',
    'TRIX_150', 'TRIX_200', 'TRIX_250', 'TRIX_300', 'TRIX_350', 'TRIX_400',
    'KAMA_150', 'KAMA_200', 'KAMA_250', 'KAMA_300', 'KAMA_350', 'KAMA_400',
]

ALL_SIGNAL_COLS = ST_COLS + MT_COLS + LT_COLS

# ── Parquet storage helpers ─────────────────────────────────────────────────────
# All tables use (Commodity, Date) as key (sim_history uses Commodity/Run_Date/
# Horizon_Day) — same drop_duplicates-upsert pattern used across every other
# Interim_Migration/LSEG project's ingest script.

def _load(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def upsert_price_history(inst: Instrument, new_df: pd.DataFrame, source: str = 'GSCI') -> None:
    """new_df: DataFrame with DatetimeIndex named Date, column CLOSE."""
    rows = new_df.reset_index()[['Date', 'CLOSE']].copy()
    rows.columns = ['Date', 'Close']
    rows.insert(0, 'Source', source)
    rows.insert(0, 'Commodity', inst.short)
    rows = rows.dropna(subset=['Close'])
    old = _load(PRICE_FILE)
    combined = pd.concat([old, rows], ignore_index=True) if not old.empty else rows
    combined = combined.drop_duplicates(subset=['Commodity', 'Source', 'Date'], keep='last')
    combined = combined.sort_values(['Commodity', 'Source', 'Date']).reset_index(drop=True)
    combined.to_parquet(PRICE_FILE, index=False)


def upsert_futures_price(inst: Instrument, new_df: pd.DataFrame) -> None:
    rows = new_df.reset_index()[['Date', 'CLOSE']].copy()
    rows.columns = ['Date', 'Close']
    rows.insert(0, 'Commodity', inst.short)
    rows = rows.dropna(subset=['Close'])
    old = _load(FUTPX_FILE)
    combined = pd.concat([old, rows], ignore_index=True) if not old.empty else rows
    combined = combined.drop_duplicates(subset=['Commodity', 'Date'], keep='last')
    combined = combined.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    combined.to_parquet(FUTPX_FILE, index=False)


def load_price_history(inst: Instrument, source: str = 'GSCI') -> pd.DataFrame:
    """Load full price history for an instrument+source. Returns DataFrame
    indexed by Date (named 'Date') with a single 'CLOSE' column, matching
    original DuckDB shape."""
    all_df = _load(PRICE_FILE)
    if all_df.empty:
        return pd.DataFrame(columns=['CLOSE'])
    df = all_df[(all_df['Commodity'] == inst.short) & (all_df['Source'] == source)][['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df.index.name = 'Date'
    df.columns = ['CLOSE']
    return df


def get_last_price_date(inst: Instrument, source: str = 'GSCI'):
    all_df = _load(PRICE_FILE)
    if all_df.empty:
        return None
    sub = all_df[(all_df['Commodity'] == inst.short) & (all_df['Source'] == source)]
    if sub.empty:
        return None
    return pd.Timestamp(pd.to_datetime(sub['Date']).max())


def upsert_active_labels(inst: Instrument, df: pd.DataFrame) -> None:
    """df: DatetimeIndex named Date, single 'Active_Label' column — the Rollex
    active-contract label (e.g. "Dec'26") as of each date. Rollex-only; no
    Source column needed since GSCI never carries this."""
    if df.empty:
        return
    rows = df.reset_index()[['Date', 'Active_Label']].copy()
    rows.insert(0, 'Commodity', inst.short)
    rows = rows.dropna(subset=['Active_Label'])
    if rows.empty:
        return
    old = _load(LABEL_FILE)
    combined = pd.concat([old, rows], ignore_index=True) if not old.empty else rows
    combined = combined.drop_duplicates(subset=['Commodity', 'Date'], keep='last')
    combined = combined.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    combined.to_parquet(LABEL_FILE, index=False)


def upsert_indicators(inst: Instrument, df: pd.DataFrame, source: str = 'GSCI') -> None:
    """Replace all indicator rows for this instrument+source. df: DatetimeIndex
    named Date, columns = CLOSE + all signal/composite columns (CLOSE is
    dropped — it's already in price_history)."""
    rows = df.reset_index().copy()
    rows = rows.rename(columns={'index': 'Date'})
    rows.insert(0, 'Source', source)
    rows.insert(0, 'Commodity', inst.short)
    signal_cols = [c for c in df.columns if c != 'CLOSE']
    rows = rows[['Commodity', 'Source', 'Date'] + signal_cols].copy()
    old = _load(IND_FILE)
    if old.empty:
        combined = rows
    else:
        # full replace for this instrument+source only
        old = old[~((old['Commodity'] == inst.short) & (old['Source'] == source))]
        combined = pd.concat([old, rows], ignore_index=True)
    combined = combined.sort_values(['Commodity', 'Source', 'Date']).reset_index(drop=True)
    combined.to_parquet(IND_FILE, index=False)


# ── Hand-rolled replacements for the 5 pandas_ta-dependent functions ───────────
#
# pandas_ta unconditionally depends on numba, which has no published wheel for
# some Python versions Streamlit Cloud may run (confirmed via PyPI's file
# listing) — that broke the dashboard's deploy the moment it needed to import
# calculate_indicators(). These replicate pandas_ta's exact formulas (read
# straight from its source: bbands, hma/wma, linreg(slope=True), trix, kama —
# all use no-TA-Lib pandas-path defaults since TA-Lib isn't installed here
# either) using plain pandas/numpy — validated to match pandas_ta's output
# exactly (max abs diff 0.0 on real KC data, all tested lengths) before this
# replaced the pandas_ta calls below. TRIX uses a plain (non presma-seeded) EMA
# — pandas_ta's own ema() seeds with an SMA, but empirically the two agree
# 100% on sign for tail rows once 1000+ periods past the series start, which is
# the only thing TRIX's sign is used for here.

def _wma(close: pd.Series, n: int) -> pd.Series:
    from numpy.lib.stride_tricks import sliding_window_view
    w = np.arange(1, n + 1, dtype=float)
    arr = close.to_numpy()
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        out[n - 1:] = sliding_window_view(arr, n) @ w * (2 / (n * n + n))
    return pd.Series(out, index=close.index)


def _hma(close: pd.Series, n: int) -> pd.Series:
    half, sq = int(n / 2), int(np.sqrt(n))
    return _wma(2 * _wma(close, half) - _wma(close, n), sq)


def _bbands(close: pd.Series, n: int):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    return mid + 2.0 * std, mid, mid - 2.0 * std  # upper, mid, lower


def _linreg_slope(close: pd.Series, n: int) -> pd.Series:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.arange(1, n + 1, dtype=float)
    x_sum = 0.5 * n * (n + 1)
    divisor = n * (x_sum * (2 * n + 1) / 3) - x_sum * x_sum
    arr = close.to_numpy()
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        windows = sliding_window_view(arr, n)
        out[n - 1:] = (n * (windows @ x) - x_sum * windows.sum(axis=1)) / divisor
    return pd.Series(out, index=close.index)


def _ema_presma(close: pd.Series, n: int) -> pd.Series:
    """EMA seeded with an SMA of the first n values (TA-Lib-style), matching
    pandas_ta's own ema(presma=True) default — used internally by pandas_ta's
    trix(). A plain close.ewm(adjust=False) from row 0 (no seeding) converges
    to the same values eventually, but on the FULL multi-year price history
    (not a short recent window) the seeding difference does not decay away
    before the dates that matter — validated to diverge by up to 0.19 over
    full history vs exact (0.0) match with presma-seeding."""
    s2 = close.copy()
    if len(s2) >= n:
        sma_seed = s2.iloc[:n].mean()
        s2.iloc[:n - 1] = np.nan
        s2.iloc[n - 1] = sma_seed
    return s2.ewm(span=n, adjust=False).mean()


def _trix_sign(close: pd.Series, n: int) -> pd.Series:
    # pandas_ta's trix(length=n) swaps length<->signal (default signal=9) when
    # length < signal, so length=5 (the only PARAMS['TRIX'] value < 9) was
    # ACTUALLY computed with an effective EMA span of 9 in the pandas_ta-based
    # production data this whole time — replicated here rather than "fixed",
    # to keep matching the existing TRIX_5 column's real historical behavior.
    eff_n = max(n, 9)
    ema1 = _ema_presma(close, eff_n)
    ema2 = _ema_presma(ema1, eff_n)
    ema3 = _ema_presma(ema2, eff_n)
    return np.sign(ema3.pct_change(1)).fillna(0)


def _kama(close: pd.Series, n: int, fast: int = 2, slow: int = 30) -> pd.Series:
    fr, sr = 2 / (fast + 1), 2 / (slow + 1)
    abs_diff = (close - close.shift(n)).abs()
    peer_diff_sum = (close - close.shift(1)).abs().rolling(n).sum()
    sc = ((abs_diff / peer_diff_sum) * (fr - sr) + sr) ** 2
    sc_arr = sc.to_numpy()
    arr = close.to_numpy()
    m = len(arr)
    result = np.full(m, np.nan)
    if m >= n:
        result[n - 1] = arr[:n].mean()
        for i in range(n, m):
            result[i] = sc_arr[i] * arr[i] + (1 - sc_arr[i]) * result[i - 1]
    return pd.Series(result, index=close.index)

# ── Stateful signal helpers ────────────────────────────────────────────────────

def _bb_signal(close: pd.Series, n: int) -> pd.Series:
    """Bollinger Band stateful signal: +1 / 0 / -1 with midline exit."""
    upper, mid, lower = _bbands(close, n)

    signals = np.zeros(len(close))
    state   = 0
    for i in range(len(close)):
        c, u, m, lo = close.iloc[i], upper.iloc[i], mid.iloc[i], lower.iloc[i]
        if np.isnan(c) or np.isnan(u):
            signals[i] = 0
            continue
        if c >= u:
            state = 1
        elif c <= lo:
            state = -1
        elif state == 1 and c <= m:
            state = 0
        elif state == -1 and c >= m:
            state = 0
        signals[i] = state
    return pd.Series(signals, index=close.index)


def _dc_signal(close: pd.Series, n: int) -> pd.Series:
    """Donchian Channel stateful signal: +1 / -1 with hold."""
    rolling_high = close.rolling(n).max()
    rolling_low  = close.rolling(n).min()

    signals = np.zeros(len(close))
    state   = 0
    for i in range(len(close)):
        c, hi, lo = close.iloc[i], rolling_high.iloc[i], rolling_low.iloc[i]
        if np.isnan(hi) or np.isnan(lo):
            signals[i] = 0
            continue
        if c >= hi:
            state = 1
        elif c <= lo:
            state = -1
        signals[i] = state
    return pd.Series(signals, index=close.index)

# ── Indicator computation ──────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all CTA trend-following signals on a price DataFrame.

    Input:  DataFrame with a CLOSE column (DatetimeIndex, weekdays only).
    Output: Same DataFrame with all indicator columns + ST/MT/LT/All averages.
    Columns are collected in a dict and concatenated once to avoid fragmentation warnings.
    """
    close = df['CLOSE'].astype('float64')  # nullable 'Float64' input -> object-dtype .to_numpy(), breaks np.isnan()
    cols: dict[str, pd.Series] = {}
    zero  = pd.Series(0.0, index=close.index)

    # ── Momentum (vol-normalised tanh) ────────────────────────────────────────
    for n in PARAMS['Mom']:
        ret       = close.pct_change(n)
        daily_vol = close.pct_change().rolling(n).std()
        cols[f'Mom_{n}'] = np.tanh(ret / (daily_vol * np.sqrt(n))).fillna(0)

    # ── SMA cross ─────────────────────────────────────────────────────────────
    for s, l in PARAMS['MA']:
        sma_s = close.rolling(s).mean()
        sma_l = close.rolling(l).mean()
        cols[f'MA_cross_({s}, {l})'] = pd.Series(
            np.where(sma_s > sma_l, 1, -1), index=close.index, dtype=float
        )

    # ── EMA cross ─────────────────────────────────────────────────────────────
    for s, l in PARAMS['EMA']:
        ema_s = close.ewm(span=s, adjust=False).mean()
        ema_l = close.ewm(span=l, adjust=False).mean()
        cols[f'EMA_cross_({s}, {l})'] = pd.Series(
            np.where(ema_s > ema_l, 1, -1), index=close.index, dtype=float
        )

    # ── HMA cross ─────────────────────────────────────────────────────────────
    for s, l in PARAMS['HMA']:
        hma_s = _hma(close, s)
        hma_l = _hma(close, l)
        sig = pd.Series(np.where(hma_s > hma_l, 1, -1), index=close.index, dtype=float)
        cols[f'HMA_cross_({s}, {l})'] = sig.where(hma_s.notna() & hma_l.notna(), other=0.0)

    # ── 3MA cross ─────────────────────────────────────────────────────────────
    def _3ma_sig(row):
        vs, vm, vl = row
        if np.isnan(vs) or np.isnan(vm) or np.isnan(vl):
            return 0
        if vs > vm > vl:
            return 1
        if vs < vm < vl:
            return -1
        return 0

    for s, m, l in PARAMS['3MA']:
        ma_s = close.rolling(s).mean()
        ma_m = close.rolling(m).mean()
        ma_l = close.rolling(l).mean()
        cols[f'3MA_cross_({s}, {m}, {l})'] = (
            pd.DataFrame({'s': ma_s, 'm': ma_m, 'l': ma_l}).apply(_3ma_sig, axis=1)
        )

    # ── Bollinger Bands (stateful) ─────────────────────────────────────────────
    for n in PARAMS['BB']:
        cols[f'BB_{n}'] = _bb_signal(close, n)

    # ── Donchian Channel (stateful) ────────────────────────────────────────────
    for n in PARAMS['DC']:
        cols[f'DC_{n}'] = _dc_signal(close, n)

    # ── Linear Regression Slope ───────────────────────────────────────────────
    for n in PARAMS['LRS']:
        slope = _linreg_slope(close, n)
        cols[f'LRS_{n}'] = np.sign(slope).fillna(0)

    # ── TRIX ──────────────────────────────────────────────────────────────────
    for n in PARAMS['TRIX']:
        cols[f'TRIX_{n}'] = _trix_sign(close, n)

    # ── KAMA vs close ─────────────────────────────────────────────────────────
    # Vol-adaptive moving average: +1 when price > KAMA (uptrend), -1 when below
    c_arr = close.astype(float).to_numpy()
    for n in PARAMS['KAMA']:
        k_arr = _kama(close, n).to_numpy()
        sig   = pd.Series(np.where(c_arr > k_arr, 1, -1), index=close.index, dtype=float)
        cols[f'KAMA_{n}'] = sig.where(~np.isnan(k_arr), other=0.0)

    # ── Concatenate all indicator columns at once (avoids fragmentation) ──────
    ind_df = pd.concat([df[['CLOSE']], pd.DataFrame(cols, index=close.index)], axis=1)

    # ── Composite averages ────────────────────────────────────────────────────
    ind_df['ST_Avg']   = ind_df[ST_COLS].mean(axis=1)
    ind_df['MT_Avg']   = ind_df[MT_COLS].mean(axis=1)
    ind_df['LT_Avg']   = ind_df[LT_COLS].mean(axis=1)
    ind_df['All_Avg']  = ind_df[ALL_SIGNAL_COLS].mean(axis=1)
    # Weighted composite: 45% MT, 35% LT, 20% ST
    ind_df['WAll_Avg'] = 0.20 * ind_df['ST_Avg'] + 0.45 * ind_df['MT_Avg'] + 0.35 * ind_df['LT_Avg']

    return ind_df

# ── Vectorized multi-path indicator engine (Monte Carlo) ────────────────────────
#
# Same formulas as calculate_indicators() above, generalized to a DataFrame
# with N columns (Monte Carlo paths sharing the same price history up to an
# anchor date, diverging only in the last `horizon` simulated days) computed
# together. pandas' rolling/ewm already vectorize across DataFrame columns for
# free — the only pieces needing an explicit rewrite are the stateful/recursive
# ones (BB/DC signal state machines, KAMA), which loop over TIME only (not
# paths), updating all N columns per step via numpy. This turns an O(N) loop of
# calculate_indicators() calls (~0.5-2s each) into one pass computing all N
# paths together in roughly the time of a handful of single-path calls.
# Validated against calculate_indicators() fed a 1-column DataFrame — exact
# match (see Code/ development notes / commit history).

def _wma_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    from numpy.lib.stride_tricks import sliding_window_view
    w = np.arange(1, n + 1, dtype=float)
    arr = close_df.to_numpy()
    out = np.full(arr.shape, np.nan)
    if arr.shape[0] >= n:
        windows = sliding_window_view(arr, n, axis=0)  # (T-n+1, P, n)
        out[n - 1:] = windows @ w * (2 / (n * n + n))
    return pd.DataFrame(out, index=close_df.index, columns=close_df.columns)


def _hma_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    half, sq = int(n / 2), int(np.sqrt(n))
    return _wma_multi(2 * _wma_multi(close_df, half) - _wma_multi(close_df, n), sq)


def _bbands_multi(close_df: pd.DataFrame, n: int):
    mid = close_df.rolling(n).mean()
    std = close_df.rolling(n).std(ddof=1)
    return mid + 2.0 * std, mid, mid - 2.0 * std


def _linreg_slope_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.arange(1, n + 1, dtype=float)
    x_sum = 0.5 * n * (n + 1)
    divisor = n * (x_sum * (2 * n + 1) / 3) - x_sum * x_sum
    arr = close_df.to_numpy()
    out = np.full(arr.shape, np.nan)
    if arr.shape[0] >= n:
        windows = sliding_window_view(arr, n, axis=0)
        out[n - 1:] = (n * (windows @ x) - x_sum * windows.sum(axis=2)) / divisor
    return pd.DataFrame(out, index=close_df.index, columns=close_df.columns)


def _ema_presma_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    s2 = close_df.copy()
    if len(s2) >= n:
        s2.iloc[n - 1] = s2.iloc[:n].mean(axis=0)
        s2.iloc[:n - 1] = np.nan
    return s2.ewm(span=n, adjust=False).mean()


def _trix_sign_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    eff_n = max(n, 9)  # see _trix_sign()'s note on pandas_ta's length<->signal swap
    ema1 = _ema_presma_multi(close_df, eff_n)
    ema2 = _ema_presma_multi(ema1, eff_n)
    ema3 = _ema_presma_multi(ema2, eff_n)
    return np.sign(ema3.pct_change(1)).fillna(0)


def _kama_multi(close_df: pd.DataFrame, n: int, fast: int = 2, slow: int = 30) -> pd.DataFrame:
    fr, sr = 2 / (fast + 1), 2 / (slow + 1)
    abs_diff = (close_df - close_df.shift(n)).abs()
    peer_diff_sum = (close_df - close_df.shift(1)).abs().rolling(n).sum()
    sc_arr = (((abs_diff / peer_diff_sum) * (fr - sr) + sr) ** 2).to_numpy()
    arr = close_df.to_numpy()
    T, P = arr.shape
    result = np.full((T, P), np.nan)
    if T >= n:
        result[n - 1] = arr[:n].mean(axis=0)
        for i in range(n, T):
            result[i] = sc_arr[i] * arr[i] + (1 - sc_arr[i]) * result[i - 1]
    return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)


def _bb_signal_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    upper, mid, lower = _bbands_multi(close_df, n)
    c, u, mid_a, lo = (a.to_numpy() for a in (close_df, upper, mid, lower))
    T, P = c.shape
    state = np.zeros(P)
    out = np.zeros((T, P))
    for i in range(T):
        ci, ui, mi, loi = c[i], u[i], mid_a[i], lo[i]
        invalid = np.isnan(ci) | np.isnan(ui)
        # Priority matches the original if/elif chain: upper-touch beats
        # lower-touch beats midline-exit — applied in reverse (lowest
        # priority first) so each np.where can be overridden by the next.
        new_state = np.where((state == -1) & (ci >= mi), 0, state)
        new_state = np.where((state == 1) & (ci <= mi), 0, new_state)
        new_state = np.where(ci <= loi, -1, new_state)
        new_state = np.where(ci >= ui, 1, new_state)
        state = np.where(invalid, state, new_state)  # NaN row: state carries over unchanged
        out[i] = np.where(invalid, 0, state)
    return pd.DataFrame(out, index=close_df.index, columns=close_df.columns)


def _dc_signal_multi(close_df: pd.DataFrame, n: int) -> pd.DataFrame:
    hi_df = close_df.rolling(n).max()
    lo_df = close_df.rolling(n).min()
    c, hi, lo = (a.to_numpy() for a in (close_df, hi_df, lo_df))
    T, P = c.shape
    state = np.zeros(P)
    out = np.zeros((T, P))
    for i in range(T):
        ci, hii, loi = c[i], hi[i], lo[i]
        invalid = np.isnan(hii) | np.isnan(loi)
        new_state = np.where(ci <= loi, -1, state)
        new_state = np.where(ci >= hii, 1, new_state)
        state = np.where(invalid, state, new_state)
        out[i] = np.where(invalid, 0, state)
    return pd.DataFrame(out, index=close_df.index, columns=close_df.columns)


def calculate_indicators_multi(close_df: pd.DataFrame, progress_cb=None) -> dict:
    """Vectorized multi-path version of calculate_indicators(). close_df: one
    column per Monte Carlo path (DatetimeIndex, weekdays only, all sharing the
    same history up to where the paths diverge). Returns the 5 composites as
    {name: DataFrame(T, P)} — {'ST_Avg','MT_Avg','LT_Avg','All_Avg','WAll_Avg'}.

    progress_cb(fraction, stage_name): optional, called after each of the 10
    indicator families finishes — real stage-by-stage progress (not a fake
    timer), for driving a UI progress bar during the Monte Carlo feature."""
    # Force plain numpy float64 — a pandas nullable 'Float64' input (e.g. from
    # a parquet round-trip) makes .to_numpy() return an object array, which
    # breaks np.isnan() in the stateful loops below.
    close_df = close_df.astype('float64')
    cols: dict[str, pd.DataFrame] = {}
    _stages = ['Mom', 'MA', 'EMA', 'HMA', '3MA', 'BB', 'DC', 'LRS', 'TRIX', 'KAMA']

    def _tick(stage: str):
        if progress_cb is not None:
            progress_cb((_stages.index(stage) + 1) / len(_stages), stage)

    for n in PARAMS['Mom']:
        ret = close_df.pct_change(n)
        vol = close_df.pct_change().rolling(n).std()
        cols[f'Mom_{n}'] = np.tanh(ret / (vol * np.sqrt(n))).fillna(0)
    _tick('Mom')

    for s, l in PARAMS['MA']:
        sma_s, sma_l = close_df.rolling(s).mean(), close_df.rolling(l).mean()
        cols[f'MA_cross_({s}, {l})'] = pd.DataFrame(
            np.where(sma_s > sma_l, 1, -1), index=close_df.index, columns=close_df.columns, dtype=float)
    _tick('MA')

    for s, l in PARAMS['EMA']:
        ema_s = close_df.ewm(span=s, adjust=False).mean()
        ema_l = close_df.ewm(span=l, adjust=False).mean()
        cols[f'EMA_cross_({s}, {l})'] = pd.DataFrame(
            np.where(ema_s > ema_l, 1, -1), index=close_df.index, columns=close_df.columns, dtype=float)
    _tick('EMA')

    for s, l in PARAMS['HMA']:
        hma_s, hma_l = _hma_multi(close_df, s), _hma_multi(close_df, l)
        sig = pd.DataFrame(np.where(hma_s > hma_l, 1, -1), index=close_df.index,
                           columns=close_df.columns, dtype=float)
        cols[f'HMA_cross_({s}, {l})'] = sig.where(hma_s.notna() & hma_l.notna(), other=0.0)
    _tick('HMA')

    for s, mm, l in PARAMS['3MA']:
        vs = close_df.rolling(s).mean().to_numpy()
        vm = close_df.rolling(mm).mean().to_numpy()
        vl = close_df.rolling(l).mean().to_numpy()
        invalid = np.isnan(vs) | np.isnan(vm) | np.isnan(vl)
        sig = np.where((vs > vm) & (vm > vl), 1, np.where((vs < vm) & (vm < vl), -1, 0))
        cols[f'3MA_cross_({s}, {mm}, {l})'] = pd.DataFrame(
            np.where(invalid, 0, sig), index=close_df.index, columns=close_df.columns, dtype=float)
    _tick('3MA')

    for n in PARAMS['BB']:
        cols[f'BB_{n}'] = _bb_signal_multi(close_df, n)
    _tick('BB')

    for n in PARAMS['DC']:
        cols[f'DC_{n}'] = _dc_signal_multi(close_df, n)
    _tick('DC')

    for n in PARAMS['LRS']:
        cols[f'LRS_{n}'] = np.sign(_linreg_slope_multi(close_df, n)).fillna(0)
    _tick('LRS')

    for n in PARAMS['TRIX']:
        cols[f'TRIX_{n}'] = _trix_sign_multi(close_df, n)
    _tick('TRIX')

    for n in PARAMS['KAMA']:
        kama = _kama_multi(close_df, n)
        sig = pd.DataFrame(np.where(close_df > kama, 1, -1), index=close_df.index,
                           columns=close_df.columns, dtype=float)
        cols[f'KAMA_{n}'] = sig.where(kama.notna(), other=0.0)
    _tick('KAMA')

    st_avg   = sum(cols[c] for c in ST_COLS) / len(ST_COLS)
    mt_avg   = sum(cols[c] for c in MT_COLS) / len(MT_COLS)
    lt_avg   = sum(cols[c] for c in LT_COLS) / len(LT_COLS)
    all_avg  = sum(cols[c] for c in ALL_SIGNAL_COLS) / len(ALL_SIGNAL_COLS)
    wall_avg = 0.20 * st_avg + 0.45 * mt_avg + 0.35 * lt_avg

    return {'ST_Avg': st_avg, 'MT_Avg': mt_avg, 'LT_Avg': lt_avg,
           'All_Avg': all_avg, 'WAll_Avg': wall_avg}

# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_price_history(inst: Instrument, full_refresh: bool = False) -> pd.DataFrame:
    """Fetch GSCI price history (Source='GSCI'), maintaining incremental parquet cache."""
    source = 'GSCI'
    last_date = None if full_refresh else get_last_price_date(inst, source)

    if LSEG_AVAILABLE:
        end_str = (date.today() + timedelta(days=1)).isoformat()
        if last_date is not None:
            start_str = last_date.strftime('%Y-%m-%d')
            print(f'  [{inst.short}] incremental fetch from {start_str}')
            try:
                new = ld.get_history(
                    universe=inst.gsci_ric, fields=['TRDPRC_1'],
                    start=start_str, end=end_str, interval='daily',
                )
                if new is not None and not new.empty:
                    new = new.rename(columns={'TRDPRC_1': 'CLOSE'})
                    new.index = pd.to_datetime(new.index)
                    new = new[new.index.dayofweek < 5]
                    new.index.name = 'Date'
                    upsert_price_history(inst, new, source=source)
            except Exception as e:
                print(f'  [{inst.short}] incremental fetch error: {e} — using cached data')
        else:
            start_str = (datetime.today() - timedelta(days=365 * 20)).strftime('%Y-%m-%d')
            print(f'  [{inst.short}] full fetch from {start_str}')
            try:
                combined = ld.get_history(
                    universe=inst.gsci_ric, fields=['TRDPRC_1'],
                    start=start_str, end=end_str, interval='daily',
                )
                if combined is None or combined.empty:
                    raise ValueError('Empty response from LSEG')
                combined = combined.rename(columns={'TRDPRC_1': 'CLOSE'})
                combined.index = pd.to_datetime(combined.index)
                combined = combined[combined.index.dayofweek < 5]
                combined.index.name = 'Date'
                upsert_price_history(inst, combined, source=source)
            except Exception as e:
                print(f'  [{inst.short}] full fetch error: {e}')
                if get_last_price_date(inst, source) is None:
                    raise
                print(f'  [{inst.short}] falling back to cached parquet data')
    else:
        if get_last_price_date(inst, source) is None:
            raise RuntimeError(f'LSEG not available and no cached data for {inst.short}')
        print(f'  [{inst.short}] LSEG unavailable — loading cached parquet data')

    price_df = load_price_history(inst, source)
    print(f'  [{inst.short}] GSCI price history: {len(price_df)} rows, last={price_df.index.max().date()}')
    return price_df


def fetch_rollex_price(inst: Instrument) -> pd.DataFrame:
    """Read the sibling LSEG-Rollex project's continuous roll-adjusted price
    (rollex_px) directly from its own parquet — CTA never fetches or writes
    this itself, Rollex's own automator keeps it current. Also reads Rollex's
    own active_label column (e.g. "Dec'26" — the currently-active contract)
    alongside rollex_px and upserts it separately into active_labels.parquet
    for display (futures price tables/charts). Returns a DataFrame indexed by
    Date with a single CLOSE column, or empty if unavailable."""
    path = ROLLEX_DB_DIR / f'rollex_{inst.short}.parquet'
    if not path.exists():
        print(f'  [{inst.short}] Rollex file not found at {path} — skipping Rollex source')
        return pd.DataFrame(columns=['CLOSE'])
    want_cols = ['rollex_px', 'active_label']
    try:
        raw = pd.read_parquet(path)
        have_cols = [c for c in want_cols if c in raw.columns]
        raw = raw[have_cols]
    except Exception as e:
        print(f'  [{inst.short}] Rollex read error: {e} — skipping Rollex source')
        return pd.DataFrame(columns=['CLOSE'])

    if 'active_label' in raw.columns:
        label_df = raw[['active_label']].rename(columns={'active_label': 'Active_Label'}).dropna()
        label_df.index = pd.to_datetime(label_df.index)
        label_df.index.name = 'Date'
        if not label_df.empty:
            upsert_active_labels(inst, label_df.sort_index())

    df = raw[['rollex_px']].rename(columns={'rollex_px': 'CLOSE'}).dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()
    if not df.empty:
        upsert_price_history(inst, df, source='Rollex')
    print(f'  [{inst.short}] Rollex price history: {len(df)} rows, '
          f'last={df.index.max().date() if not df.empty else "n/a"}')
    return df


def fetch_futures_price_history(inst: Instrument, full_refresh: bool = False) -> pd.DataFrame:
    """Fetch front-month futures price history for display in the dashboard."""
    all_df = _load(FUTPX_FILE)
    sub = all_df[all_df['Commodity'] == inst.short] if not all_df.empty else all_df
    last_date = (pd.Timestamp(pd.to_datetime(sub['Date']).max())
                if not sub.empty and not full_refresh else None)

    if LSEG_AVAILABLE:
        end_str = (date.today() + timedelta(days=1)).isoformat()
        start_str = (last_date.strftime('%Y-%m-%d') if last_date is not None
                    else (datetime.today() - timedelta(days=365 * 20)).strftime('%Y-%m-%d'))
        print(f'  [{inst.short}] futures price fetch from {start_str}')
        try:
            raw = ld.get_history(
                universe=inst.futures_ric, fields=['SETTLE'],
                start=start_str, end=end_str, interval='daily',
            )
            if raw is not None and not raw.empty:
                raw = raw.rename(columns={'SETTLE': 'CLOSE'})
                raw.index = pd.to_datetime(raw.index)
                raw = raw[raw.index.dayofweek < 5]
                raw.index.name = 'Date'
                upsert_futures_price(inst, raw)
        except Exception as e:
            print(f'  [{inst.short}] futures price fetch error: {e}')

    all_df = _load(FUTPX_FILE)
    if all_df.empty:
        return pd.DataFrame(columns=['CLOSE'])
    df = all_df[all_df['Commodity'] == inst.short][['Date', 'Close']].copy()
    if df.empty:
        return pd.DataFrame(columns=['CLOSE'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df.index.name = 'Date'
    df.columns = ['CLOSE']
    print(f'  [{inst.short}] futures price: {len(df)} rows, last={df.index.max().date()}')
    return df

# ── Simulation ────────────────────────────────────────────────────────────────

def build_simulation(
    df: pd.DataFrame, gsci_price: float, daily_vol_pct: float,
    display_price: float = None,
) -> pd.DataFrame:
    """Project indicator signals 10 business days forward under UP / DOWN / UNCH scenarios.

    Signal computation uses the GSCI index price series (gsci_price applied as % to
    the last GSCI level to extend the history). display_price (front-month futures)
    is used for the price_* columns shown in the chart — if omitted, gsci_price is used.
    """
    if display_price is None:
        display_price = gsci_price

    pricemove    = daily_vol_pct / 100.0
    base         = df.tail(1500).copy()  # 1500 rows ensures long-period indicators (TRIX_300 needs ~900)
    last_date    = base.index.max()
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=10)
    steps        = np.arange(1, 11)

    gsci_scenarios = {
        'up':   gsci_price * (1 + pricemove) ** steps,
        'down': gsci_price * (1 - pricemove) ** steps,
        'unch': np.full(10, gsci_price),
    }
    display_scenarios = {
        'up':   display_price * (1 + pricemove) ** steps,
        'down': display_price * (1 - pricemove) ** steps,
        'unch': np.full(10, display_price),
    }

    result = pd.DataFrame(index=future_dates)
    for label, gsci_prices in gsci_scenarios.items():
        ext = pd.DataFrame({'CLOSE': gsci_prices}, index=future_dates)
        combined     = pd.concat([base[['CLOSE']], ext])
        combined_ind = calculate_indicators(combined)
        tail = combined_ind.tail(10)
        result[f'ST_Avg_{label}']   = tail['ST_Avg'].values
        result[f'MT_Avg_{label}']   = tail['MT_Avg'].values
        result[f'LT_Avg_{label}']   = tail['LT_Avg'].values
        result[f'All_Avg_{label}']  = tail['All_Avg'].values
        result[f'WAll_Avg_{label}'] = tail['WAll_Avg'].values
        result[f'price_{label}']    = display_scenarios[label]

    return result

# ── Simulation history ────────────────────────────────────────────────────────

def _sim_to_rows(inst: Instrument, run_date: pd.Timestamp, sim: pd.DataFrame, source: str = 'GSCI') -> list[dict]:
    rows = []
    for day_idx, (horizon_date, row) in enumerate(sim.iterrows(), start=1):
        rows.append({
            'Commodity':    inst.short,
            'Source':       source,
            'Run_Date':     run_date,
            'Horizon_Date': pd.Timestamp(horizon_date),
            'Horizon_Day':  day_idx,
            'ST_down':    row['ST_Avg_down'],   'MT_down':   row['MT_Avg_down'],
            'LT_down':    row['LT_Avg_down'],   'All_down':  row['All_Avg_down'],
            'ST_up':      row['ST_Avg_up'],     'MT_up':     row['MT_Avg_up'],
            'LT_up':      row['LT_Avg_up'],     'All_up':    row['All_Avg_up'],
            'ST_unch':    row['ST_Avg_unch'],   'MT_unch':   row['MT_Avg_unch'],
            'LT_unch':    row['LT_Avg_unch'],   'All_unch':  row['All_Avg_unch'],
            'WAll_down':  row['WAll_Avg_down'],
            'WAll_up':    row['WAll_Avg_up'],
            'WAll_unch':  row['WAll_Avg_unch'],
            'price_down': row['price_down'],
            'price_up':   row['price_up'],
            'price_unch': row['price_unch'],
            'Actual_Close': np.nan,
        })
    return rows


def _backfill_actual_close(sim_df: pd.DataFrame, price_df_all: pd.DataFrame) -> pd.DataFrame:
    """Fill Actual_Close for past horizon dates using price_history (as-of
    backward join), matched on both Commodity AND Source — a Rollex-sourced
    sim row must be backfilled from Rollex prices, not GSCI, and vice versa."""
    if sim_df.empty or price_df_all.empty:
        return sim_df
    price_df_all = price_df_all.copy()
    price_df_all['Date'] = pd.to_datetime(price_df_all['Date'])
    needs_fill = sim_df['Actual_Close'].isna() & (sim_df['Horizon_Date'] <= pd.Timestamp(date.today()))
    if not needs_fill.any():
        return sim_df
    for inst_short, src in sim_df.loc[needs_fill, ['Commodity', 'Source']].drop_duplicates().itertuples(index=False):
        px = price_df_all[(price_df_all['Commodity'] == inst_short) & (price_df_all['Source'] == src)].sort_values('Date')
        if px.empty:
            continue
        mask = needs_fill & (sim_df['Commodity'] == inst_short) & (sim_df['Source'] == src)
        for idx in sim_df.index[mask]:
            hdate = sim_df.at[idx, 'Horizon_Date']
            eligible = px[px['Date'] <= hdate]
            if not eligible.empty:
                sim_df.at[idx, 'Actual_Close'] = eligible.iloc[-1]['Close']
    return sim_df


def append_sim_history(inst: Instrument, sim: pd.DataFrame, source: str = 'GSCI') -> None:
    """Append today's simulation to sim_history (idempotent — replaces if re-run today)."""
    today = pd.Timestamp(date.today())
    old = _load(SIM_FILE)
    if not old.empty:
        old = old[~((old['Commodity'] == inst.short) & (old['Source'] == source) & (old['Run_Date'] == today))]
    new_rows = pd.DataFrame(_sim_to_rows(inst, today, sim, source=source))
    combined = pd.concat([old, new_rows], ignore_index=True) if not old.empty else new_rows
    combined = _backfill_actual_close(combined, _load(PRICE_FILE))
    combined = combined.sort_values(['Commodity', 'Source', 'Run_Date', 'Horizon_Day']).reset_index(drop=True)
    combined.to_parquet(SIM_FILE, index=False)
    count = ((combined['Commodity'] == inst.short) & (combined['Source'] == source)).sum()
    print(f'  [{inst.short}/{source}] sim history: {count} rows total')


def backfill_sim_history(
    inst: Instrument, price_df: pd.DataFrame, futures_price_df: pd.DataFrame = None,
    lookback_bdays: int = 10, source: str = 'GSCI',
) -> None:
    """Compute simulations for any of the last `lookback_bdays` business days not yet stored."""
    sim_all = _load(SIM_FILE)
    if not sim_all.empty:
        already_run = set(
            pd.to_datetime(sim_all.loc[
                (sim_all['Commodity'] == inst.short) & (sim_all['Source'] == source), 'Run_Date'
            ]).dt.normalize()
        )
    else:
        already_run = set()

    daily_vol = price_df['CLOSE'].pct_change().rolling(20).std() * 100
    last_bdates = pd.bdate_range(end=price_df.index.max(), periods=lookback_bdays)
    dates = price_df.index[price_df.index.isin(last_bdates)]

    all_rows = []
    for run_date in dates:
        if pd.Timestamp(run_date).normalize() in already_run:
            continue

        gsci_price = price_df.loc[run_date, 'CLOSE']
        vol_pct    = daily_vol.loc[run_date]
        slice_df   = price_df.loc[price_df.index <= run_date]

        if len(slice_df) < 300 or pd.isna(gsci_price) or pd.isna(vol_pct) or vol_pct <= 0:
            continue

        display_price = None
        if futures_price_df is not None and not futures_price_df.empty:
            fut_idx = futures_price_df.index[futures_price_df.index <= run_date]
            if len(fut_idx) > 0:
                display_price = float(futures_price_df.loc[fut_idx[-1], 'CLOSE'])

        try:
            sim = build_simulation(slice_df, float(gsci_price), float(vol_pct),
                                   display_price=display_price)
        except Exception as e:
            print(f'  [{inst.short}/{source}] backfill error on {run_date.date()}: {e}')
            continue

        all_rows.extend(_sim_to_rows(inst, pd.Timestamp(run_date), sim, source=source))
        print(f'  [{inst.short}/{source}] backfilled {run_date.date()}')

    if not all_rows:
        print(f'  [{inst.short}/{source}] backfill: nothing new to add')
        return

    new_df = pd.DataFrame(all_rows)
    old = _load(SIM_FILE)
    combined = pd.concat([old, new_df], ignore_index=True) if not old.empty else new_df
    combined = _backfill_actual_close(combined, _load(PRICE_FILE))
    combined = combined.sort_values(['Commodity', 'Source', 'Run_Date', 'Horizon_Day']).reset_index(drop=True)
    combined.to_parquet(SIM_FILE, index=False)
    count = ((combined['Commodity'] == inst.short) & (combined['Source'] == source)).sum()
    print(f'  [{inst.short}/{source}] backfill done: {len(all_rows) // 10} new dates, {count} total rows')


def reset_sim_history() -> None:
    """Wipe all sim_history rows so they can be cleanly re-backfilled."""
    if SIM_FILE.exists():
        SIM_FILE.unlink()
    print('  sim_history cleared — will be rebuilt during backfill')

# ── Monte Carlo signal bands ────────────────────────────────────────────────────
#
# The deterministic UP/DOWN/UNCH scenarios above compound the same vol% move
# every day for 10 days — an extreme stress path, not a likely-range estimate.
# This bootstraps N random 10-day paths from the instrument's own recent daily
# returns (real historical returns, not a Normal-distribution assumption — keeps
# fat tails/skew) and reduces the recomputed indicator set down to per-horizon-
# day percentile bands, using calculate_indicators_multi() (all N paths at
# once — see that function's docstring) rather than looping calculate_
# indicators() N times: N=100 in ~2s, N=500 in ~11s, vs. the original
# per-path-loop's ~2.3s PER PATH (N=50 alone took ~47s). Fast enough to run
# live in the dashboard on demand — no longer needs to be an ingest-only,
# once-a-day batch step.
#
# Returns pool: last 20 trading days only (not the long-run ~2yr/500-day
# window this used before) — deliberately captures the CURRENT/latest
# volatility regime rather than averaging over a long history that may no
# longer be representative.

MC_LOOKBACK_DAYS = 20  # trading days of historical returns the bootstrap draws from

def compute_monte_carlo_bands(price_df: pd.DataFrame, n_paths: int = MC_N_PATHS,
                              horizon: int = 10, seed: int = 42, progress_cb=None) -> pd.DataFrame:
    """progress_cb(fraction, stage_name): passed straight through to
    calculate_indicators_multi() — see its docstring."""
    base = price_df['CLOSE'].tail(1500).copy()
    if len(base) < 300:
        return pd.DataFrame()
    hist_returns = base.pct_change().dropna().tail(MC_LOOKBACK_DAYS).to_numpy()
    if len(hist_returns) < 10:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    last_price = float(base.iloc[-1])
    last_date = base.index.max()
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=horizon)

    draws = rng.choice(hist_returns, size=(n_paths, horizon), replace=True)
    sim_prices = last_price * np.cumprod(1 + draws, axis=1)  # (n_paths, horizon)

    idx = base.index.append(future_dates)
    close_df = pd.DataFrame(
        {f'p{i}': np.concatenate([base.to_numpy(), sim_prices[i]]) for i in range(n_paths)},
        index=idx,
    )
    composites = calculate_indicators_multi(close_df, progress_cb=progress_cb)

    bands = pd.DataFrame({'Horizon_Date': future_dates, 'Horizon_Day': range(1, horizon + 1)})
    for c_name, df_ in composites.items():
        prefix = c_name.replace('_Avg', '')
        tail = df_.tail(horizon).to_numpy()  # (horizon, n_paths)
        for pct, label in [(10, 'p10'), (25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
            bands[f'{prefix}_{label}'] = np.percentile(tail, pct, axis=1)
    return bands

# ── Main ──────────────────────────────────────────────────────────────────────

def main(full_refresh: bool = False, reset_sim: bool = False) -> None:
    if reset_sim:
        print('Resetting sim_history…')
        reset_sim_history()

    if LSEG_AVAILABLE:
        print('Opening LSEG session…')
        ld.open_session()

    try:
        for inst in INSTRUMENTS:
            print(f'\n=== {inst.short} ({inst.label}) ===')

            # ── GSCI source — only for instruments with a GSCI sub-index ───────
            # (LCC/LSU/RC have no S&P GSCI single-commodity index at all —
            # gsci_ric is None for them, so this whole block is skipped and
            # they exist purely as Rollex-sourced instruments.)
            if inst.gsci_ric:
                price_df = fetch_price_history(inst, full_refresh=full_refresh)
                futures_df = fetch_futures_price_history(inst, full_refresh=full_refresh)

                ind_df = calculate_indicators(price_df)
                upsert_indicators(inst, ind_df, source='GSCI')
                print(f'  [{inst.short}/GSCI] indicators saved')

                backfill_sim_history(inst, price_df, futures_price_df=futures_df,
                                     lookback_bdays=10, source='GSCI')

                try:
                    gsci_price    = float(price_df['CLOSE'].dropna().iloc[-1])
                    futures_price = (float(futures_df['CLOSE'].dropna().iloc[-1])
                                     if not futures_df.empty else gsci_price)
                    daily_vol_pct = float(
                        price_df['CLOSE'].pct_change().rolling(20).std().dropna().iloc[-1] * 100
                    )
                    print(f'  [{inst.short}/GSCI] gsci={gsci_price:.5g}  futures={futures_price:.5g}'
                          f'  daily_vol={daily_vol_pct:.3f}%')
                    sim = build_simulation(price_df, gsci_price, daily_vol_pct,
                                           display_price=futures_price)
                    append_sim_history(inst, sim, source='GSCI')
                except Exception as e:
                    print(f'  [{inst.short}/GSCI] today\'s simulation error: {e}')
            else:
                print(f'  [{inst.short}] no GSCI sub-index — Rollex-only instrument')

            # ── Rollex source ────────────────────────────────────────────────
            if inst.short in ROLLEX_SHORTS:
                rollex_df = fetch_rollex_price(inst)
                if not rollex_df.empty:
                    rollex_ind_df = calculate_indicators(rollex_df)
                    upsert_indicators(inst, rollex_ind_df, source='Rollex')
                    print(f'  [{inst.short}/Rollex] indicators saved')

                    # No separate display-price fetch for Rollex — rollex_px IS
                    # the display price too (already continuous/roll-adjusted).
                    backfill_sim_history(inst, rollex_df, futures_price_df=rollex_df,
                                         lookback_bdays=10, source='Rollex')

                    try:
                        rollex_price = float(rollex_df['CLOSE'].dropna().iloc[-1])
                        rollex_vol_pct = float(
                            rollex_df['CLOSE'].pct_change().rolling(20).std().dropna().iloc[-1] * 100
                        )
                        print(f'  [{inst.short}/Rollex] price={rollex_price:.5g}  daily_vol={rollex_vol_pct:.3f}%')
                        rollex_sim = build_simulation(rollex_df, rollex_price, rollex_vol_pct,
                                                      display_price=rollex_price)
                        append_sim_history(inst, rollex_sim, source='Rollex')
                    except Exception as e:
                        print(f'  [{inst.short}/Rollex] today\'s simulation error: {e}')

    finally:
        if LSEG_AVAILABLE:
            print('\nClosing LSEG session…')
            ld.close_session()

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch TR Mapping data and compute indicators.')
    parser.add_argument('--full',      action='store_true', help='Full refresh from 20 years ago')
    parser.add_argument('--reset-sim', action='store_true',
                        help='Wipe sim_history and rebuild from scratch')
    args = parser.parse_args()
    main(full_refresh=args.full, reset_sim=args.reset_sim)
