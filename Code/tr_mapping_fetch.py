"""
tr_mapping_fetch.py — Fetch LSEG data, compute CTA trend-following indicators,
build simulation scenarios and maintain simulation history for GSCI commodity indices.

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

Two signal sources per instrument, selectable in the dashboard:
    GSCI   — S&P GSCI single-commodity sub-index (.SPGSKCP etc), all 5 instruments,
             history from ~2006. Matches Romain's original methodology exactly.
    Rollex — this desk's own continuous roll-adjusted futures price (rollex_px),
             read directly from the sibling LSEG-Rollex project's own parquet
             output (cross-repo read, no re-fetch — Rollex maintains its own
             data). Only KC/CT/SB/CC are covered (Rollex has no OJ); history
             from ~2010. Same 144-indicator math applies unchanged — it's
             return/crossing-based, so it's source-agnostic — but the actual
             signal VALUES differ between sources because GSCI's and Rollex's
             roll methodologies differ.

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
import pandas_ta as ta

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Database'
DATA_DIR.mkdir(exist_ok=True)

PRICE_FILE    = DATA_DIR / 'price_history.parquet'
FUTPX_FILE    = DATA_DIR / 'futures_price.parquet'
IND_FILE      = DATA_DIR / 'indicators.parquet'
SIM_FILE      = DATA_DIR / 'sim_history.parquet'

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
    Instrument('KC', '.SPGSKCP', 'KCv1', 0, 'Coffee'),
    Instrument('CT', '.SPGSCTP', 'CTv1', 1, 'Cotton'),
    Instrument('SB', '.SPGSSBP', 'SBv1', 1, 'Sugar'),
    Instrument('CC', '.SPGSCCP', 'CCv1', 0, 'Cocoa'),
    Instrument('OJ', '.SPGSOJP', 'OJv1', 1, 'Orange Juice'),
    # Rollex-only — no S&P GSCI single-commodity sub-index exists for these
    # London-listed ICE contracts, so gsci_ric/futures_ric are both None and
    # the GSCI half of main()'s pipeline is skipped entirely for them.
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


# ── Stateful signal helpers ────────────────────────────────────────────────────

def _bb_signal(close: pd.Series, n: int) -> pd.Series:
    """Bollinger Band stateful signal: +1 / 0 / -1 with midline exit."""
    bb    = ta.bbands(close, length=n)
    upper = bb.filter(like='BBU').iloc[:, 0]
    mid   = bb.filter(like='BBM').iloc[:, 0]
    lower = bb.filter(like='BBL').iloc[:, 0]

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
    close = df['CLOSE']
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
        hma_s = ta.hma(close, length=s)
        hma_l = ta.hma(close, length=l)
        if hma_s is None or hma_l is None:
            cols[f'HMA_cross_({s}, {l})'] = zero.copy()
            continue
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
        slope = ta.linreg(close, length=n, slope=True)
        cols[f'LRS_{n}'] = zero.copy() if slope is None else np.sign(slope).fillna(0)

    # ── TRIX ──────────────────────────────────────────────────────────────────
    for n in PARAMS['TRIX']:
        trix_df   = ta.trix(close, length=n)
        trix_cols = [] if (trix_df is None or trix_df.empty) else [
            c for c in trix_df.columns if c.startswith('TRIX_') and not c.startswith('TRIXs_')
        ]
        if not trix_cols:
            cols[f'TRIX_{n}'] = zero.copy()
        else:
            cols[f'TRIX_{n}'] = np.sign(trix_df[trix_cols[0]].astype(float)).fillna(0)

    # ── KAMA vs close ─────────────────────────────────────────────────────────
    # Vol-adaptive moving average: +1 when price > KAMA (uptrend), -1 when below
    c_arr = close.astype(float).to_numpy()
    for n in PARAMS['KAMA']:
        kama_raw = ta.kama(close, length=n)
        if kama_raw is None:
            cols[f'KAMA_{n}'] = zero.copy()
            continue
        k_arr = kama_raw.astype(float).to_numpy()
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
    this itself, Rollex's own automator keeps it current. Returns a DataFrame
    indexed by Date with a single CLOSE column, or empty if unavailable."""
    path = ROLLEX_DB_DIR / f'rollex_{inst.short}.parquet'
    if not path.exists():
        print(f'  [{inst.short}] Rollex file not found at {path} — skipping Rollex source')
        return pd.DataFrame(columns=['CLOSE'])
    try:
        raw = pd.read_parquet(path, columns=['rollex_px'])
    except Exception as e:
        print(f'  [{inst.short}] Rollex read error: {e} — skipping Rollex source')
        return pd.DataFrame(columns=['CLOSE'])
    df = raw.rename(columns={'rollex_px': 'CLOSE'}).dropna()
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
