# LSEG-CTA

CTA trend-following signal monitor for GSCI single-commodity sub-indices
(Coffee, Cotton, Sugar, Cocoa, Orange Juice). Converted to the Hardminer
architecture (Parquet data, Streamlit dashboard, GitHub repo) from Romain's
original "TR mapping old" tool (Dash + DuckDB) — indicator methodology is
unchanged.

## Structure
- `Code/tr_mapping_fetch.py` — daily LSEG fetch, ~144-signal indicator compute
  (Mom/MA/EMA/HMA/3MA-cross/BB/DC/LRS/TRIX/KAMA across ST/MT/LT buckets),
  10-day forward simulation with actual-outcome backfill.
- `Dashboard/app.py` — Streamlit dashboard (Overview / All Projections / All
  Signals / per-instrument Charts, Weekly Change, Projection tabs).
- `Database/*.parquet` — `price_history` (GSCI index, signal basis),
  `futures_price` (front-month, display only), `indicators`, `sim_history`.
- `Automator/run.bat` + `notify.py` — daily scheduled run: fetch, git push,
  email summary.

## Signal methodology
Signals are computed on the GSCI sub-index price series (e.g. `.SPGSKCP` for
Coffee), not on raw futures price (which is fetched separately, display-only).
Composites: `ST_Avg` / `MT_Avg` / `LT_Avg` (column-list means), `All_Avg`
(mean of all 144), `WAll_Avg = 0.20*ST_Avg + 0.45*MT_Avg + 0.35*LT_Avg`.
