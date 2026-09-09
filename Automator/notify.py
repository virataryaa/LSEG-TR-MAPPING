"""
notify.py — CTA (LSEG) Automator email summary
Usage: python notify.py <status> <git_status>
  status     : ok | error
  git_status : pushed | skipped | failed
"""

import sys
import datetime
import pandas as pd
from pathlib import Path

TO_EMAIL = "virat.arya@etgworld.com"
DB_DIR   = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\CTA\Database")
SHORTS   = ["KC", "RC", "CC", "LCC", "SB", "LSU", "CT", "OJ"]  # matches Dashboard/app.py's order

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def indicator_summary() -> str:
    """One line per (instrument, source) — GSCI and Rollex last-date/signal
    shown separately, same as the dashboard sidebar's 'Latest Data by
    Instrument'. Not every instrument has both (OJ is GSCI-only, LCC/LSU/RC
    are Rollex-only). Settle price joined in from price_history.parquet —
    indicators.parquet itself has no CLOSE column (dropped on write, since
    it's already stored there)."""
    lines = []
    ind_path = DB_DIR / "indicators.parquet"
    px_path  = DB_DIR / "price_history.parquet"
    if not ind_path.exists():
        return "  indicators.parquet NOT FOUND"
    df = pd.read_parquet(ind_path)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Source" not in df.columns:
        df["Source"] = "GSCI"

    px_df = pd.DataFrame(columns=["Commodity", "Source", "Date", "Close"])
    if px_path.exists():
        px_df = pd.read_parquet(px_path)
        px_df["Date"] = pd.to_datetime(px_df["Date"])

    for short in SHORTS:
        sub = df[df["Commodity"] == short]
        if sub.empty:
            lines.append(f"  {short:<4}  NO DATA")
            continue
        first = True
        for src in ("GSCI", "Rollex"):
            s = sub[sub["Source"] == src].sort_values("Date")
            if s.empty:
                continue
            last = s.iloc[-1]
            label = f"{short:<4}" if first else "    "
            px_sub = px_df[(px_df["Commodity"] == short) & (px_df["Source"] == src)
                          & (px_df["Date"] == last["Date"])]
            settle = f"{px_sub['Close'].iloc[0]:.2f}" if not px_sub.empty else "n/a"
            lines.append(
                f"  {label}  {src:<6} last={last['Date'].date()}  settle={settle:>9}   "
                f"ST={last['ST_Avg']:+.3f}  MT={last['MT_Avg']:+.3f}  "
                f"LT={last['LT_Avg']:+.3f}  WAll={last['WAll_Avg']:+.3f}"
            )
            first = False
    return "\n".join(lines)


def week_to_date_note() -> str:
    """Same Tuesday-to-Tuesday convention as the dashboard's Weekly Change
    caption — states the week runs Tuesday-to-Tuesday, and if the latest data
    is past the most recent Tuesday, calls out that partial week-to-date span."""
    path = DB_DIR / "indicators.parquet"
    if not path.exists():
        return "Week = Tuesday-to-Tuesday."
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if df.empty:
        return "Week = Tuesday-to-Tuesday."
    latest_date = df["Date"].max()
    tues_dates = df.loc[df["Date"].dt.dayofweek == 1, "Date"]
    last_tuesday = tues_dates.max() if not tues_dates.empty else None
    if last_tuesday is not None and latest_date > last_tuesday:
        return (f"Week = Tuesday-to-Tuesday  |  Latest bar is week-to-date (partial): "
                f"{last_tuesday.date()} -> {latest_date.date()}.")
    return "Week = Tuesday-to-Tuesday."


def send_outlook_email(subject: str, body: str):
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        print(f"  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


ok  = status == "ok"
tag = "[OK]" if ok else "[ERROR]"
subject = f"{tag} LSEG-CTA — {today}"

git_line = {
    "pushed":  "GitHub  : Pushed successfully",
    "skipped": "GitHub  : No changes — push skipped",
    "failed":  "GitHub  : PUSH FAILED",
}.get(git_status, f"GitHub  : {git_status}")

body = f"""LSEG CTA — Daily Update
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — fetch script failed, check run_log.txt"}
{git_line}

{"=" * 60}
CTA TREND SIGNAL SUMMARY
{"=" * 60}
{indicator_summary()}
{"=" * 60}
{week_to_date_note()}

Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\CTA\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
