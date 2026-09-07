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
SHORTS   = ["KC", "CT", "SB", "CC", "OJ"]

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def indicator_summary() -> str:
    lines = []
    path = DB_DIR / "indicators.parquet"
    if not path.exists():
        return "  indicators.parquet NOT FOUND"
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    for short in SHORTS:
        sub = df[df["Commodity"] == short].sort_values("Date")
        if sub.empty:
            lines.append(f"  {short:<4}  NO DATA")
            continue
        last = sub.iloc[-1]
        lines.append(
            f"  {short:<4}  {len(sub):>5} rows   last={last['Date'].date()}   "
            f"ST={last['ST_Avg']:+.3f}  MT={last['MT_Avg']:+.3f}  "
            f"LT={last['LT_Avg']:+.3f}  WAll={last['WAll_Avg']:+.3f}"
        )
    return "\n".join(lines)


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
Note: CTA trend-following signal pipeline, converted from Romain's "TR mapping
old" tool to the Hardminer architecture (Parquet data, Streamlit dashboard).
Indicator math (144 signals, ST/MT/LT/WAll composites) is unchanged from the
original.

Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\CTA\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
