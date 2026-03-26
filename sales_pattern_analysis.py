"""
Sales Pattern Analysis
======================
Drop new Shopify order export CSVs into a folder and run this script.
Outputs a full console report + JSON file.

Usage:
    python3 sales_pattern_analysis.py --folder "Paula Hian 365 days"
    python3 sales_pattern_analysis.py --folder ./orders --days 180
    python3 sales_pattern_analysis.py --folder ./orders --output q2_report.json

Requirements:
    pip3 install pandas python-dateutil
"""

import pandas as pd
import glob
import json
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from dateutil.relativedelta import relativedelta

MONTH_NAMES = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_orders(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(folder) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    print(f"\nLoading {len(files)} file(s) from '{folder}'...")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Rows loaded: {len(df):,}")
    return df

# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df["created_dt"] = (
        pd.to_datetime(df["Created at"], utc=True, errors="coerce")
        .dt.tz_localize(None)
    )
    max_date = df["created_dt"].max()
    cutoff   = max_date - pd.Timedelta(days=days)
    df = df[df["created_dt"] >= cutoff].copy()
    print(f"  Date window : {cutoff.date()} to {max_date.date()} ({days} days)")
    print(f"  Orders      : {df['Name'].nunique():,}")

    df["month"]   = df["created_dt"].dt.month
    df["quarter"] = df["created_dt"].dt.quarter
    df["hour"]    = (df["created_dt"].dt.hour - 4) % 24
    df["dow"]     = df["created_dt"].dt.strftime("%A")

    for col in ["Total","Lineitem price","Lineitem quantity","Discount Amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["line_revenue"]   = df["Lineitem price"] * df["Lineitem quantity"]
    df["product_family"] = df["Lineitem name"].str.split("/").str[0].str.strip()
    return df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_order_dates(df: pd.DataFrame) -> pd.Series:
    orders = df.drop_duplicates(subset="Name")[["Email","created_dt"]].copy()
    return orders.groupby("Email")["created_dt"].min()

def _new_vs_returning(df: pd.DataFrame) -> dict:
    orders     = df.drop_duplicates(subset="Name").copy()
    first_date = _first_order_dates(df)
    orders["is_new"] = orders.apply(
        lambda r: r["created_dt"] == first_date.get(r["Email"], pd.NaT), axis=1
    )
    result = {}
    for m in range(1, 13):
        subset = orders[orders["month"] == m]
        new_c  = int(subset["is_new"].sum())
        ret_c  = int((~subset["is_new"]).sum())
        new_r  = float(subset.loc[subset["is_new"], "Total"].sum())
        ret_r  = float(subset.loc[~subset["is_new"], "Total"].sum())
        result[m] = {
            "new_customers":       new_c,
            "returning_customers": ret_c,
            "new_revenue":         round(new_r, 2),
            "returning_revenue":   round(ret_r, 2),
            "new_pct":             round(new_c / (new_c + ret_c) * 100, 1) if (new_c + ret_c) else 0,
        }
    return result

def _cohort_ltv(df: pd.DataFrame) -> dict:
    orders     = df.drop_duplicates(subset="Name").copy()
    first_date = _first_order_dates(df)
    orders["first_month"] = orders["Email"].map(
        lambda e: first_date[e].to_period("M") if e in first_date.index else None
    )
    orders["order_month"] = orders["created_dt"].dt.to_period("M")
    
    # Safely compute months since first order
    def months_since_first(r):
        email_first = first_date.get(r["Email"], pd.NaT)
        if pd.notna(email_first):
            delta = relativedelta(r["created_dt"], email_first)
            return delta.years * 12 + delta.months
        return 0

    orders["months_since_first"] = orders.apply(months_since_first, axis=1)

    cohorts = {}
    for cohort_period, group in orders.groupby("first_month"):
        cohort_key = str(cohort_period)
        size       = int(group["Email"].nunique())
        if size < 5:
            continue
        rev_by_lag = group.groupby("months_since_first")["Total"].sum().to_dict()
        cohorts[cohort_key] = {
            "cohort_size": size,
            "ltv_m1":  round(float(rev_by_lag.get(0, 0)) / size, 2),
            "ltv_m3":  round(sum(float(rev_by_lag.get(l, 0)) for l in range(0, 3)) / size, 2),
            "ltv_m6":  round(sum(float(rev_by_lag.get(l, 0)) for l in range(0, 6)) / size, 2),
            "ltv_m12": round(sum(float(rev_by_lag.get(l, 0)) for l in range(0, 12)) / size, 2),
        }
    return cohorts

# ... rest of your code remains unchanged ...
