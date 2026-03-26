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
    pip3 install pandas
"""

import pandas as pd
import glob
import json
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict

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

    def months_diff(row):
        try:
            return (row["order_month"] - row["Email_first"]).n
        except Exception:
            return 0

    email_first_map = first_date.dt.to_period("M")
    orders["email_first_period"] = orders["Email"].map(
        lambda e: email_first_map[e] if e in email_first_map.index else None
    )
    orders["months_since_first"] = orders.apply(
        lambda r: (r["order_month"] - r["email_first_period"]).n
        if r["email_first_period"] is not None else 0, axis=1
    )

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


def _time_to_second_order(df: pd.DataFrame) -> dict:
    orders = (
        df.drop_duplicates(subset="Name")[["Email","created_dt"]]
        .sort_values("created_dt")
    )
    email_orders = orders.groupby("Email")["created_dt"].apply(list)
    gaps = []
    for dates in email_orders:
        if len(dates) >= 2:
            gap = (dates[1] - dates[0]).days
            if 0 < gap < 730:
                gaps.append(gap)
    if not gaps:
        return {}
    s = pd.Series(gaps)
    return {
        "mean_days":   round(float(s.mean()), 1),
        "median_days": round(float(s.median()), 1),
        "p25_days":    round(float(s.quantile(0.25)), 1),
        "p75_days":    round(float(s.quantile(0.75)), 1),
        "recommended_winback_trigger_days": round(float(s.quantile(0.25)), 0),
        "sample_size": len(gaps),
    }


def _winback_buckets(df: pd.DataFrame) -> dict:
    orders    = df.drop_duplicates(subset="Name").copy()
    max_date  = df["created_dt"].max()
    email_cnt = orders.groupby("Email")["Name"].count()
    one_timers = email_cnt[email_cnt == 1].index
    ot = orders[orders["Email"].isin(one_timers)].copy()
    ot["days_ago"] = (max_date - ot["created_dt"]).dt.days
    buckets = {
        "0_30_days":    int((ot["days_ago"] <= 30).sum()),
        "31_90_days":   int(((ot["days_ago"] > 30)  & (ot["days_ago"] <= 90)).sum()),
        "91_180_days":  int(((ot["days_ago"] > 90)  & (ot["days_ago"] <= 180)).sum()),
        "181_365_days": int(((ot["days_ago"] > 180) & (ot["days_ago"] <= 365)).sum()),
    }
    total = sum(buckets.values())
    for k in list(buckets.keys()):
        buckets[k + "_pct"] = round(buckets[k] / total * 100, 1) if total else 0
    return buckets


def _product_affinity(df: pd.DataFrame, min_support: int = 20) -> list:
    order_families = (
        df[df["line_revenue"] > 0]
        .groupby("Name")["product_family"]
        .apply(lambda x: list(set(x)))
    )
    pair_counts = defaultdict(int)
    for families in order_families:
        if len(families) >= 2:
            for a, b in combinations(sorted(families), 2):
                pair_counts[(a, b)] += 1
    pairs = [
        {"product_a": a, "product_b": b, "co_orders": c}
        for (a, b), c in pair_counts.items()
        if c >= min_support
    ]
    return sorted(pairs, key=lambda x: -x["co_orders"])[:20]


def _first_to_second_product(df: pd.DataFrame) -> dict:
    order_seq = (
        df.drop_duplicates(subset="Name")[["Email","Name","created_dt"]]
        .sort_values(["Email","created_dt"])
    )
    first_order  = order_seq.groupby("Email")["Name"].first()
    second_order = order_seq.groupby("Email")["Name"].nth(1).dropna()
    multi = second_order.index
    if len(multi) == 0:
        return {}

    first_fam = (
        df[df["Name"].isin(first_order[multi].values) & (df["line_revenue"] > 0)]
        .groupby("Name")["product_family"].first().rename("first_family")
    )
    second_fam = (
        df[df["Name"].isin(second_order.values) & (df["line_revenue"] > 0)]
        .groupby("Name")["product_family"].first().rename("second_family")
    )

    email_df = pd.DataFrame({"first_order": first_order[multi], "second_order": second_order})
    email_df = email_df.join(first_fam, on="first_order").join(second_fam, on="second_order")
    email_df = email_df.dropna(subset=["first_family","second_family"])
    email_df = email_df[email_df["first_family"] != email_df["second_family"]]

    paths = (
        email_df.groupby(["first_family","second_family"])
        .size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    result = {}
    for _, row in paths.head(30).iterrows():
        key = row["first_family"]
        if key not in result:
            result[key] = []
        result[key].append({"next_product": row["second_family"], "count": int(row["count"])})
    return result


def _discount_by_order_number(df: pd.DataFrame) -> dict:
    orders = df.drop_duplicates(subset="Name").sort_values(["Email","created_dt"]).copy()
    orders["order_number"] = orders.groupby("Email").cumcount() + 1
    orders["used_discount"] = orders["Discount Amount"] > 0
    result = {}
    for n in range(1, 6):
        subset = orders[orders["order_number"] == n]
        if len(subset) == 0:
            continue
        result[f"order_{n}"] = {
            "total_orders":    int(len(subset)),
            "discounted":      int(subset["used_discount"].sum()),
            "discount_pct":    round(float(subset["used_discount"].mean()) * 100, 1),
            "avg_order_value": round(float(subset["Total"].mean()), 2),
        }
    other = orders[orders["order_number"] >= 6]
    if len(other):
        result["order_6_plus"] = {
            "total_orders":    int(len(other)),
            "discounted":      int(other["used_discount"].sum()),
            "discount_pct":    round(float(other["used_discount"].mean()) * 100, 1),
            "avg_order_value": round(float(other["Total"].mean()), 2),
        }
    return result


def _revenue_concentration(df: pd.DataFrame) -> dict:
    orders    = df.drop_duplicates(subset="Name").copy()
    total_rev = float(orders["Total"].sum())
    cust_rev  = orders.groupby("Email")["Total"].sum().sort_values(ascending=False)
    result    = {}
    for n in [10, 50, 100, 250]:
        top_rev = float(cust_rev.head(n).sum())
        result[f"top_{n}_customers"] = {
            "revenue":     round(top_rev, 2),
            "revenue_pct": round(top_rev / total_rev * 100, 1) if total_rev else 0,
        }
    return result


def _vip_customers(df: pd.DataFrame, top_n: int = 25) -> list:
    orders = df.drop_duplicates(subset="Name").copy()
    cust   = (
        orders.groupby("Email")
        .agg(
            total_spent=("Total","sum"),
            order_count=("Name","count"),
            city=("Billing City","first"),
            state=("Billing Province","first"),
            last_order=("created_dt","max"),
        )
        .sort_values("total_spent", ascending=False)
        .head(top_n)
    )
    result = []
    for email, row in cust.iterrows():
        result.append({
            "email":       str(email)[:4] + "***",
            "total_spent": round(float(row["total_spent"]), 2),
            "order_count": int(row["order_count"]),
            "city":        str(row["city"]),
            "state":       str(row["state"]),
            "last_order":  str(row["last_order"].date()),
        })
    return result


def _emerging_markets(df: pd.DataFrame) -> list:
    orders   = df.drop_duplicates(subset="Name").copy()
    mid_date = orders["created_dt"].min() + (orders["created_dt"].max() - orders["created_dt"].min()) / 2
    first_h  = orders[orders["created_dt"] <  mid_date].groupby("Billing City")["Name"].count()
    second_h = orders[orders["created_dt"] >= mid_date].groupby("Billing City")["Name"].count()
    combined = pd.DataFrame({"first": first_h, "second": second_h}).fillna(0)
    combined = combined[combined["first"] >= 5]
    combined["growth_pct"] = ((combined["second"] - combined["first"]) / combined["first"] * 100).round(1)
    combined = combined.sort_values("growth_pct", ascending=False).head(10)
    return [
        {
            "city":        city,
            "first_half":  int(row["first"]),
            "second_half": int(row["second"]),
            "growth_pct":  float(row["growth_pct"]),
        }
        for city, row in combined.iterrows()
    ]


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame) -> dict:
    orders    = df.drop_duplicates(subset="Name").copy()
    total_rev = float(orders["Total"].sum())
    report    = {}

    # Overview
    report["overview"] = {
        "total_revenue":        round(total_rev, 2),
        "total_orders":         len(orders),
        "unique_customers":     int(orders["Email"].nunique()),
        "avg_order_value":      round(float(orders["Total"].mean()), 2),
        "marketing_opt_in_pct": round(
            (orders["Accepts Marketing"] == "yes").sum() / len(orders) * 100, 1
        ),
    }

    # Monthly
    monthly = (
        orders.groupby("month")
        .agg(order_count=("Name","count"), revenue=("Total","sum"))
        .reindex(range(1, 13), fill_value=0)
    )
    monthly["aov"] = (monthly["revenue"] / monthly["order_count"].replace(0, 1)).round(2)
    report["monthly"]           = monthly.to_dict(orient="index")
    avg_monthly_rev             = float(monthly["revenue"].mean())
    report["seasonality_index"] = {
        int(m): round(float(r) / avg_monthly_rev, 2) if avg_monthly_rev > 0 else 0
        for m, r in monthly["revenue"].items()
    }

    # Quarterly
    quarterly = (
        orders.groupby("quarter")
        .agg(order_count=("Name","count"), revenue=("Total","sum"))
    )
    quarterly["aov"] = (quarterly["revenue"] / quarterly["order_count"].replace(0, 1)).round(2)
    report["quarterly"] = quarterly.to_dict(orient="index")

    # Product families
    prod_overall = (
        df.groupby("product_family")
        .agg(qty=("Lineitem quantity","sum"), revenue=("line_revenue","sum"))
        .sort_values("revenue", ascending=False)
    )
    report["top_products"] = {
        k: {"qty": int(v["qty"]), "revenue": round(v["revenue"], 2)}
        for k, v in prod_overall.head(20).to_dict(orient="index").items()
    }
    top_families = prod_overall.head(12).index.tolist()

    # Product by quarter
    prod_by_quarter = (
        df.groupby(["quarter","product_family"])["line_revenue"]
        .sum().unstack(fill_value=0)
    )
    report["product_by_quarter"] = {
        fam: {
            int(q): round(float(prod_by_quarter.loc[q, fam]), 2)
            if q in prod_by_quarter.index and fam in prod_by_quarter.columns else 0
            for q in range(1, 5)
        }
        for fam in top_families
    }

    # Product by month
    monthly_prod = (
        df.groupby(["month","product_family"])["line_revenue"]
        .sum().unstack(fill_value=0)
    )
    report["product_by_month"] = {
        fam: {
            int(m): round(float(monthly_prod.loc[m, fam]), 2)
            if m in monthly_prod.index and fam in monthly_prod.columns else 0
            for m in range(1, 13)
        }
        for fam in top_families
    }

    # Marketing priority guide
    all_families_15 = prod_overall.head(15).index.tolist()
    priority_guide  = []
    for m in range(1, 13):
        rev      = float(monthly["revenue"].get(m, 0))
        prev_rev = float(monthly["revenue"].get(m - 1, rev)) if m > 1 else rev
        mom_pct  = round((rev - prev_rev) / prev_rev * 100, 1) if prev_rev > 0 else 0
        idx      = rev / avg_monthly_rev if avg_monthly_rev > 0 else 0
        priority = "HIGH" if idx >= 1.5 else ("MEDIUM" if idx >= 0.7 else "LOW")

        top5 = []
        if m in monthly_prod.index:
            top5 = (
                monthly_prod.loc[m].where(monthly_prod.loc[m] > 0)
                .dropna().nlargest(5).index.tolist()
            )

        surging = []
        if m > 1:
            for fam in all_families_15:
                curr_v = float(monthly_prod.loc[m, fam])   if m     in monthly_prod.index and fam in monthly_prod.columns else 0
                prev_v = float(monthly_prod.loc[m-1, fam]) if m - 1 in monthly_prod.index and fam in monthly_prod.columns else 0
                if prev_v > 0 and curr_v > 0 and (curr_v - prev_v) / prev_v * 100 >= 30:
                    surging.append({"product": fam, "mom_change_pct": round((curr_v - prev_v) / prev_v * 100, 1)})

        priority_guide.append({
            "month":             m,
            "month_name":        MONTH_NAMES[m - 1],
            "revenue":           round(rev, 2),
            "mom_change_pct":    mom_pct,
            "seasonality_index": round(idx, 2),
            "priority":          priority,
            "top_products":      top5,
            "surging_products":  sorted(surging, key=lambda x: -x["mom_change_pct"])[:4],
        })
    report["marketing_priority_guide"] = priority_guide

    # ---- NEW STRATEGIC SECTIONS ---------------------------------------------

    print("  Analyzing new vs returning customers...")
    report["new_vs_returning_by_month"] = _new_vs_returning(df)

    print("  Analyzing customer cohort LTV...")
    report["cohort_ltv"] = _cohort_ltv(df)

    print("  Analyzing time to second order...")
    report["time_to_second_order"] = _time_to_second_order(df)

    print("  Analyzing win-back buckets...")
    report["winback_buckets"] = _winback_buckets(df)

    print("  Analyzing product affinity (cart co-occurrence)...")
    report["product_affinity_pairs"] = _product_affinity(df, min_support=20)

    print("  Analyzing first-to-second product journey...")
    report["first_to_second_product"] = _first_to_second_product(df)

    print("  Analyzing discount dependency by order number...")
    report["discount_by_order_number"] = _discount_by_order_number(df)

    print("  Analyzing revenue concentration...")
    report["revenue_concentration"] = _revenue_concentration(df)

    print("  Identifying VIP customers...")
    report["vip_customers"] = _vip_customers(df, top_n=25)

    print("  Analyzing emerging markets...")
    report["emerging_markets"] = _emerging_markets(df)

    # ---- INSIGHTS -----------------------------------------------------------

    insights = []

    peak_m    = int(monthly["revenue"].idxmax())
    peak_rev  = float(monthly["revenue"].max())
    prev_name = MONTH_NAMES[peak_m - 2] if peak_m > 1 else "the prior month"
    insights.append({
        "category": "Seasonality",
        "title":    f"{MONTH_NAMES[peak_m-1]} is your peak revenue month",
        "detail":   (
            f"${peak_rev:,.0f} -- {peak_rev/avg_monthly_rev:.1f}x above the monthly average. "
            f"Ensure campaigns and inventory are fully live by end of {prev_name}."
        ),
    })

    best_jump_m, best_jump_pct = 2, 0.0
    for m in range(2, 13):
        prev = float(monthly["revenue"].get(m - 1, 0))
        curr = float(monthly["revenue"].get(m, 0))
        if prev > 0:
            chg = (curr - prev) / prev * 100
            if chg > best_jump_pct:
                best_jump_pct, best_jump_m = chg, m
    insights.append({
        "category": "Seasonality",
        "title":    f"Biggest revenue jump: {MONTH_NAMES[best_jump_m-2]} to {MONTH_NAMES[best_jump_m-1]}",
        "detail":   (
            f"+{best_jump_pct:.0f}% month-over-month. "
            f"Launch campaigns in {MONTH_NAMES[best_jump_m-2]} to ride this surge."
        ),
    })

    top2     = list(report["top_products"].items())[:2]
    combined = sum(d["revenue"] for _, d in top2)
    insights.append({
        "category": "Product mix",
        "title":    f"{top2[0][0]} & {top2[1][0]} drive {combined/total_rev*100:.0f}% of revenue",
        "detail":   (
            f"Combined ${combined:,.0f}. Anchor every peak-season campaign around these two families."
        ),
    })

    peak_q     = max(report["quarterly"], key=lambda q: report["quarterly"][q]["revenue"])
    peak_q_rev = report["quarterly"][peak_q]["revenue"]
    insights.append({
        "category": "Quarterly",
        "title":    f"Q{peak_q} is your dominant quarter",
        "detail":   (
            f"${peak_q_rev:,.0f} -- {peak_q_rev/total_rev*100:.0f}% of annual revenue. "
            f"Allocate your largest ad budgets and hero product launches here."
        ),
    })

    slow_q     = min(report["quarterly"], key=lambda q: report["quarterly"][q]["revenue"])
    slow_q_rev = report["quarterly"][slow_q]["revenue"]
    insights.append({
        "category": "Quarterly",
        "title":    f"Q{slow_q} is your slowest quarter",
        "detail":   (
            f"Only ${slow_q_rev:,.0f}. Shift Q{slow_q} spend toward retention, loyalty rewards, "
            f"and early-access campaigns rather than broad acquisition."
        ),
    })

    nvr = report["new_vs_returning_by_month"]
    peak_new_m = max(nvr, key=lambda m: nvr[m]["new_customers"])
    insights.append({
        "category": "New vs. Returning",
        "title":    f"{MONTH_NAMES[peak_new_m-1]} drives the most new customer acquisition",
        "detail":   (
            f"{nvr[peak_new_m]['new_customers']:,} new customers that month. "
            f"This is your most important acquisition window -- maximize paid spend and referral programs here."
        ),
    })

    t2 = report.get("time_to_second_order", {})
    if t2:
        trigger = int(t2.get("recommended_winback_trigger_days", 30))
        insights.append({
            "category": "Retention",
            "title":    f"Send re-engagement emails at day {trigger} after first purchase",
            "detail":   (
                f"Median time to second order is {t2['median_days']} days. "
                f"Customers in the bottom 25% repurchase within {trigger} days -- "
                f"set your post-purchase flow to fire at day {trigger} to catch buyers at peak intent."
            ),
        })

    wb   = report.get("winback_buckets", {})
    warm = wb.get("31_90_days", 0)
    cold = wb.get("181_365_days", 0)
    if warm or cold:
        insights.append({
            "category": "Win-back",
            "title":    f"{warm:,} warm lapsed buyers ready for re-engagement",
            "detail":   (
                f"{warm:,} one-time buyers purchased 31-90 days ago (warm). "
                f"{cold:,} purchased 181-365 days ago (cold, needs stronger offer). "
                f"Segment these into two distinct win-back flows with different incentives."
            ),
        })

    pairs = report.get("product_affinity_pairs", [])
    if pairs:
        top_pair = pairs[0]
        insights.append({
            "category": "Product affinity",
            "title":    f"{top_pair['product_a']} + {top_pair['product_b']} is your top bundle opportunity",
            "detail":   (
                f"Co-purchased in {top_pair['co_orders']:,} orders. "
                f"Add a 'frequently bought together' widget on both product pages "
                f"and build a dedicated bundle offer around this pairing."
            ),
        })

    rc = report.get("revenue_concentration", {})
    if rc:
        top100 = rc.get("top_100_customers", {})
        if top100:
            insights.append({
                "category": "Revenue concentration",
                "title":    f"Top 100 customers = {top100['revenue_pct']}% of revenue",
                "detail":   (
                    f"${top100['revenue']:,.0f} from just 100 buyers. "
                    f"These VIPs deserve white-glove treatment: early access, personal outreach, "
                    f"exclusive colorways. Losing even 20 of them materially impacts the business."
                ),
            })

    dbon     = report.get("discount_by_order_number", {})
    o1_disc  = dbon.get("order_1", {}).get("discount_pct", 0)
    o3_disc  = dbon.get("order_3", {}).get("discount_pct", 0)
    if dbon:
        if o3_disc > o1_disc:
            insights.append({
                "category": "Discounts",
                "title":    "Repeat buyers use MORE discounts than new buyers -- margin risk",
                "detail":   (
                    f"1st-order discount rate: {o1_disc}%. 3rd-order discount rate: {o3_disc}%. "
                    f"You may be training loyal customers to wait for promotions. "
                    f"Introduce non-discount loyalty perks (early access, free shipping) "
                    f"to reduce discount dependency among your best customers."
                ),
            })
        else:
            insights.append({
                "category": "Discounts",
                "title":    "Discounts are concentrated in first orders -- healthy pattern",
                "detail":   (
                    f"1st-order discount rate: {o1_disc}%. Repeat buyers buy closer to full price. "
                    f"Your discounting is working as an acquisition tool without eroding LTV."
                ),
            })

    em = report.get("emerging_markets", [])
    if em:
        top_em = em[0]
        insights.append({
            "category": "Geography",
            "title":    f"{top_em['city']} is your fastest-growing market (+{top_em['growth_pct']:.0f}%)",
            "detail":   (
                f"Orders grew from {top_em['first_half']} to {top_em['second_half']} half-over-half. "
                f"Consider geo-targeted campaigns, a pop-up, or local press outreach here "
                f"before competitors notice the same signal."
            ),
        })

    email_orders = orders.groupby("Email")["Name"].count()
    report["customer_retention"] = {
        "one_time":                int((email_orders == 1).sum()),
        "two_orders":              int((email_orders == 2).sum()),
        "three_plus":              int((email_orders >= 3).sum()),
        "five_plus":               int((email_orders >= 5).sum()),
        "max_orders_one_customer": int(email_orders.max()),
    }
    insights.append({
        "category": "Customer retention",
        "title":    f"{report['customer_retention']['one_time']:,} one-time buyers are a re-engagement opportunity",
        "detail":   (
            f"Only {report['customer_retention']['three_plus']:,} customers have placed 3+ orders. "
            f"A targeted win-back campaign for lapsed single-purchase customers could significantly lift LTV."
        ),
    })

    disc      = orders[orders["Discount Amount"] > 0]
    disc_rate = round(len(disc) / len(orders) * 100, 1)
    avg_disc  = round(float(disc["Discount Amount"].mean()), 2) if len(disc) else 0
    report["discounts"] = {
        "orders_with_discount": len(disc),
        "discount_rate_pct":    disc_rate,
        "avg_discount_amount":  avg_disc,
        "top_codes": (
            orders[orders["Discount Amount"] > 0]["Discount Code"]
            .value_counts().head(10).to_dict()
        ),
    }
    if disc_rate > 35:
        insights.append({
            "category": "Discounts",
            "title":    f"{disc_rate}% of orders use a discount -- watch margin impact",
            "detail":   (
                f"Average discount is ${avg_disc:.0f}. Consider tiered loyalty rewards "
                f"or early-access perks as alternatives to blanket discounting."
            ),
        })

    states = (
        orders.groupby("Billing Province")
        .agg(order_count=("Name","count"), revenue=("Total","sum"))
        .sort_values("revenue", ascending=False)
    )
    report["top_states"] = {
        k: {"orders": int(v["order_count"]), "revenue": round(v["revenue"], 2)}
        for k, v in states.head(15).to_dict(orient="index").items()
    }
    top_states_list = list(report["top_states"].items())[:2]
    top2_state_rev  = sum(d["revenue"] for _, d in top_states_list)
    insights.append({
        "category": "Geography",
        "title":    f"{top_states_list[0][0]} & {top_states_list[1][0]} are your core markets",
        "detail":   (
            f"Combined ${top2_state_rev:,.0f} ({top2_state_rev/total_rev*100:.0f}% of revenue). "
            f"Prioritize geo-targeted paid ads in these states first."
        ),
    })

    opt_in = report["overview"]["marketing_opt_in_pct"]
    insights.append({
        "category": "Customer",
        "title":    f"{opt_in}% of customers accept marketing",
        "detail":   (
            f"A high opt-in rate means email and SMS are powerful owned channels. "
            f"Invest in segmented flows by purchase history and product category."
        ),
    })

    report["insights"] = insights

    # Geography extras
    cities = (
        orders.groupby(["Billing City","Billing Province"])
        .agg(order_count=("Name","count"), revenue=("Total","sum"))
        .sort_values("revenue", ascending=False)
    )
    report["top_cities"] = {
        f"{k[0]}, {k[1]}": {"orders": int(v["order_count"]), "revenue": round(v["revenue"], 2)}
        for k, v in cities.head(15).to_dict(orient="index").items()
    }
    countries = (
        orders.groupby("Billing Country")
        .agg(order_count=("Name","count"), revenue=("Total","sum"))
        .sort_values("revenue", ascending=False)
    )
    report["top_countries"] = {
        k: {"orders": int(v["order_count"]), "revenue": round(v["revenue"], 2)}
        for k, v in countries.head(10).to_dict(orient="index").items()
    }

    report["hour_of_day"] = {
        int(h): int(c) for h, c in orders.groupby("hour")["Name"].count().items()
    }
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    report["day_of_week"] = (
        orders.groupby("dow")["Name"].count()
        .reindex(dow_order, fill_value=0).to_dict()
    )

    df["size"] = df["Lineitem name"].str.extract(r"- (XXS|XS|S|M|L|XL|XXL|XXXL|XXX)\b")
    report["size_breakdown"] = {
        str(k): int(v)
        for k, v in df.groupby("size")["Lineitem quantity"].sum()
        .sort_values(ascending=False).items()
    }

    items = df.groupby("Name")["Lineitem quantity"].sum()
    report["items_per_order"] = {
        "mean":       round(float(items.mean()), 2),
        "median":     round(float(items.median()), 2),
        "one_item":   int((items == 1).sum()),
        "two_items":  int((items == 2).sum()),
        "three_plus": int((items >= 3).sum()),
    }

    return report


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 68, indent: str = "     ") -> str:
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return ("\n" + indent).join(lines)


def print_summary(report: dict):
    SEP = "=" * 64
    ov  = report["overview"]

    print(f"\n{SEP}")
    print("  SALES PATTERN ANALYSIS")
    print(SEP)
    print(f"  Revenue          ${ov['total_revenue']:>12,.2f}")
    print(f"  Orders           {ov['total_orders']:>12,}")
    print(f"  Unique customers {ov['unique_customers']:>12,}")
    print(f"  Avg order value  ${ov['avg_order_value']:>12,.2f}")
    print(f"  Marketing opt-in {ov['marketing_opt_in_pct']:>11.1f}%")

    print(f"\n{SEP}")
    print("  QUARTERLY BREAKDOWN")
    print(SEP)
    max_q_rev = max(d["revenue"] for d in report["quarterly"].values())
    for q, d in report["quarterly"].items():
        star = "  <-- PEAK" if d["revenue"] == max_q_rev else ""
        print(f"  Q{q}  ${d['revenue']:>12,.0f}   {d['order_count']:>5,} orders   AOV ${d['aov']:,.2f}{star}")

    print(f"\n{SEP}")
    print("  SEASONALITY INDEX  (1.00 = average month)")
    print(SEP)
    for m, idx in report["seasonality_index"].items():
        bar = ">" * int(idx * 12)
        print(f"  {MONTH_SHORT[m-1]:>3}  {idx:>5.2f}x  {bar}")

    print(f"\n{SEP}")
    print("  PRODUCT FAMILY PERFORMANCE BY QUARTER")
    print("  (* = peak quarter for that product)")
    print(SEP)
    print(f"  {'Product Family':<30} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10}")
    print(f"  {'-'*62}")
    for fam, quarters in report["product_by_quarter"].items():
        peak_q = max(range(1, 5), key=lambda q: quarters.get(q, 0))
        cells  = []
        for q in range(1, 5):
            v = quarters.get(q, 0)
            s = f"${v:,.0f}"
            cells.append(f"{'*'+s:>10}" if q == peak_q else f"{s:>10}")
        print(f"  {fam[:30]:<30} {cells[0]} {cells[1]} {cells[2]} {cells[3]}")

    print(f"\n{SEP}")
    print("  NEW VS. RETURNING CUSTOMERS BY MONTH")
    print(SEP)
    print(f"  {'Month':<10} {'New':>6} {'Return':>8} {'New%':>6}  {'New Rev':>10}  {'Return Rev':>11}")
    print(f"  {'-'*58}")
    nvr = report["new_vs_returning_by_month"]
    for m in range(1, 13):
        d = nvr[m]
        print(
            f"  {MONTH_SHORT[m-1]:<10} {d['new_customers']:>6,} {d['returning_customers']:>8,} "
            f"{d['new_pct']:>5.0f}%  ${d['new_revenue']:>9,.0f}  ${d['returning_revenue']:>10,.0f}"
        )

    t2 = report.get("time_to_second_order", {})
    if t2:
        print(f"\n{SEP}")
        print("  TIME TO SECOND ORDER")
        print(SEP)
        print(f"  Median days to 2nd order    {t2['median_days']:>6.0f} days")
        print(f"  Mean days to 2nd order      {t2['mean_days']:>6.0f} days")
        print(f"  25th percentile             {t2['p25_days']:>6.0f} days")
        print(f"  75th percentile             {t2['p75_days']:>6.0f} days")
        print(f"  --> Recommended trigger     {int(t2['recommended_winback_trigger_days']):>6} days post-purchase")

    wb = report.get("winback_buckets", {})
    if wb:
        print(f"\n{SEP}")
        print("  WIN-BACK OPPORTUNITY (one-time buyers by recency)")
        print(SEP)
        print(f"  0-30 days ago    {wb.get('0_30_days',0):>6,}  ({wb.get('0_30_days_pct',0):.0f}%)  -- very warm, nurture sequence")
        print(f"  31-90 days ago   {wb.get('31_90_days',0):>6,}  ({wb.get('31_90_days_pct',0):.0f}%)  -- warm, soft re-engage")
        print(f"  91-180 days ago  {wb.get('91_180_days',0):>6,}  ({wb.get('91_180_days_pct',0):.0f}%)  -- cooling, add small offer")
        print(f"  181-365 days ago {wb.get('181_365_days',0):>6,}  ({wb.get('181_365_days_pct',0):.0f}%)  -- cold, needs strong hook")

    pairs = report.get("product_affinity_pairs", [])
    if pairs:
        print(f"\n{SEP}")
        print("  PRODUCT AFFINITY -- TOP BUNDLE OPPORTUNITIES")
        print(SEP)
        for p in pairs[:10]:
            print(f"  {p['product_a'][:24]:<24} + {p['product_b'][:24]:<24}  {p['co_orders']:>5,} co-orders")

    f2s = report.get("first_to_second_product", {})
    if f2s:
        print(f"\n{SEP}")
        print("  FIRST -> SECOND PURCHASE JOURNEY")
        print(SEP)
        for first_prod, nexts in list(f2s.items())[:8]:
            top_next = nexts[0]
            print(f"  {first_prod[:30]:<30}  -->  {top_next['next_product'][:28]:<28} ({top_next['count']:,} customers)")

    dbon = report.get("discount_by_order_number", {})
    if dbon:
        print(f"\n{SEP}")
        print("  DISCOUNT DEPENDENCY BY ORDER NUMBER")
        print(SEP)
        print(f"  {'Order #':<12} {'Orders':>8} {'Discounted':>12} {'Disc%':>8} {'AOV':>10}")
        print(f"  {'-'*54}")
        for key, d in dbon.items():
            label = key.replace("order_","#").replace("_plus","+").replace("_"," ")
            print(f"  {label:<12} {d['total_orders']:>8,} {d['discounted']:>12,} {d['discount_pct']:>7.1f}% ${d['avg_order_value']:>9,.2f}")

    rc = report.get("revenue_concentration", {})
    if rc:
        print(f"\n{SEP}")
        print("  REVENUE CONCENTRATION")
        print(SEP)
        for key, d in rc.items():
            label = key.replace("_"," ").title()
            print(f"  {label:<22}  ${d['revenue']:>10,.0f}  ({d['revenue_pct']}% of total)")

    em = report.get("emerging_markets", [])
    if em:
        print(f"\n{SEP}")
        print("  EMERGING MARKETS (fastest-growing cities)")
        print(SEP)
        for e in em[:8]:
            print(f"  {e['city'][:28]:<28}  {e['first_half']:>4} -> {e['second_half']:>4} orders   +{e['growth_pct']:.0f}%")

    print(f"\n{SEP}")
    print("  MONTH-BY-MONTH MARKETING PRIORITY GUIDE")
    print(SEP)
    for entry in report["marketing_priority_guide"]:
        mom     = entry["mom_change_pct"]
        mom_str = f"+{mom:.0f}%" if mom >= 0 else f"{mom:.0f}%"
        arrow   = "^" if mom > 10 else ("v" if mom < -10 else "-")
        print(f"\n  {entry['month_name'].upper():<12}  [{entry['priority']:<6}]  ${entry['revenue']:>10,.0f}   {arrow} {mom_str} MoM")
        if entry["top_products"][:4]:
            print(f"    Top products : {', '.join(entry['top_products'][:4])}")
        if entry["surging_products"]:
            surge_str = ",  ".join(
                f"{s['product']} ({s['mom_change_pct']:+.0f}%)" for s in entry["surging_products"][:3]
            )
            print(f"    Surging      : {surge_str}")

    print(f"\n{SEP}")
    print("  STRATEGIC INSIGHTS")
    print(SEP)
    for i, ins in enumerate(report["insights"], 1):
        print(f"\n  {i}. [{ins['category'].upper()}]")
        print(f"     {ins['title']}")
        print(f"     {_wrap(ins['detail'])}")

    print(f"\n{SEP}")
    print("  TOP STATES BY REVENUE")
    print(SEP)
    for state, d in list(report["top_states"].items())[:10]:
        print(f"  {state:<6}  ${d['revenue']:>10,.0f}   {d['orders']:,} orders")

    print(f"\n{SEP}")
    print("  TOP CITIES BY REVENUE")
    print(SEP)
    for city, d in list(report["top_cities"].items())[:10]:
        print(f"  {city[:30]:<30}  ${d['revenue']:>10,.0f}")

    print(f"\n{SEP}")
    print("  CUSTOMER RETENTION")
    print(SEP)
    cr = report["customer_retention"]
    print(f"  1-time buyers    {cr['one_time']:>8,}")
    print(f"  2-order buyers   {cr['two_orders']:>8,}")
    print(f"  3+ order buyers  {cr['three_plus']:>8,}")
    print(f"  5+ order buyers  {cr['five_plus']:>8,}")

    disc = report["discounts"]
    print(f"\n{SEP}")
    print("  DISCOUNTS")
    print(SEP)
    print(f"  Orders with discount  {disc['orders_with_discount']:>8,}  ({disc['discount_rate_pct']}%)")
    print(f"  Avg discount amount   ${disc['avg_discount_amount']:>7,.2f}")
    print(f"  Top codes:")
    for code, cnt in list(disc["top_codes"].items())[:5]:
        print(f"    {str(code)[:30]:<30}  {cnt:,} uses")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sales Pattern Analysis -- Shopify orders")
    parser.add_argument("--folder", default=".", help="Folder containing CSV exports")
    parser.add_argument("--days",   type=int, default=365, help="Rolling window in days (default: 365)")
    parser.add_argument("--output", default="sales_pattern_report.json", help="Output JSON filename")
    args = parser.parse_args()

    df_raw = load_orders(args.folder)
    df     = prepare(df_raw, args.days)
    report = analyze(df)
    print_summary(report)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Full report saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
