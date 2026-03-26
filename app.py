import streamlit as st
import pandas as pd
import json
from pathlib import Path
from sales_pattern_analysis import load_orders, prepare, analyze, print_summary  # import your existing functions

st.set_page_config(page_title="Sales Pattern Analysis", layout="wide")

st.title("Sales Pattern Analysis Dashboard")

# --- Step 1: Upload CSVs ---
uploaded_files = st.file_uploader(
    "Upload Shopify order CSVs", type="csv", accept_multiple_files=True
)

if uploaded_files:
    # Save uploaded files to a temporary folder
    temp_folder = Path("temp_uploads")
    temp_folder.mkdir(exist_ok=True)
    for f in uploaded_files:
        with open(temp_folder / f.name, "wb") as out:
            out.write(f.getbuffer())

    # Load and prepare data
    df = load_orders(temp_folder)
    df = prepare(df, days=180)  # default last 180 days

    # --- Step 2: Analyze ---
    st.subheader("Analysis Results")
    report = analyze(df)

    # Show overview metrics
    ov = report["overview"]
    st.metric("Total Revenue", f"${ov['total_revenue']:,}")
    st.metric("Total Orders", f"{ov['total_orders']:,}")
    st.metric("Unique Customers", f"{ov['unique_customers']:,}")
    st.metric("Avg Order Value", f"${ov['avg_order_value']:,}")
    st.metric("Marketing Opt-in %", f"{ov['marketing_opt_in_pct']}%")

    # Show insights
    st.subheader("Key Insights")
    for insight in report["insights"][:10]:  # show top 10 insights
        st.markdown(f"**{insight['title']}** — {insight['detail']}")

    # Optional: Download JSON report
    st.download_button(
        label="Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name="sales_report.json",
        mime="application/json"
    )
else:
    st.info("Please upload one or more Shopify CSV export files to start analysis.")
