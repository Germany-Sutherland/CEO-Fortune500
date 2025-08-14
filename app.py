import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Future CEO App", layout="wide")

st.title("🤖 Future CEO Decision & Analysis Platform")

# AutoCEO Simulator
st.header("📈 AutoCEO KPI Simulator")
kpi_name = st.text_input("Enter KPI Name", "Revenue")
current_value = st.number_input("Current Value", value=100.0)
growth_rate = st.slider("Expected Growth Rate (%)", -50, 100, 10)
years = st.slider("Years to Project", 1, 10, 5)

if st.button("Run KPI Projection"):
    projections = [current_value * ((1 + growth_rate/100) ** i) for i in range(years+1)]
    df_proj = pd.DataFrame({"Year": range(years+1), kpi_name: projections})
    st.line_chart(df_proj.set_index("Year"))

# Leadership FMEA
st.header("⚠ Leadership FMEA Analysis")
fmea_data = []
for i in range(3):
    risk = st.text_input(f"Risk #{i+1} Description", key=f"risk{i}")
    severity = st.slider(f"Severity (1-10) for Risk #{i+1}", 1, 10, 5, key=f"sev{i}")
    occurrence = st.slider(f"Occurrence (1-10) for Risk #{i+1}", 1, 10, 5, key=f"occ{i}")
    detection = st.slider(f"Detection (1-10) for Risk #{i+1}", 1, 10, 5, key=f"det{i}")
    if risk:
        rpn = severity * occurrence * detection
        fmea_data.append((risk, severity, occurrence, detection, rpn))

if fmea_data:
    df_fmea = pd.DataFrame(fmea_data, columns=["Risk", "Severity", "Occurrence", "Detection", "RPN"])
    st.dataframe(df_fmea)
    st.write("Highest RPN indicates most critical risk.")

# Ledger
st.header("💰 Revenue & Expense Ledger")
ledger_entries = []
for i in range(3):
    desc = st.text_input(f"Entry #{i+1} Description", key=f"desc{i}")
    amount = st.number_input(f"Amount for Entry #{i+1}", value=0.0, key=f"amt{i}")
    entry_type = st.selectbox(f"Type for Entry #{i+1}", ["Revenue", "Expense"], key=f"type{i}")
    if desc:
        ledger_entries.append((desc, amount if entry_type=="Revenue" else -amount))

if ledger_entries:
    df_ledger = pd.DataFrame(ledger_entries, columns=["Description", "Amount"])
    st.dataframe(df_ledger)
    st.metric("Net Total", f"{df_ledger['Amount'].sum():.2f}")

# 5W+1H Analysis
st.header("🔍 5W+1H Usefulness Analysis")
who = st.text_area("Who", "Target audience or stakeholders")
what = st.text_area("What", "Purpose or objective")
when = st.text_area("When", "Time of application")
where = st.text_area("Where", "Applicable location or domain")
why = st.text_area("Why", "Reason or benefit")
how = st.text_area("How", "Method or process")

if st.button("Generate 5W+1H Report"):
    report_df = pd.DataFrame({
        "Aspect": ["Who", "What", "When", "Where", "Why", "How"],
        "Detail": [who, what, when, where, why, how]
    })
    st.table(report_df)
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "5w1h_report.csv", "text/csv")
    txt = report_df.to_string(index=False)
    st.download_button("Download TXT", txt, "5w1h_report.txt")
