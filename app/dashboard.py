import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import config

st.set_page_config(page_title="Fraud Model Monitoring", layout="wide")

report_files = sorted(
    glob.glob(str(config.REPORTS_DIR / "monitor_*.json")),
    key=os.path.getmtime,
)

if not report_files:
    st.error("No monitoring reports found. Run `uv run python -m app.monitor` first.")
    st.stop()

with open(report_files[-1]) as f:
    report = json.load(f)

drift = report["drift"]
rolling = report["rolling_metrics"]
trigger = report["trigger"]

tab_drift, tab_metrics, tab_trigger = st.tabs(["Drift", "Rolling Metrics", "Retraining Triggers"])

with tab_drift:
    st.subheader("Feature drift (PSI)")
    psi_df = pd.DataFrame(
        {"feature": list(drift["feature_psis"].keys()), "psi": list(drift["feature_psis"].values())}
    ).sort_values("psi", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(psi_df))))
    colors = ["#d03b3b" if f in drift["drifted_features"] else "#2a78d6" for f in psi_df["feature"]]
    ax.barh(psi_df["feature"], psi_df["psi"], color=colors)
    ax.axvline(0.25, color="#898781", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.caption("Bars in red exceed the PSI > 0.25 drift threshold (dashed line).")
    st.dataframe(psi_df.sort_values("psi", ascending=False), use_container_width=True)

with tab_metrics:
    st.subheader("Rolling precision / recall")
    metrics_df = pd.DataFrame(rolling)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df.index, metrics_df["precision"], color="#2a78d6", linewidth=2, label="Precision")
    ax.plot(metrics_df.index, metrics_df["recall"], color="#1baf7a", linewidth=2, label="Recall")
    ax.set_xlabel("Window index")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.subheader("Rolling total cost")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(metrics_df.index, metrics_df["total_cost"], color="#2a78d6", linewidth=2)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Total cost ($)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.dataframe(metrics_df, use_container_width=True)

with tab_trigger:
    if trigger["triggered"]:
        st.error("🔴 Retraining triggered")
    else:
        st.success("🟢 No retraining trigger")

    if trigger["reasons"]:
        st.write("Reasons:")
        for reason in trigger["reasons"]:
            st.write(f"- {reason}")
