"""Streamlit dashboard for tactile-data-quality-inspector."""

import streamlit as st

st.set_page_config(
    page_title="Tactile Data Quality Inspector",
    page_icon="🔍",
    layout="wide",
)

st.title("Tactile Data Quality Inspector")
st.caption("tactile-qc v0.1.0 — statistical quality assessment for robotic tactile data")

st.info(
    "Dashboard 骨架已就绪。数据加载、质量指标与可视化将在后续阶段接入。",
    icon="🚧",
)
