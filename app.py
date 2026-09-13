"""Streamlit dashboard for tactile-data-quality-inspector.

Four pages (sidebar navigation):
  1. Overview            — KPIs + quality-score histogram + per-material anomaly rate
  2. Sequence Explorer   — pick a sequence, see frame thumbnails + force curve with SPC
  3. Material Comparison — box + violin of quality scores across 2-4 materials + ANOVA
  4. Data Quality Report — embedded HTML report + CSV downloads

Data sources:
  - Precomputed per-sequence results: reports/quality_scores.csv, reports/anomaly_results.csv
  - Raw frames / force traces (Sequence Explorer): data/rct/rct_dataset/rct_dataset
  - Self-contained HTML report: reports/quality_report.html

Everything heavy is cached with @st.cache_data / @st.cache_resource.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sp_stats

from tactile_qc.io import load_rct_sequences
from tactile_qc.quality import _align_force_to_frames

PROJECT = Path(__file__).resolve().parent
REPORTS = PROJECT / "reports"
# raw dataset root (contains materials/); used only by Sequence Explorer
DATA_CANDIDATES = [
    PROJECT / "data" / "rct" / "rct_dataset" / "rct_dataset",
    PROJECT / "data" / "rct" / "rct_dataset",
    PROJECT / "data" / "rct",
]

PLOT_TEMPLATE = "plotly_white"


# --------------------------------------------------------------------------- #
# cached data loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="加载质量/异常结果 …")
def load_results() -> pd.DataFrame:
    """Merge quality + anomaly per-sequence CSVs into one DataFrame."""
    q = pd.read_csv(REPORTS / "quality_scores.csv")
    a_path = REPORTS / "anomaly_results.csv"
    if a_path.exists():
        a = pd.read_csv(a_path)
        merged = q.merge(
            a[["sequence_id", "iso_label", "iso_score", "mahal_label", "mahal_score"]],
            on="sequence_id",
            how="left",
        )
    else:
        for c in ("iso_label", "iso_score", "mahal_label", "mahal_score"):
            merged = q.assign(**{c: np.nan}) if c not in q else q
    return merged


@st.cache_resource(show_spinner="加载数据集（首次约 10s） …")
def load_dataset():
    """Load the RCT dataset (metadata + force traces; no frame decode)."""
    for c in DATA_CANDIDATES:
        if (c / "materials").is_dir() and (c / "material_categories.json").exists():
            try:
                return load_rct_sequences(c)
            except Exception:
                continue
    return None


@st.cache_resource
def _seq_index():
    ds = load_dataset()
    if ds is None:
        return {}
    return {s.sequence_id: s for s in ds}


def _color_map(materials) -> dict:
    """Stable material -> color map (cheap; not cached to avoid hashing issues)."""
    palette = px.colors.qualitative.Safe
    return {m: palette[i % len(palette)] for i, m in enumerate(sorted(materials))}


@st.cache_data(show_spinner=False)
def _html_report() -> str:
    p = REPORTS / "quality_report.html"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def get_sequence(seq_id: str):
    return _seq_index().get(seq_id)


# --------------------------------------------------------------------------- #
# page config + shell
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Tactile Data Quality Inspector",
    page_icon="🔍",
    layout="wide",
)

st.title("Tactile Data Quality Inspector")
st.caption("tactile-qc v0.1.0 — 统计质量评估 Dashboard")

results = load_results()
cmap = _color_map(list(results["material"].dropna().unique()))

PAGES = ["Overview", "Sequence Explorer", "Material Comparison", "Data Quality Report"]
page = st.sidebar.radio("页面", PAGES, horizontal=False)


# --------------------------------------------------------------------------- #
# page 1 — Overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    df = results
    n = len(df)
    mean_q = float(df["quality_score"].mean())
    n_anom = int((df["iso_label"] == -1).sum()) if "iso_label" in df else 0
    anom_rate = n_anom / n if n else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总序列数", f"{n:,}")
    c2.metric("平均质量分", f"{mean_q:.1f}")
    c3.metric("异常序列数", f"{n_anom}")
    c4.metric("异常率", f"{anom_rate:.1%}")

    st.markdown("### 质量评分分布（按材料着色）")
    fig = px.histogram(
        df, x="quality_score", color="material", color_discrete_map=cmap,
        nbins=30, template=PLOT_TEMPLATE, barmode="overlay", opacity=0.8,
        labels={"quality_score": "quality_score", "material": "材料类别"},
    )
    fig.update_layout(height=420, bargap=0.02)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 按材料类别的异常率")
    g = (
        df.assign(anomaly=df["iso_label"] == -1)
        .groupby("material")["anomaly"]
        .agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "n_anomaly"})
    )
    g["anomaly_ratio"] = g["n_anomaly"] / g["n"]
    g = g.sort_values("anomaly_ratio", ascending=True).reset_index()
    bar = px.bar(
        g, x="anomaly_ratio", y="material", orientation="h",
        color="material", color_discrete_map=cmap,
        template=PLOT_TEMPLATE, text_auto=".1%",
        labels={"anomaly_ratio": "异常率", "material": "材料类别"},
    )
    bar.update_layout(height=420, showlegend=False, yaxis_tickangle=0)
    st.plotly_chart(bar, width="stretch")


# --------------------------------------------------------------------------- #
# page 2 — Sequence Explorer
# --------------------------------------------------------------------------- #
def page_explorer() -> None:
    df = results
    if _seq_index() == {}:
        st.warning(
            "未找到原始数据集（data/rct/rct_dataset/rct_dataset）。"
            "Sequence Explorer 需要原始触觉帧与力轨迹，请先解压数据。"
        )
        return

    materials = sorted(df["material"].dropna().unique())
    sel_mat = st.selectbox("材料类别", materials, key="ex_mat")
    seqs = df[df["material"] == sel_mat]["sequence_id"].tolist()
    sel_seq = st.selectbox("序列 ID", seqs, key="ex_seq")

    row = df[df["sequence_id"] == sel_seq].iloc[0]
    st.markdown(f"#### {sel_seq}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("quality_score", f"{row['quality_score']:.1f}")
    m2.metric("SNR (dB)", f"{row['snr']:.2f}")
    m3.metric("drift_slope", f"{row['drift_slope']:.4f}")
    m4.metric("saturation", f"{row['saturation_ratio']:.3f}")
    m5, m6 = st.columns(2)
    m5.metric("force_anomaly_ratio", f"{row['force_anomaly_ratio']:.3f}")
    m6.metric("spc_out_of_control", f"{row['spc_out_of_control_ratio']:.3f}")

    seq = get_sequence(sel_seq)

    st.markdown("##### 触觉帧缩略图")
    if seq is None:
        st.info("该序列的原始帧不可用。")
    else:
        frames = seq.load_frames()  # (T,H,W,3)
        ncol = 4
        for i in range(0, len(frames), ncol):
            cols = st.columns(ncol)
            for j, c in enumerate(cols):
                idx = i + j
                if idx < len(frames):
                    c.image(frames[idx], caption=f"#{idx}", width="stretch")

    st.markdown("##### 力曲线 + SPC 控制限")
    if seq is None or not seq.has_force:
        st.info("该序列无力数据。")
        return
    fmag = _align_force_to_frames(seq)
    if fmag is None or fmag.size == 0:
        st.info("无法对齐力数据。")
        return
    center = float(fmag.mean())
    spread = float(fmag.std(ddof=0))
    lo, hi = center - 3 * spread, center + 3 * spread
    ooc = (fmag < lo) | (fmag > hi)
    ooc_idx = np.where(ooc)[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=fmag, mode="lines+markers", name="||F_ext||",
            line=dict(color=cmap.get(sel_mat, "#4c78a8")),
        )
    )
    fig.add_hline(y=center, line_dash="dash", line_color="gray",
                  annotation_text=f"mean={center:.2f}", annotation_position="top left")
    fig.add_hline(y=hi, line_dash="dot", line_color="#dc2626",
                  annotation_text="+3σ", annotation_position="top left")
    fig.add_hline(y=lo, line_dash="dot", line_color="#dc2626",
                  annotation_text="-3σ", annotation_position="bottom left")
    if len(ooc_idx):
        fig.add_trace(
            go.Scatter(
                x=ooc_idx, y=fmag[ooc_idx], mode="markers",
                marker=dict(color="#dc2626", size=11, symbol="x"),
                name=f"out-of-control ({len(ooc_idx)})",
            )
        )
    fig.update_layout(
        template=PLOT_TEMPLATE, height=380,
        xaxis_title="帧序号", yaxis_title="||F_ext|| (N)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"失控帧：{int(ooc.sum())} / {len(fmag)}  ({ooc.mean():.1%})")


# --------------------------------------------------------------------------- #
# page 3 — Material Comparison
# --------------------------------------------------------------------------- #
def page_comparison() -> None:
    df = results
    materials = sorted(df["material"].dropna().unique())
    sel = st.multiselect("选择 2–4 种材料", materials, default=materials[:2], key="cmp_sel")
    if len(sel) < 2:
        st.info("请选择至少 2 种材料。")
        return
    if len(sel) > 4:
        st.warning("已超过 4 种，仅取前 4 种。")
        sel = sel[:4]

    sub = df[df["material"].isin(sel)]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 箱线图")
        bx = px.box(
            sub, x="material", y="quality_score", color="material",
            color_discrete_map=cmap, points="all", template=PLOT_TEMPLATE,
            labels={"quality_score": "quality_score", "material": ""},
        )
        bx.update_layout(height=400, showlegend=False)
        st.plotly_chart(bx, width="stretch")
    with col2:
        st.markdown("##### 小提琴图")
        vl = px.violin(
            sub, x="material", y="quality_score", color="material",
            color_discrete_map=cmap, box=True, points="all", template=PLOT_TEMPLATE,
            labels={"quality_score": "quality_score", "material": ""},
        )
        vl.update_layout(height=400, showlegend=False)
        st.plotly_chart(vl, width="stretch")

    groups = [sub.loc[sub["material"] == m, "quality_score"].dropna().values for m in sel]
    valid = [g for g in groups if len(g) >= 2]
    st.markdown("##### ANOVA 检验")
    if len(valid) >= 2:
        f_stat, p_val = sp_stats.f_oneway(*valid)
        c1, c2 = st.columns(2)
        c1.metric("F 统计量", f"{f_stat:.3f}")
        c2.metric("p 值", f"{p_val:.4g}")
        if p_val < 0.05:
            st.success(f"p = {p_val:.4g} < 0.05 → 所选材料的质量分分布**有**显著差异。")
        else:
            st.info(f"p = {p_val:.4g} ≥ 0.05 → 所选材料的质量分分布**无**显著差异。")
    else:
        st.info("所选材料中有效样本不足（每组需 ≥2），无法做 ANOVA。")

    st.markdown("##### 分组摘要")
    st.dataframe(
        sub.groupby("material")["quality_score"]
        .agg(["count", "mean", "std"])
        .round(2)
        .sort_values("mean", ascending=False),
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# page 4 — Data Quality Report
# --------------------------------------------------------------------------- #
def page_report() -> None:
    import streamlit.components.v1 as components

    st.markdown("##### HTML 报告嵌入预览")
    html = _html_report()
    if html:
        components.html(html, height=1100, scrolling=True)
    else:
        st.warning("未找到 reports/quality_report.html，请先运行 `tactile-qc run --format html`。")

    st.markdown("##### CSV 下载")
    dl = [
        ("quality_scores.csv", "质量评分"),
        ("anomaly_results.csv", "异常检测结果"),
        ("cleaned_data.csv", "清洗后数据清单"),
        ("problem_samples.csv", "问题样本清单"),
    ]
    cols = st.columns(len(dl))
    for c, (name, label) in zip(cols, dl):
        p = REPORTS / name
        with c:
            if p.exists():
                st.download_button(
                    f"⬇ {label}", data=p.read_bytes(), file_name=name, mime="text/csv",
                )
            else:
                st.button(f"⬇ {label}", disabled=True, help=f"{name} 不存在")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
if page == "Overview":
    page_overview()
elif page == "Sequence Explorer":
    page_explorer()
elif page == "Material Comparison":
    page_comparison()
else:
    page_report()
