# -*- coding: utf-8 -*-
"""生成 README 用的 Dashboard 配图（docs/assets/dashboard.png）。

忠实复刻 Streamlit Dashboard 的 Overview 页：顶部 KPI 条 + 质量分直方图
（按材料着色）+ 按材料类别的异常率条形图。数据来自真实全量结果。

排版要点（v2）：
- KPI 条改用 shape + annotation 实现，避免 go.Table 强制折行
- 图例改到右侧外部（orientation='v' + margin.r=200），不再压直方图
- 直方图 x 轴标题下方留出充足间距
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs" / "assets" / "dashboard.png"
FONT = "Microsoft YaHei, Segoe UI, Arial"

q = pd.read_csv(PROJECT / "reports" / "quality_scores.csv")
a = pd.read_csv(PROJECT / "reports" / "anomaly_results.csv")
df = q.merge(a[["sequence_id", "iso_label"]], on="sequence_id", how="left")
df["iso_label"] = df["iso_label"].fillna(1)

n = len(df)
mean_q = float(df["quality_score"].mean())
n_anom = int((df["iso_label"] == -1).sum())
anom_rate = n_anom / n
n_mat = df["material"].nunique()

palette = px.colors.qualitative.Safe
cmap = {m: palette[i % len(palette)] for i, m in enumerate(sorted(df["material"].unique()))}

# --------------------------------------------------------------------------- #
# 3 行布局：第1行 domain 占位放 KPI 横幅（用 shape + annotation 渲染）；
# 第2行直方图；第3行按材料异常率条形图。
# --------------------------------------------------------------------------- #
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.09, 0.46, 0.45],
    vertical_spacing=0.07,
    specs=[[{"type": "domain"}], [{"type": "xy"}], [{"type": "xy"}]],
)

# --- KPI 横幅：用 rect + annotation，避免 go.Table 强制折行 --- #
fig.add_shape(
    type="rect", xref="paper", yref="paper",
    x0=0.0, x1=1.0, y0=0.915, y1=0.995,
    fillcolor="rgba(245,247,250,1)",
    line=dict(color="rgba(220,225,232,1)", width=1),
    layer="below",
)
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=0.955, showarrow=False,
    xanchor="center", yanchor="middle",
    text=(
        f"<b style='color:#4C78A8;font-size:22px'>{n:,}</b> "
        f"<span style='color:#555;font-size:13px'>总序列</span> &nbsp;|&nbsp; "
        f"<b style='color:#54A24B;font-size:22px'>{n_mat}</b> "
        f"<span style='color:#555;font-size:13px'>材料类别</span> &nbsp;|&nbsp; "
        f"<b style='color:#F58518;font-size:22px'>{mean_q:.1f}</b> "
        f"<span style='color:#555;font-size:13px'>平均质量分</span> &nbsp;|&nbsp; "
        f"<b style='color:#E45756;font-size:22px'>{n_anom}</b> "
        f"<span style='color:#555;font-size:13px'>异常序列</span> "
        f"<span style='color:#E45756;font-size:13px'>({anom_rate:.1%})</span>"
    ),
    font=dict(family=FONT, size=13, color="#333"),
)

# --- (row2) 质量分直方图（按材料着色，overlay）--- #
for m in sorted(df["material"].unique()):
    sub = df[df["material"] == m]
    fig.add_trace(
        go.Histogram(
            x=sub["quality_score"], name=f"{m} (n={len(sub)})",
            marker_color=cmap[m], nbinsx=28, opacity=0.78,
            hovertemplate=f"{m}<br>质量分 %{{x}}<br>频数 %{{y}}<extra></extra>",
        ),
        row=2, col=1,
    )
fig.update_xaxes(title_text="综合质量分（0–100）", row=2, col=1)
fig.update_yaxes(title_text="序列数", row=2, col=1)
fig.update_layout(barmode="overlay", bargap=0.03)

# --- (row3) 按材料异常率横向条形图 --- #
g = (
    df.assign(anomaly=df["iso_label"] == -1)
    .groupby("material")["anomaly"]
    .agg(n="size", n_anomaly="sum")
    .reset_index()
)
g["anomaly_ratio"] = g["n_anomaly"] / g["n"]
g = g.sort_values("anomaly_ratio", ascending=True)
for _, r in g.iterrows():
    fig.add_trace(
        go.Bar(
            x=[r["anomaly_ratio"]], y=[f"{r['material']} (n={r['n']})"],
            orientation="h", marker_color=cmap[r["material"]],
            text=[f"{r['anomaly_ratio']:.1%}"], textposition="outside",
            textfont=dict(size=11, color="#444"),
            showlegend=False,
            hovertemplate=(
                f"{r['material']}<br>异常 {r['n_anomaly']}/{r['n']} "
                f"({r['anomaly_ratio']:.1%})<extra></extra>"
            ),
        ),
        row=3, col=1,
    )
fig.update_xaxes(
    title_text="异常率（IsolationForest）",
    tickformat=".0%", range=[0, 0.085],
    row=3, col=1,
)
fig.update_yaxes(tickfont=dict(size=11), row=3, col=1)

fig.update_layout(
    width=1280, height=820,
    template="plotly_white",
    font=dict(family=FONT, size=13, color="#333"),
    margin=dict(l=60, r=200, t=64, b=50),
    legend=dict(
        # 移到右侧外部，竖直排布；不再遮挡直方图
        orientation="v",
        yanchor="middle", y=0.72,
        xanchor="left", x=1.02,
        font=dict(size=10, color="#444"),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="rgba(220,225,232,1)",
        borderwidth=1,
    ),
    title=dict(
        text="tactile-qc Dashboard · Overview（RCT 全量 1,832 序列）",
        font=dict(size=17), x=0.01,
    ),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(OUT, scale=2)
print(f"saved: {OUT}")