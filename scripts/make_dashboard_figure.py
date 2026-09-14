# -*- coding: utf-8 -*-
"""生成 README 用的 Dashboard 配图（docs/assets/dashboard.png）。

忠实复刻 Streamlit Dashboard 的 Overview 页：顶部 KPI 卡片 + 质量分直方图
（按材料堆叠）+ 按材料类别的异常率条形图。数据来自真实全量结果。

风格：统一深墨绿暗色主题（scripts/figure_style.py）。
排版要点（v3）：
- KPI 改为 4 张卡片（paper 坐标 shape + annotation），大数字 + mono 标签
- 直方图堆叠（barmode=stack），绿色系明度渐变，图例右侧独立区域
- 条形图省略 x 刻度，异常率由条端数值标签直接表达
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    AMBER, BG, BORDER, GREEN, GREEN_RAMP, GREEN_SOFT, MONO, PANEL, SERIF,
    SANS, TEXT, TEXT_DIM, TEXT_FAINT, apply_theme, card, hex_to_rgba,
    theme_axes,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs" / "assets" / "dashboard.png"

q = pd.read_csv(PROJECT / "reports" / "quality_scores.csv")
a = pd.read_csv(PROJECT / "reports" / "anomaly_results.csv")
df = q.merge(a[["sequence_id", "iso_label"]], on="sequence_id", how="left")
df["iso_label"] = df["iso_label"].fillna(1)

n = len(df)
mean_q = float(df["quality_score"].mean())
n_anom = int((df["iso_label"] == -1).sum())
anom_rate = n_anom / n
n_mat = df["material"].nunique()

materials = sorted(df["material"].unique())
cmap = {m: GREEN_RAMP[i % len(GREEN_RAMP)] for i, m in enumerate(materials)}

fig = go.Figure()

# --------------------------------------------------------------------------- #
# 布局：margin=0，paper 坐标 = 整张画布（0–1），全部子区用手动 domain 排布。
#   标题 y 0.965–1.0 | KPI 卡片 y 0.80–0.935
#   直方图 y [0.40, 0.735] x [0.06, 0.70] + 图例右侧
#   条形图 y [0.055, 0.28] x [0.16, 0.985]（左侧留材质标签）
# --------------------------------------------------------------------------- #
apply_theme(
    fig, width=1280, height=960,
    margin=dict(l=0, r=0, t=0, b=0),
    eyebrow="TACTILE-QC · DASHBOARD OVERVIEW",
    title="RCT 全量 1,832 序列 · 质量总览",
    subtitle="五维指标（SNR / 漂移 / 饱和 / 力异常 / SPC）+ IsolationForest 异常标记 · 基于真实全量数据复刻",
)

# --- KPI 卡片区：4 张卡片 --- #
kpis = [
    (f"{n:,}", "总序列", TEXT),
    (f"{n_mat}", "材料类别", TEXT),
    (f"{mean_q:.1f}", "平均质量分", GREEN),
    (f"{n_anom}", f"异常序列 · {anom_rate:.1%}", AMBER),
]
CARD_Y0, CARD_Y1 = 0.775, 0.895
for i, (value, label, color) in enumerate(kpis):
    x0 = 0.022 + i * 0.2375
    x1 = x0 + 0.225
    card(fig, x0, x1, CARD_Y0, CARD_Y1)
    fig.add_annotation(
        xref="paper", yref="paper", x=(x0 + x1) / 2, y=CARD_Y1 - 0.045,
        showarrow=False, xanchor="center", yanchor="middle",
        text=f"<span style='font-family:{SERIF};font-size:27px;color:{color}'>"
             f"{value}</span>",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=(x0 + x1) / 2, y=CARD_Y0 + 0.028,
        showarrow=False, xanchor="center", yanchor="middle",
        text=f"<span style='font-family:{MONO};font-size:10.5px;"
             f"color:{TEXT_DIM}'>{label.upper()}</span>",
    )

# --- 直方图：按材料堆叠，绿色系明度渐变 --- #
for m in materials:
    sub = df[df["material"] == m]
    fig.add_trace(
        go.Histogram(
            x=sub["quality_score"], name=f"{m} (n={len(sub)})",
            marker=dict(color=cmap[m], line=dict(color=BG, width=0.6)),
            nbinsx=28, xaxis="x", yaxis="y",
            hovertemplate=f"{m}<br>质量分 %{{x}}<br>频数 %{{y}}<extra></extra>",
        ),
    )

fig.update_layout(
    barmode="stack", bargap=0.04,
    xaxis=dict(domain=[0.06, 0.70], anchor="y",
               tick0=20, dtick=10,
               title=dict(text="综合质量分（0–100）",
                          font=dict(family=SANS, size=13, color=TEXT_DIM))),
    yaxis=dict(domain=[0.40, 0.735], anchor="x",
               title=dict(text="序列数",
                          font=dict(family=SANS, size=13, color=TEXT_DIM))),
    legend=dict(
        orientation="v",
        xref="paper", yref="paper",
        xanchor="left", x=0.735,
        yanchor="top", y=0.735,
        font=dict(family=SANS, size=11.5, color=TEXT_DIM),
        bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
        title=dict(text="<span style='font-family:%s;font-size:10.5px;"
                        "color:%s'>材料</span>" % (MONO, TEXT_DIM)),
    ),
)
theme_axes(fig, tick_size=11.5)

# --- 条形图：按材料异常率（横向，升序） --- #
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
            orientation="h", xaxis="x2", yaxis="y2",
            marker=dict(color=GREEN),
            text=[f"{r['anomaly_ratio']:.1%}"], textposition="outside",
            textfont=dict(family=MONO, size=11, color=GREEN_SOFT),
            showlegend=False,
            hovertemplate=(
                f"{r['material']}<br>异常 {r['n_anomaly']}/{r['n']} "
                f"({r['anomaly_ratio']:.1%})<extra></extra>"
            ),
        ),
    )

fig.update_layout(
    xaxis2=dict(domain=[0.16, 0.985], anchor="y2", range=[0, 0.085],
                showticklabels=False, showgrid=False),
    yaxis2=dict(domain=[0.055, 0.28], anchor="x2",
                tickfont=dict(family=SANS, size=11.5, color=TEXT_DIM)),
)
fig.add_annotation(
    xref="paper", yref="paper", x=0.985, y=0.305,
    showarrow=False, xanchor="right", yanchor="middle",
    text=f"<span style='font-family:{MONO};font-size:10.5px;"
         f"color:{TEXT_FAINT}'>ISOLATIONFOREST 异常率 →</span>",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(OUT, scale=2)
print(f"saved: {OUT}")
