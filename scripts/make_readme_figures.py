# -*- coding: utf-8 -*-
"""生成 README 用的质量分布图（docs/assets/quality_distribution.png）。

数据源：reports/quality_scores.csv（RCT 全量 1832 序列评分结果）。
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs" / "assets" / "quality_distribution.png"
FONT = "Microsoft YaHei, Segoe UI, Arial"

df = pd.read_csv(PROJECT / "reports" / "quality_scores.csv")

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.42, 0.58],
    horizontal_spacing=0.09,
    subplot_titles=(
        f"综合质量分分布（n={len(df)}）",
        "分材料类别质量分箱线图",
    ),
)

# (a) histogram
fig.add_trace(
    go.Histogram(
        x=df["quality_score"], nbinsx=40, marker_color="#4C78A8",
        hovertemplate="质量分 %{x}<br>频数 %{y}<extra></extra>",
    ),
    row=1, col=1,
)
med = float(df["quality_score"].median())
fig.add_vline(
    x=med, line_dash="dash", line_color="#E45756", line_width=2,
    annotation_text=f"中位数 {med:.1f}",
    annotation_font=dict(size=13, color="#E45756", family=FONT),
    row=1, col=1,
)

# (b) box plots by material (sorted by median)
order = (
    df.groupby("material")["quality_score"].median()
    .sort_values(ascending=False).index.tolist()
)
for _i, m in enumerate(order):
    sub = df[df["material"] == m]
    fig.add_trace(
        go.Box(
            y=sub["quality_score"], name=f"{m}<br>(n={len(sub)})",
            marker_color="#4C78A8", line_color="#2b5d8a",
            fillcolor="rgba(76,120,168,0.45)", boxmean=True,
            showlegend=False, hovertemplate="%{y:.1f}<extra>" + m + "</extra>",
        ),
        row=1, col=2,
    )

fig.update_xaxes(title_text="综合质量分（0–100）", row=1, col=1)
fig.update_xaxes(tickangle=35, tickfont=dict(size=10), row=1, col=2)
fig.update_yaxes(title_text="序列数", row=1, col=1)
fig.update_yaxes(title_text="综合质量分", range=[-2, 102], row=1, col=2)

fig.update_layout(
    width=1150, height=470,
    template="plotly_white",
    font=dict(family=FONT, size=13, color="#333"),
    margin=dict(l=60, r=30, t=70, b=90),
    title=dict(
        text="RCT 触觉数据集 — 每序列综合质量分分布（SNR / 漂移 / 饱和 / 力异常 / SPC 加权）",
        font=dict(size=16),
        x=0.02,
    ),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(OUT, scale=2)
print(f"saved: {OUT}")
