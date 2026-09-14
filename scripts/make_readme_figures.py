# -*- coding: utf-8 -*-
"""生成 README 用的质量分布图（docs/assets/quality_distribution.png）。

数据源：reports/quality_scores.csv（RCT 全量 1832 序列评分结果）。
风格：统一深墨绿暗色主题（scripts/figure_style.py）。
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    AMBER, GREEN, GREEN_RAMP, MONO, SERIF, SANS,
    TEXT, TEXT_DIM, apply_theme, hex_to_rgba, theme_axes,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs" / "assets" / "quality_distribution.png"

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
        x=df["quality_score"], nbinsx=40,
        marker=dict(color=hex_to_rgba(GREEN, 0.55),
                    line=dict(color=GREEN, width=1)),
        hovertemplate="质量分 %{x}<br>频数 %{y}<extra></extra>",
    ),
    row=1, col=1,
)
med = float(df["quality_score"].median())
fig.add_vline(
    x=med, line_dash="dash", line_color=AMBER, line_width=2,
    annotation_text=f"中位数 {med:.1f}",
    annotation_font=dict(size=13, color=AMBER, family=MONO),
    row=1, col=1,
)

# (b) box plots by material (sorted by median, 浅→深绿)
order = (
    df.groupby("material")["quality_score"].median()
    .sort_values(ascending=False).index.tolist()
)
for i, m in enumerate(order):
    sub = df[df["material"] == m]
    c = GREEN_RAMP[i % len(GREEN_RAMP)]
    fig.add_trace(
        go.Box(
            y=sub["quality_score"], name=f"{m} (n={len(sub)})",
            marker_color=c, line_color=c,
            fillcolor=hex_to_rgba(c, 0.32), boxmean=True,
            showlegend=False, hovertemplate="%{y:.1f}<extra>" + m + "</extra>",
        ),
        row=1, col=2,
    )

fig.update_xaxes(title_text="综合质量分（0–100）", row=1, col=1)
# 右图 x 轴必须显式设标题，否则 plotly 会继承左图的"综合质量分（0–100）"
fig.update_xaxes(
    title_text="材料类别", tickangle=30,
    tickfont=dict(family=MONO, size=9.5, color=TEXT_DIM),
    row=1, col=2,
)
fig.update_yaxes(title_text="序列数", row=1, col=1)
fig.update_yaxes(title_text="综合质量分", range=[-2, 102], row=1, col=2)

apply_theme(
    fig, width=1180, height=580,
    margin=dict(l=70, r=46, t=150, b=110),
    eyebrow="TACTILE-QC · QUALITY DISTRIBUTION",
    title="RCT 触觉数据集 · 每序列综合质量分分布",
    subtitle="SNR / 漂移 / 饱和 / 力异常 / SPC 加权评分 · 全量 1,832 序列",
)
theme_axes(fig)
# 全局关掉 legend；否则 plotly 会给无名箱线图显示 "trace 0"
fig.update_layout(showlegend=False)

# subplot 标题统一为衬线浅色
for ann in fig["layout"]["annotations"]:
    ann["font"] = dict(family=SERIF, size=15, color=TEXT)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(OUT, scale=2)
print(f"saved: {OUT}")
