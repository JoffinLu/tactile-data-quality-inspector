# -*- coding: utf-8 -*-
"""生成 README 补充可视化（3 张，存 docs/assets/）。

1. radar_materials.png    材料质量指纹雷达（2x4 small multiples，分位归一化）
2. anomaly_agreement.png  双检测器一致性散点（IsolationForest vs Mahalanobis）
3. correlation_heatmap.png 五维指标与综合分的相关热力图

数据源：reports/quality_scores.csv + reports/anomaly_results.csv（真实全量）。
风格：统一深墨绿暗色主题（scripts/figure_style.py）。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    AMBER, AMBER_RAMP, AMBER_SOFT, BG, BORDER, GRID, GRID_STRONG, GREEN,
    GREEN_SOFT, MONO, PANEL, SERIF, SANS, TEXT, TEXT_DIM, TEXT_FAINT,
    apply_theme, hex_to_rgba, theme_axes,
)

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "docs" / "assets"

q = pd.read_csv(PROJECT / "reports" / "quality_scores.csv")
a = pd.read_csv(PROJECT / "reports" / "anomaly_results.csv")

materials = sorted(q["material"].unique())

# --------------------------------------------------------------------------- #
# Figure 1 — 材料质量指纹雷达（2x4 small multiples）
# 每维取数据集内分位（0~1，1 = 该维最优），方向统一为"越大越好"
# --------------------------------------------------------------------------- #
DIMS = ["SNR", "低漂移", "低饱和", "力一致", "低越限"]
qn = pd.DataFrame(index=q.index)
qn["SNR"] = q["snr"].rank(pct=True)
qn["低漂移"] = 1 - q["drift_slope"].abs().rank(pct=True)
qn["低饱和"] = 1 - q["saturation_ratio"].rank(pct=True)
qn["力一致"] = 1 - q["force_anomaly_ratio"].rank(pct=True)
qn["低越限"] = 1 - q["spc_out_of_control_ratio"].rank(pct=True)
qn["material"] = q["material"]

mat_mean = qn.groupby("material")[DIMS].mean()
counts = q["material"].value_counts()

fig1 = make_subplots(
    rows=2, cols=4,
    specs=[[{"type": "polar"}] * 4, [{"type": "polar"}] * 4],
    subplot_titles=[f"{m}（n={counts[m]}）" for m in materials],
    horizontal_spacing=0.10,
    vertical_spacing=0.20,
)

for i, m in enumerate(materials):
    row, col = i // 4 + 1, i % 4 + 1
    vals = mat_mean.loc[m, DIMS].tolist()
    fig1.add_trace(
        go.Scatterpolar(
            r=vals + vals[:1], theta=DIMS + DIMS[:1],
            fill="toself",
            fillcolor=hex_to_rgba(GREEN, 0.16),
            line=dict(color=GREEN, width=2),
            marker=dict(size=4, color=GREEN_SOFT),
            showlegend=False,
            name=m,
            hovertemplate="%{theta}: %{r:.2f}<extra>" + m + "</extra>",
        ),
        row=row, col=col,
    )

polar_cfg: dict = {}
for i in range(1, 9):
    key = "polar" if i == 1 else f"polar{i}"
    polar_cfg[key] = dict(
        angularaxis=dict(
            categoryorder="array", categoryarray=DIMS,
            tickfont=dict(family=SANS, size=9.5, color=TEXT_DIM),
            rotation=90, direction="clockwise",
        ),
        radialaxis=dict(
            # 各材料分位均值集中在 0.5 附近；range 压缩到 [0.30, 0.80]
            # 放大材料间差异（分位语义不变：外侧 = 更优）
            range=[0.30, 0.80],
            tickvals=[0.4, 0.5, 0.6, 0.7],
            tickfont=dict(family=MONO, size=7.5, color=TEXT_FAINT),
            gridcolor=GRID,
            linecolor=BORDER,
        ),
        bgcolor=PANEL,
    )
fig1.update_layout(**polar_cfg)
for ann in fig1["layout"]["annotations"]:
    ann["font"] = dict(family=SERIF, size=13.5, color=TEXT)
apply_theme(
    fig1, width=1280, height=780,
    margin=dict(l=75, r=75, t=140, b=30),
    eyebrow="TACTILE-QC · MATERIAL FINGERPRINT",
    title="材料质量指纹",
    subtitle="五维指标分位雷达 · 分位 1.0 = 全数据集该维最优 · 径向范围 0.30–0.80",
)

# --------------------------------------------------------------------------- #
# Figure 2 — 双检测器一致性散点
# x: iso_score（越小越异常，阈值 0）；y: log10(-mahal_score)（越大越异常）
# --------------------------------------------------------------------------- #
a["mahal_log"] = np.log10(-a["mahal_score"])
mah_thr = float(np.log10(-a.loc[a["mahal_label"] == -1, "mahal_score"].max()))

mask_normal = (a["iso_label"] == 1) & (a["mahal_label"] == 1)
mask_both = (a["iso_label"] == -1) & (a["mahal_label"] == -1)
mask_mahal_only = (a["iso_label"] == 1) & (a["mahal_label"] == -1)
mask_iso_only = (a["iso_label"] == -1) & (a["mahal_label"] == 1)

groups = [
    ("正常（双检测器一致）", mask_normal, TEXT_FAINT, 5, 0.35),
    ("仅 Mahalanobis 标记", mask_mahal_only, AMBER_SOFT, 9, 0.75),
    ("仅 IsolationForest 标记", mask_iso_only, AMBER_RAMP[0], 9, 0.85),
    ("双检测器共同标记", mask_both, AMBER, 9, 0.95),
]

fig2 = go.Figure()
for label, mask, color, size, opac in groups:
    if int(mask.sum()) == 0:
        continue
    sub = a[mask]
    fig2.add_trace(
        go.Scatter(
            x=sub["iso_score"], y=sub["mahal_log"], mode="markers",
            name=f"{label}（{int(mask.sum())}）",
            marker=dict(color=color, size=size, opacity=opac,
                        line=dict(width=0.5, color=BG)),
            hovertemplate=(
                "ISO 分 %{x:.4f} | Mahal log₁₀ %{y:.2f}"
                f"<extra>{label}</extra>"
            ),
        )
    )

fig2.add_vline(
    x=0.0, line_dash="dash", line_color=GRID_STRONG, line_width=1.2,
    annotation_text="IF 阈值",
    annotation_font=dict(family=MONO, size=10, color=TEXT_DIM),
    annotation_position="top",
)
fig2.add_hline(
    y=mah_thr, line_dash="dash", line_color=GRID_STRONG, line_width=1.2,
    annotation_text="Mahalanobis 阈值",
    annotation_font=dict(family=MONO, size=10, color=TEXT_DIM),
    annotation_position="bottom right",
)

apply_theme(
    fig2, width=1280, height=700,
    margin=dict(l=70, r=40, t=130, b=60),
    eyebrow="TACTILE-QC · DETECTOR AGREEMENT",
    title="双检测器一致性",
    subtitle="IsolationForest ⊂ Mahalanobis · κ = 0.615 · 92 / 1,832 异常",
)
theme_axes(fig2)
fig2.update_xaxes(
    title=dict(text="IsolationForest 得分（越小越异常）",
               font=dict(family=SANS, size=13, color=TEXT_DIM)),
    range=[-0.26, 0.19],
)
fig2.update_yaxes(
    title=dict(text="Mahalanobis 距离（log₁₀ 尺度）",
               font=dict(family=SANS, size=13, color=TEXT_DIM)),
)
fig2.update_layout(
    legend=dict(
        yanchor="bottom", y=0.02, xanchor="left", x=0.02,
        bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
        font=dict(family=SANS, size=12, color=TEXT_DIM),
    ),
)

# --------------------------------------------------------------------------- #
# Figure 3 — 五维指标与综合分相关热力图
# --------------------------------------------------------------------------- #
cols = ["snr", "drift_slope", "saturation_ratio",
        "force_anomaly_ratio", "spc_out_of_control_ratio", "quality_score"]
labels = ["SNR", "|漂移斜率|", "饱和率", "力异常率", "SPC 越限", "综合质量分"]
corr_src = q[cols].copy()
corr_src["drift_slope"] = corr_src["drift_slope"].abs()
corr = corr_src.corr()

# 自定义发散色标：琥珀（负相关/警示）→ 墨绿（0）→ 亮绿（正相关）
diverging = [[0.0, AMBER_RAMP[3]], [0.5, PANEL], [1.0, GREEN]]

fig3 = go.Figure(
    go.Heatmap(
        z=corr.values, x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=diverging,
        colorbar=dict(
            title=dict(text="Pearson r",
                       font=dict(family=MONO, size=11, color=TEXT_DIM)),
            thickness=14, len=0.85,
            tickfont=dict(family=MONO, size=10.5, color=TEXT_DIM),
            outlinecolor=BORDER, outlinewidth=1,
        ),
        xgap=4, ygap=4,
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    )
)
# 数值用 annotation 逐格叠加：亮色格（|r| 大）配深墨绿字，暗格配浅灰绿字；
# 对角线自相关恒为 1 且无信息量，跳过
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            continue
        zval = float(corr.values[i, j])
        fig3.add_annotation(
            x=labels[j], y=labels[i],
            text=f"{zval:.2f}", showarrow=False,
            font=dict(family=MONO, size=12,
                      color=BG if abs(zval) > 0.55 else TEXT_DIM),
        )
fig3.update_yaxes(autorange="reversed")
fig3.update_xaxes(tickfont=dict(family=SANS, size=11.5, color=TEXT_DIM))
fig3.update_yaxes(tickfont=dict(family=SANS, size=11.5, color=TEXT_DIM))
# 画布与其他 README 图等宽（1280）；热力图区域用 xaxis domain 收窄并居中，
# 保持格子接近正方形，避免全宽拉伸后格子变扁
fig3.update_layout(
    xaxis=dict(domain=[0.13, 0.69], anchor="y"),
)
apply_theme(
    fig3, width=1280, height=820,
    margin=dict(l=110, r=130, t=130, b=70),
    eyebrow="TACTILE-QC · CORRELATION STRUCTURE",
    title="指标相关结构",
    subtitle="Pearson r · n = 1,832 · SPC 越限是拖分主力（r = −0.92）",
)

# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
ASSETS.mkdir(parents=True, exist_ok=True)
for fig, name in [
    (fig1, "radar_materials.png"),
    (fig2, "anomaly_agreement.png"),
    (fig3, "correlation_heatmap.png"),
]:
    out = ASSETS / name
    fig.write_image(out, scale=2)
    print(f"saved: {out}")
