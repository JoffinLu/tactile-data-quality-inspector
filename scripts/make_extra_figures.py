# -*- coding: utf-8 -*-
"""生成 README 补充可视化（3 张，存 docs/assets/）。

1. radar_materials.png    材料质量指纹雷达（2x4 small multiples，分位归一化）
2. anomaly_agreement.png  双检测器一致性散点（IsolationForest vs Mahalanobis）
3. correlation_heatmap.png 五维指标与综合分的相关热力图

数据源：reports/quality_scores.csv + reports/anomaly_results.csv（真实全量）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "docs" / "assets"
FONT = "Microsoft YaHei, Segoe UI, Arial"

q = pd.read_csv(PROJECT / "reports" / "quality_scores.csv")
a = pd.read_csv(PROJECT / "reports" / "anomaly_results.csv")

palette = px.colors.qualitative.Safe
materials = sorted(q["material"].unique())
cmap = {m: palette[i % len(palette)] for i, m in enumerate(materials)}


def _rgb_tuple(color: str) -> tuple[int, int, int]:
    """把 plotly 颜色（'#RRGGBB' 或 'rgb(r,g,b)'）解析为 (r, g, b)。"""
    if color.startswith("#"):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    body = color[color.index("(") + 1: color.index(")")]
    parts = [int(p) for p in body.split(",")[:3]]
    return (parts[0], parts[1], parts[2])


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
    r_, g_, b_ = _rgb_tuple(cmap[m])
    fig1.add_trace(
        go.Scatterpolar(
            r=vals + vals[:1], theta=DIMS + DIMS[:1],
            fill="toself",
            fillcolor=f"rgba({r_},{g_},{b_},0.28)",
            line=dict(color=cmap[m], width=2),
            marker=dict(size=4, color=cmap[m]),
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
            tickfont=dict(size=9, color="#555"),
            rotation=90, direction="clockwise",
        ),
        radialaxis=dict(
            # 各材料分位均值集中在 0.5 附近；range 压缩到 [0.30, 0.80]
            # 放大材料间差异（分位语义不变：外侧 = 更优）
            range=[0.30, 0.80],
            tickvals=[0.4, 0.5, 0.6, 0.7],
            tickfont=dict(size=7, color="#999"),
            gridcolor="rgba(160,160,160,0.30)",
            linecolor="rgba(160,160,160,0.40)",
        ),
        bgcolor="rgba(248,249,251,0.6)",
    )
fig1.update_layout(**polar_cfg)
for ann in fig1["layout"]["annotations"]:
    ann["font"] = dict(size=12, color="#333", family=FONT)
fig1.update_layout(
    width=1280, height=740,
    template="plotly_white",
    font=dict(family=FONT, size=12, color="#333"),
    title=dict(
        text="材料质量指纹 — 五维指标分位雷达（分位 1.0 = 全数据集该维最优，径向范围 0.30–0.80）",
        font=dict(size=16), x=0.02,
    ),
    margin=dict(l=40, r=40, t=90, b=30),
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
    ("正常（双检测器一致）", mask_normal, "#B0BEC5", 5, 0.40),
    ("仅 Mahalanobis 标记", mask_mahal_only, "#E67E22", 9, 0.85),
    ("仅 IsolationForest 标记", mask_iso_only, "#F4D03F", 9, 0.90),
    ("双检测器共同标记", mask_both, "#C0392B", 9, 0.95),
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
                        line=dict(width=0.5, color="white")),
            hovertemplate=(
                "ISO 分 %{x:.4f} | Mahal log₁₀ %{y:.2f}"
                f"<extra>{label}</extra>"
            ),
        )
    )

fig2.add_vline(
    x=0.0, line_dash="dash", line_color="#888", line_width=1.2,
    annotation_text="IF 阈值", annotation_font=dict(size=10, color="#888"),
    annotation_position="top",
)
fig2.add_hline(
    y=mah_thr, line_dash="dash", line_color="#888", line_width=1.2,
    annotation_text="Mahalanobis 阈值", annotation_font=dict(size=10, color="#888"),
    annotation_position="bottom right",
)

fig2.update_layout(
    width=1000, height=620,
    template="plotly_white",
    font=dict(family=FONT, size=13, color="#333"),
    title=dict(
        text="双检测器一致性 — IsolationForest ⊂ Mahalanobis（κ = 0.615，92/1832 异常）",
        font=dict(size=16), x=0.02,
    ),
    xaxis=dict(title="IsolationForest 得分（越小越异常）", range=[-0.26, 0.19]),
    yaxis=dict(title="Mahalanobis 距离（log₁₀ 尺度）"),
    legend=dict(
        yanchor="bottom", y=0.02, xanchor="left", x=0.02,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(200,200,200,0.6)", borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=70, r=40, t=80, b=60),
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

fig3 = go.Figure(
    go.Heatmap(
        z=corr.values, x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Pearson r", thickness=15, len=0.85),
        xgap=2, ygap=2,
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    )
)
# 数值用 annotation 逐格叠加：深色格（|r|>0.6）自动换白字，浅色格用深灰字；
# 对角线自相关恒为 1 且无信息量，跳过
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            continue
        zval = float(corr.values[i, j])
        fig3.add_annotation(
            x=labels[j], y=labels[i],
            text=f"{zval:.2f}", showarrow=False,
            font=dict(size=12, color="#ffffff" if abs(zval) > 0.6 else "#222"),
        )
fig3.update_yaxes(autorange="reversed")
fig3.update_layout(
    width=820, height=680,
    template="plotly_white",
    font=dict(family=FONT, size=12, color="#333"),
    title=dict(
        text="五维指标与综合分的相关结构（Pearson r，n=1832）",
        font=dict(size=16), x=0.02,
    ),
    margin=dict(l=90, r=60, t=80, b=60),
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