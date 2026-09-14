# -*- coding: utf-8 -*-
"""README 可视化统一主题（深墨绿暗色风格）。

设计规范（参照高级暗色产品页）：
- 背景双层：纸面深墨绿 BG + 面板绿 PANEL，卡片用半透明细边框
- 强调色：亮绿 GREEN 为主，琥珀 AMBER 做警示/对比（异常、中位线、负相关）
- 字体三档：标题衬线 SERIF（Georgia + 中文宋体）、正文无衬线 SANS、
  标签等宽 MONO（Consolas，大写 + 手动字距）
- 所有图表共用 apply_theme()：三行式标题（eyebrow / 主标题 / 副题）+ 外框

用法：
    from figure_style import *          # 或 sys.path 追加 scripts/ 后 import
    fig = go.Figure(...)
    apply_theme(fig, width=..., height=..., margin=...,
                eyebrow="TACTILE-QC · QUALITY DISTRIBUTION",
                title="综合质量分分布",
                subtitle="RCT 全量 1,832 序列")
    theme_axes(fig)                     # 统一网格/轴线/刻度颜色
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 调色板
# --------------------------------------------------------------------------- #
BG = "#0A1A12"                            # 纸面：深墨绿
PANEL = "#0F2418"                         # 面板：稍亮墨绿
PANEL_HI = "#143021"                      # 卡片高光面
GRID = "rgba(134, 239, 172, 0.07)"        # 网格：极淡绿
GRID_STRONG = "rgba(134, 239, 172, 0.16)"
BORDER = "rgba(167, 243, 208, 0.17)"      # 卡片细边框
BORDER_SOFT = "rgba(167, 243, 208, 0.08)"

GREEN = "#4ADE80"                         # 主强调
GREEN_SOFT = "#86EFAC"
GREEN_DIM = "#2FA85C"
GREEN_DEEP = "#177245"

AMBER = "#E8B44A"                         # 警示 / 对比
AMBER_SOFT = "#F3D28C"

TEXT = "#ECF5EF"                          # 主文字
TEXT_DIM = "#9DB8A8"                      # 次文字
TEXT_FAINT = "#5F7A6A"                    # 三级文字

# 材料分类：绿色系 8 档明度渐变（浅 → 深）
GREEN_RAMP = [
    "#D9F7E4", "#A9EFC5", "#7CE6A8", "#4ADE80",
    "#33C369", "#24A456", "#198545", "#116636",
]
# 琥珀单色渐变（用于需要"警示浓度"编码的场合）
AMBER_RAMP = ["#F3D28C", "#EDC264", "#E8B44A", "#D99B2E", "#C2821C"]

# --------------------------------------------------------------------------- #
# 字体
# --------------------------------------------------------------------------- #
SERIF = "Georgia, STZhongsong, SimSun, serif"          # 标题
SANS = "Segoe UI, Microsoft YaHei, sans-serif"         # 正文
MONO = "Consolas, Microsoft YaHei, monospace"          # 标签 / 刻度


def spaced(text: str) -> str:
    """给 eyebrow 文本加手动字距（plotly 不支持 letter-spacing）。"""
    return " ".join(list(text.replace(" ", "  ")))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' → 'rgba(r,g,b,alpha)'（plotly 填充半透明用）。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
# 标题
# --------------------------------------------------------------------------- #
def title_html(eyebrow: str, title: str, subtitle: str = "") -> str:
    """三行式标题：mono eyebrow（亮绿）→ 衬线主标题（近白）→ 副题（灰绿）。"""
    parts = [
        f"<span style='color:{GREEN};font-family:{MONO};font-size:11.5px;'>"
        f"◆ {spaced(eyebrow)}</span>",
        f"<span style='color:{TEXT};font-family:{SERIF};font-size:23px;'>"
        f"{title}</span>",
    ]
    if subtitle:
        parts.append(
            f"<span style='color:{TEXT_DIM};font-family:{SANS};font-size:12.5px;'>"
            f"{subtitle}</span>"
        )
    return "<br>".join(parts)


# --------------------------------------------------------------------------- #
# 主题应用
# --------------------------------------------------------------------------- #
def apply_theme(
    fig,
    *,
    width: int,
    height: int,
    margin: dict,
    eyebrow: str,
    title: str,
    subtitle: str = "",
    frame: bool = True,
):
    """统一纸面/字体/三行标题/外框。调用后再用 theme_axes 微调坐标轴。"""
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        template="none",
        font=dict(family=SANS, size=13, color=TEXT_DIM),
        title=dict(
            text=title_html(eyebrow, title, subtitle),
            x=0.018,
            y=0.982,
            xanchor="left",
            yanchor="top",
            pad=dict(t=6, b=0),
        ),
        margin=margin,
    )
    if frame:
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, x1=1, y0=0, y1=1,
            line=dict(color=BORDER, width=1),
            layer="above",
        )


def theme_axes(fig, tick_size: float = 11.5, **kwargs):
    """统一坐标轴：淡绿网格、无零线、轴线用细边框色、mono 刻度。"""
    base = dict(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=BORDER,
        ticks="outside",
        tickcolor=BORDER,
        tickfont=dict(family=MONO, size=tick_size, color=TEXT_DIM),
        **kwargs,
    )
    fig.update_xaxes(**base)
    fig.update_yaxes(**base)


def card(fig, x0: float, x1: float, y0: float, y1: float,
         fill: str = PANEL, line: str = BORDER):
    """在 paper 坐标画一张卡片底板。"""
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=fill, line=dict(color=line, width=1),
        layer="below",
    )
