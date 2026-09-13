"""Quality-report generation: self-contained HTML + CSV exports.

The reports consume the per-sequence DataFrames produced by
:func:`tactile_qc.quality.compute_quality_score` and
:func:`tactile_qc.anomaly.detect_anomalies` (no dataset reload needed), so
they are fast and decoupled from the heavy data-loading path.

Public API
----------
- :func:`generate_html_report`  -> single self-contained ``.html`` file
- :func:`generate_csv_export`  -> ``cleaned_data.csv`` + ``problem_samples.csv``
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.metrics import cohen_kappa_score

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 1080px; margin: 0 auto; padding: 32px 24px; color: #1f2328;
       background: #ffffff; line-height: 1.55; }
h1 { font-size: 26px; margin: 0 0 4px 0; }
h2 { font-size: 19px; margin: 28px 0 10px 0; border-bottom: 1px solid #e6e8eb; padding-bottom: 6px; }
.meta { color: #6b7280; font-size: 13px; margin-bottom: 8px; }
.stat { display: inline-block; background: #f3f4f6; border-radius: 8px; padding: 8px 14px;
        margin: 4px 8px 4px 0; font-size: 14px; }
.stat b { font-size: 18px; display: block; color: #2f6fed; }
ul.findings { padding-left: 20px; }
ul.findings li { margin-bottom: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }
th, td { border: 1px solid #e6e8eb; padding: 6px 10px; text-align: left; }
th { background: #f3f4f6; }
tr.anom { background: #fff5f5; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; }
.badge-anom { background: #fee2e2; color: #b91c1c; }
.badge-ok { background: #dcfce7; color: #166534; }
footer { color: #9ca3af; font-size: 12px; margin-top: 32px; border-top: 1px solid #e6e8eb; padding-top: 10px; }
"""


def _key_findings(quality_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> list[str]:
    """Derive 3-5 text findings directly from the per-sequence frames."""
    findings: list[str] = []
    q = quality_df["quality_score"]
    n_cat = quality_df["material"].nunique()
    findings.append(
        f"共评估 <b>{len(quality_df)}</b> 个序列、覆盖 <b>{n_cat}</b> 个材料类别；"
        f"质量评分均值 <b>{q.mean():.1f}</b>（标准差 {q.std():.1f}），"
        f"范围 [{q.min():.1f}, {q.max():.1f}]。"
    )
    low = int((q < 30).sum())
    high = int((q > 80).sum())
    findings.append(
        f"低分序列（&lt;30）{low} 个（{100 * low / len(q):.1f}%）、"
        f"高分序列（&gt;80）{high} 个（{100 * high / len(q):.1f}%）。"
    )
    cat = quality_df.groupby("material")["quality_score"].mean().sort_values()
    findings.append(
        f"质量评分最低的材料类别：<b>{cat.index[0]}</b>（均值 {cat.iloc[0]:.1f}）；"
        f"最高：<b>{cat.index[-1]}</b>（均值 {cat.iloc[-1]:.1f}）。"
    )
    iso_n = int((anomaly_df["iso_label"] == -1).sum())
    mah_n = int((anomaly_df["mahal_label"] == -1).sum())
    findings.append(
        f"IsolationForest 标记异常 <b>{iso_n}</b> 个"
        f"（{100 * iso_n / len(anomaly_df):.1f}%）；"
        f"Mahalanobis 标记 <b>{mah_n}</b> 个。"
    )
    if {"iso_label", "mahal_label"}.issubset(anomaly_df.columns):
        kappa = float(cohen_kappa_score(anomaly_df["iso_label"], anomaly_df["mahal_label"]))
        agree = float((anomaly_df["iso_label"] == anomaly_df["mahal_label"]).mean())
        findings.append(
            f"两方法一致性：Cohen&#39;s κ = <b>{kappa:.3f}</b>，整体一致率 {agree:.1%}。"
        )
    return findings


def _overview_stats(
    quality_df: pd.DataFrame, dataset_stats: Optional[Mapping[str, Any]] = None
) -> tuple[int, Optional[int], Optional[int]]:
    """Return (sequence_count, material_count, total_frames)."""
    n_seq = len(quality_df)
    n_mat = dataset_stats.get("material_count") if dataset_stats else None
    n_frames = dataset_stats.get("total_frames") if dataset_stats else None
    return n_seq, n_mat, n_frames


def generate_html_report(
    quality_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    output_path: str | Path,
    dataset_stats: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a self-contained HTML quality report.

    Parameters
    ----------
    quality_df : pandas.DataFrame
        Output of :func:`tactile_qc.quality.compute_quality_score`
        (must have ``material`` and ``quality_score``).
    anomaly_df : pandas.DataFrame
        Output of :func:`tactile_qc.anomaly.detect_anomalies` ``per_sequence``
        (``sequence_id, material, iso_label, iso_score, mahal_label, mahal_score``).
    output_path : str | Path
        Destination ``.html`` file.
    dataset_stats : Mapping, optional
        Optional richer overview (``material_count``, ``total_frames``) from the
        loaded dataset — the per-sequence frames alone only carry category
        labels, so material/frame totals are passed in by the caller.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_seq, n_mat, n_frames = _overview_stats(quality_df, dataset_stats)
    stats_html = (
        f'<span class="stat">序列数<b>{n_seq}</b></span>'
        f'<span class="stat">材料数<b>{n_mat if n_mat is not None else "—"}</b></span>'
        f'<span class="stat">总帧数<b>{n_frames if n_frames is not None else "—"}</b></span>'
        f'<span class="stat">材料类别数<b>{quality_df["material"].nunique()}</b></span>'
    )

    findings = _key_findings(quality_df, anomaly_df)
    findings_html = "<ul class='findings'>" + "".join(f"<li>{f}</li>" for f in findings) + "</ul>"

    # --- plot 1: quality-score histogram ---
    hist = go.Figure(
        go.Histogram(x=quality_df["quality_score"], nbinsx=30, marker_color="#4c78a8")
    )
    hist.update_layout(
        title="质量评分分布", xaxis_title="quality_score",
        yaxis_title="序列数", bargap=0.02, height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # --- plot 2: quality-score box by material ---
    box = go.Figure()
    cat_means = quality_df.groupby("material")["quality_score"].mean().sort_values()
    for mat in cat_means.index:
        box.add_trace(
            go.Box(
                y=quality_df.loc[quality_df["material"] == mat, "quality_score"],
                name=mat,
                boxmean=True,
            )
        )
    box.update_layout(
        title="按材料类别的质量评分", boxmode="group", height=380,
        xaxis_tickangle=-25, yaxis_title="quality_score",
        margin=dict(l=40, r=20, t=50, b=80),
        showlegend=False,
    )

    # --- top-20 anomalies by IsolationForest score (lower = more anomalous) ---
    top = anomaly_df.sort_values("iso_score", ascending=True).head(20).copy()
    rows_html = []
    for _, r in top.iterrows():
        iso_bad = r["iso_label"] == -1
        mah_bad = r["mahal_label"] == -1
        row_open = "<tr class='anom'>" if iso_bad else "<tr>"
        rows_html.append(
            row_open
            + f"<td>{r['sequence_id']}</td><td>{r.get('material', '')}</td>"
            + (
                f"<td><span class='badge badge-anom'>异常</span></td>"
                if iso_bad
                else "<td><span class='badge badge-ok'>正常</span></td>"
            )
            + f"<td>{r['iso_score']:.4f}</td>"
            + (
                f"<td><span class='badge badge-anom'>异常</span></td>"
                if mah_bad
                else "<td><span class='badge badge-ok'>正常</span></td>"
            )
            + f"<td>{r['mahal_score']:.2f}</td></tr>"
        )
    table_html = (
        "<table><thead><tr>"
        "<th>sequence_id</th><th>material</th><th>IF</th>"
        "<th>iso_score</th><th>Mahal</th><th>mahal_score</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )

    plotly_js = get_plotlyjs()
    hist_div = hist.to_html(full_html=False, include_plotlyjs=False)
    box_div = box.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RCT 触觉数据质量报告</title>
<style>{_CSS}</style>
<script>{plotly_js}</script>
</head>
<body>
<h1>RCT 触觉数据质量报告</h1>
<div class="meta">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · tactile-qc</div>

<h2>数据集概览</h2>
<div>{stats_html}</div>

<h2>关键统计发现</h2>
{findings_html}

<h2>质量评分分布</h2>
{hist_div}

<h2>按材料类别的质量评分</h2>
{box_div}

<h2>异常序列 Top-20（按 IsolationForest 分数升序）</h2>
{table_html}

<footer>tactile-data-quality-inspector · 自包含 HTML 报告，含内联 Plotly.js</footer>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def generate_csv_export(
    quality_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export a cleaned per-sequence manifest and a problem-sample list.

    Parameters
    ----------
    quality_df, anomaly_df : pandas.DataFrame
        Per-sequence frames from the quality and anomaly stages.
    output_dir : str | Path
        Destination directory (created if missing).

    Returns
    -------
    dict[str, pathlib.Path]
        ``{"cleaned_data": ..., "problem_samples": ...}``.

    Notes
    -----
    A sequence is listed in ``problem_samples.csv`` when any of:
    IsolationForest flags it, its quality_score is below 30, or its
    SPC out-of-control ratio exceeds 0.1. The ``reason`` column records why.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    anom_cols = [
        "sequence_id",
        "iso_label",
        "iso_score",
        "mahal_label",
        "mahal_score",
    ]
    anom_cols = [c for c in anom_cols if c in anomaly_df.columns]
    merged = quality_df.merge(anomaly_df[anom_cols], on="sequence_id", how="left")

    cleaned_path = out / "cleaned_data.csv"
    merged.to_csv(cleaned_path, index=False)

    def _reasons(row) -> str:
        flags = []
        if row.get("iso_label") == -1:
            flags.append("iso_anomaly")
        if row["quality_score"] < 30:
            flags.append("low_quality(<30)")
        if pd.notna(row.get("spc_out_of_control_ratio")) and row["spc_out_of_control_ratio"] > 0.1:
            flags.append("spc_out_of_control>0.1")
        return ";".join(flags)

    problems = merged.copy()
    problems["reason"] = problems.apply(_reasons, axis=1)
    mask = (
        (problems["iso_label"] == -1)
        | (problems["quality_score"] < 30)
        | (problems["spc_out_of_control_ratio"] > 0.1)
    )
    problems = problems.loc[mask].sort_values("quality_score")
    prob_path = out / "problem_samples.csv"
    problems.to_csv(prob_path, index=False)

    return {"cleaned_data": cleaned_path, "problem_samples": prob_path}
