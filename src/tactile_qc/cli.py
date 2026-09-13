"""Command-line interface for tactile-qc.

Examples
--------
::

    tactile-qc explore --data-dir ./data/rct/rct_dataset
    tactile-qc run --data-dir ./data/rct --output-dir ./reports --format both
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import click


def _resolve_dataset_dir(path: str | Path) -> Path:
    """Locate the RCT dataset root (the dir containing ``materials/``).

    Accepts either the dataset root itself or a parent directory; the common
    extraction layouts ``<root>`` and ``<root>/rct_dataset`` and
    ``<root>/rct_dataset/rct_dataset`` are all tried. When several candidates
    are valid, the one with the most material sub-directories is chosen, so a
    parent dir containing both a partial and a complete extraction resolves to
    the complete one.
    """
    p = Path(path)
    candidates = [p, p / "rct_dataset", p / "rct_dataset" / "rct_dataset"]
    valid = [
        c
        for c in candidates
        if (c / "materials").is_dir() and (c / "material_categories.json").exists()
    ]
    if not valid:
        raise click.ClickException(
            f"未在 {path} 下找到 RCT 数据集根目录（需含 materials/ 与 "
            f"material_categories.json）。"
        )
    valid.sort(
        key=lambda c: sum(
            1 for _ in (c / "materials").rglob("*") if _.is_file()
        ),
        reverse=True,
    )
    return valid[0]


@click.group()
@click.version_option(package_name="tactile-qc")
def cli() -> None:
    """tactile-qc — statistical quality assessment for robotic tactile data."""


def _resolve_workers(workers: int) -> int | None:
    """Map the --workers flag to an ``n_jobs`` value.

    ``1`` -> ``None`` (serial); ``0`` or negative -> auto (half the CPU
    cores, capped at 8); ``> 1`` -> that many processes.
    """
    if workers == 1:
        return None
    if workers <= 0:
        import os

        return max(2, min(8, (os.cpu_count() or 1) // 2))
    return workers


@cli.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="RCT 数据集根目录（含 materials/），或其父目录。",
)
@click.option(
    "--output-dir",
    default="reports",
    type=click.Path(),
    help="报告输出目录（默认 reports）。",
)
@click.option(
    "--contamination",
    default=0.05,
    type=float,
    show_default=True,
    help="异常检测的预期异常比例。",
)
@click.option(
    "--workers",
    "-j",
    default=0,
    type=int,
    show_default=True,
    help="并行进程数：0=自动（CPU 核数一半，上限 8），1=串行，N=N 进程。",
)
@click.option(
    "--format",
    "fmt",
    default="both",
    type=click.Choice(["html", "csv", "both"]),
    show_default=True,
    help="输出格式。",
)
def run(
    data_dir: str,
    output_dir: str,
    contamination: float,
    workers: int,
    fmt: str,
) -> None:
    """运行完整质检管线：质量评分 + 异常检测 + 报告生成（单遍扫描）。"""
    from . import anomaly, io, quality, report

    data_dir = _resolve_dataset_dir(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = _resolve_workers(workers)

    click.echo(f"[1/4] 加载数据集: {data_dir}")
    ds = io.load_rct_sequences(data_dir)
    stats = {
        "sequence_count": len(ds),
        "material_count": len({s.material_id for s in ds}),
        "total_frames": sum(len(s) for s in ds),
    }
    click.echo(
        f"      {stats['sequence_count']} 序列 / "
        f"{stats['material_count']} 材料 / {stats['total_frames']} 帧"
    )

    mode = "串行" if n_jobs is None else f"{n_jobs} 进程并行"
    click.echo(f"[2/4] 单遍评估（质量指标 + 异常特征，{mode}） ...")
    step = {"next": max(1, len(ds) // 10)}

    def _progress(done: int, total: int) -> None:
        if done == total or done >= step["next"]:
            step["next"] = done + max(1, total // 10)
            click.echo(f"      {done}/{total} 序列")

    assessments = quality.assess_sequences(ds, n_jobs=n_jobs, progress=_progress)

    qdf = quality.compute_quality_score(assessments=assessments)
    qpath = out / "quality_scores.csv"
    qdf.to_csv(qpath, index=False)
    click.echo(f"      -> {qpath} （均值 {qdf['quality_score'].mean():.1f}）")

    click.echo("[3/4] 异常检测（复用已解码帧，无二次加载） ...")
    rep = anomaly.detect_anomalies(
        assessments=assessments, contamination=contamination
    )
    adf = rep.per_sequence
    apath = out / "anomaly_results.csv"
    adf.to_csv(apath, index=False)
    iso_n = int((adf["iso_label"] == -1).sum())
    click.echo(
        f"      -> {apath} （IF 异常 {iso_n} = {100 * iso_n / len(adf):.1f}%）"
    )

    click.echo("[4/4] 生成报告 ...")
    if fmt in ("html", "both"):
        hpath = report.generate_html_report(
            qdf, adf, out / "quality_report.html", dataset_stats=stats
        )
        click.echo(f"      html -> {hpath}")
    if fmt in ("csv", "both"):
        paths = report.generate_csv_export(qdf, adf, out)
        for k, v in paths.items():
            click.echo(f"      csv  -> {v} ({k})")
    click.echo("完成。")


@cli.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="RCT 数据集根目录（含 materials/），或其父目录。",
)
def explore(data_dir: str) -> None:
    """快速输出数据集概览到终端。"""
    from . import io

    data_dir = _resolve_dataset_dir(data_dir)
    ds = io.load_rct_sequences(data_dir)
    n_mat = len({s.material_id for s in ds})
    n_frames = sum(len(s) for s in ds)
    n_force = sum(s.has_force for s in ds)
    click.echo(f"数据集: {data_dir}")
    click.echo(f"  序列数    : {len(ds)}")
    click.echo(f"  材料数    : {n_mat}")
    click.echo(f"  总帧数    : {n_frames}")
    click.echo(f"  有力轨迹  : {n_force}  (缺失 {len(ds) - n_force})")
    cat = Counter(s.material_label for s in ds)
    click.echo("  按材料类别:")
    for k, v in cat.most_common():
        click.echo(f"    {k:20s} {v}")
    click.echo(
        f"  帧数/序列 : 均值 {n_frames / max(len(ds), 1):.1f}"
    )


if __name__ == "__main__":
    cli()
