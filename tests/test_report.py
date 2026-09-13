"""Tests for tactile_qc.report (HTML/CSV generation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tactile_qc import report


def _sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = pd.DataFrame(
        {
            "sequence_id": [f"seq_{i}" for i in range(6)],
            "material": ["A", "A", "B", "B", "C", "C"],
            "quality_score": [95.0, 88.0, 72.0, 25.0, 60.0, 50.0],
            "snr": [20, 18, 16, 5, 15, 14],
            "drift_slope": [0.01, -0.02, 0.1, 0.5, 0.05, 0.03],
            "saturation_ratio": [0.0, 0.0, 0.01, 0.2, 0.0, 0.0],
            "force_anomaly_ratio": [0.0, 0.0, 0.05, 0.1, 0.0, 0.02],
            "spc_out_of_control_ratio": [0.0, 0.05, 0.06, 0.2, 0.0, 0.0],
        }
    )
    anomaly = pd.DataFrame(
        {
            "sequence_id": [f"seq_{i}" for i in range(6)],
            "material": ["A", "A", "B", "B", "C", "C"],
            "iso_label": [1, 1, 1, -1, 1, 1],
            "iso_score": [0.2, 0.15, 0.1, -0.3, 0.05, 0.0],
            "mahal_label": [1, 1, -1, -1, 1, 1],
            "mahal_score": [-1.0, -2.0, -8.0, -9.0, -1.5, -2.5],
        }
    )
    return quality, anomaly


class TestHtmlReport:
    def test_writes_selfcontained_html(self, tmp_path: Path) -> None:
        q, a = _sample_dfs()
        out = tmp_path / "report.html"
        p = report.generate_html_report(
            q, a, out, dataset_stats={"material_count": 3, "total_frames": 96}
        )
        assert p == out and out.exists() and out.stat().st_size > 1000
        text = out.read_text(encoding="utf-8")
        assert "Plotly" in text  # inline plotly js present
        assert "seq_3" in text  # top anomaly listed
        assert "质量评分分布" in text


class TestCsvExport:
    def test_cleaned_and_problem_files(self, tmp_path: Path) -> None:
        q, a = _sample_dfs()
        paths = report.generate_csv_export(q, a, tmp_path)
        assert paths["cleaned_data"].exists()
        assert paths["problem_samples"].exists()
        cleaned = pd.read_csv(paths["cleaned_data"])
        assert len(cleaned) == 6
        assert "iso_label" in cleaned.columns and "quality_score" in cleaned.columns

    def test_problem_samples_reasons(self, tmp_path: Path) -> None:
        q, a = _sample_dfs()
        paths = report.generate_csv_export(q, a, tmp_path)
        prob = pd.read_csv(paths["problem_samples"])
        # seq_3: iso anomaly + low quality + spc>0.1
        assert "seq_3" in set(prob["sequence_id"])
        r = prob.loc[prob["sequence_id"] == "seq_3", "reason"].iloc[0]
        assert "iso_anomaly" in r and "low_quality(<30)" in r and "spc_out_of_control" in r

    def test_csv_export_without_spc_column(self, tmp_path: Path) -> None:
        # quality frame missing the SPC column must not crash; seq_3 is
        # still flagged via iso_anomaly + low_quality
        q, a = _sample_dfs()
        q = q.drop(columns=["spc_out_of_control_ratio"])
        paths = report.generate_csv_export(q, a, tmp_path)
        prob = pd.read_csv(paths["problem_samples"])
        assert "seq_3" in set(prob["sequence_id"])
        r = prob.loc[prob["sequence_id"] == "seq_3", "reason"].iloc[0]
        assert "iso_anomaly" in r and "low_quality(<30)" in r
        assert "spc" not in r
