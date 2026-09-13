"""Tests for tactile_qc.anomaly.

Synthetic, hermetic — no dependency on the real RCT dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tactile_qc import anomaly


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_frames(t: int, h: int = 8, w: int = 8, base: float = 120.0,
                noise: float = 0.0, drift: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(t)[:, None, None]
    frames = np.full((t, h, w), float(base), dtype=np.float64) + drift * idx
    if noise:
        frames = frames + rng.normal(0.0, noise, frames.shape)
    return np.clip(frames, 0, 255).astype(np.uint8)


def fake_seq(seq_id: str, frames: np.ndarray, material: str = "Test",
             force: float = 1.0) -> SimpleNamespace:
    n = frames.shape[0]
    forces = np.zeros((n, 6), dtype=float)
    forces[:, 2] = force  # constant Fz
    return SimpleNamespace(
        sequence_id=seq_id,
        material_label=material,
        has_force=True,
        forces=forces,
        z_positions=np.arange(n, dtype=float),
        depth_values=np.arange(n, dtype=float),
        load_frames=lambda fr=frames: fr,
    )


@pytest.fixture
def small_dataset() -> list:
    # 6 sequences; one is a clear outlier (very noisy)
    seqs = []
    for i in range(6):
        noise = 30.0 if i == 5 else float(i) * 2.0
        seqs.append(fake_seq(f"seq_{i}", make_frames(8, noise=noise, seed=i)))
    return seqs


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #
class TestExtractFeatures:
    def test_shape_and_names(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=10, img_size=8)
        assert feats.X.shape[0] == 6
        # 10 pca + 4 force + 2 extra
        assert feats.X.shape[1] == 10 + 4 + 2
        assert len(feats.feature_names) == feats.X.shape[1]
        assert feats.sequence_ids == [f"seq_{i}" for i in range(6)]

    def test_pca_components_clamped(self, small_dataset) -> None:
        # ask for more components than frames available -> clamped
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=100, img_size=8)
        assert feats.n_pca <= 100

    def test_no_force_imputed(self) -> None:
        frames = make_frames(8, noise=1.0, seed=0)
        seq = SimpleNamespace(
            sequence_id="no_force", material_label="X", has_force=False,
            forces=None, z_positions=None, depth_values=np.arange(8, dtype=float),
            load_frames=lambda fr=frames: fr,
        )
        feats = anomaly.extract_sequence_features([seq, fake_seq("b", make_frames(8, noise=2.0, seed=1))],
                                                  pca_components=5, img_size=8)
        assert not np.isnan(feats.X).any()  # NaN force stats imputed


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
class TestDetectors:
    def test_isolation_forest_labels(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=5, img_size=8)
        res = anomaly.isolation_forest_anomaly(feats.X, contamination=0.2)
        assert set(np.unique(res.labels)).issubset({-1, 1})
        assert res.scores.shape == (6,)

    def test_mahalanobis_labels(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=5, img_size=8)
        res = anomaly.mahalanobis_anomaly(feats.X, contamination=0.2)
        assert set(np.unique(res.labels)).issubset({-1, 1})
        assert res.scores.shape == (6,)

    def test_outlier_flagged(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=5, img_size=8)
        iso = anomaly.isolation_forest_anomaly(feats.X, contamination=0.2)
        # the noisy outlier (seq_5) is likely flagged
        assert iso.labels[5] == -1

    def test_too_few_samples(self) -> None:
        X = np.zeros((1, 3))
        res = anomaly.isolation_forest_anomaly(X)
        assert res.labels[0] == 1
        res2 = anomaly.mahalanobis_anomaly(X)
        assert res2.labels[0] == 1


# --------------------------------------------------------------------------- #
# stratified evaluation + comparison
# --------------------------------------------------------------------------- #
class TestEvalAndComparison:
    def test_stratified_table_columns(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=5, img_size=8)
        iso = anomaly.isolation_forest_anomaly(feats.X, contamination=0.2)
        ev = anomaly.stratified_anomaly_table(iso.labels, feats.materials)
        for col in ("material", "n", "n_anomaly", "n_normal", "anomaly_ratio"):
            assert col in ev.table.columns
        assert ev.table["n"].sum() == 6
        assert 0.0 <= ev.table["anomaly_ratio"].max() <= 1.0

    def test_kappa_range(self, small_dataset) -> None:
        feats = anomaly.extract_sequence_features(small_dataset, pca_components=5, img_size=8)
        iso = anomaly.isolation_forest_anomaly(feats.X, contamination=0.2)
        mah = anomaly.mahalanobis_anomaly(feats.X, contamination=0.2)
        cmp = anomaly.compare_methods(iso.labels, mah.labels)
        assert -1.0 <= cmp.cohen_kappa <= 1.0
        assert cmp.n_both_anomaly + cmp.n_both_normal + cmp.n_disagree == 6

    def test_perfect_agreement(self) -> None:
        a = np.array([-1, 1, 1, -1])
        cmp = anomaly.compare_methods(a, a)
        assert cmp.cohen_kappa == 1.0
        assert cmp.agreement == 1.0
        assert cmp.n_disagree == 0


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
class TestDetectAnomalies:
    def test_report_structure(self, small_dataset) -> None:
        rep = anomaly.detect_anomalies(
            small_dataset, contamination=0.2, pca_components=5, img_size=8
        )
        assert list(rep.per_sequence.columns) == [
            "sequence_id", "material", "iso_label", "iso_score",
            "mahal_label", "mahal_score",
        ]
        assert len(rep.per_sequence) == 6
        assert hasattr(rep, "stratified")
        assert hasattr(rep, "comparison")
        assert -1.0 <= rep.comparison.cohen_kappa <= 1.0


# --------------------------------------------------------------------------- #
# detection rate under known anomaly injection
# --------------------------------------------------------------------------- #
class TestDetectionRate:
    def _injected_matrix(self, seed: int, n_normal: int = 200, n_anom: int = 20,
                         dim: int = 6, shift: float = 10.0):
        rng = np.random.default_rng(seed)
        normal = rng.normal(0, 1, (n_normal, dim))
        anomalies = rng.normal(0, 1, (n_anom, dim)) + shift
        X = np.vstack([normal, anomalies])
        return X, n_normal, n_anom

    def test_isolation_forest_high_recall(self) -> None:
        X, n_normal, n_anom = self._injected_matrix(seed=42)
        res = anomaly.isolation_forest_anomaly(
            X, contamination=n_anom / (n_normal + n_anom)
        )
        recall = int(((res.labels == -1)[n_normal:]).sum()) / n_anom
        assert recall >= 0.8  # >=80% of injected anomalies detected

    def test_isolation_forest_low_false_positive_rate(self) -> None:
        # few normal samples mis-flagged alongside the true anomalies
        X, n_normal, n_anom = self._injected_matrix(seed=7)
        res = anomaly.isolation_forest_anomaly(
            X, contamination=n_anom / (n_normal + n_anom)
        )
        fpr = int(((res.labels == -1)[:n_normal]).sum()) / n_normal
        assert fpr < 0.05  # <5% of normals mis-flagged

    def test_recall_stable_across_seeds(self) -> None:
        # detection rate must not hinge on one lucky seed
        recalls = []
        for seed in (1, 2, 3, 4, 5):
            X, n_normal, n_anom = self._injected_matrix(seed=seed)
            res = anomaly.isolation_forest_anomaly(
                X, contamination=n_anom / (n_normal + n_anom)
            )
            recalls.append(int(((res.labels == -1)[n_normal:]).sum()) / n_anom)
        assert min(recalls) >= 0.8

    def test_anomaly_scores_rank_injected_lower(self) -> None:
        # decision_function: higher = more normal -> injected anomalies rank lower
        X, n_normal, n_anom = self._injected_matrix(seed=11)
        res = anomaly.isolation_forest_anomaly(
            X, contamination=n_anom / (n_normal + n_anom)
        )
        assert res.scores[n_normal:].mean() < res.scores[:n_normal].mean()

    def test_mahalanobis_high_recall(self) -> None:
        X, n_normal, n_anom = self._injected_matrix(seed=0, dim=4, shift=8.0)
        res = anomaly.mahalanobis_anomaly(
            X, contamination=n_anom / (n_normal + n_anom)
        )
        recall = int(((res.labels == -1)[n_normal:]).sum()) / n_anom
        assert recall >= 0.7

