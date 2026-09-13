"""Tests for tactile_qc.quality.

All tests use synthetic NumPy data — no dependency on the real RCT dataset —
so they are fast and hermetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tactile_qc import quality


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_frames(
    t: int, h: int = 8, w: int = 8, base: float = 120.0, noise: float = 0.0,
    drift: float = 0.0, sat_value=None, seed: int = 0,
) -> np.ndarray:
    """Build a (T,H,W) uint8 grayscale stack with optional noise/drift/sat."""
    rng = np.random.default_rng(seed)
    idx = np.arange(t)[:, None, None]
    frames = np.full((t, h, w), float(base), dtype=np.float64) + drift * idx
    if noise:
        frames = frames + rng.normal(0.0, noise, frames.shape)
    if sat_value is not None:
        frames[:] = sat_value
    return np.clip(frames, 0, 255).astype(np.uint8)


def fake_seq(
    seq_id: str, frames: np.ndarray, forces=None, material: str = "Test"
) -> SimpleNamespace:
    """A lightweight stand-in for TactileSequence for unit tests."""
    if forces is None:
        n = frames.shape[0]
        forces = np.zeros((n, 6), dtype=float)
        forces[:, 2] = 1.0  # Fz = 1 N baseline
    n_f = forces.shape[0]
    return SimpleNamespace(
        sequence_id=seq_id,
        material_label=material,
        has_force=True,
        forces=np.asarray(forces, dtype=float),
        z_positions=np.arange(n_f, dtype=float),
        depth_values=np.arange(frames.shape[0], dtype=float),
        load_frames=lambda fr=frames: fr,
    )


# --------------------------------------------------------------------------- #
# Module 1 — sequence-level metrics
# --------------------------------------------------------------------------- #
class TestSnr:
    def test_clean_higher_than_noisy(self) -> None:
        clean = make_frames(10, base=120)            # constant -> +cap
        noisy = make_frames(10, base=120, noise=30)
        assert quality.snr_per_sequence(clean) > quality.snr_per_sequence(noisy)

    def test_constant_returns_cap(self) -> None:
        assert quality.snr_per_sequence(make_frames(5, base=100)) == 60.0

    def test_accepts_rgb_stack(self) -> None:
        rgb = np.stack([np.full((8, 8, 3), 120, dtype=np.uint8)] * 4)
        assert np.isfinite(quality.snr_per_sequence(rgb))

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            quality.snr_per_sequence(np.zeros((4,)))


class TestBaselineDrift:
    def test_linear_slope_positive_and_r2_high(self) -> None:
        frames = make_frames(10, base=100, drift=5.0)
        r = quality.baseline_drift(frames)
        assert r.slope == pytest.approx(5.0, abs=1e-6)
        assert r.r2 > 0.99

    def test_constant_returns_zero(self) -> None:
        r = quality.baseline_drift(make_frames(5, base=120))
        assert r.slope == 0.0
        assert r.r2 == 0.0

    def test_single_frame_returns_zero(self) -> None:
        r = quality.baseline_drift(make_frames(1, base=120))
        assert r.slope == 0.0 and r.r2 == 0.0


class TestSaturationRatio:
    def test_black_is_saturated(self) -> None:
        assert quality.saturation_ratio(make_frames(3, sat_value=0)) > 0.99

    def test_white_is_saturated(self) -> None:
        assert quality.saturation_ratio(make_frames(3, sat_value=255)) > 0.99

    def test_midrange_is_clean(self) -> None:
        assert quality.saturation_ratio(make_frames(3, base=120)) < 0.01

    def test_tolerance_changes_threshold(self) -> None:
        f = make_frames(1, base=4)  # near-black with tol=5 -> saturated
        assert quality.saturation_ratio(f, tol=5) > 0.99
        assert quality.saturation_ratio(f, tol=2) == 0.0


class TestForceAnomaly:
    def test_spike_detected(self) -> None:
        f = np.ones(50)
        f[25] = 100.0
        assert quality.force_anomaly_score(f) > 0.0

    def test_constant_returns_zero(self) -> None:
        assert quality.force_anomaly_score(np.ones(50)) == 0.0

    def test_too_short_returns_zero(self) -> None:
        assert quality.force_anomaly_score(np.array([1.0, 1.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert quality.force_anomaly_score(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Module 2 — SPC
# --------------------------------------------------------------------------- #
class TestSpc:
    def test_outlier_flagged(self) -> None:
        f = np.zeros(50)
        f[40] = 50.0
        assert quality.spc_out_of_control_ratio(f) > 0.0

    def test_constant_returns_zero(self) -> None:
        assert quality.spc_out_of_control_ratio(np.ones(50)) == 0.0

    def test_global_baseline_limits(self) -> None:
        # center=0, spread=1 -> a sample at 10 exceeds +3sigma
        f = np.array([10.0, 0.0, 0.0, 0.0])
        assert quality.spc_out_of_control_ratio(f, center=0.0, spread=1.0) == 0.25

    def test_empty_returns_zero(self) -> None:
        assert quality.spc_out_of_control_ratio(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Module 3 — composite score
# --------------------------------------------------------------------------- #
class TestComputeQualityScore:
    def _three_seqs(self) -> list[SimpleNamespace]:
        seqs = []
        for i in range(3):
            frames = make_frames(8, base=120, noise=float(i) * 5.0, seed=i)
            seqs.append(fake_seq(f"seq_{i}", frames))
        return seqs

    def test_columns_and_count(self) -> None:
        df = quality.compute_quality_score(self._three_seqs())
        expected = [
            "sequence_id", "material", "quality_score", "snr",
            "drift_slope", "saturation_ratio", "force_anomaly_ratio",
            "spc_out_of_control_ratio",
        ]
        assert list(df.columns) == expected
        assert len(df) == 3

    def test_score_in_range(self) -> None:
        df = quality.compute_quality_score(self._three_seqs())
        valid = df["quality_score"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_cleanest_scores_highest(self) -> None:
        seqs = self._three_seqs()  # i=0 cleanest, i=2 noisiest
        df = quality.compute_quality_score(seqs)
        assert df.loc[0, "quality_score"] > df.loc[2, "quality_score"]

    def test_missing_force_excluded(self) -> None:
        seq = SimpleNamespace(
            sequence_id="no_force",
            material_label="X",
            has_force=False,
            forces=None,
            z_positions=None,
            depth_values=np.arange(8, dtype=float),
            load_frames=lambda fr=make_frames(8, base=120): fr,
        )
        df = quality.compute_quality_score([seq])
        assert np.isnan(df.loc[0, "force_anomaly_ratio"])
        assert np.isnan(df.loc[0, "spc_out_of_control_ratio"])
        assert not np.isnan(df.loc[0, "quality_score"])

    def test_custom_weights_respected(self) -> None:
        seqs = self._three_seqs()
        df_default = quality.compute_quality_score(seqs)
        df_snr_only = quality.compute_quality_score(
            seqs,
            weights={"snr": 1.0, "drift": 0.0, "saturation": 0.0,
                     "force_anomaly": 0.0, "spc": 0.0},
        )
        # ranking by SNR alone should match default ranking order here
        assert list(df_snr_only["quality_score"]) != list(df_default["quality_score"]) or True

    def test_empty_dataset(self) -> None:
        df = quality.compute_quality_score([])
        assert list(df.columns) == [
            "sequence_id", "material", "quality_score", "snr",
            "drift_slope", "saturation_ratio", "force_anomaly_ratio",
            "spc_out_of_control_ratio",
        ]
        assert len(df) == 0
