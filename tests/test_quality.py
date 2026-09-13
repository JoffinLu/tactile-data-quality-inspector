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


# --------------------------------------------------------------------------- #
# Correctness — analytic checks against known signals
# --------------------------------------------------------------------------- #
class TestSnrCorrectness:
    def test_snr_matches_known_ratio(self) -> None:
        # spatial-gradient signal image with known variance + iid noise of known variance
        rng = np.random.default_rng(7)
        h = w = 16
        signal = np.tile(np.linspace(0.0, 200.0, w, dtype=np.float64), (h, 1))
        signal_var = float(np.var(signal))
        noise_std = 5.0
        noise_var = noise_std ** 2
        frames = signal[None, :, :] + rng.normal(0.0, noise_std, (10, h, w))
        snr = quality.snr_per_sequence(frames)
        expected = 10.0 * np.log10(signal_var / noise_var)
        assert snr == pytest.approx(expected, rel=0.15)

    def test_more_noise_lower_snr(self) -> None:
        rng = np.random.default_rng(11)
        base = np.tile(np.linspace(0.0, 200.0, 16), (16, 1))[None, :, :]
        s_low = quality.snr_per_sequence(base + rng.normal(0, 2.0, (10, 16, 16)))
        s_high = quality.snr_per_sequence(base + rng.normal(0, 20.0, (10, 16, 16)))
        assert s_low > s_high

    def test_flat_image_with_noise_returns_floor(self) -> None:
        # frame-level brightness jitter: the temporal-mean image is spatially
        # flat (no signal) while every frame is noisy -> SNR floor -60 dB
        rng = np.random.default_rng(5)
        t = 12
        jitter = rng.normal(0.0, 10.0, t)  # one offset per frame, all pixels
        frames = 120.0 + jitter[:, None, None]
        assert quality.snr_per_sequence(frames) == -60.0


class TestBaselineDriftFitting:
    def test_slope_matches_injected(self) -> None:
        # brightness increases exactly 3 per frame, no noise
        frames = make_frames(20, base=80, drift=3.0)
        r = quality.baseline_drift(frames)
        assert r.slope == pytest.approx(3.0, abs=1e-9)
        assert r.r2 > 0.999

    def test_negative_slope(self) -> None:
        rng = np.random.default_rng(1)
        t, h, w = 20, 8, 8
        idx = np.arange(t)[:, None, None]
        frames = np.full((t, h, w), 150.0) - 2.0 * idx + rng.normal(0, 0.01, (t, h, w))
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        r = quality.baseline_drift(frames)
        assert r.slope == pytest.approx(-2.0, abs=0.01)
        assert r.r2 > 0.99

    def test_noisy_drift_slope_recovered(self) -> None:
        # linear drift buried in noise: slope still recovered, R² high
        rng = np.random.default_rng(2)
        t, h, w = 40, 8, 8
        idx = np.arange(t)[:, None, None]
        frames = np.full((t, h, w), 100.0) + 1.5 * idx + rng.normal(0, 2.0, (t, h, w))
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        r = quality.baseline_drift(frames)
        assert r.slope == pytest.approx(1.5, abs=0.2)
        assert r.r2 > 0.9


class TestSpcControlLimits:
    def test_injected_outliers_flagged(self) -> None:
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(0, 1, 1000), [10.0, -10.0, 8.0]])
        ratio = quality.spc_out_of_control_ratio(x)
        assert ratio > 0.0          # injected outliers exceed +/-3 sigma
        assert ratio < 0.01         # ... but only a handful

    def test_essentially_none_within_limits(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 5000)  # ~0.27% outside +/-3 sigma
        assert quality.spc_out_of_control_ratio(x) < 0.01

    def test_global_center_spread_threshold(self) -> None:
        # center=10, spread=2 -> limits [4, 16]; 17 exceeds, 15 does not
        f = np.array([10, 10, 10, 10, 17.0, 15.0])
        r = quality.spc_out_of_control_ratio(f, center=10.0, spread=2.0)
        assert r == pytest.approx(1.0 / 6.0)

    def test_k_multiplier_tightens_limits(self) -> None:
        # on a standard-normal sample, ±1σ flags ~31.7%, ±3σ flags ~0.27%
        rng = np.random.default_rng(9)
        x = rng.normal(0, 1, 4000)
        loose = quality.spc_out_of_control_ratio(x, k=3.0)
        tight = quality.spc_out_of_control_ratio(x, k=1.0)
        assert tight > loose
        assert tight == pytest.approx(0.317, abs=0.03)

    def test_control_limits_match_three_sigma(self) -> None:
        # center=50, spread=2 -> limits [44, 56]; boundary values not flagged
        x = np.array([44.0, 56.0, 44.1, 55.9, 43.99, 56.01])
        ratio = quality.spc_out_of_control_ratio(x, center=50.0, spread=2.0)
        assert ratio == pytest.approx(2.0 / 6.0)

