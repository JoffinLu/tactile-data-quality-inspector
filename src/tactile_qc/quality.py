"""Statistical quality metrics for tactile sequences.

This module implements three layers of quality assessment for the RCT dataset:

1. Sequence-level metrics
   - :func:`snr_per_sequence`        signal-to-noise ratio of a contact sequence
   - :func:`baseline_drift`           linear drift of inter-frame brightness
   - :func:`saturation_ratio`         fraction of near-black / near-white pixels
   - :func:`force_anomaly_score`      sliding-window 3-sigma force breakpoints

2. Statistical process control (SPC)
   - :func:`spc_out_of_control_ratio` X-bar control-chart out-of-control ratio

3. Composite quality score
   - :func:`compute_quality_score`    weighted, normalized 0-100 score per sequence

Shape assumptions (IMPORTANT)
-----------------------------
``frames`` : np.ndarray of shape ``(T, H, W)``, dtype uint8 or float
    A stack of ``T`` grayscale tactile frames from one contact sequence
    (DIGIT-style single-channel sensor). A ``(T, H, W, C)`` multi-channel
    array is accepted and reduced to grayscale via channel-mean.

``forces`` : np.ndarray of shape ``(T,)``, dtype float
    A 1-D per-frame force signal (e.g. the magnitude ||F_ext|| or the axial
    Fz aligned to the tactile frames). The raw RCT force trace is longer
    than the frame count (it spans approach + contact); callers must align
    it to frames before passing it here. See :func:`_align_force_to_frames`.

Higher quality == higher ``snr`` and lower ``drift_slope`` /
``saturation_ratio`` / ``force_anomaly_ratio`` / ``spc_out_of_control_ratio``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: Default component weights. The spec's "force anomaly 0.3" is realised as the
#: force sub-component split into ``force_anomaly`` (0.15) + ``spc`` (0.15) so
#: that SPC contributes to the score while the category totals still match
#: SNR 0.3 / drift 0.2 / saturation 0.2 / force 0.3 (sum = 1.0).
DEFAULT_WEIGHTS: dict[str, float] = {
    "snr": 0.3,
    "drift": 0.2,
    "saturation": 0.2,
    "force_anomaly": 0.15,
    "spc": 0.15,
}

#: finite SNR cap (dB) used to keep normalization well-behaved on noise-free data
_SNR_CAP_DB = 60.0


def _to_gray(frames: np.ndarray) -> np.ndarray:
    """Coerce ``frames`` to a float64 grayscale array of shape (T, H, W)."""
    arr = np.asarray(frames)
    if arr.ndim == 4:  # (T, H, W, C)
        arr = arr.mean(axis=-1)
    elif arr.ndim != 3:
        raise ValueError(
            f"frames must be (T,H,W) or (T,H,W,C); got shape {arr.shape}"
        )
    return arr.astype(np.float64)


# --------------------------------------------------------------------------- #
# shared per-sequence feature helpers
# (used by the anomaly stage; defined here so that process-pool workers
#  only need this module and never import sklearn)
# --------------------------------------------------------------------------- #
def _downscale_gray(gray: np.ndarray, size: int) -> np.ndarray:
    """Downscale a (T,H,W) grayscale stack to (T, size*size) float32."""
    from PIL import Image

    out = np.empty((gray.shape[0], size * size), dtype=np.float32)
    for i, fr in enumerate(gray):
        im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8), mode="L")
        im = im.resize((size, size))
        out[i] = np.asarray(im, dtype=np.float32).ravel()
    return out


def _force_stats(force: np.ndarray | None) -> np.ndarray:
    """``[mean, std, skew, kurt]`` of a force signal; NaNs if unavailable."""
    if force is None or force.size == 0:
        return np.full(4, np.nan)
    x = np.asarray(force, dtype=np.float64).ravel()
    std = float(np.std(x)) if x.size > 1 else 0.0
    if x.size > 2 and np.std(x) > 0:
        import warnings

        from scipy import stats  # lazy: keeps module import light

        # near-constant force traces can trigger a harmless precision-loss
        # RuntimeWarning from the moment calculation; the finite guards below
        # already handle any degraded result.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = float(stats.skew(x, bias=False))
            kurt = float(stats.kurtosis(x, bias=False))
    else:
        skew, kurt = 0.0, 0.0
    # robust to scipy returning inf/nan on degenerate inputs
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    return np.array([float(x.mean()), std, skew, kurt])


def _frame_diff_mean(gray: np.ndarray) -> float:
    """Mean absolute pixel difference between consecutive frames."""
    if gray.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(gray, axis=0)).mean())


# --------------------------------------------------------------------------- #
# Module 1 — sequence-level metrics
# --------------------------------------------------------------------------- #
def snr_per_sequence(frames: np.ndarray) -> float:
    """Signal-to-noise ratio (dB) of a contact sequence.

    The temporal-mean image is the *signal*; frame-to-frame pixel variation
    is the *noise*::

        signal_var = Var( mean_t(frame) )          # spatial var of mean image
        noise_var  = mean_t( Var_pixel(frame) )    # mean pixel-wise temporal var
        SNR_dB     = 10 * log10( signal_var / noise_var )

    Returns a capped finite value in ``[-60, 60]`` dB. Noise-free sequences
    (constant across time) yield the cap ``+60``; flat mean image with noise
    yields ``-60``.

    Parameters
    ----------
    frames : np.ndarray
        ``(T, H, W)`` grayscale frames (or ``(T,H,W,C)``).
    """
    arr = _to_gray(frames)
    t = arr.shape[0]
    signal = arr.mean(axis=0)  # (H, W)
    signal_var = float(np.var(signal))
    noise_var = float(np.mean(np.var(arr, axis=0))) if t > 1 else 0.0
    if noise_var <= 0.0:
        return _SNR_CAP_DB  # no temporal noise -> "clean"
    if signal_var <= 0.0:
        return -_SNR_CAP_DB  # flat mean image but noisy -> no signal
    snr = 10.0 * np.log10(signal_var / noise_var)
    return float(np.clip(snr, -_SNR_CAP_DB, _SNR_CAP_DB))


@dataclass(frozen=True)
class DriftResult:
    """Linear-regression drift of inter-frame mean brightness."""

    slope: float  # brightness change per frame index
    r2: float  # goodness of fit in [0, 1]


def baseline_drift(frames: np.ndarray) -> DriftResult:
    """Linear-regression fit of inter-frame brightness trend.

    For each frame computes its mean brightness, then fits
    ``brightness = slope * t + intercept`` (``t`` = frame index) and returns
    the slope and the coefficient of determination R².

    Parameters
    ----------
    frames : np.ndarray
        ``(T, H, W)`` grayscale frames.

    Returns
    -------
    DriftResult
        ``slope`` (brightness/frame) and ``r2``. Sequences with ``T < 2`` or
        constant brightness return ``DriftResult(0.0, 0.0)``.
    """
    arr = _to_gray(frames)
    t = arr.shape[0]
    if t < 2:
        return DriftResult(0.0, 0.0)
    brightness = arr.mean(axis=(1, 2))  # (T,)
    if np.allclose(brightness, brightness[0]):
        return DriftResult(0.0, 0.0)
    idx = np.arange(t, dtype=np.float64)
    slope, _intercept = np.polyfit(idx, brightness, 1)
    pred = slope * idx + _intercept
    ss_res = float(np.sum((brightness - pred) ** 2))
    ss_tot = float(np.sum((brightness - brightness.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DriftResult(float(slope), float(r2))


def saturation_ratio(frames: np.ndarray, tol: int = 5) -> float:
    """Fraction of pixels near black (<= ``tol``) or near white (>= ``255-tol``).

    Parameters
    ----------
    frames : np.ndarray
        ``(T, H, W)`` grayscale frames.
    tol : int, default 5
        Tolerance band around 0 / 255 counted as "saturated".

    Returns
    -------
    float
        Fraction in ``[0, 1]`` of saturated pixels across all frames.
    """
    arr = _to_gray(frames)
    flat = arr.reshape(arr.shape[0], -1)
    sat = (flat <= tol) | (flat >= 255 - tol)
    return float(sat.mean())


def force_anomaly_score(
    forces: np.ndarray, window: int = 11, k: float = 3.0
) -> float:
    """Detect force-signal breakpoints with a sliding window + 3-sigma rule.

    A centered rolling window (size ``window``) estimates a local mean and
    standard deviation; any sample whose deviation from the local mean
    exceeds ``k`` local sigmas is flagged as an anomaly.

    Parameters
    ----------
    forces : np.ndarray
        ``(T,)`` per-frame force signal.
    window : int, default 11
        Odd sliding-window size (clamped to the signal length).
    k : float, default 3.0
        Number of local standard deviations for the cutoff.

    Returns
    -------
    float
        Fraction in ``[0, 1]`` of flagged samples. Constant signals or
        signals shorter than 3 samples return ``0.0``.
    """
    f = np.asarray(forces, dtype=np.float64).ravel()
    n = f.size
    if n == 0:
        return 0.0
    if n < 3 or window < 3:
        return 0.0
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        sd = float(f.std())
        if sd <= 0:
            return 0.0
        return float(np.mean(np.abs(f - f.mean()) > k * sd))
    half = w // 2
    padded = np.pad(f, half, mode="edge")  # (n + 2*half,)
    # centered rolling mean / std via cumulative sums
    csum = np.concatenate(([0.0], np.cumsum(padded)))
    csum2 = np.concatenate(([0.0], np.cumsum(padded * padded)))
    rolls = csum[w:] - csum[:-w]  # length n
    rolls2 = csum2[w:] - csum2[:-w]
    mean = rolls / w
    var = np.clip(rolls2 / w - mean * mean, 0.0, None)
    sd = np.sqrt(var)
    flagged = np.where(sd > 0, np.abs(f - mean) > k * sd, False)
    return float(flagged.mean())


# --------------------------------------------------------------------------- #
# Module 2 — statistical process control (SPC)
# --------------------------------------------------------------------------- #
def spc_out_of_control_ratio(
    forces: np.ndarray,
    center: float | None = None,
    spread: float | None = None,
    k: float = 3.0,
) -> float:
    """X-bar control-chart out-of-control ratio for a force signal.

    Control limits are ``center +/- k * spread``. By default ``center`` and
    ``spread`` are the sequence's own mean and standard deviation; passing
    both lets you use a same-material global baseline instead.

    Parameters
    ----------
    forces : np.ndarray
        ``(T,)`` per-frame force signal.
    center : float, optional
        Control-chart center line (defaults to the signal mean).
    spread : float, optional
        Process spread (defaults to the signal standard deviation, ddof=0).
    k : float, default 3.0
        Control-limit multiplier (typically 3 for X-bar charts).

    Returns
    -------
    float
        Fraction in ``[0, 1]`` of samples outside the control limits. Empty
        signals or zero-spread signals return ``0.0``.
    """
    f = np.asarray(forces, dtype=np.float64).ravel()
    n = f.size
    if n == 0:
        return 0.0
    mu = float(f.mean()) if center is None else float(center)
    sd = float(f.std(ddof=0)) if spread is None else float(spread)
    if sd <= 0.0:
        return 0.0
    lo, hi = mu - k * sd, mu + k * sd
    return float(np.mean((f < lo) | (f > hi)))


# --------------------------------------------------------------------------- #
# single-pass assessment (quality metrics + anomaly feature inputs)
# --------------------------------------------------------------------------- #
@dataclass
class SequenceAssessment:
    """One-pass per-sequence result: quality metrics + anomaly feature inputs.

    Produced by :func:`assess_sequences`. Consumed by
    :func:`compute_quality_score` (the metrics) and
    :func:`tactile_qc.anomaly.extract_sequence_features` (the downscaled
    pixel stacks / force statistics), so a quality+anomaly run decodes each
    sequence's contact frames exactly once instead of twice.
    """

    sequence_id: str
    material: str
    snr: float
    drift_slope: float
    saturation_ratio: float
    force_anomaly_ratio: float  # NaN when no usable force signal
    spc_out_of_control_ratio: float  # NaN when no usable force signal
    downscaled: np.ndarray  # (T, img_size**2) float32 — PCA input rows
    n_frames: int
    frame_diff_mean: float
    force_stats: np.ndarray  # (4,) mean/std/skew/kurt of aligned ||F_ext||


@dataclass(frozen=True)
class _AssessParams:
    """Picklable parameter bundle for the assessment worker."""

    sat_tol: int = 5
    anomaly_window: int = 11
    anomaly_k: float = 3.0
    spc_k: float = 3.0
    img_size: int = 64


def _assess_one(seq: Any, p: _AssessParams) -> SequenceAssessment:
    """Decode one sequence's frames once and compute everything needed."""
    frames = seq.load_frames()
    gray = _to_gray(frames)
    drift = baseline_drift(gray)
    snr = snr_per_sequence(gray)
    sat = saturation_ratio(gray, tol=p.sat_tol)
    fmag = _align_force_to_frames(seq)
    if fmag is not None and fmag.size > 0:
        fa = force_anomaly_score(fmag, window=p.anomaly_window, k=p.anomaly_k)
        spc = spc_out_of_control_ratio(fmag, k=p.spc_k)
    else:
        fa = float("nan")
        spc = float("nan")
    return SequenceAssessment(
        sequence_id=seq.sequence_id,
        material=getattr(seq, "material_label", "Unknown"),
        snr=snr,
        drift_slope=drift.slope,
        saturation_ratio=sat,
        force_anomaly_ratio=fa,
        spc_out_of_control_ratio=spc,
        downscaled=_downscale_gray(gray, p.img_size),
        n_frames=int(gray.shape[0]),
        frame_diff_mean=_frame_diff_mean(gray),
        force_stats=_force_stats(fmag),
    )


def _assess_task(payload: tuple[Any, _AssessParams]) -> SequenceAssessment:
    """Top-level worker entry for :class:`ProcessPoolExecutor`."""
    seq, params = payload
    return _assess_one(seq, params)


def _can_parallel(seqs: list[Any], n_jobs: int | None) -> bool:
    """Parallel mode requires picklable, file-backed sequences."""
    return (
        n_jobs is not None
        and n_jobs > 1
        and len(seqs) > 1
        and all(hasattr(s, "frame_paths") for s in seqs)
    )


def _assess_parallel(
    seqs: list[Any],
    params: _AssessParams,
    n_jobs: int,
    progress: Callable[[int, int], None],
) -> list[SequenceAssessment]:
    """Run :func:`_assess_one` across a process pool, preserving order.

    Falls back to serial execution with a warning when the pool cannot be
    used (e.g. the calling script lacks an ``if __name__ == "__main__"``
    guard on Windows, or the pool workers die) — correctness beats speed.
    """
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    total = len(seqs)
    chunk = max(1, total // (n_jobs * 4))
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            gen = pool.map(
                _assess_task, ((s, params) for s in seqs), chunksize=chunk
            )
            out: list[SequenceAssessment] = []
            for i, res in enumerate(gen, start=1):
                out.append(res)
                progress(i, total)
            return out
    except (RuntimeError, BrokenProcessPool) as exc:
        warnings.warn(
            f"process pool unavailable ({exc}); falling back to serial "
            "assessment. Hint: on Windows/macOS call assess_sequences from "
            "inside an `if __name__ == '__main__':` block when using "
            "n_jobs > 1.",
            stacklevel=2,
        )
    out = []
    for i, seq in enumerate(seqs, start=1):
        out.append(_assess_one(seq, params))
        progress(i, total)
    return out


def assess_sequences(
    dataset: Iterable[Any],
    sat_tol: int = 5,
    anomaly_window: int = 11,
    anomaly_k: float = 3.0,
    spc_k: float = 3.0,
    img_size: int = 64,
    n_jobs: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[SequenceAssessment]:
    """Assess every sequence in a single pass over the decoded frames.

    This is the one-pass core of the pipeline: each sequence's contact
    frames are decoded exactly once and yield BOTH the quality metrics and
    the raw inputs needed by the anomaly stage (downscaled pixel stacks,
    force statistics, frame-diff proxy). Running quality + anomaly from one
    ``assess_sequences`` call halves the frame-decoding I/O of running the
    two stages separately.

    Parameters
    ----------
    dataset : Iterable
        Sequence objects exposing ``load_frames()`` (see
        :func:`compute_quality_score` for the duck-typed interface).
    sat_tol, anomaly_window, anomaly_k, spc_k :
        Forwarded to the underlying metric functions.
    img_size : int, default 64
        Downscale size for the anomaly-stage pixel features.
    n_jobs : int, optional
        Number of worker processes. ``None`` or ``<= 1`` runs serially in
        the current process (default). ``> 1`` uses a
        :class:`concurrent.futures.ProcessPoolExecutor`; sequences that are
        not file-backed (no ``frame_paths`` attribute, e.g. synthetic
        in-memory datasets) always run serially because they may not be
        picklable. If the pool cannot start (e.g. the calling script lacks
        an ``if __name__ == "__main__"`` guard on Windows), the run falls
        back to serial with a warning.
    progress : callable, optional
        ``progress(done, total)`` invoked from the parent process after
        every completed sequence (serial) or returned result (parallel).

    Returns
    -------
    list[SequenceAssessment]
        One assessment per sequence, in input order.
    """

    def _no_progress(done: int, total: int) -> None:  # pragma: no cover
        pass

    cb = progress if progress is not None else _no_progress
    seqs = list(dataset)
    params = _AssessParams(
        sat_tol=sat_tol,
        anomaly_window=anomaly_window,
        anomaly_k=anomaly_k,
        spc_k=spc_k,
        img_size=img_size,
    )
    if _can_parallel(seqs, n_jobs):
        return _assess_parallel(seqs, params, n_jobs, cb)
    out: list[SequenceAssessment] = []
    total = len(seqs)
    for i, seq in enumerate(seqs, start=1):
        out.append(_assess_one(seq, params))
        cb(i, total)
    return out


# --------------------------------------------------------------------------- #
# Module 3 — composite quality score
# --------------------------------------------------------------------------- #
def _align_force_to_frames(seq: Any) -> np.ndarray | None:
    """Align a sequence's force trace to its contact frames.

    The RCT force trace covers approach + contact (longer than the frame
    count); this maps each tactile frame (by its depth value) to the force
    sample with the nearest robot z-position and returns the external-force
    magnitude at those frames. Returns ``None`` when the sequence has no
    force data.
    """
    forces = getattr(seq, "forces", None)
    z = getattr(seq, "z_positions", None)
    depths = getattr(seq, "depth_values", None)
    if forces is None or z is None or depths is None:
        return None
    forces = np.asarray(forces, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    if forces.size == 0 or z.size == 0 or depths.size == 0:
        return None
    if forces.ndim == 1 or forces.shape[-1] < 3:
        mag = np.abs(forces if forces.ndim == 1 else forces[:, 0])
    else:
        mag = np.linalg.norm(forces[:, :3], axis=-1)  # ||(fx,fy,fz)||
    # nearest robot z for each frame depth
    idx = np.argmin(np.abs(z[:, None] - depths[None, :]), axis=0)
    return mag[idx]  # (n_frames,)


def _minmax(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Min-max normalize to [0,1]; NaNs preserved; zero-range -> all best."""
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.full(x.shape, np.nan)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo < 1e-12:
        return np.ones(x.shape)  # no discrimination -> treat as best
    norm = (x - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return norm if higher_is_better else (1.0 - norm)


def compute_quality_score(
    dataset: Iterable[Any] | None = None,
    weights: Mapping[str, float] | None = None,
    sat_tol: int = 5,
    anomaly_window: int = 11,
    anomaly_k: float = 3.0,
    spc_k: float = 3.0,
    n_jobs: int | None = None,
    assessments: list[SequenceAssessment] | None = None,
) -> pd.DataFrame:
    """Compute a composite 0-100 quality score for every sequence.

    Each sequence is scored on five axes, min-max normalized across the
    dataset (so the score is *relative*), and combined with configurable
    weights::

        quality_score = 100 * sum(w_i * norm_i) / sum(w_i available)

    Sequences missing force data keep their image-metric score (the force
    components are NaN and excluded from both numerator and denominator).

    Parameters
    ----------
    dataset : Iterable, optional
        Anything iterable of sequence objects. Each sequence must expose
        ``load_frames()`` -> (T,H,W[,C]) array, ``sequence_id``,
        ``material_label``, ``forces`` (T,6), ``z_positions`` (T,) and
        ``depth_values`` (T,) — i.e. :class:`tactile_qc.io.TactileSequence`.
        May be omitted when ``assessments`` is given.
    weights : Mapping[str, float], optional
        Component weights with keys in
        ``{snr, drift, saturation, force_anomaly, spc}``.
        Defaults to :data:`DEFAULT_WEIGHTS`.
    sat_tol, anomaly_window, anomaly_k, spc_k :
        Forwarded to the underlying metric functions (ignored when
        ``assessments`` is given — they were already applied there).
    n_jobs : int, optional
        Worker processes for :func:`assess_sequences` (serial by default).
    assessments : list of SequenceAssessment, optional
        Reuse a previous :func:`assess_sequences` run instead of decoding
        the frames again (single-pass quality + anomaly pipelines).

    Returns
    -------
    pandas.DataFrame
        Columns: ``sequence_id, material, quality_score, snr, drift_slope,
        saturation_ratio, force_anomaly_ratio, spc_out_of_control_ratio``.
        ``quality_score`` is in ``[0, 100]`` (NaN if no metric available).
    """
    if assessments is None:
        if dataset is None:
            raise ValueError("either dataset or assessments must be provided")
        assessments = assess_sequences(
            dataset,
            sat_tol=sat_tol,
            anomaly_window=anomaly_window,
            anomaly_k=anomaly_k,
            spc_k=spc_k,
            n_jobs=n_jobs,
        )
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update({k: float(v) for k, v in weights.items()})

    rows: list[dict[str, Any]] = [
        {
            "sequence_id": a.sequence_id,
            "material": a.material,
            "snr": a.snr,
            "drift_slope": a.drift_slope,
            "saturation_ratio": a.saturation_ratio,
            "force_anomaly_ratio": a.force_anomaly_ratio,
            "spc_out_of_control_ratio": a.spc_out_of_control_ratio,
        }
        for a in assessments
    ]

    cols = [
        "sequence_id",
        "material",
        "quality_score",
        "snr",
        "drift_slope",
        "saturation_ratio",
        "force_anomaly_ratio",
        "spc_out_of_control_ratio",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)

    norm = {
        "snr": _minmax(df["snr"].to_numpy(), higher_is_better=True),
        "drift": _minmax(
            np.abs(df["drift_slope"].to_numpy()), higher_is_better=False
        ),
        "saturation": _minmax(
            df["saturation_ratio"].to_numpy(), higher_is_better=False
        ),
        "force_anomaly": _minmax(
            df["force_anomaly_ratio"].to_numpy(), higher_is_better=False
        ),
        "spc": _minmax(
            df["spc_out_of_control_ratio"].to_numpy(), higher_is_better=False
        ),
    }

    scores: list[float] = []
    for i in range(len(df)):
        num = 0.0
        den = 0.0
        for key in ("snr", "drift", "saturation", "force_anomaly", "spc"):
            v = norm[key][i]
            if np.isnan(v):
                continue
            num += w[key] * float(v)
            den += w[key]
        s = (num / den * 100.0) if den > 0 else float("nan")
        scores.append(round(float(np.clip(s, 0.0, 100.0)), 4))
    df["quality_score"] = scores

    return df[cols]
