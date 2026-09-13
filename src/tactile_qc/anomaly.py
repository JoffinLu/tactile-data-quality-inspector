"""Unsupervised anomaly detection for RCT tactile sequences.

Pipeline
--------
1. **Feature extraction** — :func:`extract_sequence_features`
   Builds a ``(n_sequences, 26)`` feature matrix per sequence:
     * 20 PCA components of the grayscale frame pixels (frames are downscaled
       to ``img_size x img_size`` before PCA to keep memory bounded)
     * 4 force statistics of the per-frame ||F_ext||: mean / std / skew /
       kurtosis (NaN when a sequence has no force trace; imputed with the
       column mean before modelling)
     * 1 sequence length (number of contact frames)
     * 1 mean frame-to-frame pixel difference (an optical-flow proxy)

2. **Isolation Forest** — :func:`isolation_forest_anomaly`
   :class:`sklearn.ensemble.IsolationForest`, unsupervised, ``contamination``
   configurable (default 0.05). Outputs ``-1`` for anomaly / ``1`` for normal
   plus the ``decision_function`` score (higher = more normal).

3. **Mahalanobis distance** — :func:`mahalanobis_anomaly`
   Covariance-based distance, thresholded at the chi-square quantile
   ``(1 - contamination, df=n_features)``. Same label convention.

4. **Stratified evaluation** — :func:`stratified_anomaly_table`
   Per-material-category anomaly ratio with a
   :func:`scipy.stats.chi2_contingency` test for category differences.

5. **Method comparison** — :func:`compare_methods`
   Agreement between IsolationForest and Mahalanobis via Cohen's kappa.

Shape assumptions
-----------------
``frames`` : ``(T, H, W)`` grayscale or ``(T, H, W, C)`` (reduced to gray).
``forces`` : ``(T,)`` per-frame force magnitude (aligned to frames by
:func:`tactile_qc.quality._align_force_to_frames`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from .quality import _align_force_to_frames

#: force statistical feature names
_FORCE_FEATURES = ("force_mean", "force_std", "force_skew", "force_kurt")
#: extra (non-PCA) feature names
_EXTRA_FEATURES = ("n_frames", "frame_diff_mean")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _to_gray(frames: np.ndarray) -> np.ndarray:
    """Coerce ``frames`` to a float64 grayscale array of shape (T, H, W)."""
    arr = np.asarray(frames)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    elif arr.ndim != 3:
        raise ValueError(f"frames must be (T,H,W) or (T,H,W,C); got {arr.shape}")
    return arr.astype(np.float64)


def _downscale_gray(gray: np.ndarray, size: int) -> np.ndarray:
    """Downscale a (T,H,W) grayscale stack to (T, size*size) float32."""
    from PIL import Image

    out = np.empty((gray.shape[0], size * size), dtype=np.float32)
    for i, fr in enumerate(gray):
        im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8), mode="L")
        im = im.resize((size, size))
        out[i] = np.asarray(im, dtype=np.float32).ravel()
    return out


def _force_stats(force: Optional[np.ndarray]) -> np.ndarray:
    """``[mean, std, skew, kurt]`` of a force signal; NaNs if unavailable."""
    if force is None or force.size == 0:
        return np.full(4, np.nan)
    x = np.asarray(force, dtype=np.float64).ravel()
    std = float(np.std(x)) if x.size > 1 else 0.0
    if x.size > 2 and np.std(x) > 0:
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
# 1. feature extraction
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyFeatures:
    """Extracted feature matrix + metadata for a dataset of sequences."""

    X: np.ndarray  # (n_seq, d) standardized
    sequence_ids: list[str]
    materials: list[str]
    feature_names: list[str]
    n_pca: int = 0
    pca: Optional[IncrementalPCA] = None
    scaler: Optional[StandardScaler] = None


def extract_sequence_features(
    dataset: Iterable[Any],
    pca_components: int = 20,
    img_size: int = 64,
    random_state: int = 42,
) -> AnomalyFeatures:
    """Build the per-sequence feature matrix.

    Parameters
    ----------
    dataset : Iterable
        Iterable of sequence objects (see
        :func:`tactile_qc.quality.compute_quality_score` for the duck-typed
        interface — ``load_frames()``, ``sequence_id``, ``material_label``,
        ``forces``, ``z_positions``, ``depth_values``).
    pca_components : int, default 20
        Number of PCA dimensions. Clamped to ``min(n_frames_total, img_size**2)``.
    img_size : int, default 64
        Frames are downscaled to ``img_size x img_size`` grayscale before PCA.
    random_state : int, default 42

    Returns
    -------
    AnomalyFeatures
        ``X`` is standardized (zero mean / unit variance). Force statistics
        that are NaN (no force trace) are imputed with the column mean before
        standardization.
    """
    seq_ids: list[str] = []
    materials: list[str] = []
    frame_stacks: list[np.ndarray] = []
    force_rows: list[np.ndarray] = []
    len_list: list[int] = []
    diff_list: list[float] = []

    for seq in dataset:
        raw = seq.load_frames()
        gray = _to_gray(raw)
        ds = _downscale_gray(gray, img_size)
        frame_stacks.append(ds)
        seq_ids.append(seq.sequence_id)
        materials.append(getattr(seq, "material_label", "Unknown"))
        force_rows.append(_force_stats(_align_force_to_frames(seq)))
        len_list.append(int(gray.shape[0]))
        diff_list.append(_frame_diff_mean(gray))

    all_frames = np.concatenate(frame_stacks, axis=0)  # (sum_T, img_size**2)
    n_comp = min(pca_components, all_frames.shape[0], all_frames.shape[1])
    pca = IncrementalPCA(
        n_components=n_comp,
        batch_size=min(512, all_frames.shape[0]),
    )
    pca.fit(all_frames)

    pca_rows = [pca.transform(ds).mean(axis=0) for ds in frame_stacks]
    X_pca = np.stack(pca_rows)  # (n_seq, n_comp)
    force_arr = np.stack(force_rows)  # (n_seq, 4)
    extra = np.column_stack(
        [np.asarray(len_list, dtype=float), np.asarray(diff_list, dtype=float)]
    )
    X_raw = np.hstack([X_pca, force_arr, extra]).astype(np.float64)

    # impute NaN force features with column means
    col_mean = np.nanmean(X_raw, axis=0)
    nan_idx = np.where(np.isnan(X_raw))
    X_raw[nan_idx] = np.take(col_mean, nan_idx[1])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    names = (
        [f"pca_{i}" for i in range(n_comp)]
        + list(_FORCE_FEATURES)
        + list(_EXTRA_FEATURES)
    )
    return AnomalyFeatures(
        X=X,
        sequence_ids=seq_ids,
        materials=materials,
        feature_names=names,
        n_pca=n_comp,
        pca=pca,
        scaler=scaler,
    )


# --------------------------------------------------------------------------- #
# 2 & 3. detectors
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    """Labels (-1 anomaly / 1 normal) and decision scores for a detector."""

    labels: np.ndarray  # (n,)
    scores: np.ndarray  # (n,) higher = more normal


def isolation_forest_anomaly(
    X: np.ndarray, contamination: float = 0.05, random_state: int = 42
) -> AnomalyResult:
    """Unsupervised IsolationForest anomaly detection.

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` feature matrix (standardized recommended).
    contamination : float, default 0.05
        Expected fraction of anomalies.
    random_state : int, default 42

    Returns
    -------
    AnomalyResult
        ``labels`` in ``{-1, 1}``; ``scores`` from ``decision_function``
        (higher = more normal).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return AnomalyResult(
            np.ones(n, dtype=int), np.zeros(n, dtype=float)
        )
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    iso.fit(X)
    return AnomalyResult(
        labels=iso.predict(X), scores=iso.decision_function(X)
    )


def mahalanobis_anomaly(
    X: np.ndarray, contamination: float = 0.05
) -> AnomalyResult:
    """Covariance-based Mahalanobis-distance anomaly detection.

    Distance ``d2 = (x - mu) Σ^-1 (x - mu)``; a sample is flagged when
    ``d2`` exceeds the chi-square quantile ``chi2.ppf(1 - contamination, df=d)``.

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` feature matrix.
    contamination : float, default 0.05

    Returns
    -------
    AnomalyResult
        ``labels`` in ``{-1, 1}``; ``scores = -d2`` (higher = more normal,
        consistent with IsolationForest).
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if n < 2:
        return AnomalyResult(np.ones(n, dtype=int), np.zeros(n, dtype=float))
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    if cov.shape != (p, p):
        cov = np.eye(p)
    # regularize an ill-conditioned covariance
    try:
        if np.linalg.cond(cov) > 1e10:
            cov = cov + np.eye(p) * 1e-6
    except np.linalg.LinAlgError:
        cov = cov + np.eye(p) * 1e-6
    inv = np.linalg.pinv(cov)
    diff = X - mu
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    d2 = np.clip(d2, 0.0, None)
    if n > p and p >= 1:
        thresh = float(stats.chi2.ppf(1.0 - contamination, df=p))
    else:
        thresh = float(np.quantile(d2, 1.0 - contamination))
    labels = np.where(d2 > thresh, -1, 1)
    return AnomalyResult(labels=labels, scores=-d2)


# --------------------------------------------------------------------------- #
# 4. stratified evaluation
# --------------------------------------------------------------------------- #
@dataclass
class StratifiedEval:
    """Per-category anomaly rates + chi-square independence test."""

    table: pd.DataFrame
    chi2: float
    p_value: float
    dof: int


def stratified_anomaly_table(
    labels: np.ndarray, materials: Sequence[str]
) -> StratifiedEval:
    """Per-material-category anomaly ratio + chi2_contingency test.

    Parameters
    ----------
    labels : np.ndarray
        Detector labels (``-1`` = anomaly, ``1`` = normal).
    materials : sequence of str
        Category per sequence.

    Returns
    -------
    StratifiedEval
        ``table`` columns: ``material, n, n_anomaly, n_normal,
        anomaly_ratio`` (sorted by ratio desc). ``chi2`` / ``p_value`` /
        ``dof`` from the independence test.
    """
    lab = np.asarray(labels)
    is_anom = lab == -1
    df = pd.DataFrame({"material": list(materials), "anomaly": is_anom})
    g = (
        df.groupby("material")["anomaly"]
        .agg(n="size", n_anomaly="sum")
        .reset_index()
    )
    g["n_normal"] = g["n"] - g["n_anomaly"]
    g["anomaly_ratio"] = (g["n_anomaly"] / g["n"]).round(4)
    g = g.sort_values("anomaly_ratio", ascending=False).reset_index(drop=True)

    contingency = g[["n_anomaly", "n_normal"]].to_numpy()
    # chi2_contingency needs a >=2x2 table; guard degenerate cases
    if contingency.shape[0] >= 2 and contingency.sum() > 0:
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        chi2, p, dof = float(chi2), float(p), int(dof)
    else:
        chi2, p, dof = float("nan"), float("nan"), 0
    return StratifiedEval(table=g, chi2=chi2, p_value=p, dof=dof)


# --------------------------------------------------------------------------- #
# 5. method comparison
# --------------------------------------------------------------------------- #
@dataclass
class MethodComparison:
    """Agreement between two anomaly detectors."""

    cohen_kappa: float
    agreement: float
    n_both_anomaly: int
    n_both_normal: int
    n_disagree: int


def compare_methods(
    labels_a: np.ndarray, labels_b: np.ndarray
) -> MethodComparison:
    """Compare two label vectors via Cohen's kappa.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray
        Label vectors in ``{-1, 1}`` (same length).

    Returns
    -------
    MethodComparison
        ``cohen_kappa`` (``-1..1``), raw ``agreement`` fraction and confusion counts.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    kappa = float(cohen_kappa_score(a, b)) if len(a) else float("nan")
    return MethodComparison(
        cohen_kappa=kappa,
        agreement=float(np.mean(a == b)) if len(a) else float("nan"),
        n_both_anomaly=int(np.sum((a == -1) & (b == -1))),
        n_both_normal=int(np.sum((a == 1) & (b == 1))),
        n_disagree=int(np.sum(a != b)),
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyReport:
    """Full anomaly-detection result for a dataset of sequences."""

    per_sequence: pd.DataFrame
    stratified: StratifiedEval
    comparison: MethodComparison
    features: AnomalyFeatures


def detect_anomalies(
    dataset: Iterable[Any],
    contamination: float = 0.05,
    pca_components: int = 20,
    img_size: int = 64,
    random_state: int = 42,
) -> AnomalyReport:
    """End-to-end anomaly detection: features -> IF + Mahalanobis -> eval.

    Parameters
    ----------
    dataset : Iterable
        Iterable of sequence objects (see
        :func:`extract_sequence_features`).
    contamination : float, default 0.05
        Expected anomaly fraction (forwarded to both detectors).
    pca_components, img_size, random_state :
        Forwarded to :func:`extract_sequence_features` /
        :func:`isolation_forest_anomaly`.

    Returns
    -------
    AnomalyReport
        ``per_sequence`` DataFrame columns: ``sequence_id, material,
        iso_label, iso_score, mahal_label, mahal_score``;
        ``stratified`` (IsolationForest) ; ``comparison`` (IF vs Mahalanobis).
    """
    feats = extract_sequence_features(
        dataset,
        pca_components=pca_components,
        img_size=img_size,
        random_state=random_state,
    )
    iso = isolation_forest_anomaly(
        feats.X, contamination=contamination, random_state=random_state
    )
    mah = mahalanobis_anomaly(feats.X, contamination=contamination)
    strat = stratified_anomaly_table(iso.labels, feats.materials)
    cmp = compare_methods(iso.labels, mah.labels)
    per_seq = pd.DataFrame(
        {
            "sequence_id": feats.sequence_ids,
            "material": feats.materials,
            "iso_label": iso.labels,
            "iso_score": np.round(iso.scores, 6),
            "mahal_label": mah.labels,
            "mahal_score": np.round(mah.scores, 6),
        }
    )
    return AnomalyReport(
        per_sequence=per_seq, stratified=strat, comparison=cmp, features=feats
    )
