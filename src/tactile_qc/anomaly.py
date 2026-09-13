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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from .quality import SequenceAssessment

#: force statistical feature names
_FORCE_FEATURES = ("force_mean", "force_std", "force_skew", "force_kurt")
#: extra (non-PCA) feature names
_EXTRA_FEATURES = ("n_frames", "frame_diff_mean")


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
    pca: IncrementalPCA | None = None
    scaler: StandardScaler | None = None


def _pca_batches(
    stacks: list[np.ndarray], batch_size: int, min_rows: int
) -> Iterable[np.ndarray]:
    """Yield ``batch_size``-row chunks of the virtual concatenation of *stacks*.

    Batch boundaries sit at exact multiples of ``batch_size`` — matching what
    ``IncrementalPCA.fit`` would slice internally on the concatenated matrix —
    but the concatenation is never materialised, so peak memory holds one
    batch instead of a full second copy of all stacks. A final short chunk is
    merged into the previous batch so every yielded batch has at least
    ``min_rows`` rows (``partial_fit`` requires ``n_samples >= n_components``).
    """
    batches: list[list[np.ndarray]] = []
    cur: list[np.ndarray] = []
    cur_rows = 0
    for s in stacks:
        view = s
        while view.shape[0]:
            if cur_rows >= batch_size:
                batches.append(cur)
                cur, cur_rows = [], 0
            take = batch_size - cur_rows
            part, view = view[:take], view[take:]
            cur.append(part)
            cur_rows += part.shape[0]
    if cur_rows:
        batches.append(cur)
    if len(batches) >= 2 and sum(b.shape[0] for b in batches[-1]) < min_rows:
        batches[-2].extend(batches.pop())
    for parts in batches:
        yield np.concatenate(parts, axis=0)


def extract_sequence_features(
    dataset: Iterable[Any] | None = None,
    pca_components: int = 20,
    img_size: int = 64,
    random_state: int = 42,
    n_jobs: int | None = None,
    assessments: list[SequenceAssessment] | None = None,
) -> AnomalyFeatures:
    """Build the per-sequence feature matrix.

    Parameters
    ----------
    dataset : Iterable, optional
        Iterable of sequence objects (see
        :func:`tactile_qc.quality.compute_quality_score` for the duck-typed
        interface — ``load_frames()``, ``sequence_id``, ``material_label``,
        ``forces``, ``z_positions``, ``depth_values``). May be omitted when
        ``assessments`` is given.
    pca_components : int, default 20
        Number of PCA dimensions. Clamped to ``min(n_frames_total, img_size**2)``.
    img_size : int, default 64
        Expected downscale size of the assessments. Ignored when
        ``dataset`` is given (the assessments are created at this size);
        when ``assessments`` is given it must match their size or a
        ``ValueError`` is raised.
    random_state : int, default 42
        Kept for API stability (IncrementalPCA is deterministic).
    n_jobs : int, optional
        Worker processes for :func:`tactile_qc.quality.assess_sequences`
        (serial by default; only used when ``dataset`` is given).
    assessments : list of SequenceAssessment, optional
        Reuse a previous :func:`tactile_qc.quality.assess_sequences` run —
        the frames are not decoded a second time.

    Returns
    -------
    AnomalyFeatures
        ``X`` is standardized (zero mean / unit variance). Force statistics
        that are NaN (no force trace) are imputed with the column mean before
        standardization.
    """
    from .quality import assess_sequences

    if assessments is None:
        if dataset is None:
            raise ValueError("either dataset or assessments must be provided")
        assessments = assess_sequences(
            dataset, img_size=img_size, n_jobs=n_jobs
        )
    if not assessments:
        raise ValueError("no sequences to analyse — dataset is empty")

    seq_ids = [a.sequence_id for a in assessments]
    materials = [a.material for a in assessments]
    frame_stacks = [a.downscaled for a in assessments]
    force_arr = np.stack([a.force_stats for a in assessments])  # (n_seq, 4)
    extra = np.column_stack(
        [
            np.asarray([a.n_frames for a in assessments], dtype=float),
            np.asarray([a.frame_diff_mean for a in assessments], dtype=float),
        ]
    )

    n_feat = frame_stacks[0].shape[1]
    if img_size is not None and n_feat and n_feat != img_size * img_size:
        from math import isqrt

        raise ValueError(
            f"assessments were computed at img_size={isqrt(n_feat)} but "
            f"img_size={img_size} was requested; re-run assess_sequences "
            "with the matching size"
        )

    total_frames = sum(s.shape[0] for s in frame_stacks)
    n_comp = min(pca_components, total_frames, n_feat)
    if n_comp < 1:
        raise ValueError(
            "no decodable frames found — cannot fit PCA (check the "
            "dataset's contact_frames/ images)"
        )
    batch_size = min(512, total_frames)
    pca = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
    for batch in _pca_batches(frame_stacks, batch_size, min_rows=n_comp):
        pca.partial_fit(batch)

    pca_rows = [pca.transform(ds).mean(axis=0) for ds in frame_stacks]
    X_pca = np.stack(pca_rows)  # (n_seq, n_comp)
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
    if n > p >= 1:
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
    dataset: Iterable[Any] | None = None,
    contamination: float = 0.05,
    pca_components: int = 20,
    img_size: int = 64,
    random_state: int = 42,
    n_jobs: int | None = None,
    assessments: list[SequenceAssessment] | None = None,
) -> AnomalyReport:
    """End-to-end anomaly detection: features -> IF + Mahalanobis -> eval.

    Parameters
    ----------
    dataset : Iterable, optional
        Iterable of sequence objects (see
        :func:`extract_sequence_features`). May be omitted when
        ``assessments`` is given.
    contamination : float, default 0.05
        Expected anomaly fraction (forwarded to both detectors).
    pca_components, img_size, random_state :
        Forwarded to :func:`extract_sequence_features` /
        :func:`isolation_forest_anomaly`.
    n_jobs : int, optional
        Worker processes for :func:`tactile_qc.quality.assess_sequences`
        (serial by default; only used when ``dataset`` is given).
    assessments : list of SequenceAssessment, optional
        Reuse a previous :func:`tactile_qc.quality.assess_sequences` run —
        the frames are not decoded a second time.

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
        n_jobs=n_jobs,
        assessments=assessments,
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
