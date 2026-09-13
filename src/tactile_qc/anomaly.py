"""Anomaly detection for tactile data."""

from __future__ import annotations


def detect_anomalies(data, **kwargs):
    """Detect anomalous frames / channels in tactile data.

    Planned strategies (later stage):
    - statistical outlier detection (z-score / modified z-score)
    - unsupervised models (IsolationForest, etc.)
    - temporal anomaly detection
    """
    raise NotImplementedError("Anomaly detection arrives in a later stage")
