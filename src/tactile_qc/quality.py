"""Statistical quality metrics for tactile data."""

from __future__ import annotations


def compute_quality_metrics(data):
    """Compute statistical quality metrics for tactile data.

    Planned metric families (stage 2):
    - per-channel statistics: mean / variance / dynamic range
    - saturation & dead-pixel rates
    - noise floor estimates
    - temporal stability (drift, frame-to-frame delta)
    """
    raise NotImplementedError("Quality metrics are implemented in stage 2")
