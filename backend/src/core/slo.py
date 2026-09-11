"""Small in-process SLO recorder for request-latency evidence.

The audit table remains the durable source of truth. This bounded recorder makes the
current P85/P95 visible through /health without retaining user content.
"""

from collections import deque
from statistics import quantiles
from threading import Lock
from typing import Any


class SloRecorder:
    def __init__(self, max_samples: int = 500):
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = Lock()

    def observe(self, elapsed_ms: float) -> None:
        with self._lock:
            self._samples.append(max(0.0, elapsed_ms))

    def snapshot(self, p85_target_ms: float, p95_target_ms: float, minimum_samples: int) -> dict[str, Any]:
        with self._lock:
            values = sorted(self._samples)

        def percentile(percent: float) -> float | None:
            if not values:
                return None
            if len(values) == 1:
                return round(values[0], 2)
            return round(quantiles(values, n=100, method="inclusive")[int(percent) - 1], 2)

        p85 = percentile(85)
        p95 = percentile(95)
        sufficient = len(values) >= minimum_samples

        return {
            "sample_count": len(values),
            "minimum_samples": minimum_samples,
            "p85_ms": p85,
            "p95_ms": p95,
            "p85_target_ms": p85_target_ms,
            "p95_target_ms": p95_target_ms,
            "sufficient_sample_size": sufficient,
            "p85_within_slo": sufficient and p85 is not None and p85 <= p85_target_ms,
            "p95_within_slo": sufficient and p95 is not None and p95 <= p95_target_ms,
        }


slo_recorder = SloRecorder()
