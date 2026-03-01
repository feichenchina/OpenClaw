"""Core scheduling strategies for PD disaggregation."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from .models import (
    InstanceRole,
    PDInstance,
    ScheduleRequest,
    ScheduleResult,
    ScheduleStrategy,
    SchedulerConfig,
)

logger = logging.getLogger(__name__)


class PDScheduler:
    """Dynamic scheduler that routes requests to P/D instances.

    Supports multiple scheduling strategies:
    - round_robin: Rotate through available instances
    - least_load: Pick the instance with the fewest pending requests
    - weighted: Score instances by load, KV-cache usage, and latency
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._rr_counters: dict[str, int] = {
            InstanceRole.PREFILL.value: 0,
            InstanceRole.DECODE.value: 0,
        }

    @property
    def strategy(self) -> ScheduleStrategy:
        return self._config.strategy

    @strategy.setter
    def strategy(self, value: ScheduleStrategy) -> None:
        self._config.strategy = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        request: ScheduleRequest,
        candidates: List[PDInstance],
    ) -> ScheduleResult:
        """Pick the best instance for *request* from *candidates*.

        Parameters
        ----------
        request:
            The incoming scheduling request (prefill or decode phase).
        candidates:
            Pre-filtered list of **healthy** instances whose role matches
            ``request.phase``.

        Returns
        -------
        ScheduleResult with the chosen instance, or ``success=False``
        if no candidate is available.
        """
        start = time.time()

        available = [c for c in candidates if c.is_available]
        if not available:
            return ScheduleResult(
                request_id=request.request_id,
                instance=None,
                success=False,
                error="No available instances for role " + request.phase.value,
            )

        chosen = self._pick(request.phase, available)
        if chosen is None:
            return ScheduleResult(
                request_id=request.request_id,
                instance=None,
                success=False,
                error="Scheduling strategy returned no instance",
            )

        chosen.pending_requests += 1
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            "Scheduled request %s → %s (%s) [%.2f ms]",
            request.request_id,
            chosen.instance_id,
            request.phase.value,
            elapsed_ms,
        )

        return ScheduleResult(
            request_id=request.request_id,
            instance=chosen,
            success=True,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _pick(self, role: InstanceRole, candidates: List[PDInstance]) -> Optional[PDInstance]:
        strategy = self._config.strategy
        if strategy == ScheduleStrategy.ROUND_ROBIN:
            return self._round_robin(role, candidates)
        if strategy == ScheduleStrategy.LEAST_LOAD:
            return self._least_load(candidates)
        if strategy == ScheduleStrategy.WEIGHTED:
            return self._weighted(candidates)
        # Fallback
        return self._least_load(candidates)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _round_robin(self, role: InstanceRole, candidates: List[PDInstance]) -> PDInstance:
        key = role.value
        idx = self._rr_counters.get(key, 0) % len(candidates)
        self._rr_counters[key] = idx + 1
        return candidates[idx]

    @staticmethod
    def _least_load(candidates: List[PDInstance]) -> PDInstance:
        return min(candidates, key=lambda c: c.pending_requests)

    def _weighted(self, candidates: List[PDInstance]) -> PDInstance:
        """Score each candidate and pick the one with the lowest score."""

        def _score(inst: PDInstance) -> float:
            load_score = inst.pending_requests
            kv_score = inst.kv_cache_usage  # 0.0 – 1.0
            # Use time since last heartbeat as a rough latency proxy
            latency_score = time.time() - inst.last_heartbeat
            return (
                self._config.load_weight * load_score
                + self._config.kv_cache_weight * kv_score
                + self._config.latency_weight * latency_score
            )

        return min(candidates, key=_score)
