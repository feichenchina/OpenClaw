"""Request router – the primary entry point for PD scheduling."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .health import HealthChecker
from .models import (
    InstanceRole,
    PDInstance,
    ScheduleRequest,
    ScheduleResult,
    SchedulerConfig,
)
from .scheduler import PDScheduler

logger = logging.getLogger(__name__)


def _http_post(url: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Perform a simple HTTP POST with JSON body and parse JSON response."""
    data = json.dumps(body).encode()
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 – URL is from trusted config
        return json.loads(resp.read().decode())


class PDRouter:
    """Routes inference requests through prefill → decode pipeline.

    Usage::

        router = PDRouter(config)
        # Register instances (typically from config)
        router.register_instance(PDInstance(...))
        # Route a request
        result = router.route_request({"prompt": "Hello", ...})
    """

    def __init__(
        self,
        config: SchedulerConfig,
        http_post: Optional[Callable] = None,
        http_get: Optional[Callable] = None,
    ) -> None:
        self._config = config
        self._scheduler = PDScheduler(config)
        self._health = HealthChecker(
            instances=[],
            check_interval=config.health_check_interval,
            check_timeout=config.health_check_timeout,
            http_get=http_get,
        )
        self._http_post = http_post or _http_post

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def register_instance(self, instance: PDInstance) -> None:
        self._health.add_instance(instance)

    def unregister_instance(self, instance_id: str) -> Optional[PDInstance]:
        return self._health.remove_instance(instance_id)

    def drain_instance(self, instance_id: str) -> bool:
        return self._health.drain_instance(instance_id)

    def get_instances(self, role: Optional[InstanceRole] = None) -> List[PDInstance]:
        """Return registered instances, optionally filtered by role."""
        instances = list(self._health.instances.values())
        if role is not None:
            instances = [i for i in instances if i.role == role]
        return instances

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def run_health_checks(self) -> Dict[str, str]:
        """Run health checks and return {instance_id: status}."""
        results = self._health.check_all()
        return {k: v.value for k, v in results.items()}

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    def route_request(
        self,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> ScheduleResult:
        """Route a full inference request through the P→D pipeline.

        1. Schedule the prefill phase to a P-instance.
        2. Forward the request to the chosen P-instance.
        3. On prefill completion, schedule the decode phase to a D-instance.
        4. Forward to the chosen D-instance.

        For simplicity this method runs synchronously.  In production,
        each step would be async with retries.
        """
        rid = request_id or uuid.uuid4().hex[:12]

        # --- Step 1: Schedule prefill ---
        prefill_req = ScheduleRequest(
            request_id=rid,
            phase=InstanceRole.PREFILL,
            payload=payload,
        )
        prefill_candidates = self._health.get_healthy_instances(
            role=InstanceRole.PREFILL.value,
        )
        prefill_result = self._scheduler.schedule(prefill_req, prefill_candidates)

        if not prefill_result.success:
            logger.error("Prefill scheduling failed for %s: %s", rid, prefill_result.error)
            return prefill_result

        # --- Step 2: Forward to prefill instance ---
        prefill_inst = prefill_result.instance
        assert prefill_inst is not None
        try:
            prefill_resp = self._http_post(
                f"{prefill_inst.address}/v1/completions",
                body={**payload, "phase": "prefill"},
                timeout=self._config.health_check_timeout * 10,
            )
        except (URLError, OSError) as exc:
            prefill_inst.pending_requests = max(0, prefill_inst.pending_requests - 1)
            return ScheduleResult(
                request_id=rid,
                instance=prefill_inst,
                success=False,
                error=f"Prefill forward failed: {exc}",
            )
        finally:
            prefill_inst.pending_requests = max(0, prefill_inst.pending_requests - 1)

        kv_token = prefill_resp.get("kv_cache_token")

        # --- Step 3: Schedule decode ---
        decode_req = ScheduleRequest(
            request_id=rid,
            phase=InstanceRole.DECODE,
            payload=payload,
            prefill_instance_id=prefill_inst.instance_id,
            kv_cache_token=kv_token,
        )
        decode_candidates = self._health.get_healthy_instances(
            role=InstanceRole.DECODE.value,
        )
        decode_result = self._scheduler.schedule(decode_req, decode_candidates)

        if not decode_result.success:
            logger.error("Decode scheduling failed for %s: %s", rid, decode_result.error)
            return decode_result

        # --- Step 4: Forward to decode instance ---
        decode_inst = decode_result.instance
        assert decode_inst is not None
        try:
            self._http_post(
                f"{decode_inst.address}/v1/completions",
                body={
                    **payload,
                    "phase": "decode",
                    "kv_cache_token": kv_token,
                    "prefill_instance": prefill_inst.instance_id,
                },
                timeout=self._config.health_check_timeout * 10,
            )
        except (URLError, OSError) as exc:
            decode_inst.pending_requests = max(0, decode_inst.pending_requests - 1)
            return ScheduleResult(
                request_id=rid,
                instance=decode_inst,
                success=False,
                error=f"Decode forward failed: {exc}",
            )
        finally:
            decode_inst.pending_requests = max(0, decode_inst.pending_requests - 1)

        return ScheduleResult(
            request_id=rid,
            instance=decode_inst,
            success=True,
            latency_ms=prefill_result.latency_ms + decode_result.latency_ms,
        )
