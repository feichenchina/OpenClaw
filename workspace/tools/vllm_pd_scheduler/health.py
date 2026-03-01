"""Health checking for vLLM PD instances."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import InstanceStatus, PDInstance

logger = logging.getLogger(__name__)


def _http_get(url: str, timeout: float) -> Dict[str, Any]:
    """Perform a simple HTTP GET and parse JSON response."""
    import json

    req = Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 – URL is from trusted config
        return json.loads(resp.read().decode())


class HealthChecker:
    """Monitors health of vLLM instances via their /health endpoint."""

    def __init__(
        self,
        instances: List[PDInstance],
        check_interval: float = 5.0,
        check_timeout: float = 2.0,
        on_status_change: Optional[Callable[[PDInstance, InstanceStatus, InstanceStatus], None]] = None,
        http_get: Optional[Callable] = None,
    ) -> None:
        self._instances = {inst.instance_id: inst for inst in instances}
        self._check_interval = check_interval
        self._check_timeout = check_timeout
        self._on_status_change = on_status_change
        self._running = False
        self._http_get = http_get or _http_get

    @property
    def instances(self) -> Dict[str, PDInstance]:
        return dict(self._instances)

    def add_instance(self, instance: PDInstance) -> None:
        self._instances[instance.instance_id] = instance

    def remove_instance(self, instance_id: str) -> Optional[PDInstance]:
        return self._instances.pop(instance_id, None)

    def check_instance(self, instance: PDInstance) -> InstanceStatus:
        """Check health of a single instance synchronously."""
        try:
            data = self._http_get(
                f"{instance.address}/health",
                timeout=self._check_timeout,
            )
            old_status = instance.status
            instance.status = InstanceStatus.HEALTHY
            instance.last_heartbeat = time.time()

            # Update metrics if returned by vLLM
            if "num_requests_running" in data:
                instance.pending_requests = data["num_requests_running"]
            if "gpu_cache_usage_perc" in data:
                instance.kv_cache_usage = data["gpu_cache_usage_perc"]

            if old_status != InstanceStatus.HEALTHY and self._on_status_change:
                self._on_status_change(instance, old_status, InstanceStatus.HEALTHY)

            return InstanceStatus.HEALTHY
        except (URLError, OSError, ValueError, KeyError) as exc:
            old_status = instance.status
            instance.status = InstanceStatus.UNHEALTHY
            logger.warning("Health check failed for %s: %s", instance.instance_id, exc)

            if old_status != InstanceStatus.UNHEALTHY and self._on_status_change:
                self._on_status_change(instance, old_status, InstanceStatus.UNHEALTHY)

            return InstanceStatus.UNHEALTHY

    def check_all(self) -> Dict[str, InstanceStatus]:
        """Run health checks on all registered instances."""
        results = {}
        for inst in self._instances.values():
            if inst.status == InstanceStatus.DRAINING:
                results[inst.instance_id] = InstanceStatus.DRAINING
                continue
            results[inst.instance_id] = self.check_instance(inst)
        return results

    def get_healthy_instances(self, role: Optional[str] = None) -> List[PDInstance]:
        """Return list of healthy instances, optionally filtered by role."""
        result = []
        for inst in self._instances.values():
            if inst.status != InstanceStatus.HEALTHY:
                continue
            if role and inst.role.value != role:
                continue
            result.append(inst)
        return result

    def drain_instance(self, instance_id: str) -> bool:
        """Mark an instance as draining (no new requests)."""
        inst = self._instances.get(instance_id)
        if inst is None:
            return False
        old_status = inst.status
        inst.status = InstanceStatus.DRAINING
        if self._on_status_change:
            self._on_status_change(inst, old_status, InstanceStatus.DRAINING)
        return True
