"""Data models for the PD scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InstanceRole(str, Enum):
    """Role of a vLLM instance in PD disaggregation."""

    PREFILL = "prefill"
    DECODE = "decode"


class InstanceStatus(str, Enum):
    """Health status of a vLLM instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class ScheduleStrategy(str, Enum):
    """Scheduling strategy for request routing."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    WEIGHTED = "weighted"


@dataclass
class PDInstance:
    """Represents a single vLLM instance (prefill or decode worker)."""

    instance_id: str
    role: InstanceRole
    host: str
    port: int
    status: InstanceStatus = InstanceStatus.UNKNOWN
    weight: float = 1.0
    # Runtime metrics
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    kv_cache_usage: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_available(self) -> bool:
        return self.status == InstanceStatus.HEALTHY


@dataclass
class SchedulerConfig:
    """Configuration for the PD scheduler."""

    prefill_instances: List[Dict[str, Any]] = field(default_factory=list)
    decode_instances: List[Dict[str, Any]] = field(default_factory=list)
    strategy: ScheduleStrategy = ScheduleStrategy.LEAST_LOAD
    health_check_interval: float = 5.0
    health_check_timeout: float = 2.0
    max_retries: int = 3
    drain_timeout: float = 30.0
    # Weighted strategy parameters
    load_weight: float = 0.5
    kv_cache_weight: float = 0.3
    latency_weight: float = 0.2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        strategy_str = data.get("strategy", "least_load")
        return cls(
            prefill_instances=data.get("prefill_instances", []),
            decode_instances=data.get("decode_instances", []),
            strategy=ScheduleStrategy(strategy_str),
            health_check_interval=data.get("health_check_interval", 5.0),
            health_check_timeout=data.get("health_check_timeout", 2.0),
            max_retries=data.get("max_retries", 3),
            drain_timeout=data.get("drain_timeout", 30.0),
            load_weight=data.get("load_weight", 0.5),
            kv_cache_weight=data.get("kv_cache_weight", 0.3),
            latency_weight=data.get("latency_weight", 0.2),
        )


@dataclass
class ScheduleRequest:
    """A request to be scheduled to a PD instance."""

    request_id: str
    phase: InstanceRole
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # For decode phase: carries KV cache transfer info from prefill
    prefill_instance_id: Optional[str] = None
    kv_cache_token: Optional[str] = None


@dataclass
class ScheduleResult:
    """Result of a scheduling decision."""

    request_id: str
    instance: Optional[PDInstance]
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
