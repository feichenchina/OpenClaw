"""Unit tests for the PD scheduler core logic."""

from __future__ import annotations

import json
import time
import unittest

from vllm_pd_scheduler.config import load_config
from vllm_pd_scheduler.health import HealthChecker
from vllm_pd_scheduler.models import (
    InstanceRole,
    InstanceStatus,
    PDInstance,
    ScheduleRequest,
    ScheduleStrategy,
    SchedulerConfig,
)
from vllm_pd_scheduler.router import PDRouter
from vllm_pd_scheduler.scheduler import PDScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance(
    iid: str = "p1",
    role: InstanceRole = InstanceRole.PREFILL,
    host: str = "127.0.0.1",
    port: int = 8000,
    status: InstanceStatus = InstanceStatus.HEALTHY,
    pending: int = 0,
    kv_usage: float = 0.0,
) -> PDInstance:
    inst = PDInstance(instance_id=iid, role=role, host=host, port=port)
    inst.status = status
    inst.pending_requests = pending
    inst.kv_cache_usage = kv_usage
    return inst


def _make_request(
    rid: str = "req-1",
    phase: InstanceRole = InstanceRole.PREFILL,
) -> ScheduleRequest:
    return ScheduleRequest(request_id=rid, phase=phase, payload={"prompt": "hi"})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels(unittest.TestCase):
    def test_pd_instance_address(self):
        inst = _make_instance(host="10.0.0.1", port=9000)
        self.assertEqual(inst.address, "http://10.0.0.1:9000")

    def test_pd_instance_is_available(self):
        inst = _make_instance(status=InstanceStatus.HEALTHY)
        self.assertTrue(inst.is_available)
        inst.status = InstanceStatus.UNHEALTHY
        self.assertFalse(inst.is_available)
        inst.status = InstanceStatus.DRAINING
        self.assertFalse(inst.is_available)

    def test_scheduler_config_from_dict(self):
        data = {
            "strategy": "round_robin",
            "prefill_instances": [{"id": "p1", "host": "h", "port": 1}],
            "decode_instances": [],
            "health_check_interval": 10.0,
        }
        cfg = SchedulerConfig.from_dict(data)
        self.assertEqual(cfg.strategy, ScheduleStrategy.ROUND_ROBIN)
        self.assertEqual(len(cfg.prefill_instances), 1)
        self.assertEqual(cfg.health_check_interval, 10.0)

    def test_scheduler_config_defaults(self):
        cfg = SchedulerConfig.from_dict({})
        self.assertEqual(cfg.strategy, ScheduleStrategy.LEAST_LOAD)
        self.assertEqual(cfg.health_check_interval, 5.0)


# ---------------------------------------------------------------------------
# Scheduler strategies
# ---------------------------------------------------------------------------

class TestPDScheduler(unittest.TestCase):
    def test_round_robin(self):
        cfg = SchedulerConfig(strategy=ScheduleStrategy.ROUND_ROBIN)
        sched = PDScheduler(cfg)
        candidates = [_make_instance(iid=f"p{i}") for i in range(3)]
        results = []
        for i in range(6):
            req = _make_request(rid=f"r{i}")
            res = sched.schedule(req, candidates)
            self.assertTrue(res.success)
            results.append(res.instance.instance_id)
        self.assertEqual(results, ["p0", "p1", "p2", "p0", "p1", "p2"])

    def test_least_load(self):
        cfg = SchedulerConfig(strategy=ScheduleStrategy.LEAST_LOAD)
        sched = PDScheduler(cfg)
        c1 = _make_instance(iid="p1", pending=5)
        c2 = _make_instance(iid="p2", pending=2)
        c3 = _make_instance(iid="p3", pending=8)
        req = _make_request()
        res = sched.schedule(req, [c1, c2, c3])
        self.assertTrue(res.success)
        self.assertEqual(res.instance.instance_id, "p2")

    def test_weighted_prefers_low_score(self):
        cfg = SchedulerConfig(
            strategy=ScheduleStrategy.WEIGHTED,
            load_weight=1.0,
            kv_cache_weight=0.0,
            latency_weight=0.0,
        )
        sched = PDScheduler(cfg)
        c1 = _make_instance(iid="p1", pending=10)
        c2 = _make_instance(iid="p2", pending=1)
        req = _make_request()
        res = sched.schedule(req, [c1, c2])
        self.assertTrue(res.success)
        self.assertEqual(res.instance.instance_id, "p2")

    def test_no_candidates(self):
        cfg = SchedulerConfig()
        sched = PDScheduler(cfg)
        req = _make_request()
        res = sched.schedule(req, [])
        self.assertFalse(res.success)
        self.assertIn("No available instances", res.error)

    def test_unhealthy_candidates_filtered(self):
        cfg = SchedulerConfig()
        sched = PDScheduler(cfg)
        c1 = _make_instance(iid="p1", status=InstanceStatus.UNHEALTHY)
        req = _make_request()
        res = sched.schedule(req, [c1])
        self.assertFalse(res.success)

    def test_schedule_increments_pending(self):
        cfg = SchedulerConfig(strategy=ScheduleStrategy.LEAST_LOAD)
        sched = PDScheduler(cfg)
        c1 = _make_instance(iid="p1", pending=0)
        req = _make_request()
        res = sched.schedule(req, [c1])
        self.assertTrue(res.success)
        self.assertEqual(c1.pending_requests, 1)


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------

class TestHealthChecker(unittest.TestCase):
    def test_check_healthy(self):
        """Simulate a healthy response."""
        inst = _make_instance(status=InstanceStatus.UNKNOWN)

        def fake_get(url, timeout):
            return {"status": "ok", "num_requests_running": 3, "gpu_cache_usage_perc": 0.45}

        hc = HealthChecker([inst], http_get=fake_get)
        status = hc.check_instance(inst)
        self.assertEqual(status, InstanceStatus.HEALTHY)
        self.assertEqual(inst.pending_requests, 3)
        self.assertAlmostEqual(inst.kv_cache_usage, 0.45, places=2)

    def test_check_unhealthy(self):
        """Simulate a failed health check."""
        inst = _make_instance(status=InstanceStatus.HEALTHY)

        def fake_get(url, timeout):
            raise OSError("connection refused")

        hc = HealthChecker([inst], http_get=fake_get)
        status = hc.check_instance(inst)
        self.assertEqual(status, InstanceStatus.UNHEALTHY)
        self.assertEqual(inst.status, InstanceStatus.UNHEALTHY)

    def test_check_all_skips_draining(self):
        inst = _make_instance(status=InstanceStatus.DRAINING)

        def fake_get(url, timeout):
            raise AssertionError("Should not be called for draining instances")

        hc = HealthChecker([inst], http_get=fake_get)
        results = hc.check_all()
        self.assertEqual(results[inst.instance_id], InstanceStatus.DRAINING)

    def test_get_healthy_instances_by_role(self):
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL, status=InstanceStatus.HEALTHY)
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE, status=InstanceStatus.HEALTHY)
        d2 = _make_instance(iid="d2", role=InstanceRole.DECODE, status=InstanceStatus.UNHEALTHY)

        hc = HealthChecker([p1, d1, d2])
        prefills = hc.get_healthy_instances(role="prefill")
        self.assertEqual(len(prefills), 1)
        self.assertEqual(prefills[0].instance_id, "p1")

        decodes = hc.get_healthy_instances(role="decode")
        self.assertEqual(len(decodes), 1)
        self.assertEqual(decodes[0].instance_id, "d1")

    def test_drain_instance(self):
        inst = _make_instance(iid="p1", status=InstanceStatus.HEALTHY)
        hc = HealthChecker([inst])
        ok = hc.drain_instance("p1")
        self.assertTrue(ok)
        self.assertEqual(inst.status, InstanceStatus.DRAINING)
        # Draining unknown id
        self.assertFalse(hc.drain_instance("nonexistent"))

    def test_status_change_callback(self):
        """Verify the on_status_change callback fires."""
        changes = []

        def on_change(inst, old, new):
            changes.append((inst.instance_id, old, new))

        inst = _make_instance(status=InstanceStatus.UNKNOWN)

        def fake_get(url, timeout):
            return {}

        hc = HealthChecker([inst], on_status_change=on_change, http_get=fake_get)
        hc.check_instance(inst)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0], (inst.instance_id, InstanceStatus.UNKNOWN, InstanceStatus.HEALTHY))


# ---------------------------------------------------------------------------
# Router (integration-style)
# ---------------------------------------------------------------------------

class TestPDRouter(unittest.TestCase):
    def _make_router(self, *, prefill_ok=True, decode_ok=True):
        """Create a router with mock HTTP calls."""
        cfg = SchedulerConfig(strategy=ScheduleStrategy.LEAST_LOAD)

        def mock_post(url, body, timeout):
            if "prefill" in url or body.get("phase") == "prefill":
                if not prefill_ok:
                    raise OSError("prefill instance down")
                return {"kv_cache_token": "tok-abc"}
            if not decode_ok:
                raise OSError("decode instance down")
            return {"text": "hello world"}

        def mock_get(url, timeout):
            return {"status": "ok"}

        router = PDRouter(cfg, http_post=mock_post, http_get=mock_get)
        return router

    def test_route_request_success(self):
        router = self._make_router()
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL)
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE)
        router.register_instance(p1)
        router.register_instance(d1)
        # Mark healthy
        router.run_health_checks()

        result = router.route_request({"prompt": "Hello"}, request_id="r1")
        self.assertTrue(result.success)

    def test_route_request_no_prefill(self):
        router = self._make_router()
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE)
        router.register_instance(d1)
        router.run_health_checks()

        result = router.route_request({"prompt": "Hello"}, request_id="r2")
        self.assertFalse(result.success)
        self.assertIn("No available instances", result.error)

    def test_route_request_prefill_forward_fails(self):
        router = self._make_router(prefill_ok=False)
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL)
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE)
        router.register_instance(p1)
        router.register_instance(d1)
        router.run_health_checks()

        result = router.route_request({"prompt": "Hello"}, request_id="r3")
        self.assertFalse(result.success)
        self.assertIn("Prefill forward failed", result.error)

    def test_route_request_decode_forward_fails(self):
        router = self._make_router(decode_ok=False)
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL)
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE)
        router.register_instance(p1)
        router.register_instance(d1)
        router.run_health_checks()

        result = router.route_request({"prompt": "Hello"}, request_id="r4")
        self.assertFalse(result.success)
        self.assertIn("Decode forward failed", result.error)

    def test_get_instances_by_role(self):
        router = self._make_router()
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL)
        d1 = _make_instance(iid="d1", role=InstanceRole.DECODE)
        router.register_instance(p1)
        router.register_instance(d1)

        prefills = router.get_instances(role=InstanceRole.PREFILL)
        self.assertEqual(len(prefills), 1)
        decodes = router.get_instances(role=InstanceRole.DECODE)
        self.assertEqual(len(decodes), 1)
        all_inst = router.get_instances()
        self.assertEqual(len(all_inst), 2)

    def test_drain_instance(self):
        router = self._make_router()
        p1 = _make_instance(iid="p1", role=InstanceRole.PREFILL)
        router.register_instance(p1)
        router.run_health_checks()

        ok = router.drain_instance("p1")
        self.assertTrue(ok)
        # After draining, routing should fail
        result = router.route_request({"prompt": "Hello"}, request_id="r5")
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        import os
        import tempfile

        data = {
            "strategy": "weighted",
            "prefill_instances": [{"id": "p1", "host": "10.0.0.1", "port": 8000}],
            "decode_instances": [{"id": "d1", "host": "10.0.0.2", "port": 8001}],
            "health_check_interval": 3.0,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            self.assertEqual(cfg.strategy, ScheduleStrategy.WEIGHTED)
            self.assertEqual(len(cfg.prefill_instances), 1)
            self.assertEqual(len(cfg.decode_instances), 1)
            self.assertEqual(cfg.health_check_interval, 3.0)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    def test_main_missing_config_exits(self):
        from vllm_pd_scheduler.__main__ import main

        with self.assertRaises(FileNotFoundError):
            main(["--config", "/nonexistent/path.json"])


if __name__ == "__main__":
    unittest.main()
