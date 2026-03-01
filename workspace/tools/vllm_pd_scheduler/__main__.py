"""CLI entry point for the vLLM PD scheduler.

Usage:
    python -m vllm_pd_scheduler --config scheduler_config.json
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .config import load_config
from .models import InstanceRole, PDInstance
from .router import PDRouter

logger = logging.getLogger("vllm_pd_scheduler")


def _build_instances(config):
    """Create PDInstance objects from raw config dicts."""
    instances = []
    for item in config.prefill_instances:
        instances.append(
            PDInstance(
                instance_id=item["id"],
                role=InstanceRole.PREFILL,
                host=item["host"],
                port=item["port"],
                weight=item.get("weight", 1.0),
            )
        )
    for item in config.decode_instances:
        instances.append(
            PDInstance(
                instance_id=item["id"],
                role=InstanceRole.DECODE,
                host=item["host"],
                port=item["port"],
                weight=item.get("weight", 1.0),
            )
        )
    return instances


def main(argv=None):
    parser = argparse.ArgumentParser(description="vLLM PD Disaggregation Dynamic Scheduler")
    parser.add_argument("--config", required=True, help="Path to scheduler config JSON file")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Health check interval in seconds (overrides config)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.interval is not None:
        config.health_check_interval = args.interval

    router = PDRouter(config)

    # Register instances
    for inst in _build_instances(config):
        router.register_instance(inst)
        logger.info("Registered %s instance %s at %s", inst.role.value, inst.instance_id, inst.address)

    # Graceful shutdown
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("Received signal %s, shutting down…", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(
        "Scheduler started – strategy=%s, interval=%.1fs",
        config.strategy.value,
        config.health_check_interval,
    )

    # Health-check loop
    while running:
        results = router.run_health_checks()
        healthy = sum(1 for v in results.values() if v == "healthy")
        logger.debug("Health check: %d/%d healthy", healthy, len(results))
        time.sleep(config.health_check_interval)

    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
