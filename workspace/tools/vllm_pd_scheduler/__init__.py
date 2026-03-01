"""vLLM PD Disaggregation Dynamic Scheduler.

This module implements dynamic scheduling for Prefill-Decode (PD) disaggregated
inference based on vLLM. It routes prefill requests to P-workers and decode
requests to D-workers, with health monitoring and load-based scheduling.
"""

__version__ = "0.1.0"
