"""Configuration loading for the PD scheduler."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .models import SchedulerConfig


def load_config(path: str) -> SchedulerConfig:
    """Load scheduler configuration from a JSON file.

    Parameters
    ----------
    path:
        Absolute or relative path to a JSON configuration file.

    Returns
    -------
    SchedulerConfig parsed from the file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    resolved = os.path.expanduser(path)
    with open(resolved, encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return SchedulerConfig.from_dict(data)
