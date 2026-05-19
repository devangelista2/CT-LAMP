"""Configuration loading and override utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a nested dict.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Nested dictionary of configuration values.
    """
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted-key overrides to a config dict (mutates a deep copy).

    Args:
        cfg: Base configuration dictionary.
        overrides: List of strings like "sampling.num_steps=50" or
                   "operator.kernel_size=61".

    Returns:
        New dict with overrides applied.

    Raises:
        ValueError: If an override string is malformed.
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override!r}")
        key, _, value_str = override.partition("=")
        keys = key.strip().split(".")
        # Parse value: try int, float, bool, then string
        value = _parse_value(value_str.strip())
        # Navigate and set
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return cfg


def _parse_value(s: str) -> Any:
    """Try to parse a string as int, float, bool, None, then fall back to str."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def resolve_device(cfg: dict[str, Any]) -> str:
    """Resolve 'auto' device to the best available device string.

    Uses IPPy's get_device() if available, otherwise falls back to
    torch.cuda.is_available() logic.
    """
    device = cfg.get("experiment", {}).get("device", "auto")
    if device != "auto":
        return device
    try:
        from IPPy.utilities import get_device
        return str(get_device())
    except Exception:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
