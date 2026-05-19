"""Logging setup utilities."""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to write log output to (in addition to stderr).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Silence known third-party deprecation warnings that are outside this project.
    warnings.filterwarnings(
        "ignore",
        message=r".*cuda\.cudart module is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cuda\.amp\.autocast\(args\.\.\.\) is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"generative\.networks\.layers\.vector_quantizer",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"cuda(\..*)?",
    )

    # Silence noisy third-party libraries that flood the log at INFO level
    for _noisy in ("httpx", "httpcore", "datasets", "huggingface_hub",
                   "filelock", "urllib3", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
