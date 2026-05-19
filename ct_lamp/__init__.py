"""CT-LAMP package initialization."""

from __future__ import annotations

import warnings

# Suppress third-party FutureWarnings at package import time, before MONAI /
# generative modules are imported by submodules.
warnings.simplefilter("ignore", FutureWarning)
