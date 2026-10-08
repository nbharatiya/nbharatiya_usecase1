"""Retry helpers for transient LLM/API failures."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def retry_call(operation: Callable[[], T], attempts: int = 3, base_delay: float = 1.0) -> T:
    """Retry an operation with 1s, 2s, 4s exponential backoff by default."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            delay = base_delay * (2**attempt)
            LOGGER.warning("Attempt %s/%s failed; retrying in %.1fs: %s", attempt + 1, attempts, delay, exc)
            time.sleep(delay)
    assert last_error is not None
    raise last_error
