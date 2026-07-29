"""Small retry helper with exponential backoff (no external deps)."""

import time
from collections.abc import Callable
from typing import TypeVar

from app.core.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    label: str = "operation",
) -> T:
    """Call ``func`` retrying on failure with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retry_on as exc:  # noqa: PERF203
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, attempts, exc, delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
