"""Structured JSON logging.

Logs are emitted as one JSON object per line to stdout. That is the 12-factor
approach: the application does not care where logs end up; the platform (Docker,
Kubernetes) or a collector (Promtail shipping to Grafana Loki) captures stdout and
persists it. JSON lines stay greppable locally and queryable once shipped.
"""

from __future__ import annotations

import json
import logging
import sys

from app import config


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line, including any `extra` fields."""

    _RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge structured context passed via `logger.info(..., extra={...})`.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
