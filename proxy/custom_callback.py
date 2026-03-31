"""
custom_callback.py — LiteLLM proxy callback (async-only).

This module defines a custom LiteLLM callback that logs one JSONL line per
request using the `standard_logging_object` (SLO).
"""

import asyncio
import json
import os
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger

OUTPUT_PATH = os.environ.get("METRICS_JSONL", "runs/traces.jsonl")


class MetricsCallback(CustomLogger):
    """LiteLLM callback that logs each request as one JSONL line."""

    def __init__(self) -> None:
        """Initialize output path and async write lock."""
        self._path = Path(OUTPUT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log successful LLM call."""
        await self._log(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log failed LLM call."""
        await self._log(kwargs)

    async def _log(self, kwargs) -> None:
        """Extract SLO from kwargs and append JSON line to output file.
        Thread-safe via async lock."""
        slo = kwargs.get("standard_logging_object")
        if not slo:
            return

        line = self._safe_json(slo) + "\n"

        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        """Append line to output file."""
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _safe_json(self, data: dict) -> str:
        """JSON serialize dict, returns error payload on failure."""
        try:
            return json.dumps(data)
        except Exception:
            return json.dumps(
                {"error": "non-serializable standard_logging_object", "raw": str(data)}
            )


# Required by LiteLLM proxy
proxy_handler_instance = MetricsCallback()
