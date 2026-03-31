"""
Managed LiteLLM proxy subprocess with automatic metrics collection.
"""

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .parse_metrics import parse

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_S = 30
HEALTH_INTERVAL_S = 0.5
FLUSH_WAIT_S = 2.0
RUNS_DIR = Path("runs")


# ── Health check ──────────────────────────────────────────────────────────────


def _wait_healthy(proxy_url: str, timeout: float = HEALTH_TIMEOUT_S) -> None:
    """Poll /health endpoint until HTTP 200 or timeout.
    Raises TimeoutError if proxy doesn't become healthy."""
    health_url = proxy_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout

    logger.info(f"Waiting for {health_url} ...")

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    logger.info("Ready.")
                    return
        except Exception:
            pass

        time.sleep(HEALTH_INTERVAL_S)

    raise TimeoutError(f"Proxy did not become healthy within {timeout}s — check logs.")


# ── Proxy ─────────────────────────────────────────────────────────────────────


class LiteLLMProxy:
    """Managed LiteLLM proxy subprocess with automatic lifecycle and metrics."""

    def __init__(
        self,
        model_url: str,
        model_name: str,
        model_provider: str,
        api_key: str,
        proxy_url: str = "http://0.0.0.0:14000",
    ):
        """
        Args:
            model_url (str): Full URL to the model API base (e.g., https://...)
            model_name (str): Model name to expose via proxy
            model_provider (str): Provider/model identifier (e.g., groq/llama-3.3-70b-versatile)
            api_key (str): API key for model provider
            proxy_url (str): Full base URL where the proxy will listen
        """
        self.model_url = model_url.rstrip("/")
        self.model_name = model_name
        self.model_provider = model_provider
        self.api_key = api_key
        self.proxy_url = proxy_url.rstrip("/")

        self.run_dir: Path | None = None
        self.traces_path: Path | None = None
        self.metrics: dict | None = None

        self._proc = None
        self._config_file = None

    def start(self) -> "LiteLLMProxy":
        """Launch proxy subprocess and wait for health check.
        Raises TimeoutError if proxy doesn't start."""
        run_id = uuid.uuid4().hex[:8]
        self.run_dir = RUNS_DIR / f"run-{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.run_dir / "traces.jsonl"

        logger.info(f"Run directory: {self.run_dir}")

        env = os.environ.copy()
        env["MODEL_URL"] = self.model_url
        env["MODEL_NAME"] = self.model_name
        env["MODEL_PROVIDER"] = self.model_provider
        env["MODEL_API_KEY"] = self.api_key
        env["METRICS_JSONL"] = str(self.traces_path.resolve())

        # Use static config directly from proxy/ directory
        self._config_file = Path(__file__).parent / "config.yaml"
        project_root = Path(__file__).parent.parent.resolve()

        proxy_host = urlparse(self.proxy_url).hostname or ""
        proxy_port = urlparse(self.proxy_url).port or 0

        cmd = [
            "litellm",
            "--host",
            proxy_host,
            "--port",
            str(proxy_port),
            "--config",
            str(self._config_file),
        ]

        logger.info(f"Starting on {self.proxy_url} → model at {self.model_url}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(project_root),
        )

        def _stream():
            for line in self._proc.stdout:
                logger.debug(f"[litellm] {line.rstrip()}")

        threading.Thread(target=_stream, daemon=True).start()

        try:
            _wait_healthy(self.proxy_url)
        except TimeoutError:
            self.stop()
            raise

        return self

    def stop(self) -> None:
        """Stop proxy and generate metrics.json."""
        if self._proc and self._proc.poll() is None:
            logger.info("Stopping ...")
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        time.sleep(FLUSH_WAIT_S)
        logger.info(f"Stopped. Traces → {self.traces_path}")

        self._parse()

    def _parse(self) -> None:
        """Parse traces.jsonl into metrics.json."""
        if not self.traces_path or not self.traces_path.exists():
            logger.warning("No traces file found — skipping parse.")
            self.metrics = {}
            return

        self.metrics = parse(self.traces_path)
        metrics_path = self.run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self.metrics, indent=2))
        logger.info(f"Metrics written to {metrics_path}")

    def __enter__(self) -> "LiteLLMProxy":
        """Enter context manager and start proxy."""
        return self.start()

    def __exit__(self, *_) -> None:
        """Exit context manager and stop proxy."""
        self.stop()
