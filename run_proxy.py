#!/usr/bin/env python3
"""
CLI entrypoint for running the LiteLLM proxy.

Example: MODEL_URL="https://..." MODEL_API_KEY="..." python run_proxy.py
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

from proxy.proxy import LiteLLMProxy

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint for running the proxy in standalone mode."""
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Start LiteLLM proxy with per-run metrics output.")
    parser.add_argument("--model-url", default=os.environ.get("MODEL_URL"))
    parser.add_argument("--proxy-url", default=os.environ.get("PROXY_URL"))
    args = parser.parse_args()

    # Validate required environment variables
    model_url = args.model_url
    api_key = os.environ.get("MODEL_API_KEY")

    if not model_url:
        logger.error("Must provide --model-url or set MODEL_URL")
        sys.exit(1)

    if not api_key:
        logger.error("Must set MODEL_API_KEY environment variable")
        sys.exit(1)

    proxy = LiteLLMProxy(
        model_url=model_url,
        model_name=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
        model_provider=os.environ.get("MODEL_PROVIDER", "groq/llama-3.3-70b-versatile"),
        api_key=api_key,
        proxy_url=args.proxy_url or "http://0.0.0.0:14000",
    )

    proxy.start()

    logger.info(f"Listening at {proxy.proxy_url}")
    logger.info(f"Traces → {proxy.traces_path}")
    logger.info("Ctrl+C to stop — metrics parsed automatically.")

    signal.signal(signal.SIGTERM, lambda *_: (proxy.stop(), sys.exit(0)))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        proxy.stop()
        if proxy.metrics:
            logger.info(json.dumps(proxy.metrics, indent=2))


if __name__ == "__main__":
    main()
