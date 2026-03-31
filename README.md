# llm-otel

A lightweight, metrics-focused wrapper around the **LiteLLM proxy**. It manages
the proxy as a subprocess, logs one JSONL trace per request via a custom
callback, and aggregates those traces into a `metrics.json` on shutdown — so
every run leaves behind a self-contained record of tokens, latency, tool calls,
and errors.

Runs without authentication, for local development and request observability.

## How it works

```
client → LiteLLM proxy → model provider
              │
              ▼
   proxy/custom_callback.py   writes runs/run-<id>/traces.jsonl (one line/request)
              │
         (on shutdown)
              ▼
   proxy/parse_metrics.py     aggregates traces.jsonl → runs/run-<id>/metrics.json
```

- `run_proxy.py` — CLI entry point.
- `proxy/proxy.py` — `LiteLLMProxy`, the managed subprocess + lifecycle.
- `proxy/config.yaml` — static LiteLLM config (reads `os.environ/*`).
- `proxy/custom_callback.py` — LiteLLM callback that writes JSONL traces.
- `proxy/parse_metrics.py` — trace → metrics aggregation.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quick start

```bash
MODEL_URL="https://api.groq.com/openai/v1" \
MODEL_API_KEY="gsk_..." \
uv run run_proxy.py
```

Press `Ctrl+C` to stop — the proxy flushes logs, parses traces into
`metrics.json`, and prints the aggregate.

### Configuration

| Variable          | Required | Default                          | Description                          |
| ----------------- | -------- | -------------------------------- | ------------------------------------ |
| `MODEL_URL`       | **yes**  | —                                | API base URL of the model provider   |
| `MODEL_API_KEY`   | **yes**  | —                                | Provider API key                     |
| `MODEL_NAME`      | no       | `llama-3.3-70b-versatile`        | Model name the proxy exposes         |
| `MODEL_PROVIDER`  | no       | `groq/llama-3.3-70b-versatile`   | LiteLLM provider/model identifier    |
| `PROXY_URL`       | no       | `http://0.0.0.0:14000`           | Address the proxy listens on         |

`MODEL_URL` and `PROXY_URL` may also be passed as `--model-url` / `--proxy-url`.

### Making requests

No auth is required — any `api_key` value works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:14000", api_key="dummy")

resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Output

Each run writes a unique directory:

```
runs/run-<id>/
├── traces.jsonl   # one LiteLLM standard_logging_object per request
└── metrics.json   # aggregated tokens, latency, tool calls, error rate
```

Re-aggregate an existing trace file at any time:

```bash
uv run python proxy/parse_metrics.py runs/run-<id>/traces.jsonl
```

## Other providers

```bash
# OpenAI
MODEL_URL="https://api.openai.com/v1" MODEL_API_KEY="sk-..." \
MODEL_NAME="gpt-4o" MODEL_PROVIDER="openai/gpt-4o" uv run run_proxy.py

# Anthropic
MODEL_URL="https://api.anthropic.com/v1" MODEL_API_KEY="sk-ant-..." \
MODEL_NAME="claude-sonnet-4-5" MODEL_PROVIDER="anthropic/claude-sonnet-4-5" \
uv run run_proxy.py
```

## Tests

`tests/test_all.py` validates the callback + parser end-to-end using LiteLLM mock
responses — no network or API key needed:

```bash
uv run python tests/test_all.py
```

## Lint

```bash
uv run ruff check .
uv run ruff format .
```

CI (`.github/workflows/ci.yml`) runs the lint and tests on every push and pull
request.

## License

MIT — see [LICENSE](LICENSE).
