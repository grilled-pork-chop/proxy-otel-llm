#!/usr/bin/env python3
"""
test_all.py — end-to-end validation of callback + parser.

Tests:
  1. Success          — slo written, tokens + latency correct
  2. Single tool call — tool name extracted by parser
  3. Multi tool call  — all names extracted, correct count
  4. Failure          — error written, tokens zero
  5. No duplicates    — 2 requests → exactly 2 records
  6. Parser           — correct aggregated output
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import litellm
import litellm.exceptions
import litellm.utils

sys.path.insert(0, str(Path(__file__).parent))

# Point callback at a temp file before importing
tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
os.environ["METRICS_JSONL"] = tmp.name
tmp.close()

from proxy.custom_callback import MetricsCallback  # noqa: E402
from proxy.parse_metrics import parse  # noqa: E402

TRACES = Path(os.environ["METRICS_JSONL"])
handler = MetricsCallback()
litellm.callbacks = [handler]


# ── Helpers ───────────────────────────────────────────────────────────────────


def records() -> list[dict]:
    return [json.loads(line) for line in TRACES.open() if line.strip()]


async def wait_for(total: int, timeout: float = 5.0) -> None:
    """Poll until the trace file holds at least `total` records (the callback
    writes asynchronously). Raises TimeoutError instead of racing a fixed sleep."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(records()) >= total:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"expected >= {total} records, got {len(records())} after {timeout}s")


def check(label: str, got, expected) -> bool:
    if got != expected:
        print(f"  FAIL  {label}: got {got!r}, expected {expected!r}")
        return False
    print(f"  OK    {label}")
    return True


def make_tool_response(names: list[str]) -> litellm.utils.ModelResponse:
    return litellm.utils.ModelResponse(
        id="chatcmpl-tool",
        choices=[
            litellm.utils.Choices(
                finish_reason="tool_calls",
                index=0,
                message=litellm.utils.Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        litellm.utils.ChatCompletionMessageToolCall(
                            id=f"call-{i}",
                            type="function",
                            function=litellm.utils.Function(name=n, arguments="{}"),
                        )
                        for i, n in enumerate(names)
                    ],
                ),
            )
        ],
        model="fake",
        usage=litellm.utils.Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


async def run():
    results = []

    print("\n── Test 1: success ──────────────────────────────────────────────")
    before = len(records())
    await litellm.acompletion(
        model="openai/fake",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hello world",
    )
    await wait_for(before + 1)
    r = records()[before]
    results.append(
        all(
            [
                check("status", r["status"], "success"),
                check("prompt_tokens", r["prompt_tokens"], 10),
                check("completion_tokens", r["completion_tokens"], 20),
                check("response_time > 0", r["response_time"] > 0, True),
                check("model set", bool(r["model"]), True),
            ]
        )
    )

    print("\n── Test 2: single tool call ─────────────────────────────────────")
    before = len(records())
    await litellm.acompletion(
        model="openai/fake",
        messages=[{"role": "user", "content": "weather?"}],
        mock_response=make_tool_response(["get_weather"]),
    )
    await wait_for(before + 1)
    r = records()[before]
    metrics = parse(TRACES)
    choices = (r.get("response") or {}).get("choices") or []
    tcs = (choices[0].get("message") or {}).get("tool_calls") or [] if choices else []
    results.append(
        all(
            [
                check(
                    "finish_reason in slo",
                    choices[0].get("finish_reason") if choices else None,
                    "tool_calls",
                ),
                check(
                    "tool call name in slo",
                    tcs[0]["function"]["name"] if tcs else None,
                    "get_weather",
                ),
                check("parser raw_calls", metrics["tools"]["raw_calls"], 1),
                check("parser breakdown", metrics["tools"]["breakdown"].get("get_weather"), 1),
            ]
        )
    )

    print("\n── Test 3: multiple tool calls ──────────────────────────────────")
    before = len(records())
    await litellm.acompletion(
        model="openai/fake",
        messages=[{"role": "user", "content": "do things"}],
        mock_response=make_tool_response(["search", "calculator"]),
    )
    await wait_for(before + 1)
    metrics = parse(TRACES)
    # Cumulative at this point: get_weather(1) + search(1) + calculator(1) = 3 total
    results.append(
        all(
            [
                check("raw_calls cumulative", metrics["tools"]["raw_calls"], 3),
                check("distinct_calls cumulative", metrics["tools"]["distinct_calls"], 3),
                check("search in breakdown", "search" in metrics["tools"]["breakdown"], True),
                check("calc in breakdown", "calculator" in metrics["tools"]["breakdown"], True),
            ]
        )
    )

    print("\n── Test 4: failure ──────────────────────────────────────────────")
    before = len(records())
    try:
        await litellm.acompletion(
            model="openai/fake",
            messages=[{"role": "user", "content": "fail"}],
            mock_response=litellm.exceptions.BadRequestError(
                message="test error", model="fake", llm_provider="openai"
            ),
        )
    except Exception:
        pass
    await wait_for(before + 1)
    r = records()[before]
    results.append(
        all(
            [
                check("status", r["status"], "failure"),
                check("prompt_tokens", r["prompt_tokens"], 0),
                check("has error_str", bool(r.get("error_str")), True),
            ]
        )
    )

    print("\n── Test 5: no duplicate writes ──────────────────────────────────")
    before = len(records())
    for _ in range(2):
        await litellm.acompletion(
            model="openai/fake",
            messages=[{"role": "user", "content": "ping"}],
            mock_response="pong",
        )
    await wait_for(before + 2)
    await asyncio.sleep(0.1)  # settle, so a duplicate write would surface
    results.append(check("2 requests = 2 records", len(records()) - before, 2))

    print("\n── Test 6: parser aggregation ───────────────────────────────────")
    all_recs = records()
    metrics = parse(TRACES)
    n_failure = sum(1 for r in all_recs if r["status"] == "failure")
    exp_prompt = sum(r["prompt_tokens"] for r in all_recs if r["status"] == "success")
    exp_comp = sum(r["completion_tokens"] for r in all_recs if r["status"] == "success")
    print(f"  {json.dumps(metrics, indent=4)}")
    results.append(
        all(
            [
                check("llm.calls", metrics["llm"]["calls"], len(all_recs)),
                check("tokens.prompt", metrics["llm"]["tokens"]["prompt"], exp_prompt),
                check("tokens.completion", metrics["llm"]["tokens"]["completion"], exp_comp),
                check("requests.total", metrics["requests"]["total"], len(all_recs)),
                check("requests.errors", metrics["requests"]["errors"], n_failure),
                check("latency > 0", metrics["llm"]["latency"]["total_ms"] > 0, True),
            ]
        )
    )

    print(f"\n{'=' * 60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} passed")
    TRACES.unlink(missing_ok=True)
    if passed < len(results):
        sys.exit(1)


asyncio.run(run())
