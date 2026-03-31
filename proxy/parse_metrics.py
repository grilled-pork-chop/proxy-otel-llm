"""
Parse traces.jsonl into aggregated metrics.json.
"""

import json
from pathlib import Path


def _tool_calls(slo: dict) -> list[str]:
    """Extract tool function names from SLO response, returns empty list if none."""
    names: list[str] = []
    response = slo.get("response") or {}

    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        for tc in tool_calls:
            name = (tc.get("function") or {}).get("name")
            if name:
                names.append(name)

    return names


def parse(traces_path: Path) -> dict:
    """
    Aggregate JSONL traces into metrics (tokens, latency, tools, errors).

    Returns empty dict if no records. Safely handles missing fields.
    """
    records = [json.loads(line) for line in traces_path.open() if line.strip()]
    if not records:
        return {}

    success = [r for r in records if r.get("status") == "success"]
    failures = [r for r in records if r.get("status") == "failure"]

    timestamps = [r.get("startTime") for r in records if r.get("startTime")]
    if timestamps:
        time_range = {
            "start": min(timestamps),
            "end": max(timestamps),
            "duration_s": round(max(timestamps) - min(timestamps), 2),
        }
    else:
        time_range = {"start": 0, "end": 0, "duration_s": 0}

    prompt = sum(r.get("prompt_tokens", 0) for r in success)
    completion = sum(r.get("completion_tokens", 0) for r in success)

    lats_ms = [r["response_time"] * 1000 for r in records if r.get("response_time") is not None]

    all_tool_calls = [name for r in success for name in _tool_calls(r)]
    distinct_tools = set(all_tool_calls)

    tool_errors = [r for r in failures if _tool_calls(r)]

    def lat(ms: list[float]) -> dict:
        """Compute latency summary (total, avg, max)."""
        if not ms:
            return {"total_ms": 0, "avg_ms": 0, "max_ms": 0}

        return {
            "total_ms": round(sum(ms)),
            "avg_ms": round(sum(ms) / len(ms)),
            "max_ms": round(max(ms)),
        }

    total = len(records)

    return {
        "time_range": time_range,
        "llm": {
            "calls": total,
            "tokens": {
                "prompt": prompt,
                "completion": completion,
                "total": prompt + completion,
            },
            "latency": lat(lats_ms),
        },
        "tools": {
            "raw_calls": len(all_tool_calls),
            "distinct_calls": len(distinct_tools),
            "breakdown": {n: all_tool_calls.count(n) for n in distinct_tools},
            "errors": len(tool_errors),
        },
        "requests": {
            "total": total,
            "errors": len(failures),
            "error_rate": round(len(failures) / total, 4) if total else 0,
        },
    }
