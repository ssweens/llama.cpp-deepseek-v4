#!/usr/bin/env python3
"""DeepSeek-V4 endpoint regression harness.

This script intentionally does not start Corral, Docker, or llama-server. Point it
at an already-running OpenAI-compatible endpoint, preferably the systemctl-managed
Corral service for integration validation.

Core checks lock in the failure modes we previously fixed:
  - simple non-tool chat sanity
  - basic arithmetic/coherence sanity
  - minimal structured tool calls (auto + required)
  - multi-turn tool-call replay fixture (non-streaming + streaming)
  - prompt-cache reuse on repeated tool-call replay

Optional exact antirez/ds4 short-vector checks are available via
--include-official-vectors. They are deterministic byte-level checks and are best
suited to BF16/reference-style runs, not lossy quants.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_REPLAY = REPO_ROOT / "tasks" / "replays" / "dsv4_agent_tool_replay_turn02_request.json"
DEFAULT_DS4_ROOT = Path(os.environ.get("DS4_ROOT", str(REPO_ROOT.parent / "ds4")))
DEFAULT_SHORT_VECTOR_CASES = {"short_reasoning_plain", "short_code_completion"}


class RegressionError(Exception):
    pass


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str
    elapsed_s: float


def post_json(base_url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", "replace")
            return json.loads(payload)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RegressionError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RegressionError(f"request failed: {e}") from e


def post_stream(base_url: str, body: dict[str, Any], timeout: int) -> list[dict[str, Any]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data_s = line[6:]
                if data_s == "[DONE]":
                    break
                events.append(json.loads(data_s))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RegressionError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RegressionError(f"stream request failed: {e}") from e
    return events


def choice_message(resp: dict[str, Any]) -> dict[str, Any]:
    try:
        return resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise RegressionError(f"missing choices[0].message in response: {json.dumps(resp)[:1000]}") from e


def finish_reason(resp: dict[str, Any]) -> str | None:
    try:
        return resp["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def usage_cached_tokens(resp: dict[str, Any]) -> int:
    usage = resp.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    val = details.get("cached_tokens")
    return int(val or 0)


def make_bash_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute shell command and capture output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_directory": {"type": "string", "description": "Working directory"},
                },
                "required": ["command"],
            },
        },
    }


def assert_exact_content(resp: dict[str, Any], expected: str) -> str:
    msg = choice_message(resp)
    content = msg.get("content") or ""
    got = content.strip()
    if got != expected:
        raise RegressionError(f"expected content {expected!r}, got {got!r}; reasoning_len={len(msg.get('reasoning_content') or '')}")
    return f"content={got!r}, reasoning_len={len(msg.get('reasoning_content') or '')}, finish={finish_reason(resp)}"


def parse_tool_args(args_s: str) -> dict[str, Any]:
    try:
        parsed = json.loads(args_s)
    except json.JSONDecodeError as e:
        raise RegressionError(f"tool arguments are not valid JSON: {args_s!r}") from e
    if not isinstance(parsed, dict):
        raise RegressionError(f"tool arguments JSON is not an object: {parsed!r}")
    return parsed


def assert_bash_ls_tool_call(resp: dict[str, Any], *, require_finish_reason: bool = True) -> str:
    msg = choice_message(resp)
    calls = msg.get("tool_calls") or []
    if not calls:
        content = (msg.get("content") or "")[:300]
        raise RegressionError(f"expected structured tool_calls, got none; content_prefix={content!r}")
    call = calls[0]
    fn = call.get("function") or {}
    name = fn.get("name")
    if name != "bash":
        raise RegressionError(f"expected first tool name 'bash', got {name!r}; calls={calls!r}")
    args = parse_tool_args(fn.get("arguments") or "")
    command = str(args.get("command") or "")
    if "ls" not in command or "-la" not in command:
        raise RegressionError(f"expected bash command containing 'ls -la', got {command!r}; args={args!r}")
    fin = finish_reason(resp)
    if require_finish_reason and fin != "tool_calls":
        raise RegressionError(f"expected finish_reason='tool_calls', got {fin!r}")
    return f"tool=bash, command={command!r}, finish={fin}, reasoning_len={len(msg.get('reasoning_content') or '')}, cached_tokens={usage_cached_tokens(resp)}"


def aggregate_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    finish: str | None = None
    usage: dict[str, Any] | None = None

    for event in events:
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish = choice.get("finish_reason") or finish
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0))
            dst = tool_parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                dst["id"] += tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                dst["name"] += fn["name"]
            if fn.get("arguments"):
                dst["arguments"] += fn["arguments"]

    calls = []
    for idx in sorted(tool_parts):
        part = tool_parts[idx]
        calls.append({
            "type": "function",
            "id": part["id"],
            "function": {
                "name": part["name"],
                "arguments": part["arguments"],
            },
        })

    return {
        "choices": [{
            "finish_reason": finish,
            "message": {
                "role": "assistant",
                "content": "".join(content),
                "reasoning_content": "".join(reasoning),
                "tool_calls": calls,
            },
        }],
        "usage": usage or {},
    }


def with_model_and_limits(body: dict[str, Any], model: str, max_tokens: int, stream: bool) -> dict[str, Any]:
    out = json.loads(json.dumps(body))
    out["model"] = model
    out["max_tokens"] = max_tokens
    out["stream"] = stream
    # Keep cache_prompt explicit for the cache regression, and harmless for direct llama-server.
    out["cache_prompt"] = True
    return out


def run_test(name: str, fn) -> TestResult:
    start = time.time()
    try:
        detail = fn()
        return TestResult(name, True, detail, time.time() - start)
    except Exception as e:  # keep going to report all failures
        return TestResult(name, False, str(e), time.time() - start)


@dataclass
class OfficialStep:
    selected: bytes


@dataclass
class OfficialCase:
    case_id: str
    ctx: int
    steps: list[OfficialStep]
    prompt_path: Path


def parse_official_vec(path: Path) -> list[OfficialCase]:
    cases: list[OfficialCase] = []
    cur_id: str | None = None
    cur_ctx = 0
    cur_prompt: Path | None = None
    cur_steps: list[OfficialStep] = []

    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "case":
            if cur_id is not None:
                raise RegressionError(f"nested official vector case: {line}")
            cur_id = parts[1]
            cur_ctx = int(parts[2])
            cur_prompt = Path(parts[4])
            cur_steps = []
        elif parts[0] == "step":
            if cur_id is None:
                raise RegressionError(f"official vector step outside case: {line}")
            cur_steps.append(OfficialStep(selected=bytes.fromhex(parts[2])))
        elif parts[0] == "top":
            continue
        elif parts[0] == "end":
            if cur_id is None or cur_prompt is None:
                raise RegressionError("official vector end outside case")
            cases.append(OfficialCase(cur_id, cur_ctx, cur_steps, cur_prompt))
            cur_id = None
            cur_prompt = None
            cur_steps = []
        else:
            raise RegressionError(f"unrecognized official vector line: {line}")
    return cases


def token_bytes_from_logprobs(resp: dict[str, Any]) -> list[bytes]:
    content = resp.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
    out: list[bytes] = []
    for item in content:
        if item.get("bytes") is not None:
            out.append(bytes(int(x) for x in item["bytes"]))
        else:
            out.append(str(item.get("token", "")).encode("utf-8"))
    return out


def run_official_vector_case(case: OfficialCase, ds4_root: Path, base_url: str, model: str, timeout: int) -> str:
    prompt_path = case.prompt_path if case.prompt_path.is_absolute() else ds4_root / case.prompt_path
    prompt = prompt_path.read_text(encoding="utf-8")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": len(case.steps),
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }
    resp = post_json(base_url, body, timeout)
    got = token_bytes_from_logprobs(resp)
    if len(got) < len(case.steps):
        raise RegressionError(f"case {case.case_id}: expected {len(case.steps)} logprob tokens, got {len(got)}")
    mismatches = []
    for i, step in enumerate(case.steps):
        if got[i] != step.selected:
            mismatches.append((i, step.selected, got[i]))
    if mismatches:
        i, exp, actual = mismatches[0]
        raise RegressionError(f"case {case.case_id}: token {i} mismatch expected={exp.hex()} got={actual.hex()}")
    return f"case={case.case_id}, tokens={len(case.steps)}"


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9999", help="OpenAI-compatible endpoint base URL")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS", help="model id to send in requests")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tool-replay", type=Path, default=DEFAULT_TOOL_REPLAY)
    ap.add_argument("--skip-cache-reuse", action="store_true", help="skip repeated-replay cached_tokens assertion")
    ap.add_argument("--skip-stream", action="store_true", help="skip streaming tool-call replay")
    ap.add_argument("--include-official-vectors", action="store_true", help="run exact antirez/ds4 short vector checks")
    ap.add_argument("--ds4-root", type=Path, default=DEFAULT_DS4_ROOT, help="path to antirez/ds4 checkout; defaults to DS4_ROOT or ../ds4")
    ap.add_argument("--official-case", action="append", help="official vector case id; default short cases")
    args = ap.parse_args(list(argv))

    if not args.tool_replay.exists():
        print(f"missing tool replay fixture: {args.tool_replay}", file=sys.stderr)
        return 2

    replay_body = json.loads(args.tool_replay.read_text(encoding="utf-8"))
    results: list[TestResult] = []

    def simple_ok() -> str:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly OK and nothing else."}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        return assert_exact_content(post_json(args.base_url, body, args.timeout), "OK")

    def basic_math() -> str:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        return assert_exact_content(post_json(args.base_url, body, args.timeout), "4")

    def minimal_tool_auto() -> str:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You have tools. When the user asks for a directory listing, call the bash tool instead of writing shell XML or prose-only commands."},
                {"role": "user", "content": "Use the bash tool to list the current directory with ls -la."},
            ],
            "tools": [make_bash_tool()],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        return assert_bash_ls_tool_call(post_json(args.base_url, body, args.timeout))

    def minimal_tool_required() -> str:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You have tools. When the user asks for a directory listing, call the bash tool instead of writing shell XML or prose-only commands."},
                {"role": "user", "content": "Use the bash tool to list the current directory with ls -la."},
            ],
            "tools": [make_bash_tool()],
            "tool_choice": {"type": "function", "function": {"name": "bash"}},
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        return assert_bash_ls_tool_call(post_json(args.base_url, body, args.timeout))

    def tool_replay_nonstream() -> str:
        body = with_model_and_limits(replay_body, args.model, args.max_tokens, stream=False)
        return assert_bash_ls_tool_call(post_json(args.base_url, body, args.timeout))

    def tool_replay_stream() -> str:
        body = with_model_and_limits(replay_body, args.model, args.max_tokens, stream=True)
        events = post_stream(args.base_url, body, args.timeout)
        if not events:
            raise RegressionError("stream produced no SSE events")
        resp = aggregate_stream(events)
        return assert_bash_ls_tool_call(resp, require_finish_reason=False) + f", sse_events={len(events)}"

    def tool_replay_cache_reuse() -> str:
        body = with_model_and_limits(replay_body, args.model, args.max_tokens, stream=False)
        first = post_json(args.base_url, body, args.timeout)
        assert_bash_ls_tool_call(first)
        second = post_json(args.base_url, body, args.timeout)
        assert_bash_ls_tool_call(second)
        cached = usage_cached_tokens(second)
        if cached <= 0:
            raise RegressionError("expected cached_tokens > 0 on repeated tool-call replay, got 0")
        return f"second_cached_tokens={cached}, first_cached_tokens={usage_cached_tokens(first)}"

    for name, fn in [
        ("simple_ok", simple_ok),
        ("basic_math", basic_math),
        ("minimal_tool_auto", minimal_tool_auto),
        ("minimal_tool_required", minimal_tool_required),
        ("tool_replay_turn02_nonstream", tool_replay_nonstream),
    ]:
        results.append(run_test(name, fn))

    if not args.skip_stream:
        results.append(run_test("tool_replay_turn02_stream", tool_replay_stream))
    if not args.skip_cache_reuse:
        results.append(run_test("tool_replay_turn02_cache_reuse", tool_replay_cache_reuse))

    if args.include_official_vectors:
        vec_path = args.ds4_root / "tests" / "test-vectors" / "official.vec"
        cases = parse_official_vec(vec_path)
        wanted = set(args.official_case or DEFAULT_SHORT_VECTOR_CASES)
        selected = [case for case in cases if case.case_id in wanted]
        for case in selected:
            results.append(run_test(
                f"official_vector_{case.case_id}",
                lambda case=case: run_official_vector_case(case, args.ds4_root, args.base_url, args.model, args.timeout),
            ))

    for res in results:
        mark = "PASS" if res.ok else "FAIL"
        print(f"{mark:4} {res.name:34} {res.elapsed_s:8.2f}s  {res.detail}")

    failed = [res for res in results if not res.ok]
    if failed:
        print(f"\n{len(failed)} regression(s) failed", file=sys.stderr)
        return 1
    print(f"\nall {len(results)} regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
