# DeepSeek-V4 Regression Harness — Handoff Report

## How to Run

### Prerequisites
- Working tree on the DeepSeek4 branch under test.
- Mounted-code container build exists at `build-dsv4-container/`.
- Preferred strict regression model: `deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS`.
- IQ1_M is useful for smoke/coherence checks but is too lossy for strict structured tool-call regression acceptance.

### Regression materials
- `scripts/dsv4_regression.py` — endpoint regression harness.
- `tasks/replays/dsv4_agent_tool_replay_turn02_request.json` — multi-turn tool-call/cache replay fixture.
- `tasks/dsv4_resident_bench_server.md` — optional mounted-code resident server workflow for ad hoc validation.

### Start an endpoint

For final Corral/Pi validation, use the systemd-managed Corral service. For ad hoc mounted-code validation, start the resident benchmark server documented in `tasks/dsv4_resident_bench_server.md`.

### Run the suite

```bash
cd "$LLAMA_DSV4_ROOT"

scripts/dsv4_regression.py \
  --base-url http://127.0.0.1:18089 \
  --model deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --max-tokens 2048 \
  --timeout 900 \
  --tool-replay tasks/replays/dsv4_agent_tool_replay_turn02_request.json
```

`--base-url` should be the endpoint root. The harness appends `/v1/chat/completions` itself.

Optional: add `--include-official-vectors` for exact antirez/ds4 byte-level checks (best with BF16, may drift on quants). Set `DS4_ROOT=/path/to/ds4` or pass `--ds4-root /path/to/ds4` if the DS4 checkout is not a sibling of this repository.

---

## Results (2026-05-09, IQ2_XXS, mounted-code build aafa0ad91 + uncommitted prefill changes)

Command:

```bash
scripts/dsv4_regression.py \
  --base-url http://127.0.0.1:18089 \
  --model deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --timeout 1200
```

| Test | Result | Detail |
|------|--------|--------|
| simple_ok | **PASS** | content='OK', finish=stop |
| basic_math | **PASS** | content='4', finish=stop |
| minimal_tool_auto | **PASS** | structured `bash` tool call |
| minimal_tool_required | **PASS** | structured `bash` tool call |
| tool_replay_turn02_nonstream | **PASS** | structured `bash` tool call |
| tool_replay_turn02_stream | **PASS** | structured `bash` tool call; `cached_tokens=1408` |
| tool_replay_turn02_cache_reuse | **PASS** | second replay structured; `cached_tokens=1408` |

**7/7 passed.**

### IQ1_M note

The same suite against IQ1_M failed strict structured tool-call assertions, frequently emitting literal XML `<tool_call>` text. Since IQ2_XXS passes fresh, streaming, and checkpoint-restored tool replay cases on the same mounted-code build, treat the IQ1_M failures as quant-quality/model-behavior limitations rather than a blocking server/parser regression for the usual DeepSeek4 acceptance model.

---

## Current Interpretation

No server/parser regression is currently proven with the usual IQ2_XXS acceptance model.

The earlier IQ1_M run showed literal XML `<tool_call>` text in minimal, streaming, and cache-reuse cases. Re-running the same mounted-code build with IQ2_XXS produced structured `tool_calls` for all cases, including checkpoint-restored tool replay. That rules out a broad checkpoint-restore parser bug for the standard regression model.

Working interpretation:

- IQ1_M is too lossy for strict structured tool-call acceptance and should not be the default tool-call regression model.
- IQ2_XXS remains the standard regression target for DeepSeek4 server/tool/cache correctness.
- If IQ1_M support is desired, track it separately as a quant-quality / parser-robustness improvement, not as a blocker for the prefill-speed branch.

---

## Next Steps

1. Continue prefill-speed validation with IQ2_XXS as the default regression model.
2. Keep the mounted-code container workflow for rapid branch validation.
3. When the prefill-speed branch is ready, rebuild `llamatrifecta_deepseekv4:latest` from a clean commit and validate through systemctl-managed Corral.
4. Optional separate work: if IQ1_M must support strict structured tool calling, investigate prompt/parser robustness for very low-bit quants separately.

---

## Files Added in This Session

- `scripts/dsv4_regression.py` — endpoint regression harness
- `tasks/replays/dsv4_agent_tool_replay_turn02_request.json` — multi-turn tool replay fixture
- `tasks/dsv4_regression_handoff.md` — this file

## Container Status

No regression server is expected to stay running from this handoff. For ad hoc resident-server lifecycle commands, see `tasks/dsv4_resident_bench_server.md`.
