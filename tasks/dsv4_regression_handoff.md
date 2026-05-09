# DeepSeek-V4 Regression Harness — Handoff Report

## How to Run

### Prerequisites
- Working tree on `work/dsv4-prefill-speedup` branch (or equivalent)
- Mounted-code container build exists at `build-dsv4-container/`
- Preferred regression model: IQ2_XXS at `/mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-IQ2_XXS.gguf`
- IQ1_M is useful for smoke/coherence checks but is too lossy for strict structured tool-call regression acceptance.

### Start the mounted-code server (if not already running)

```bash
cd /home/bigkahuna/src/llama.cpp-deepseek-v4

# Server is already running as dsv4-iq2xxs-mounted-server on port 18089
# If you need to restart it:
docker stop dsv4-iq2xxs-mounted-server 2>/dev/null || true

docker run --rm --init -d \
  --name dsv4-iq2xxs-mounted-server \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /home/bigkahuna/models:/home/bigkahuna/models \
  -v /mnt/models:/mnt/models \
  -v /mnt/supmodels:/mnt/supmodels \
  -v /home/bigkahuna/src/llama.cpp-deepseek-v4:/src \
  --gpus all -p 18089:18089 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LD_LIBRARY_PATH=/src/build-dsv4-container/bin:/opt/cuda/lib64:/usr/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
  llamatrifecta_deepseekv4:latest \
  bash -lc 'cd /src && ./build-dsv4-container/bin/llama-server \
    --host 0.0.0.0 --port 18089 \
    -m /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-IQ2_XXS.gguf \
    -dev CUDA0,CUDA1,CUDA2 -ngl 99 -c 65536 -ts 17,17,10 \
    -fit off -sm layer -fa on --no-mmap \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 2 --threads-batch 2 -b 4096 -ub 1024 \
    --jinja \
    --chat-template-file /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-chat_template.jinja \
    --temp 0.0 --top-p 1.0 --repeat-penalty 1.05 \
    --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
    --reasoning on --reasoning-budget 1024 \
    --reasoning-budget-message "... thinking budget exceeded, let'"'"'s answer now." \
    >/tmp/llama-server-18089.log 2>&1'

# Wait for ready (~60s)
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:18089/health | grep -q ok && echo "READY at ${i}s" && break
  sleep 1
done
```

### Run the suite

```bash
cd /home/bigkahuna/src/llama.cpp-deepseek-v4

scripts/dsv4_regression.py \
  --base-url http://127.0.0.1:18089 \
  --model deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --timeout 1200
```

Optional: add `--include-official-vectors` for exact antirez/ds4 byte-level checks (best with BF16, may drift on quants).

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
| mina_turn02_nonstream_tool | **PASS** | structured `bash` tool call |
| mina_turn02_stream_tool | **PASS** | structured `bash` tool call; `cached_tokens=1408` |
| mina_turn02_cache_reuse | **PASS** | second replay structured; `cached_tokens=1408` |

**7/7 passed.**

### IQ1_M note

The same suite against IQ1_M failed strict structured tool-call assertions, frequently emitting literal XML `<tool_call>` text. Since IQ2_XXS passes fresh, streaming, and checkpoint-restored Mina cases on the same mounted-code build, treat the IQ1_M failures as quant-quality/model-behavior limitations rather than a blocking server/parser regression for the usual DeepSeek4 acceptance model.

---

## Current Interpretation

No server/parser regression is currently proven with the usual IQ2_XXS acceptance model.

The earlier IQ1_M run showed literal XML `<tool_call>` text in minimal, streaming, and cache-reuse cases. Re-running the same mounted-code build with IQ2_XXS produced structured `tool_calls` for all cases, including checkpoint-restored Mina replay. That rules out a broad checkpoint-restore parser bug for the standard regression model.

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
- `tasks/replays/dsv4_mina_turn02_tools_request.json` — Mina turn-02 fixture
- `tasks/dsv4_regression_handoff.md` — this file

## Container Status

`dsv4-iq2xxs-mounted-server` running on port 18089 for further debugging.
