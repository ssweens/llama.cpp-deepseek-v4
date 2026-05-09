# DeepSeek4 resident benchmark server

Use this only for ad hoc DeepSeek4 performance/debug sweeps where repeatedly loading the full model would dominate iteration time. Do **not** use this as final Corral validation; final Corral/Pi validation must go through the systemd-managed Corral service.

## When to use

- Multi-variant DeepSeek4 prefill/decode sweeps with `llama-benchy`.
- Endpoint regressions against a mounted working tree after backend changes.
- Short profiling runs that need the full model resident.

Avoid this for:

- Final product validation through Corral.
- Corral-managed `corral-llama-*` containers.
- Repeated full-model `docker run llama-bench` sweeps.

## Start normal mounted-code server

This uses the DeepSeekV4 image only as a runtime base and runs the mounted build from `LLAMA_DSV4_ROOT`.

Set local paths first:

```bash
export LLAMA_DSV4_ROOT=/path/to/llama.cpp-deepseek-v4
export LLAMA_BENCHY_ROOT=/path/to/llama-benchy
export MODEL_ROOT=/path/to/models
export SUPMODEL_ROOT=/path/to/supmodels
```

Start the server:

```bash
docker run -d --name dsv4-iq2xxs-benchy-server \
  --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -p 18089:18089 \
  -v "$MODEL_ROOT:/models:ro" \
  -v "$SUPMODEL_ROOT:/supmodels:ro" \
  -v "$LLAMA_DSV4_ROOT:/src" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LD_LIBRARY_PATH=/src/build-dsv4-container/bin:/opt/cuda/lib64:/usr/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
  llamatrifecta_deepseekv4:latest \
  bash -lc 'cd /src && exec ./build-dsv4-container/bin/llama-server \
    --host 0.0.0.0 --port 18089 \
    -m /models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-IQ2_XXS.gguf \
    -dev CUDA0,CUDA1,CUDA2 -ngl 99 -c 65536 -ts 17,17,10 \
    -fit off -sm layer -fa on --no-mmap \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 2 --threads-batch 2 -b 4096 -ub 1024 \
    --jinja \
    --chat-template-file /models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-chat_template.jinja \
    --temp 0.0 --top-p 1.0 --repeat-penalty 1.05 \
    --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
    --reasoning on --reasoning-budget 1024 \
    --reasoning-budget-message "... thinking budget exceeded, let us answer now."'
```

Wait for readiness:

```bash
for i in $(seq 1 180); do
  curl -fsS -m 2 http://127.0.0.1:18089/health && echo && break
  sleep 1
done
```

OpenAI-compatible base URL for API clients:

```text
http://127.0.0.1:18089/v1
```

Regression harness base URL, because `scripts/dsv4_regression.py` appends `/v1/chat/completions` itself:

```text
http://127.0.0.1:18089
```

## Stop / remove

Only stop this ad hoc dev container. Do not directly stop Corral-managed `corral-llama-*` containers.

```bash
docker stop dsv4-iq2xxs-benchy-server
docker rm dsv4-iq2xxs-benchy-server
```

## Rebuild mounted binaries

After source changes, rebuild before restarting the resident server:

```bash
docker run --rm \
  --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$LLAMA_DSV4_ROOT:/src" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LD_LIBRARY_PATH=/src/build-dsv4-container/bin:/opt/cuda/lib64:/usr/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
  llamatrifecta_deepseekv4:latest \
  bash -lc 'cd /src && cmake --build build-dsv4-container --target llama-bench llama-server -j 8'
```

## Coherence-enabled API baseline

Do not pass `--skip-coherence`; coherence is enabled by default. `--no-cache` sends `cache_prompt=false` for cold-prefill comparability.

```bash
cd "$LLAMA_BENCHY_ROOT"
.venv/bin/llama-benchy \
  --base-url http://127.0.0.1:18089/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash \
  --pp 2048 8192 \
  --tg 1 \
  --runs 3 \
  --concurrency 1 \
  --no-cache \
  --no-warmup \
  --no-adapt-prompt \
  --latency-mode none \
  --save-result /tmp/dsv4_cuda_prefill_profile/benchy_pp2048_8192_tg1_coherence_runs3.json \
  --format json
```

## Endpoint regression

DeepSeek4 generated reasoning must count against budget; keep `--max-tokens 2048` for these tests.

```bash
cd "$LLAMA_DSV4_ROOT"
python3 scripts/dsv4_regression.py \
  --base-url http://127.0.0.1:18089 \
  --model deepseek-ai/DeepSeek-V4-Flash-IQ2_XXS \
  --max-tokens 2048 \
  --timeout 900 \
  --tool-replay tasks/replays/dsv4_agent_tool_replay_turn02_request.json
```

## MMQ profiling mode

For routed expert MMQ work, add these environment variables to the `docker run` command and restart the server:

```bash
-e GGML_PROFILE_MMQ_ID=1 \
-e GGML_CUDA_DISABLE_GRAPHS=1 \
```

`GGML_PROFILE_MMQ_ID=1` prints routed-expert component timing and expert tile population on server shutdown. Graphs are disabled because the profiler synchronizes inside the graph path and is for diagnostics only.

Save logs after stopping the profiling server:

```bash
docker logs dsv4-iq2xxs-benchy-server > /tmp/dsv4_cuda_prefill_profile/mmq_profile_server.log 2>&1
```
