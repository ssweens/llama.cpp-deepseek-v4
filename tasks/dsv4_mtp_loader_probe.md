# DeepSeek4 MTP Loader/Probe Notes

Branch: `work/dsv4-mtp-loader-probe`
Date: 2026-05-09

## Implemented in this branch

- Added DeepSeek4 MTP validation/probe plumbing behind the existing speculative interface:
  - `--spec-type mtp` selects MTP;
  - existing `--model-draft FNAME` carries the DeepSeek4 MTP support GGUF path;
  - existing `--draft-max N` / `common_params_speculative::n_max` carries the future draft cap.
- Added server startup validation for DeepSeek4 MTP sidecar GGUFs.
- Added a default-off MTP state output hook for one-token decode graphs; DeepSeek4 currently supplies final HC state as that opaque handoff tensor.
- Added a host-side copy buffer for that MTP handoff state in `llama_context` for the next draft-one probe step.
- Added env-gated sidecar tensor data loading into a persistent backend weight buffer.
- Added an env-gated projection/top-1 probe that feeds base token embedding + captured target HC through sidecar `enorm/e_proj`, `hnorm/h_proj`, sidecar HC head/norm, and base output, then logs projection top-1 vs target argmax.
- Fixed CPU-only DeepSeek4 reserve/probe issues found by the real IQ1_M target run: BF16 activation inputs are cast back to F32 for non-BF16 weight matmuls so CPU supports the op, and the disconnected projection-top1 probe tensor is explicitly expanded into the graph before decode copies it.
- Kept runtime MTP drafting/speculative commit disabled. Passing `--spec-type mtp --model-draft <MTP.gguf>` validates only and logs that drafting is not enabled yet; `DSV4_MTP_PROBE=1` additionally loads sidecar tensor data and enables state/projection probe plumbing. Server slot initialization intentionally skips `common_speculative_init()` for MTP until the full runtime drafter is wired.
- Documented the new server flags in `tools/server/README.md`.

## Staged sidecar

Downloaded sidecar:

```text
/mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Verified local header:

```text
size = 3,807,602,400 bytes
general.architecture = deepseek4_mtp_support
general.name = DeepSeek V4 Flash MTP support
deepseek4.expert_count = 256
deepseek4.mtp_layer_count = 1
deepseek4.nextn_predict_layers = 1
tensors = 32
```

## Validation behavior

The server validator is metadata/tensor-directory only in default startup mode. It does not load sidecar tensor data unless `DSV4_MTP_PROBE=1` is also set. Probe mode loads the already-validated sidecar tensors into a persistent backend weight buffer and runs the projection-only probe graph, but still does not run the full MTP transformer block or alter emitted tokens.

Validation requires:

- target model architecture is `LLM_ARCH_DEEPSEEK4`;
- target model has token embedding and output tensors for MTP sharing;
- sidecar file exists;
- sidecar `general.architecture == deepseek4_mtp_support`;
- sidecar `deepseek4.mtp_layer_count == 1`;
- sidecar `deepseek4.nextn_predict_layers == 1`;
- sidecar `deepseek4.expert_count` matches target `n_expert`;
- all 32 expected `mtp.0.*` tensors exist with exact hparams-derived shapes and accepted types.

Accepted routed expert sidecar types mirror DS4 authority: `IQ2_XXS`, `Q2_K`, or `Q4_K`; the current downloaded sidecar uses `Q4_K`.

## Quality gates run

Build:

```text
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release --target llama-server -j 8
```

Help check:

```text
llama-server --help | rg -- '--spec-type|--model-draft|--draft-max'
```

Result: existing speculative flags expose the MTP control surface.

Unsupported-target startup check:

```text
DSV4_MTP_PROBE=1 llama-server -m /tmp/stories260K.gguf \
  --spec-type mtp \
  --model-draft /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --ctx-size 32 --no-warmup --port 18191
```

Result: server exits with rc `1`, reports no sidecar tensor-data load, and emits clean error:

```text
failed to validate DeepSeek4 MTP sidecar '...DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf': MTP speculative decoding is only supported with DeepSeek4 target models in this build
```

Default no-MTP startup check:

```text
llama-server -m /tmp/stories260K.gguf --ctx-size 32 --no-warmup --port 18192
curl http://127.0.0.1:18192/health
```

Result: health returned `{"status":"ok"}` after 2 seconds.

Real DeepSeek4 target + MTP sidecar probe check:

```text
DSV4_MTP_PROBE=1 llama-server \
  -m /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/deepseek-ai__DeepSeek-V4-Flash-IQ1_M.gguf \
  --spec-type mtp \
  --model-draft /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --draft-max 1 \
  --ctx-size 32 --batch-size 16 --ubatch-size 1 --parallel 1 \
  --no-warmup --threads 8 --threads-batch 8 -fit off --flash-attn off --port 18202
curl http://127.0.0.1:18202/health
curl http://127.0.0.1:18202/completion -d '{"prompt":"Hi","n_predict":1,"temperature":0,"cache_prompt":false}'
```

Result: health returned `{"status":"ok"}` after 82 seconds; one-token completion returned newline after ~6 seconds and logged:

```text
validated DeepSeek4 MTP sidecar '...DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf' (draft=1)
load_dsv4_mtp_sidecar: loaded 32 DeepSeek4 MTP sidecar tensors into CPU_Mapped buffer ( 3631.21 MiB)
dsv4 mtp projection probe: target_argmax=201 projection_top1=20219 match=0
```

The smaller `/mnt/supmodels/gguf/deepseek-ai__DeepSeek-V4-Flash-Q2_K_S.with-template.gguf` target was rejected as corrupted/incomplete before MTP validation (`blk.4.ffn_down_exps.weight` out of file bounds), so the real probe used the valid IQ1_M target.

Whitespaces:

```text
git diff --check
```

Result: passed.

Note: whole-file `clang-format --dry-run` on `tools/server/server-context.cpp` reports many pre-existing formatting deviations outside touched lines. `git-clang-format --force origin/main -- ...` was applied to changed lines only.

## Not implemented yet

- No full MTP transformer-block graph.
- No draft token generation.
- No speculative verification/commit.

## HC-state probe plumbing

When a valid DeepSeek4 target is loaded with `--spec-type mtp --model-draft <MTP.gguf>` and `DSV4_MTP_PROBE=1`:

- server startup loads the MTP sidecar GGUF tensor data into a persistent backend weight buffer, using the target model HC/output tensor buffer type;
- server startup sets a private context probe flag;
- graph reuse accounts for that flag;
- DeepSeek4 marks the final per-token `hc_state` as generic MTP state output only for `n_tokens == 1 && n_outputs == 1` graphs, avoiding prompt-prefill HC output blowups;
- decode copies that F32 handoff state into `llama_context::get_mtp_state()`;
- the graph computes a projection-only top-1 through the sidecar input projections and sidecar output head, and the server logs it against the target argmax.

This is intentionally only a handoff/probe surface. It does not alter emitted tokens and does not run the one-layer MTP transformer block yet, so the logged projection top-1 is not a speculative draft token.

## Full-block design notes

The DS4 authority path (`metal_graph_eval_mtp_draft_from_hc`) shows the full MTP draft is:

1. base token embedding for the current token;
2. sidecar `enorm/e_proj` projected and repeated across HC rows;
3. target final HC state normalized row-wise by sidecar `hnorm` and projected by sidecar `h_proj`;
4. sum of the two projections as the MTP block input HC state;
5. one sidecar transformer block using sidecar attention/FFN/HC tensors;
6. sidecar HC head + sidecar `norm`;
7. base output projection.

Important constraints for the next implementation:

- The MTP block must not reuse or mutate target model KV/cache state. It needs private MTP raw-window/cache/state tensors analogous to DS4 authority's `mtp_raw_cache`, `mtp_n_raw`, and per-layer compressed/indexer state.
- The DS4 authority calls the MTP block with logical layer id `1`; the llama.cpp graph should preserve the same DS4 rope/compression schedule instead of assuming logical layer `0` from the sidecar tensor prefix.
- The current projection/top-1 probe intentionally stops before the transformer block. Its top-1 log validates sidecar tensor loading, target HC handoff, graph outputs, and server logging, but it is not a useful draft-quality metric.

## Next exact step

Create a follow-up branch for draft-one probing, e.g.:

```text
work/dsv4-mtp-hc-probe
```

Recommended scope:

1. Build the missing one-token MTP transformer-block graph and its private cache/state handling between the now-working sidecar input projection and sidecar output head.
2. Under `DSV4_MTP_PROBE=1`, log full draft top-1 and compare it against the target argmax without changing emitted tokens.
3. Do not implement speculative commit until deterministic no-MTP vs probe token streams match exactly.
