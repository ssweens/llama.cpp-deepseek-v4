# DeepSeek4 MTP Loader/Probe Notes

Branch: `work/dsv4-mtp-loader-probe`
Date: 2026-05-09

## Current status update (2026-05-09)

- MTP runtime is no longer probe-only: `--spec-type mtp --model-draft <MTP-support.gguf>` loads the MTP draft GGUF, routes drafting through `common/speculative.cpp`, and uses the existing server verifier/accept loop.
- `DSV4_MTP_PROBE=1` is diagnostics-only. Normal MTP runtime works without it; diagnostics mode keeps the full HC/raw handoff copies and detailed preview logging.
- Code organization was reset to mirror existing patterns: speculative orchestration lives in `common/speculative.cpp` / server verifier flow, DeepSeek4 MTP metadata/tensor validation lives behind `src/models/deepseek4.*`, and the server no longer calls a `src/models/models.h` MTP hook.
- Boundary cleanup: generic server/speculative code no longer contains DS4/MTP env names, timing probes, or DS4 MTP log/comment text; `llama_context` uses generic model/memory hooks for MTP draft validation, raw-window policy, and recurrent-state replay instead of direct DS4 references.
- Docker mounted-code validation shows real accepted drafts and correct visible content on smoke prompts. After sidecar depth-2 plus verifier full-accept cleanup skip, a raw deterministic ctx8192 prompt improved from target-only `46.94 tok/s` to MTP `52.12 tok/s` with matching visible text and `20/20` accepted/generated drafts.
- Standard IQ2_XXS `llama-benchy pp2048/tg32` is still not solved: freecheck MTP generates 0-1 drafts on the natural bench prompt, so tg remains below the prior target-only baseline. This means the remaining blocker is MTP draft quality/alignment on long natural prompts, not verifier cleanup overhead.

## Implemented in this branch

- Added DeepSeek4 MTP validation/probe plumbing behind the existing speculative interface:
  - `--spec-type mtp` selects MTP;
  - existing `--model-draft FNAME` carries the DeepSeek4 MTP support GGUF path;
  - existing `--draft-max N` / `common_params_speculative::n_max` carries the future draft cap.
- Added MTP draft/support GGUF validation behind `llama_model` dispatch; DeepSeek4-specific tensor/schema checks live in `src/models/deepseek4.*`.
- Added a default-off MTP state output hook for one-token decode graphs; DeepSeek4 currently supplies final HC state as that opaque handoff tensor.
- Added a host-side copy buffer for that MTP handoff state in `llama_context` for the next draft-one probe step.
- Added MTP draft GGUF tensor data loading into a persistent backend weight buffer using existing draft placement knobs such as `--device-draft`.
- Added an env-gated projection/top-1 probe that feeds base token embedding + captured target HC through MTP draft `enorm/e_proj`, `hnorm/h_proj`, HC head/norm, and base output, then logs projection top-1 vs target argmax.
- Extended the probe/runtime graph to run a one-token MTP transformer block using MTP draft attention, FFN, HC, and output-head tensors before logging draft top-1 vs target argmax. The block now consumes a private host-backed raw-window cache on continuation steps.
- Fixed CPU-only DeepSeek4 reserve/probe issues found by the real IQ1_M target run: BF16 activation inputs are cast back to F32 for non-BF16 weight matmuls so CPU supports the op, and disconnected probe top-1 tensors are explicitly expanded into the graph before decode copies them.
- Runtime MTP drafting now uses `common_speculative_init()` and the existing server speculative verifier/accept loop. Accepted-token private raw commit exists for the integrated probe path, but the viable speed path still requires a sidecar/MTP-only graph instead of target-graph-integrated MTP blocks.
- Documented the new server flags in `tools/server/README.md`.

## Staged MTP draft/support GGUF

Downloaded MTP draft/support GGUF:

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

The context loader validates metadata/tensor-directory shape and loads MTP draft tensor data during normal MTP runtime. `DSV4_MTP_PROBE=1` enables additional diagnostics copies/logging; it is not required for runtime drafting. Accepted-token private raw/HC commit is still incomplete.

Validation requires:

- target model architecture is `LLM_ARCH_DEEPSEEK4`;
- target model has token embedding and output tensors for MTP sharing;
- MTP draft GGUF file exists;
- MTP draft GGUF `general.architecture == deepseek4_mtp_support`;
- MTP draft GGUF `deepseek4.mtp_layer_count == 1`;
- MTP draft GGUF `deepseek4.nextn_predict_layers == 1`;
- MTP draft GGUF `deepseek4.expert_count` matches target `n_expert`;
- all 32 expected `mtp.0.*` tensors exist with exact hparams-derived shapes and accepted types.

Accepted routed expert MTP draft types mirror DS4 authority: `IQ2_XXS`, `Q2_K`, or `Q4_K`; the current downloaded GGUF uses `Q4_K`.

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

After the raw-current one-token MTP block probe was added, the same target/sidecar command reached health after 88 seconds; one-token completion returned newline after ~9 seconds and logged:

```text
dsv4 mtp block probe: target_argmax=201 draft_top1=2390 match=0
```

After the private host-backed raw-window cache was added, an `n_predict=2` run reached health after 94 seconds, returned `"\n\t"`, and logged two probe rows. The second row used one previous private MTP raw-cache row instead of current-token-only attention:

```text
dsv4 mtp block probe: target_argmax=201 draft_top1=2390 match=0
dsv4 mtp block probe: target_argmax=200 draft_top1=5 match=0
```

After the recursive draft-2 probe was added and switched to `ggml_argmax` for stable graph-internal greedy token IDs, a `--draft-max 2` run reached health after 80 seconds, returned the same `"\n\t"`, and logged:

```text
dsv4 mtp block probe: target_argmax=201 draft_top1=2390 draft2_top1=42 match=0
dsv4 mtp block probe: target_argmax=200 draft_top1=5 draft2_top1=23166 match=0
```

The smaller `/mnt/supmodels/gguf/deepseek-ai__DeepSeek-V4-Flash-Q2_K_S.with-template.gguf` target was rejected as corrupted/incomplete before MTP validation (`blk.4.ffn_down_exps.weight` out of file bounds), so the real probe used the valid IQ1_M target.

Whitespaces:

```text
git diff --check
```

Result: passed.

Note: whole-file `clang-format --dry-run` on `tools/server/server-context.cpp` reports many pre-existing formatting deviations outside touched lines. `git-clang-format --force origin/main -- ...` was applied to changed lines only.

## Not implemented yet

- No Qwen-PR-style sidecar/MTP-only graph or separate MTP context fed by target hidden state.
- No deterministic token-level parity harness for target-only vs MTP.
- No benchmark-proven speedup; current integrated MTP accepts tokens but is slower than target-only.

## HC-state probe plumbing

When a valid DeepSeek4 target is loaded with `--spec-type mtp --model-draft <MTP.gguf>` and `DSV4_MTP_PROBE=1`:

- server startup loads the MTP sidecar GGUF tensor data into a persistent backend weight buffer, using the target model HC/output tensor buffer type;
- server startup sets a private context probe flag;
- graph reuse accounts for that flag;
- DeepSeek4 marks the final per-token `hc_state` as generic MTP state output only for `n_tokens == 1 && n_outputs == 1` graphs, avoiding prompt-prefill HC output blowups;
- decode copies that F32 handoff state into `llama_context::get_mtp_state()`;
- the graph computes a one-token MTP block through the sidecar input projections, logical layer-1 raw attention, FFN, sidecar HC head/norm, and base output, and the server logs it against the target argmax;
- decode now copies both the target HC handoff and the sidecar block's output HC handoff, preparing recursive MTP draft probing;
- when `--draft-max > 1`, the probe unrolls a second sidecar block in the same graph: draft[0] comes from target HC + current token, draft[1] comes from sidecar HC + draft[0] token, with no target-layer execution and no emitted-token changes;
- the recursive block's raw K row is copied to host as a draft raw-row candidate for future accept-time private raw-cache commit; it is not appended to persistent raw state yet;
- server logging keeps probe-only verifier-preview counters: draft[1] is compared with the current target argmax, while the previous step's draft[2] is compared with the current target argmax only if previous draft[1] matched the actual token that entered this target decode step;
- continuation steps feed prior private MTP raw rows from host state into the block and copy the current MTP raw row back after compute; the host cache stores at most `raw_window - 1` prior rows because the graph concatenates the current row separately;
- the private raw state resets on discontinuities/non-single-token probe batches and on server slot prompt/reset paths, and remains separate from target KV/cache state.

This is intentionally still a handoff/probe surface. It does not alter emitted tokens. DS4 authority confirmed the MTP sidecar uses logical layer id 1, whose DeepSeek4 compression ratio is zero, so the sidecar drafter itself needs private raw-window and HC state, not private compressed/indexer state. Target compressed/indexer frontier rollback is covered by existing hybrid-ISWA partial state checkpoints: `llama_kv_cache_iswa` handles SWA KV, `mem_recr` carries compressor/indexer frontiers, and `dsv4_state_write/read` carries sequence-local compressed attention/indexer cache rows.

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

- The MTP block must not reuse or mutate target model KV/cache state. It now has a private host-backed raw-window cache analogous to DS4 authority's `mtp_raw_cache` / `mtp_n_raw`. The sidecar block is logical layer 1 and dense (`compress_ratio == 0`), so compressed/indexer state belongs to the target verifier checkpoint/rollback path, not the MTP drafter block.
- The DS4 authority calls the MTP block with logical layer id `1`; the llama.cpp graph preserves that schedule for the current raw-one block probe.
- The current block probe uses private raw-window attention only. Its top-1 log validates sidecar tensor loading, target HC handoff, sidecar attention/FFN graph wiring, private raw-cache handoff, recursive sidecar-HC handoff for draft-2, graph outputs, and server logging, but it is not yet a complete speculative draft-quality metric.
- `ggml_argsort_top_k(k=1)` is fine as a disconnected probe output, but feeding it into a recursive MTP block perturbed the copied first top-1 on CPU. The recursive probe uses `ggml_argmax` instead, matching greedy backend sampling and preserving the draft[0] value seen in non-recursive probe mode.

## Next exact step

Pivot from target-graph-integrated MTP to a sidecar/MTP-only graph, matching Qwen PR #22673 and `../ds4/`:

1. Make the normal DeepSeek4 target graph expose/copy the last target HC row cheaply, without running MTP blocks in the target graph.
2. Add an MTP-only graph/context path that consumes `(target_or_mtp_hc, token, pos, private raw cache)` and returns `(top1, next_hc, raw row)` using the loaded MTP GGUF tensors.
3. Drive that graph from `common_speculative_state_mtp`, recurse up to `--draft-max`, and trim/commit private raw rows based on existing verifier accept counts.
4. Re-run deterministic parity smoke and the standard IQ2_XXS resident `llama-benchy` benchmark before claiming progress.
