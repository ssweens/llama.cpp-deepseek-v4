# Lessons

- When the user rejects a workaround and asks for the real fix, do not present a runtime flag fallback (`disable FA`, manual script cleanup, etc.) as the answer.
  - Trace the backend/runtime rejection path to the exact code gate first.
  - Only present a workaround as temporary diagnostic evidence, never as the implementation target.
- Do not rebuild reflexively after a small edit when the immediate task is read-only diagnosis.
  - First state exactly what code changed and why a rebuild is or is not required.
  - Keep investigation steps separate from implementation validation so the user never has to ask what triggered a build.
- Do not launch heavy local reproduction runs on huge models during a remote-pipeline incident unless the user has explicitly asked for local reproduction.
  - First exhaust the remote logs, sentinels, serial console, and exact command traces already available.
  - If a local run is truly needed, say exactly why it is necessary before starting it.

- When validating a new model family, do not guess the chat/output format from nearby models.
  - Trace the official source of truth end-to-end first: HF README/docs, encoding or chat-template files, converter metadata injection, GGUF metadata, and llama.cpp parser selection.
  - If the upstream release does not ship a Jinja chat template, do not treat a nearby built-in template as equivalent without proof.
  - Separate parser-format bugs from model-graph bugs by validating both:
    1. structured parsing with the correct official format, and
    2. raw token output with chat parsing disabled.
- Do not present response-quality conclusions from runs that used an unverified template or parser path.
- When the repository already includes an additional implementation reference (for example MLX for the same model family), do not tunnel on a single upstream implementation.
  - Cross-check PyTorch/HF behavior against the MLX implementation before concluding a parity bug is isolated.
  - Especially for precision, cache/state propagation, and attention-path semantics, compare all available reference implementations before changing runtime math.
- When debugging a new architecture, do not infer precision/layout semantics from checkpoint artifacts alone before reading the model paper / official architecture writeup.
  - Read the paper first, then reconcile the paper with HF inference code, converter behavior, and checkpoint tensors.
  - Treat surprising checkpoint details (extra scale tensors, mixed dtypes, etc.) as hypotheses until the paper and official implementation confirm their role.
- When the user asks to "not stop until all gaps are covered", do not pause at intermediate stabilization points.
  - Close every currently identified functional gap in the active path before proposing commit/push.
  - Treat crash-fix + partial parity as "in progress" until the remaining known semantic gaps are either implemented or explicitly disproven with tests.
- When the user points out that a model implementation "should be close" and says to use references more, immediately re-diff against every available source of truth.
  - Use the paper for required semantics, the official HF inference for exact runtime behavior, MLX for an independent implementation, and external forks for llama.cpp integration details.
  - Do not accept runtime stability as semantic progress unless deterministic outputs or logits are checked against a reference.
  - Pull the live branch/PR references, not stale local snapshots; for DeepSeek-V4 MLX specifically, fetch `ml-explore/mlx-lm` PR 1192 before drawing conclusions.

## DSv4 BF16 garbage output (2026-04-26)
- **Symptom:** Multilingual gibberish from `llama-completion` despite all graph code, op kernels, GGUF metadata keys, and tensor names matching antirez fork verbatim. Both our binary and antirez's CPU build produced the same garbage on our v2 GGUF.
- **Investigation that almost missed it:** Spent hours auditing graph-construction divergences. Confirmed tensor shapes matched HF, then "verified" tensor data against HF using `f.get_tensor(name).float()` — but for `torch.float8_e4m3fn` tensors, `.float()` reinterprets each FP8 byte as its FP8 numeric value (no scale applied). Both sides looked equal, but BOTH were wrong.
- **Actual root cause:** Our DSv4 `prepare_tensors()` overrode the base method to handle 3-tuple yields from `modify_tensors` (for MXFP4 expert packing) — and forgot to call `self.dequant_model()`. That meant FP8 dense weights (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, indexer `wq_b`, shared expert MLP) had their raw FP8 byte values cast directly to BF16 with **no E8M0 block scale applied**. Magnitudes were ~3000× too large per layer × 43 layers = pure noise logits.
- **Diagnostic that found it:** Compared GGUF tensor data byte-for-byte against HF `torch.float8_e4m3fn` weight × E8M0 scale (proper dequantization), instead of against the raw FP8-as-FP32 reinterpretation. Match failed for every FP8 tensor.
- **Lesson 1 — verification on quantized formats:** When validating quantized weight conversion, dequantize using the official scale formula. `torch.float8_e4m3fn → .float()` returns the FP8 numeric value, not the dequantized real-world value. Always multiply by the block scale.
- **Lesson 2 — overriding `prepare_tensors`:** If you override `prepare_tensors` in a `TextModel` subclass and don't call `super().prepare_tensors()`, you skip `self.dequant_model()` (which dequantizes FP8/FP4/GPTQ/etc.). Either call `super()` or explicitly invoke `self.dequant_model()` at the start.
- **Lesson 3 — when both forks fail identically, the bug is in shared inputs (GGUF), not graph code.** Antirez's CPU binary produced ~identical garbage to ours on our GGUF. That should have been the first signal to inspect the GGUF data path, not graph divergence — instead of spending hours auditing graph code that was actually correct.
- **Lesson 4 — magnitude sanity check:** Before deep audits, dump a few weight values from the GGUF and compare to plausible NN weight magnitudes (typically O(0.01–1)). Values like `-128.0`, `-80.0` for an attention projection are an immediate red flag for missing dequant.

## DSv4 CUDA decode degeneration after first compressed slot (2026-04-27)
- **Symptom:** Built `build-cuda` produced coherent first ~3 generated tokens then drifted into repetition (`Let's break it step-by-step: ... 3 apples: 3 apples: ...`). True CPU-only build (`build-cuda` with `CUDA_VISIBLE_DEVICES=` cleared, or a fresh `build-cpu` with `GGML_CUDA=OFF`) produced fully coherent output for the same GGUF and prompt.
- **First narrowing:** Token-by-token n=2/3/4 comparison showed first divergence at exactly the decode step where the first ratio-4 compressed slot is read after being written. Initial hypothesis was a read-after-write scheduling issue between `cpy_k` and the compressed-cache read.
- **Confirming hypothesis was wrong:** Adding explicit graph dependencies (concat of old cache rows + just-computed compressed row, viewing `cpy_k`’s output tensor for the raw SWA read) did not change CUDA output. CPU output already matched antirez. So the bug had to be inside the CUDA backend.
- **Localizing CUDA:** Running the same `build-cuda` binary with `CUDA_VISIBLE_DEVICES=` produced perfect output. With CUDA visible, even `-ngl 0 -fa 0` showed `CUDA0 compute buffer size = 1270.50 MiB` because the scheduler offloads ops to CUDA when possible. So the corruption came from CUDA kernels participating in DSv4 graph nodes, not from `-ngl` placement.
- **Root cause:** `ggml/src/ggml-cuda/dsv4.cu` `kernel_dsv4_rope_tail_f32`’s YaRN math was wrong:
  - `rope_yarn_ramp_device` returned `clamp((i0/2 - low)/(high - low), 0, 1)`. CPU `rope_yarn_ramp` returns `1 - clamp(...)`. Sign of the ramp was inverted.
  - `rope_yarn_device` set `theta_scaled = theta * freq_scale` and then “blended” it against `theta_scaled / mix * mix`, which algebraically equals `theta_scaled`. Result: zero YaRN extrapolation. CPU instead blends `theta_interp = freq_scale * theta_extrap` against `theta_extrap` using `ramp_mix = ramp * ext_factor`.
  - CPU also applies `mscale *= 1 + 0.1 * log(1 / freq_scale)` when `ext_factor != 0`. CUDA did not. Our `get_rope_cfg` divides `attn_factor` by that exact factor in advance assuming the runtime would multiply it back. With CUDA never multiplying, the on-CUDA magnitude was systematically too small.
- **Fix:** Rewrite the CUDA `rope_yarn_ramp_device` and `rope_yarn_device` to mirror `ggml-cpu/ops.cpp`’s `rope_yarn_ramp` and `rope_yarn` exactly: use `1 - clamp(...)`, blend `theta_interp` against `theta_extrap`, and apply the magnitude correction when `ext_factor != 0`.
- **Lesson 1 — kernel parity:** When porting a math kernel (especially RoPE, YaRN, softmax_ext, sinks) from CPU reference to CUDA, mirror the CPU function _line for line_ first. Optimize only after byte-level/numerical parity is verified by `test-backend-ops` or by full-model decode equality.
- **Lesson 2 — scheduler reach:** `-ngl 0` is not equivalent to “CPU only” in a `GGML_CUDA=ON` build. Ops without explicit placement still get scheduled to CUDA if available. To validate “CPU semantics,” either build with `GGML_CUDA=OFF` or run with `CUDA_VISIBLE_DEVICES=` cleared. Don’t conflate `-ngl 0` with “CUDA out of the picture.”
- **Lesson 3 — magnitude correction contracts:** When the call site pre-cancels a YaRN magnitude factor (`attn_factor /= 1 + 0.1 * log(1/freq_scale)`), the kernel _must_ apply that factor internally for the cancellation to work. Document this contract in code at both ends. We added a comment block above the CUDA YaRN helpers stating this contract.

## DSv4 Q2 quantization, full GPU offload, and tooling fixes (2026-04-27)

### Things that broke until fixed
1. **`llama-quantize` failed on i32 hash-MoE table.** `blk.X.ffn_gate_tid2eid.weight` is type i32 (per-token expert lookup) and must not be quantized. Added skip in `tensor_allows_quantization` in `src/llama-quant.cpp` next to the existing `ffn_gate_inp.weight` skip.
2. **`llama-imatrix` segfaulted on ARANGE ops.** Our DSv4 graph emits `ggml_arange` (e.g. for grouped-output ids) which has no `src[0]`. The imatrix collector dereferenced `src0->name` unconditionally. Added a `if (src0 == nullptr) return false;` guard.
3. **`llama-imatrix`/`llama-perplexity` warmup hit `n_seqs == 1` assertion.** Our hash-MoE branch did not gate on `!cparams.warmup` like antirez. Added the warmup guard to the `selected_experts_in` block in `src/models/deepseek4.cpp`. Important documented contract: during warmup `t_inp_tokens` may exist but is not a real token stream and the shape relationship between `ffn_gate_inp_b` and `n_expert_used` cannot be relied on.
4. **Multi-GPU pipeline parallelism hit `GGML_SCHED_MAX_SPLIT_INPUTS=30` assertion.** Our DSv4 graph emitted ~5 mask input tensors per ratio>0 layer. With 41 such layers that is ~200 distinct graph inputs, and the upstream ggml scheduler caps split inputs at 30. The first instinct to bump the constant was wrong — that is a shared global constant not a model-specific knob. Correct fix: cache mask tensors by `(kind, ratio, n_tokens, n_topk)` within a single graph build. All layers with the same parameters now share one input tensor. Reduced graph input count to ~7 well under any plausible cap. Implemented as an `unordered_map<string, ggml_tensor*>` cache scoped to the graph constructor in `src/models/deepseek4.cpp`.

### Imatrix workflow lessons
- IQ2_XXS strictly requires imatrix (`llama_model_quantize: failed to quantize: this quantization requires an imatrix!`). Q2_K does not, but Q2_K is a *mixed* recipe that promotes `ffn_down` to Q3_K, blowing up DSv4 size to 96 GiB. Pure Q2_K is 87 GiB. To fit DSv4-Flash (284B params, 142B active) on 88 GiB total VRAM with KV/compute headroom, IQ2_XXS (70 GiB) is the right target.
- Imatrix capture under `--no-mmap` blows up RAM (model is 154 GiB, RAM is 124 GiB) — keep mmap enabled.
- Imatrix on DSv4 takes ~85 min for 32 chunks at -ngl 8 BF16. Acceptable.
- `--process-output` is required if you want to capture imatrix for `output.weight` and the output HC tensors. Without it you must override those tensors via `--tensor-type` to a non-imatrix-needing type (Q8_0, Q4_K, Q6_K).
- Our grouped-output projection reshapes `wq` from 2D to 3D for `ggml_mul_mat_id`. The imatrix collector records the shape under the reshaped tensor's name, which has a `" (reshaped)"` suffix and 8x rolled-up size mismatch with the underlying 2D weight. Two workarounds:
  - Strip the suffix in the imatrix file by rewriting via `gguf` Python (not byte-edit; offsets matter).
  - Delete `attn_output_a` entries and override the tensor via `--tensor-type-file` to Q4_K (which does not require imatrix).
- `llama-bench` uses `/` as `--tensor-split` separator while `llama-cli` and `llama-completion` use `,`. Easy to miss.
- `--ngl 99` is "CPU only" trap-worthy in a `GGML_CUDA=ON` build: even with no model layers offloaded, the scheduler will still place ops on CUDA. To validate CPU semantics, use `CUDA_VISIBLE_DEVICES=` or build with `GGML_CUDA=OFF`. (Already in lessons; restated because the user reasonably questioned whether I was customizing.)

### Performance summary (post-fixes, IQ2_XXS, fully on 3 GPUs)
- pp128: **146 t/s**
- pp512: **185 t/s**
- tg32:  **48 t/s**
- tg128: **54 t/s**
- 16K context summary task: 174 t/s prefill on 6569 input tokens, 10 t/s decode
- IQ2_XXS quality: greedy non-thinking chat is coherent (`6 apples` math correct with EOS). Thinking-mode greedy decoding can loop because 2.06 bpw is too lossy for sustained long-form reasoning.

## DSv4 unified sparse-gather attention op (2026-04-30)

### Final architecture (after a long debugging arc)
- DSv4-Flash uses MLA + window+compressed attention. The previous llama.cpp port concatenated `Kcur` (window) and the compressed cache into a single `Kall` and built a dense `[n_kv, n_tokens]` mask containing both causal -infs AND interspersed -infs (for non-top-k compressed positions). This was fed to `ggml_flash_attn_ext` via `build_attn_mha`.
- The interspersed -inf pattern is incompatible with FA's tile architecture (see preceding lesson). Pragmatic NaN guards in FA kernels did not fix the corruption — the bug is in tile-level softmax reductions over Swiss-cheese masks, not just fully-masked columns.
- **Fix:** new `GGML_OP_DSV4_SPARSE_ATTN` op that gathers KV by index (matching HF's `sparse_attn_kernel`) instead of using a dense mask. The op handles BOTH the contiguous window AND the topk-gathered compressed positions plus the per-head learned attention sink, in one pass with FlashAttention-style online softmax. No -inf mask is ever built.
- DSv4 attention now has exactly two paths:
  - sparse op when compressor + indexer + n_comp > 0
  - standard `build_attn` (window only) when not
- Performance note: the Phase-1 naive CUDA kernel uses one block per (token, head, batch) with warp-reduction scalar dot products. It is correctness-first and slower than tensor-core FA on small/dense KV. Phase 2 will move to MMA tensor cores. The bug fix justifies the perf cost; the unified architecture eliminates ~150 lines of dense-mask scaffolding.

### Lessons
- **Custom kernel + one architectural path beat heroic kernel-level workarounds.** Once the bug class is "FA can't handle this kind of mask", do not patch FA — replace the call.
- **Dead graph inputs cause `set_input` crashes.** When you bypass a code path that was building input tensors via `cache_mask` + `add_input`, those tensors get pruned by the scheduler (no buffer) but the input handler still tries to write to them (`buffer_is_host` assert). The right fix is to NOT build the input tensors when going down the bypass path — not to add `if (buffer == nullptr) return;` band-aids in every handler. Eliminate the upstream cause.
- **"Fall through" cleanly to existing standard paths.** When refactoring DSv4 to remove its custom `build_attn_mha` calls, the cleanest pattern was to leave a flag (`use_prompt_sparse_attn`) at false in the non-sparse case and let the existing standard `build_attn` block at the end of the function handle it. Don't recreate window-only attention; reuse what's already there.
- **Compressor cache writes must happen even when this step does not use sparse attention.** Future decode iterations will need newly-compressed positions in the DSv4 attn_k / index_k caches. Run cache writes unconditionally; gate only the attention computation on sparse eligibility.
- **Pri-perf trade-off can be small enough to accept.** The all-visible decode shortcut saved maybe 10% on the first ~2K tokens of context. Eliminating it (in exchange for one unified architecture and ~150 fewer lines) was worth it. Always quantify before committing to perf-optimal multi-path.
- **Window phase needs causality even though it looks dense.** First long-prompt test of the sparse op produced repetition: every prompt query was attending to all window positions including FUTURE tokens. Adding an optional `window_mask` input (causal+SWA pattern for prompt, SWA validity mask for decode) fixed it. The sparse op is "sparse on the compressed cache" — the window phase is still dense and still needs the standard causal mask the rest of llama.cpp uses.
- **Measure before MMA.** The first-cut naive sparse kernel measured at ~0.81 tok/s during BF16 8-layer-offload decode and looked like an obvious tensor-core optimization target. But IQ2_XS with full offload (`-ngl 99`) hit 40 tok/s on the same kernel — the slow BF16 number was CPU bottleneck from 32 unfilled layers, not kernel slowness. Always profile in the configuration you actually ship before committing to a kernel rewrite.
- **Profile before optimizing.** Per-op cudaStreamSynchronize-based profiling of DSv4 IQ2_XS decode at full GPU offload showed `DSV4_SPARSE_ATTN` as only 2.48% of total compute time. The actual cost centers are dense `MUL_MAT` (28%, cuBLAS, hard to improve) and MoE `MUL_MAT_ID` (12%, 256-expert top-8 dispatch). Phase 2b MMA on the sparse-attn kernel would have gained at most 2-3% TPS for ~16 hours of work — a poor trade. The Phase 2a warp-parallel optimization captured the meaningful kernel-level win without a full MMA rewrite. Profiler is committed (gated on `GGML_CUDA_PROFILE_OPS=1`) for future bottleneck discovery.

## DSv4 IQ2_XS long-prompt collapse (2026-04-30)

### Root cause: indexer + compressor weights quantized too aggressively
- **Symptom:** IQ2_XS produces correct short-prompt math ("answer":"6") at 40 tok/s decode, but on long prompts (438 tokens, both structured and simple) collapses into sentence-level repetition after a coherent prefix. BF16 with the same sparse-attn kernel produces perfect output on the same prompts.
- **Initial misattribution:** I attributed this to the previously-documented IQ2_XXS "thinking-mode collapse" quality ceiling. The user pushed back ("I don't believe this to be true") and the diagnostic plan in todo.md isolated the actual cause.
- **Diagnostic plan (executed):** Same loop reproduces with `-fa off` (rules out sparse-op bug). Same loop with temp=0.7 (rules out greedy-decode pathology). Same loop on a simple essay prompt (rules out instruction recursion). Different loop with IQ2_S (2.6bpw) but still loops (rules in quant level).
- **Root cause located via tensor-type inspection of the IQ2_XS GGUF:** The `indexer.attn_q_b`, `attn_compressor_kv`, `attn_compressor_gate`, `indexer_compressor_kv`, `indexer_compressor_gate`, and `indexer.proj` (some) tensors are critical for selecting which compressed positions to attend to. Quantizing them to IQ2_XS (2.4 bpw) introduces enough noise into the indexer scores that argsort-top-k picks wrong positions, which then feeds into the sparse attention with bad indices. The model sees nearly-correct attention output most of the time but degenerates into repetition under sustained generation.
- **Fix:** Architecture-aware override in `llama_tensor_get_type_impl()` (`src/llama-quant.cpp`). When `arch == LLM_ARCH_DEEPSEEK4` and the tensor name contains `indexer` or `compressor`, force the target to `GGML_TYPE_MXFP4` if the recipe would otherwise pick an aggressive (sub-4 bpw) type. MXFP4 is 4 bpw structured FP with E8M0 group scale — it matches the natural precision of the FP4 QAT-trained DSv4 weights, and it's higher precision than IQ2 without the size cost of Q8_0. Users can still override with `--tensor-type` if they want a different target.
- **Why this is the right place** (not a bash wrapper, not GGUF-key magic): the architecture is already loaded into `qs.model.arch` from the GGUF metadata at this point, the function already has arch-aware branches for FALCON, the existing `--tensor-type` regex mechanism is preserved as the user-facing escape hatch, and the fix is automatic for any DSv4 GGUF without external configuration.
- **Lesson 1 — don't accept "quant ceiling" without proving it.** When something looks like a known-issue family member, run the actual diagnostic before filing it under that label. The user's pushback saved a real bug from being closed as expected behavior.
- **Lesson 2 — some tensors are quant-fragile.** For DSv4 specifically, the indexer is a small but architecturally critical component. Its outputs steer attention. Even small precision degradation translates into large attention errors.
- **Lesson 3 — inspect the GGUF, not the recipe.** The recipe documentation may not match the actual file (overrides, defaults, missing rules). When a quant misbehaves, dump the per-tensor types from the GGUF and look for surprises in critical components.

## DSv4 Flash Attention sparse mask incompatibility (2026-04-30)

### Root cause: FA tile architecture cannot handle interspersed -inf masks
- **Symptom:** BF16 `-fa on` produces angle-bracket spam (`<<<<<<`) immediately after prompt eval. BF16 `-fa off` produces perfect coherent structured output on the same prompt/model/GPU config.
- **Investigation path:**
  1. Bisected to commit `b50d0af2d` (multi-seq compressed attention) as the regression boundary for short prompts with FA on.
  2. Discovered that commit `eb99b6dcf` (pre-b50) enabled FA for 512-dim heads for the first time — before that, FA silently fell back to CPU softmax path, masking the issue.
  3. Attempted fix: clamp `KQ_max_new <= -1e9f → 0.0f` after warp reductions in all FA kernel variants (CUDA MMA/WMMA/TILE, SYCL/HIP tile/vec/softmax, CUDA softmax). This prevents NaN from `expf(-inf - (-inf))` when entire columns are masked.
  4. Result: NaN prevention works for fully-masked columns, but bracket spam persists. The issue is interspersed `-inf` entries *within* FA tiles, not just fully-masked columns.
- **Root cause:** The CUDA FA kernels (`fattn-mma-f16`, `fattn-wmma-f16`, `fattn-tile`) use tiled matrix multiplication with warp-level reductions that assume a dense causal mask (lower triangle of 0/-inf). DSv4's sparse compressed attention creates "Swiss cheese" masks where arbitrary positions within a tile are -inf. The tile-based softmax reductions handle this pattern incorrectly — partial sums and max values within a tile become corrupted when -inf values are interspersed with valid attention scores in non-contiguous patterns.
- **Fix:** A dedicated sparse-gather attention kernel (`GGML_OP_DSV4_SPARSE_ATTN`) that gathers KV by index (matching HF's `sparse_attn_kernel` in `kernel.py`) instead of using a dense mask. See `tasks/dsv4_sparse_attn_kernel_plan.md`.

### Lessons
- **FA kernels are not general-purpose masked attention.** They are optimized for dense causal masks. Any model that needs arbitrary sparse masks (like DSv4's compressed attention with top-k selection) requires a dedicated kernel.
- **"Just add a guard" rarely fixes a tiled-compute incompatibility.** The NaN-prevention guard handled one edge case (fully-masked columns) but the fundamental problem was the tile architecture's inability to process interspersed -inf. Always verify the fix addresses the actual tile-level data flow, not just the most visible symptom.
- **Bisecting across FA-enable commits is critical.** The regression appeared at `b50d0af2d` but the actual enablement was `eb99b6dcf`. Without tracing the FA dispatch path, we would have blamed the multi-seq refactor instead of the FA kernel.
- **Never run `git checkout` on uncommitted WIP files.** Multiple reckless `git checkout` commands destroyed hours of uncommitted work (BF16 cache fixes, debug instrumentation, HC dtype fixes). Always use targeted `Edit` tool or `git stash` before any destructive git operation.

## DSv4 GCP IQ2_XXS corruption (2026-04-29)

### Root cause: -Wmaybe-uninitialized in our quant fallback patch
A previous commit added two graceful-fallback branches in `llama_model_quantize_impl` (missing imatrix → keep original; already-quantized + no `--allow-requantize` → keep original). The patch left a control-flow shape where `new_data` and `new_size` were only assigned conditionally inside an `if (f32_data) { ... }` block whose predicate the compiler could not statically prove non-null. GCC 13.3.0/-O3 (the GCP toolchain) emitted `-Wmaybe-uninitialized` and produced code that, on the F32 / dequant `else` paths, sometimes used the previous loop iteration's stack values. The result: 93 tensors (full layers 21/32/42 plus partials around 20/31/41) had all-zero raw bytes on disk while the log claimed normal quantization, because `fout.write` was writing from anonymous-mmap zero-filled `work` regions instead of the freshly quantized output.

### How we found it
1. Confirmed BF16 source bytes were identical local↔GCS (rules out source corruption).
2. Diffed tensor types between original (working) and GCP (broken) IQ2_XXS files — only meaningful regression was `attn_output_a` Q4_K→IQ2_XXS, which alone could not produce all-zero output.
3. Dequantized BF16 vs broken IQ2_XXS in Python via `gguf.quants.dequantize`; computed RMS error per layer. Layers 21/32/42 had 100% RMS error (i.e. literally zero output).
4. Verified raw on-disk bytes were zero, not a dequantization artifact.
5. Searched the GCP build log for warnings/errors and found exactly the `-Wmaybe-uninitialized` for `new_data` and `new_size` at the use sites.

### Lesson
- **Treat compiler warnings in modified code as production bugs.** Especially `-Wmaybe-uninitialized` in performance-critical -O3 paths. The local GCC 15 build emitted the same warning but happened to not exhibit the UB; the GCP GCC 13 build did. Stale-stack reuse is the typical manifestation.
- **For graceful fallbacks, restructure don't decorate.** When adding new "keep original" branches alongside existing "compute and replace" branches, restructure the surrounding code so each branch unconditionally assigns the variables consumed by downstream code. Don't add a post-hoc `if (sentinel)` guard that the compiler must prove always-true.
- **Validate quant artifacts numerically, not just by file size and tensor count.** A 70 GiB GGUF that loads cleanly and produces tokens can still have entire layers zeroed; the only reliable check is dequantize-and-compare (or at minimum, raw-byte zero scan) per tensor.
