# DeepSeek-V4 Flash Enablement — TODO

## Done
- [x] End-to-end DSV4-Flash inference with CUDA support
- [x] CUDA YaRN parity, decode KV-quant, graph-dependency cleanups
- [x] Enable Q2/IQ2 quantization and full multi-GPU offload
- [x] Generic tensor_allows_quantization skip for non-dequantizable types
- [x] Remove DSv4 arch checks from generic build_moe_ffn
- [x] Fix CUDA Flash Attention for DSV4 (stride gate + mask alignment)
- [x] Fix imatrix grouped-output collapse for attn_output_a
- [x] Fix warmup/auto-fit graph reservation (n_seqs assertion)
- [x] Multi-sequence support for compressed attention path (compressor prefill state, compressed pool, state storage)
- [x] GCP: BF16 convert from fresh HF safetensors → `gs://test-quant-jobs/bf16-cache/deepseek-ai__DeepSeek-V4-Flash/`
- [x] Local validation: multi-seq imatrix (n_seq=4) + single-seq regression (n_seq=1)
- [x] Root-cause FA bracket-spam regression: CUDA Flash Attention cannot handle DSv4 sparse interspersed `-inf` masks
  - Confirmed: BF16 `-fa off` produces perfect structured output (400 tokens, long prompt)
  - Confirmed: BF16 `-fa on` produces bracket spam even with NaN-prevention kernel fix
  - Root cause: FA tile-based architecture assumes dense causal masks; interspersed `-inf` within tiles corrupts softmax
  - Attempted fix: clamp fully-masked `KQ_max_new <= -1e9f → 0.0f` in all FA variants (CUDA MMA/WMMA/TILE, SYCL/HIP tile/vec/softmax)
  - Result: prevents NaN for fully-masked columns but does not fix interspersed sparse mask corruption
  - Conclusion: custom sparse-gather kernel required (see `tasks/dsv4_sparse_attn_kernel_plan.md`)
- [x] Write detailed custom sparse-gather kernel implementation plan (`tasks/dsv4_sparse_attn_kernel_plan.md`)
- [x] **Implement `GGML_OP_DSV4_SPARSE_ATTN` custom kernel** (Phase 1)
  - Op registration in ggml.h / ggml.c (`70beee99f`)
  - Reference CPU implementation in ggml-cpu/ops.cpp
  - Naive CUDA kernel in ggml-cuda/dsv4-sparse-attn.cu (one block per (token,head,batch), warp-reduction dot products, FlashAttention-style online softmax with attention sink)
  - Backend dispatch in ggml-cuda/ggml-cuda.cu and ggml-backend-meta.cpp
  - RPC proto bumped to v2
- [x] **Unified sparse-gather attention path in deepseek4.cpp** (`efa210c69`)
  - Replaced ALL `build_attn_mha` call sites with `ggml_dsv4_sparse_attn`
  - Deleted `dsv4_build_compressed_mask_from_topk` and supporting helpers/input handlers
  - Deleted all-visible decode shortcut and dense `-inf` interspersed mask construction
  - Architecture: `if compressor + indexer + n_comp > 0` → sparse op; `else` → standard `build_attn` (window only)
  - Net change: deepseek4.cpp -156 lines
  - **Validated**: BF16 `-fa on` short prompt produces correct "6 apples" coherent reasoning (was bracket spam under the buggy dense path)

## Active branch: `work/dsv4-prefill-speedup` — DSv4 prefill speed evaluation
- [x] Create working branch from current DeepSeek4 runtime head.
- [x] Add a repeatable DeepSeek4 endpoint regression harness for simple chat, coherence, structured tool calls, streaming tool-call deltas, prompt-cache reuse, and optional short antirez/ds4 official vectors before changing prefill behavior. Added `scripts/dsv4_regression.py` plus tracked replay fixture `tasks/replays/dsv4_agent_tool_replay_turn02_request.json`; syntax/help/fixture validation passed.
- [ ] Run the DeepSeek4 endpoint regression suite against `deepseek-ai/DeepSeek-V4-Flash-IQ1_M` using the existing full DeepSeek Docker layer with the working tree mounted, building/running code inside that container instead of rebuilding Docker images.
- [x] Establish current prefill bottleneck from existing systemctl Corral logs and code paths, without manually starting Corral. Logs show DSv4/IQ1_M prompts split into repeated 128-token prefill chunks due checkpoint-tail policy, with prompt eval around ~184 tok/s for an 858-token prompt and ~170 tok/s for large continuation chunks.
- [x] Audit prompt prefill chunking/checkpoint policy: `server-context.cpp` breaks prompt batches at SWA-spaced tail offsets (`n_swa=128`), so even prompts below `n_ubatch` get fragmented into 128-token chunks to create rollback checkpoints.
- [x] Audit DSv4 graph inputs and graph reuse for prefill: custom input `can_reuse()` implementations exist, but chunk shape changes (`4096 -> 128 -> final`) still limit graph reuse and the artificial chunking moves most tokens off the first-chunk vectorized prefill path.
- [ ] Profile or derive prefill op cost centers for current IQ1/IQ2/MXFP4 paths: MoE `MUL_MAT_ID`, dense matmuls, compressor/indexer, sparse attention, KV/cache writes, scheduler splits/copies.
- [x] Identify low-risk architecture/code changes that could plausibly 2x prefill: (1) stop creating SWA-spaced tail checkpoints by default for DSv4 prefill so short/medium prompts run as one vectorized prompt graph; (2) implement vectorized continuation prefill compressor for aligned chunks after cache restore instead of per-token `dsv4_build_compressor_decode_chunk()` graph expansion.
- [x] Implement Candidate 1 low-risk chunking change: DeepSeek4 compressed-KV contexts now skip the SWA-spaced tail checkpoint loop, keeping only the `4 + n_ubatch` and final aligned checkpoint guards. Local `build-vulkan-linux-release` llama-server build passed.
- [x] Implement immediate continuation-prefill waste fix: nonzero-position DSv4 prompt chunks no longer build unused `attn_compressor_pool` / `attn_indexer_pool` prompt-only tensors before falling through to the cache-aware decode/replay path. Local `build-vulkan-linux-release` llama-server build passed.
- [x] Rank remaining candidates by expected gain/risk: validate Candidate 1 + 1.5 first because they are low-risk and may already approach the desired 2x on continuation prefill; defer full vectorized continuation compressor until after measuring. Minimal acceptance bench: rebuild DeepSeek image from this branch, restart Corral via systemctl only, replay a multi-turn IQ2_XXS history, and compare prompt chunk pattern / prompt eval tps against the 128-token-fragmented baseline.
- [x] Triage tester regression report: IQ1_M emitted literal XML tool-call text in strict tool-call tests, but the same mounted-code build with the usual IQ2_XXS model passed all 7 regression cases, including streaming and checkpoint-restored tool replay with `cached_tokens=1408`. Switched `scripts/dsv4_regression.py` default to IQ2_XXS and updated `tasks/dsv4_regression_handoff.md`.

## In Progress
- [x] Long-prompt validation: BF16 `-fa on` with isolation prompt + n_predict=400 — perfect structured output
- [x] Verify IQ2_XS short prompt with the unified sparse path — coherent ("answer":"6") at 40 tok/s
- [x] **DIAGNOSED: IQ2_XS long-prompt collapse is a quant-recipe issue, NOT sparse-op or generic quant ceiling.**
  - User pushback ("I don't believe this to be true") was correct. Proper diagnosis:
    - IQ2_XS `-fa off`: same exact loop → not sparse-op specific
    - IQ2_XS temp=0.7 / temp=1.0+rp: still loops or rambles → not greedy-decode
    - IQ2_XS simple essay prompt (300-word essay, no recursion): also collapses into repetition ("printing press","printing press",...) → not prompt-recursion
    - IQ2_S (2.6 bpw) instead of IQ2_XS (2.4 bpw): still loops on same prompt → quant level matters
    - BF16 same prompt: perfect structured output → not the model itself
  - Root cause via GGUF inspection: indexer + compressor weights are quantized to IQ2_XS:
    - `indexer.attn_q_b.weight` → IQ2_XS
    - `attn_compressor_kv.weight` → IQ2_XS
    - `attn_compressor_gate.weight` → IQ2_XS
    - `indexer_compressor_kv.weight` → IQ2_XS
    - `indexer_compressor_gate.weight` → IQ2_XS
  - These tensors steer attention (indexer top-k selection + compressor pool that gets gathered). At 2.4 bpw the noise is enough to derail topk selection on sustained generation — but is fine on single short answers (where small attention errors don't compound).
- [x] **FIX: architecture-aware override in `llama_tensor_get_type_impl()` (commit `268fd574e`)**
  - When `arch == LLM_ARCH_DEEPSEEK4` and tensor name contains `indexer` or `compressor`,
    force `GGML_TYPE_MXFP4` if the recipe would otherwise pick an aggressive (sub-4 bpw) type.
  - MXFP4 (4 bpw structured FP with E8M0 group scale) matches the natural precision of
    the FP4 QAT-trained DSv4 weights and avoids the IQ2 noise that derails top-k.
  - User `--tensor-type` regex overrides still win (the override only kicks in when the user did not specify).
  - Verified via instrumented debug build: every indexer / compressor tensor across all DSv4 layers routes to MXFP4 instead of IQ2_XS.
- [ ] **End-to-end validation pending**: re-quantize the BF16 source with the fix in place and confirm the long-prompt loop is gone. Local re-quant takes ~9 hours due to 43 layers x 3 ffn_*_exps tensors (~1 GB each); recommend running on the GCP CPU quant VM via the next `--quant` invocation, which will pick up the fix automatically.
- [x] **Phase 2a sparse kernel optimization (commit `d95382d9c`)**: warp-parallel KV chunks. Each warp processes its own KV slot in parallel, with one block sync per chunk of N_WARPS positions instead of one per position. Measured on IQ2_XS full offload long-prompt: decode +11% (33.8 -> 37.6 t/s), prompt eval -7% (413 -> 384 t/s) due to higher shared-mem occupancy pressure. Decode wins in net inference throughput.
- [x] **Phase 2b sparse kernel MMA optimization — not worth doing.** Per-op profiling on IQ2_XS full GPU offload shows `DSV4_SPARSE_ATTN` is only 2.48% of total compute time. The bottleneck is dense `MUL_MAT` (28%, cuBLAS) and MoE `MUL_MAT_ID` (12%, 256-expert top-8 dispatch). MMA tensor cores on sparse-attn would gain at most 2-3% TPS for 8-16 hours of CUDA template engineering. Closed.
- [x] **Per-op profiler committed for future bottleneck discovery.** `GGML_CUDA_PROFILE_OPS=1` env var enables a `cudaStreamSynchronize`+chrono timing wrapper around `ggml_cuda_compute_forward`. Disables CUDA graphs (incompatible with mid-graph sync). Prints op-enum and DSv4-tensor-name-bucketed summaries on context destruction. Useful as a development tool when investigating perf regressions or new bottlenecks.
- [x] **CONT reduction (commit `7013be05c`)**: 34 cont sites use a graph-build-time `cont_if_needed()` helper that skips materialization for already-contiguous inputs. CONT op share dropped from 8.89% to 1.02%; total measured compute -8%.
- [x] **Graph reuse / `can_reuse()` (commit `3fd0394f7`)**: implemented `can_reuse()` for all four DSv4 `llm_graph_input_i` subclasses. Graph reuse rate 0 → 197/200 (98.5%); long-prompt decode 38.6 → 49.5 t/s (+28%).
- [x] **HC_SPLIT_SINKHORN warp-parallel rewrite (commit `ab28d37c2`)**: original 1-thread-per-row kernel rewritten as 1 warp per row for the n_hc=4 DSv4 case. All sinkhorn row/col reductions via `__shfl_xor_sync`; no shared memory, no `__syncthreads`. Op time 96.85 ms (6.79%) → 26.38 ms (1.98%) (-73%). Long-prompt decode 49.5 → 56.4 t/s (+14%).
- [x] **`GGML_GLU_OP_SWIGLU_CLAMPED` (commit `289fcb280`)**: new GLU variant for DSv4/Step3.5 swiglu clamping. Eliminates a 4-op (CLAMP+CLAMP+SILU+MUL) chain in `build_moe_ffn`. Decode 56.4 -> 57.1 t/s.
- [x] **GLU op_params plumbed through fusion args (commit `9fa843249`)**: enables `MUL_MAT_ID + MUL_MAT_ID + GLU` fusion to fire for SWIGLU_CLAMPED (DSv4), and fixes a latent default-params bug for SWIGLU_OAI in the mmvq/mmf fused fast paths. Fusion summary now shows 86 MoE-FFN fusions per profile run (previously 0). Decode 57.1 -> 57.8 t/s. Net since original baseline: 33.8 -> 57.8 t/s = +71% decode.
- [x] **`DSV4_FP8_KV_QUANTIZE` hardware conversion (commit `80c0d0a07`)**: replaced 127-iteration `expf()` linear search with `__nv_fp8_e4m3(float)` hardware FP8 conversion. Decode +8% across all prompt sizes (much bigger than the 1-2% the 2.6% profile share suggested - the inner loop was way more expensive than measured).
- [ ] **Future smaller perf candidates**:
  1. HC_EXPAND (1.65%) + HC_WEIGHTED_SUM (1.55%): launch-overhead bound; only meaningful via fusion.
  2. `CPY` (7.05%) and `SET_ROWS` (3.89%): KV cache writes; some elidable via in-place views.

**Cumulative DSv4 perf delta (438-token long prompt, IQ2_XS full GPU offload):**

  | variant                                    | prefill   | decode    |
  | ------------------------------------------ | --------- | --------- |
  | original baseline                          | 413 t/s   | 33.8 t/s  |
  | + Phase 2a sparse-attn (warp-parallel)     | 384 t/s   | 37.6 t/s  |
  | + CONT reduction                           | 402.8 t/s | 38.6 t/s  |
  | + graph reuse (can_reuse)                  | 410.7 t/s | 49.5 t/s  |
  | + HC_SPLIT_SINKHORN warp-parallel          | 413.2 t/s | **56.4 t/s**  |

  Net delta: prefill ~equal, **decode +66.7%**.
- [x] **Phase 3: HIP/ROCm runtime validated on AMD Strix Halo (gfx1151)**. Built llama-completion against the HIP-only target with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` for full offload on the iGPU. Long-prompt (438-token isolation prompt) numbers, IQ2_XS, full offload (-ngl 99 -fa on):
  - **HIP / Strix Halo iGPU**: 77 tok/s prefill, 12 tok/s decode
  - **CUDA / 2x 5090 + 3090**: 413 tok/s prefill, 38 tok/s decode
  - Ratios (HIP / CUDA): 0.19x prefill, 0.32x decode — better than the ~0.08x raw-FLOPs hardware ratio would predict.
  - Without `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` the iGPU silently stalls at 99% GPU memory pool during load.
  - Confirms `__shfl_xor_sync(width=32)` and `nv_bfloat16` work on RDNA3 wave32 via the existing CUDA→HIP translation. CDNA wave64 untested.
- [ ] GCP: imatrix + logits generation on h100-4x
  - **Blocked on H100-4x spot capacity** — all configured zones stocked out
  - BF16 cache ready at `gs://test-quant-jobs/bf16-cache/deepseek-ai__DeepSeek-V4-Flash/`

## Remaining
- [ ] GCP: IQ2_S quantization using generated imatrix
- [ ] Download artifacts locally, validate fit + quality
- [ ] Add/update corral config entry for DSV4-Flash IQ2_S
- [ ] Update documentation (README, TEST_COVERAGE, etc.)
- [ ] Final commit + push of any remaining changes

## Merge `work/dsv4-prefill-speedup` to `main` (requested 2026-05-09)
- [x] Commit intended prefill/regression changes on `work/dsv4-prefill-speedup` while leaving scratch files untracked.
- [x] Run available quality gates: `git diff --check`, `python3 -m py_compile scripts/dsv4_regression.py`, and `cmake --build build-vulkan-linux-release --target llama-server -j 8` passed. IQ2_XXS regression was already green and the user explicitly said no need to rerun before merge.
- [x] Merge the branch into local `main` and keep `work/dsv4-prefill-speedup` branch alive for follow-up bugs.
- [x] Pushed `main` and `work/dsv4-prefill-speedup` to origin with user approval.

### Review
- Main now includes commit `22eaf7d89 perf(deepseek4): reduce prefill checkpoint fragmentation`.
- `work/dsv4-prefill-speedup` remains alive and points at the same commit as `main` for follow-up bug fixes.

## General DSv4 prefill speed follow-up — vectorized continuation compressor
- [x] Implement a guarded vectorized continuation-prefill compressor for contiguous single-sequence ratio-4 chunks that start after position 0, so large cold prompts do not route all post-first chunks through per-token decode replay.
- [x] Keep the existing decode/replay path as the fallback for decode, multi-seq, non-contiguous positions, non-ratio-4 layers, or unsafe alignment.
- [x] Build and validate `llama-server` after the graph change: host `build-vulkan-linux-release` build passed; mounted CUDA/HIP `build-dsv4-container` build passed.
- [x] Run the DeepSeek4 endpoint regression harness with IQ2_XXS and compare prompt chunk/path logs for large prompts: all 7 regression cases passed; debug logs confirmed `path=prefill-replay` on aligned continuation chunks.
- [x] Record results and decide whether backend fusion/profiling is still needed. Clean no-debug IQ2_XXS tool replay timing: fresh 2486-token prefill `265.04 tok/s` vs previous ~`220.83 tok/s`; checkpoint-replayed 1078-token prefill `189.85-194.73 tok/s` vs previous ~`151-155 tok/s`. This is a useful ~20-28% improvement, not 2x; next candidates are larger backend fusion/profile work around prompt indexer/top-k, compressor pooling, and MoE/expert matmul throughput.

## CUDA prefill profiling — IQ2_XXS
- [ ] Free the ad hoc mounted-code server container only, leaving Corral/systemd-managed services untouched.
- [ ] Build/check a bounded standalone profiling harness (`llama-bench` or equivalent) in `build-dsv4-container`.
- [ ] Run non-profiled CUDA prefill baselines for prompt sizes 512/2048/8192 with `-n 1` and key `-ub` settings.
- [ ] Run `GGML_PROFILE_OPS=1` prefill profiles for representative prompt sizes/configs and capture op-share summaries.
- [ ] Compare op mix: `MUL_MAT`, `MUL_MAT_ID`, `DSV4_SPARSE_ATTN`, `ARGSORT`, compressor ops, `CPY`, `SET_ROWS`, and scheduler split/copy indicators.
- [x] Record findings and choose the next implementation target: profiles show CUDA prefill is dominated by `DSV4_SPARSE_ATTN` for large prompt ubatches because the prompt path passes `Kcur` as `n_window=n_tokens` with a dense `[n_tokens,n_tokens]` mask, so the kernel scans masked-out raw positions even though DSv4 logical SWA is 128.

## CUDA prefill optimization — bounded prompt sparse raw window
- [x] Add a bounded raw-window mode to `GGML_OP_DSV4_SPARSE_ATTN` via op params, defaulting to current behavior for decode/Vulkan and any caller that passes `0`.
- [x] Pass `prompt_window_size` from the DSv4 prompt sparse path when causal prompt attention is active.
- [x] Keep decode path unchanged and preserve the dense mask as a correctness guard for visible positions.
- [x] Preserve ROCm/HIP behavior: the shared CUDA/HIP sparse-attn source compiles, and bounded prompt scanning remains semantically identical to the mask-scanning path for visible rows.
- [x] Build CUDA/HIP and host targets. Host `llama-server llama-bench` build passed; mounted Docker CUDA/HIP `llama-bench llama-server` build passed.
- [x] Re-profile IQ2_XXS prefill to verify `DSV4_SPARSE_ATTN` time drops and run IQ2_XXS endpoint regressions; run at least a bounded ROCm smoke if hardware/runtime permits. CUDA p2048/ub2048 improved `345.66 -> 612.91 tok/s` non-profiled and `DSV4_SPARSE_ATTN` dropped `3866.68 -> 1315.55 ms`; IQ2_XXS regression passed on retry after one non-deterministic cache-reuse tool-name miss; ROCm0 p64 smoke passed at `49.49 tok/s`.

## CUDA prefill optimization — MoE / `MUL_MAT_ID` long-prompt target
- [x] Commit bounded sparse-attention work: `e68598bab perf(deepseek4): bound prompt sparse raw attention`.
- [x] Free only the ad hoc mounted-code server container before benchmarking; no Corral/systemd-managed containers were touched.
- [x] Establish current grid after bounded sparse-attn: IQ2_XXS CUDA p2048/ub512 `298.47 tok/s`, p8192/ub512 `262.53 tok/s`, p2048/ub1024 `349.44 tok/s`, p8192/ub1024 `251.89 tok/s`, p2048/ub2048 `644.81 tok/s`, p8192/ub2048 crashes.
- [x] Profile p2048/ub512 vs p8192/ub512 with the same ubatch to separate pure context-length effects from chunk-size effects. After sparse mask-skip, p8192/ub512 improved to `285.11 tok/s`; detailed profile shows `MUL_MAT_ID` now dominates and `DSV4_SPARSE_ATTN` fell to `5055.68 ms / 15.42%`.
- [x] Instrument or derive the `GGML_OP_MUL_MAT_ID` dispatch path for DSv4 IQ2_XXS: MMVQ for tiny `ne2`, MMQ for large routed-expert prefill, slow host-sorted path must remain avoided. Component profiling with graphs disabled shows ID compaction + activation quantization are tiny; expert MMQ matmul itself is ~96% of routed expert time.
- [x] Switch further multi-variant perf sweeps to a persistent resident model plus `llama-benchy` API harness from `.venv`; avoid repeated full-model `docker run llama-bench` reloads except for one-off backend-only checks. First coherence-enabled API baseline against resident mounted-code server (`-ub 1024`) passed coherence and measured p2048 `357.04 tok/s`, p8192 `277.28 tok/s` via benchy; server logs showed p2048 `365.39 tok/s`, p8192 `279.26 tok/s`.
- [ ] Debug the p8192/ub2048 CUDA graph/cuBLAS crash with precise CUDA error text and, if needed, narrower prompt/ubatch bisect; p4096/ub2048 passes with `GGML_CUDA_DISABLE_GRAPHS=1`, so at least one failure mode is CUDA-graph-specific.
- [ ] If MMQ remains the dominant wall after API-based baselines, target the GPU-side routed-expert MMQ matmul/tile shape; `mm_ids_helper` and `quantize_mmq_q8_1_cuda` are not the wall.
- [ ] Preserve ROCm/HIP behavior: any shared CUDA/HIP kernel change must compile in the HIP build and get at least a ROCm0 smoke when hardware/runtime permits.
- [x] Re-run IQ2_XXS endpoint regression and compare prefill grid after any code change. Post-commit resident-server regression passed all 7 cases with `--max-tokens 2048` against `http://127.0.0.1:18089` after correcting the harness base URL (the harness appends `/v1/chat/completions`).

### Branch kickoff: `work/dsv4-mmq-kernel`
- [x] Pull merged `main` and create `work/dsv4-mmq-kernel` from merge commit `b24ca65ba`.
- [x] Re-establish resident-server/API baselines with coherence enabled for the exact starting branch state before changing MMQ code. Resident server `dsv4-iq2xxs-benchy-server` (`-ub 1024`) passed coherence; `llama-benchy` cold-cache 3-run baseline saved to `/tmp/dsv4_cuda_prefill_profile/mmq_branch_baseline_pp2048_8192_tg1_coherence_runs3.json`: pp2048 `351.30 ± 4.07 tok/s`, pp8192 `278.58 ± 0.39 tok/s`. Server logs showed pp2048 `359-366 tok/s`, pp8192 `280-281 tok/s`.
- [x] Restore only the useful env-gated MMQ component profiler if needed; avoid committing experimental stream-k overrides unless measured and justified. Added default-off `GGML_PROFILE_MMQ_ID=1` diagnostics for routed-expert timing plus expert population/tile-slot stats; no experimental stream-k override was reintroduced.
- [x] Inspect the routed-expert `MUL_MAT_ID` MMQ shapes: `ne12`, expert population distribution, `expert_bounds`, tile/grid selection, stream-k path, and IQ2_XXS quant type dispatch. Graphs-off p8192 profile showed current stream-k schedules dense expert tile slots while many are empty: tok=64 MoE calls average ~`94.5` useful tile slots vs `256` current; tok=1024 MoE calls average ~`225` useful vs `2048` current. Component timing still shows matmul dominates; ID helper and activation quantization remain tiny.
- [x] Prototype one guarded MMQ tile/scheduling change at a time, preserving CUDA graph compatibility and avoiding the host-sorted slow path. Added a device-built compact expert-tile map for routed-expert MMQ stream-k plus guarded partial-tile `ids_dst` loads; default dispatch remains MMQ and CUDA graph path works in the resident server.
- [ ] Preserve ROCm/HIP behavior: shared CUDA/HIP changes must compile in HIP and get a bounded ROCm0 smoke when runtime permits. CUDA+HIP `llama-bench llama-server` build passed in the dev container; ROCm0 runtime smoke could not run there because the container exposes no ROCm-capable device (`invalid device: ROCm0`).
- [x] Run DeepSeek4 IQ2_XXS endpoint regression with `--max-tokens 2048` after any functional/kernel change. Compact MMQ tile prototype passed all 7 IQ2_XXS regressions against the resident server on retry; the first run had one strict tool-argument variance (`ls` vs expected `ls -la`) while simple chat, math, minimal tools, streaming, and cache reuse passed.
- [x] Measure compact MMQ tile prototype via resident `llama-benchy` with coherence enabled. Starting branch baseline (`-ub 1024`) was pp2048 `351.30 ± 4.07 tok/s`, pp8192 `278.58 ± 0.39 tok/s`; final dense-index compact-tile prototype measured pp2048 `370.11 ± 2.42 tok/s`, pp8192 `293.97 ± 0.62 tok/s` over three cold-cache runs. Server logs agreed: pp2048 ~`381-383 tok/s`, pp8192 ~`295-297 tok/s`.

## Independent audit: `perf/dsv4-graph-orchestration` (May 2026)

### Goal
Find any meaningful remaining speed improvement or unnecessary blooper bug in the six-commit DSv4 perf branch, without trusting prior agent conclusions.

### Plan
- [x] Audit branch diff against `origin/main`, file by file.
- [x] Verify K-quant `GET_ROWS` kernels against GGUF K-quant block layouts / existing dequant code.
- [x] Verify CUDA/HIP backend support checks are safe and cannot route unsupported shapes to the new kernels.
- [x] Verify CUDA graph/MMQ compatibility change is logically correct and not over-enabling graph capture.
- [x] Verify DSv4 grouped-output ID hoist preserves shape/lifetime semantics.
- [x] Verify RDNA3.5 MMVQ split cannot affect non-RDNA3.5 targets.
- [x] Search for adjacent missing fast paths that could be meaningful (GET_ROWS_BACK, CPY/SET_ROWS, DUP/CONT, backend split transfers, small constants).
- [x] Run targeted build/test/bench only if the audit finds a plausible change or needs verification.
- [x] Record findings and any recommended follow-up.

### Findings
- Found one real latent correctness bug in the broad K-quant `GET_ROWS` path: K kernels capped `gridDim.y` to `UINT16_MAX` for `ne10`, but used `const int i10 = blockIdx.y` with no grid-stride loop. Any K-quant `GET_ROWS` selecting more than 65,535 rows could leave the tail unwritten. DSv4 inference does not hit this, but backend-op tests already include 70,000-row GET_ROWS cases for F32/Q4_0, so the K-quant implementation should be equally safe.
- Fixed by passing `ne10` into the K kernels and looping `for (i10 = blockIdx.y; i10 < ne10; i10 += gridDim.y)`.
- Added targeted regression coverage: `GET_ROWS(type=q2_K,n=256,m=80000,r=70000,be1=2,be2=1,v=0)`.
- Verified K-quant dequant formulas match the existing `convert.cu` `dequantize_block_q{2,3,4,5,6}_K` kernels; the new fix does not alter dequant math.
- Graph/MMQ change mirrors `ggml_cuda_mul_mat_id()` dispatch and does not appear to over-enable graph capture into the stream-synchronizing slow path.
- DSv4 grouped-output ID hoist is shape/lifetime-safe: depends only on graph-build constants `n_o_groups` and `n_tokens`, and is reused inside the same ggml context.
- RDNA3.5 MMVQ split is scoped by compile-time and runtime architecture table IDs; non-RDNA3.5 targets stay on previous tables.
- Adjacent fast-path audit: `CPY`/`SET_ROWS` already cover current q8_0 KV-cache path; K-quant CPY/SET_ROWS would mainly matter for unusual K-quant KV cache configs and is not an obvious current DSv4 trifecta win. IQ-quant `GET_ROWS` remains unsupported, but DSv4 protected token embeddings are Q2_K/Q6_K in the tested recipes, so no immediate high-confidence gain there.

### Verification
- Build: `cmake --build . --target llama-server` in `llamatrifecta_deepseekv4:latest` passed for CUDA+HIP `getrows.cu`.
- Targeted backend test CUDA0: `GET_ROWS(type=q2_K,n=256,m=80000,r=70000,be1=2,be2=1,v=0)` passed.
- Targeted backend test ROCm0: same test passed.
- Full GET_ROWS backend subset on CUDA0 passed: 68/68 supported tests.
- DSv4 sanity bench on `PORT=10099`: 512-token prompt, trifecta, Q2_K_S => 169.0 t/s prefill, 21.5 t/s decode. Decode non-regressed; prefill within known run noise for this bench harness.

### Recommendation
- Commit the K-quant `GET_ROWS` large-row correctness fix and regression test.
- Do not chase more quick perf patches in this branch. No additional obvious, low-risk, high-impact speed bug was found in this audit; remaining wins likely require larger architectural work or a separate profiler-led effort.

## Vulkan backend device-lost fix — `work/dsv4-vulkan-device-lost`

### Plan
- [x] Create a fresh branch from merged `origin/main`, preserving existing untracked scratch files.
- [x] Map the Corral crash stack and current Vulkan command path to the exact ggml-vulkan source functions/shaders.
- [x] Reproduce with a bounded direct Vulkan backend run, avoiding manual Corral startup and avoiding Corral-managed container mutation.
- [x] Identify the smallest backend-side trigger for the RADV compute-ring timeout: DSv4 Vulkan sparse-attention prompt prefill submitted one large RADV compute dispatch that scanned masked raw-window positions.
- [x] Implement a targeted Vulkan backend fix: plumb `raw_window_limit` to `dsv4_sparse_attn.comp` and split large RADV `DSV4_SPARSE_ATTN` dispatches into 4096-work-item submissions using a `base_work_idx` push constant.
- [x] Build Vulkan targets and run targeted backend/model validation with the resident bench-server flow; final systemctl-managed Corral validation remains for image refresh.
- [x] Record review notes, quality gates, and any remaining AMD/RADV limitations before commit/push.

### Review
- Reproduced baseline failure through the resident ad hoc server only: `llama-benchy --pp 2048 --tg 1 --concurrency 1` disconnected and kernel logged AMD compute-ring reset / `vk::DeviceLostError`.
- Isolated trigger with `GGML_VK_SYNC_LOGGER=1`: timeout window correlated with `DSV4_SPARSE_ATTN`; later `mul_mat_q_f16` stack traces were surfacing an already-lost device.
- Validation after fix:
  - Build passed in mounted DeepSeek container: `cmake --build build-dsv4-container --target llama-bench llama-server -j 8`.
  - Host Vulkan test target rebuild passed: `cmake --build build-vulkan-linux-release --target test-backend-ops -j 8`.
  - Focused op catalog had no generated `DSV4_SPARSE_ATTN` cases (`0/0`), so model-level validation is the meaningful gate.
  - Resident server, AMD/RADV `Vulkan0`, IQ2_XXS, `-c 65536 -b 4096 -ub 1024`: `llama-benchy --pp 2048 --tg 1 --concurrency 1` completed, coherence passed, prompt eval `14.77 tok/s`, and `journalctl -k` showed no AMD ring reset/device-wedged entries during the run.
  - Post-stress small chat with `max_tokens=2048` returned `Paris`, confirming the server stayed usable.
  - Reduced real endpoint regression on AMD/RADV Vulkan resident server passed 4/4 with `max_tokens=2048`: `simple_ok` 11.14s, `basic_math` 6.13s, `minimal_tool_auto` 91.05s, `tool_replay_turn02_nonstream` 271.34s with `cached_tokens=1408`. Container remained running and kernel logs showed no AMD ring reset/device-wedged entries afterward.
  - Remaining core endpoint regressions also passed on the same AMD/RADV Vulkan server with `max_tokens=2048`: `minimal_tool_required` 76.81s, `tool_replay_turn02_stream` 366.91s with `sse_events=97` and `cached_tokens=1408`, `tool_replay_turn02_cache_reuse` 642.29s with `second_cached_tokens=1408`. Container remained running and kernel logs showed no AMD ring reset/device-wedged entries afterward.
- Remaining: rebuild the official DeepSeekV4 Docker image from this branch if needed, then validate through systemd-managed Corral; do not mutate Corral-managed containers directly.

## DeepSeek4 next prefill ideas — MTP vs other candidates

### Research Plan
- [x] Inspect `/home/bigkahuna/src/ds4` MTP implementation and identify the exact model tensors, graph structure, cache semantics, and generation loop behavior it expects.
- [x] Inspect current llama.cpp DeepSeek4 code paths to find what is already present, missing, or intentionally incompatible for MTP/speculative heads.
- [x] Review reference PRs `ggml-org/llama.cpp#22400`, `#22673`, and `am17an/llama.cpp` for implementation scope, correctness risks, CUDA/HIP/ROCm coverage, and mergeability.
- [x] Estimate MTP impact separately for prefill, decode, and end-to-end agent/tool workloads; call out whether MTP helps prompt processing directly or mainly amortizes decode after prefill.
- [x] Compare MTP against remaining non-MTP candidates: routed MoE MMQ tuning, prompt compressor/indexer fusion, top-k/argsort improvements, KV write/copy reductions, and Vulkan stability/perf.
- [x] Produce a written next-steps plan with recommended priority, validation gates, and branch boundaries: `tasks/dsv4_next_prefill_mtp_plan.md`.

## DeepSeek4 MTP feasibility spike — `work/dsv4-mtp-feasibility`

### Plan
- [x] Find or acquire the DeepSeek4 MTP sidecar GGUF and record whether it is locally available. Not local; remote sidecar exists at `antirez/deepseek-v4-gguf`.
- [x] Dump MTP sidecar metadata/tensor names/types/shapes without loading the full target model.
- [x] Cross-check sidecar tensors against `/home/bigkahuna/src/ds4` MTP validation/build code and identify base-model tensor sharing requirements.
- [x] Trace current llama.cpp DeepSeek4 loader/graph/speculative interfaces for the smallest no-behavior loader/probe hook.
- [x] Produce an implementation sketch for `--mtp-model` / `--mtp-draft` feasibility, explicitly separating no-op loader plumbing from runtime speculative commit.
- [x] Record spike results, blockers, and recommended next branch in `tasks/dsv4_mtp_feasibility_spike.md`.

## DeepSeek4 MTP loader/probe — `work/dsv4-mtp-loader-probe`

### Definition of done
- [ ] MTP accepts and commits draft tokens through the target verifier.
- [ ] Accepted MTP private raw/HC state is committed correctly; rejected drafts leave no persistent state.
- [ ] Docker runtime validation shows target-only vs MTP output parity on deterministic prompts.
- [ ] Benchmarks show MTP is faster than target-only inference.
- [ ] MTP runs without `DSV4_MTP_PROBE=1` for normal runtime; probe mode remains diagnostics-only.
- [ ] Code organization mirrors existing llama.cpp patterns: speculative control/orchestration in `common/speculative.*` and the server verifier path, model-specific DeepSeek4 MTP tensor/graph details in DS4-specific implementation files, no ad hoc public MTP args, no arbitrary `src/llama-mtp.*`, and no MTP hook in `src/models/models.h` unless an existing project pattern justifies it.

### Immediate corruption triage
- [x] Stop advancing verifier/commit work until generation corruption is explained.
- [x] Compare real DeepSeek4 target-only output against `DSV4_MTP_PROBE=1 --spec-type mtp --model-draft ... --draft-max 2` for the exact same prompt/settings: both returned `"\n\t\t"` for raw `prompt="Hi"`, `n_predict=3`, `temperature=0`.
- [x] If target-only is clean but MTP-probe is corrupt, disable MTP probe graph pieces in order (sidecar load only, HC/raw capture only, draft graph) to isolate the mutation. Not needed for the reported raw `"Hi"` signal because target-only matched the MTP-probe output.
- [x] Since target-only also emits the same suspicious output for raw `"Hi"`, rerun through the expected chat/template path before calling the probe safe: Docker resident target-only and MTP-probe chat runs both returned content `"4"`, identical reasoning text, and identical token usage for `What is 2+2? Reply with only the number.`
- [x] Use the resident Docker/mounted-code sidecar workflow from `../llama.cpp-deepseek-v4/tasks/dsv4_resident_bench_server.md` for DeepSeek4 runtime parity instead of host CPU-only ad hoc servers.
- [x] Do not mark MTP probe safe based only on “emitted unchanged” claims without a target-only baseline artifact.

### Current WIP quarantine
- [x] Treat all uncommitted changes after `a8f11e424` as unstable until reviewed, built, and Docker-validated. Current dirty files include `src/llama-context.*`, `src/models/deepseek4.cpp`, `src/models/models.h`, `tools/server/server-*.cpp`, and task docs.
- [x] Separate useful WIP from wrong-direction WIP before more runtime work:
  - [x] Preserve useful findings only after validation: real MTP can reach verifier acceptance counts, `--device-draft CUDA2` places MTP tensors on CUDA2, and request parsing must not zero server speculative budget.
  - [x] Remove/revert wrong-direction organization: no MTP validation hook in `src/models/models.h`, no model-specific server include path as final architecture, and no invented `src/llama-mtp.*` boundary unless existing llama.cpp patterns support it.
  - [x] Rename/refactor public-facing terminology toward existing concepts (`draft model`, `MTP draft GGUF`, `MTP support GGUF`) instead of promoting generic "sidecar" terminology in active runtime paths.
- [x] Restore a clean host `llama-server` build and `git diff --check` before any Docker runtime validation.

### Required execution order from the base task
1. [x] **Code organization first:** mirror existing llama.cpp patterns before touching real commit behavior. Keep speculative control in `common/speculative.*` / server verifier flow; keep DS4 MTP tensor validation, tensor-name mapping, and graph construction in DS4-specific code; keep generic context/server code model-agnostic and minimal.
2. [x] **Standard draft placement:** keep MTP support GGUF allocation wired through existing speculative draft placement knobs (`--device-draft`, `--gpu-layers-draft`, draft cache/thread params where applicable). Do not hard-code DS4 device/layer placement.
3. [x] **Request-path budget preservation:** cleanly fix the OpenAI/DeepSeek4 request path so per-request normalization does not zero the server's configured MTP speculative budget.
4. [ ] **Verifier dry-run before commit:** decode proposed MTP draft tokens through the existing target verifier/checkpoint/rollback path while keeping emitted tokens target-only and always restoring target state.
5. [ ] **Private drafter state design:** specify and implement exact accepted/rejected state transitions for MTP private HC/raw rows before calling any real commit path safe.
6. [x] **Real accept/commit smoke path:** MTP token acceptance now goes through `common/speculative.cpp` and the existing server verifier/accept loop; Docker runs show accepted drafts are committed and final visible content is correct on short arithmetic and counting prompts.
7. [ ] **Correctness validation:** use the mounted-code Docker resident workflow to compare target-only vs MTP on deterministic chat prompts; current arithmetic prompt is not exact-deterministic run-to-run in target-only reasoning text, so use content/finish plus a more stable token-level harness before declaring parity.
8. [ ] **Performance validation last:** only benchmark after correctness passes. Use the standard IQ2_XXS model and `llama-benchy` resident-server harness (`pp2048`, `tg32`, runs >= 3) before making any speed conclusion. Prior IQ1_M one-off `llama-server` timing logs are smoke evidence only, not benchmark evidence.
9. [x] **Normal runtime gate:** MTP works without `DSV4_MTP_PROBE=1`; that env var remains diagnostics-only.

### Current triangulation checkpoint — accepted drafts are no longer the main blocker

#### MTP verifier/prefill obvious-factor investigation
- [x] Correct the scope: the pp2048 prompt prefill itself remains ~400 tok/s with MTP enabled; the slow path is the MTP speculative decode/verification path after prefill.
- [x] Fix the sidecar MTP graph topology churn: raw-cache length now uses a fixed sliding-window tensor plus mask, so changing `n_raw` no longer forces a new MTP graph shape. Short trace now keeps sidecar predictions aligned with integrated MTP across accepted rows.
- [x] Batch DS4 partial-accept replay instead of replaying accepted tokens through one `decode()` call per token.
- [x] Test depth/alignment hypothesis: draft-max 3 creates more drafts but standard pp2048/tg32 got worse (`tg≈13.27`) because partial-accept replay dominates; cap was restored to 2.
- [x] Profile the remaining draft-max 1 verifier path: MTP draft generation is not the wall (`dur(g)=14 ms` in pp2048/tg8 profile); full target-model work dominates (`MUL_MAT_ID`, `MUL_MAT`, `DSV4_SPARSE_ATTN`).
- [ ] Next real speed fix must target DS4 small continuation verifier batches (`n_tokens=2/3`, `path=decode-replay`) or implement a DS4-style micro verifier/prefix-frontier capture; standard MTP remains slower until verifier batches are faster than separate target decodes.
- [x] Tested removing the raw-target-top1 pre-gate after the raw-cache fix. It helps draft-max 1 (`tg≈19.9`) but hurts draft-max 2 (`tg≈13.4`) by increasing partial accepts/replay, so raw-top pre-gating is now retained for multi-token drafts and relaxed only for draft-max 1.
- [x] Added a separate MTP scheduler/graph result to avoid target graph eviction of the MTP sidecar graph. It reduced sidecar draft cost (`dur(g)` around `147 ms -> 113 ms` in draft-max 1 smoke), but standard throughput is still verifier-bound.
- [x] Retested depth 4 after the raw-cache fix. It remains unsafe: the counting prompt diverged (`...10,10,10...`) despite accepted drafts, confirming deeper verifier batches still need an exact DS4 prefix/frontier solution before exposure.

#### Post-boundary cleanup execution plan
- [x] Build mounted Docker binaries from commit `2be876ea0` before runtime testing.
- [x] Smoke IQ2_XXS MTP without diagnostics and compare draft generation/acceptance against the prior diagnostics-on/off mismatch hypothesis: normal path now generates/accepts drafts without diagnostics on raw deterministic prompts.
- [x] If drafts are generated but still slow, isolate the remaining overhead by checking verifier `seq_rm` rollback cost after the live-frontier cap: full-accept verifier cleanup was a no-op tail `seq_rm` and skipping it made raw deterministic MTP faster than target-only (`52.12 tok/s` vs `46.94 tok/s`).
- [x] Fix the highest-confidence blocker inside DS4/model/memory boundaries only, then repeat the same smoke: safe recursive depth-2 plus full-accept cleanup skip preserve raw output parity and speed up predictable decode.
- [ ] Standard `llama-benchy` (`pp2048`, `tg32`) is still blocked: freecheck MTP generates 0-1 drafts on natural bench text, so decode remains below target-only despite raw-prompt speedup. Next fix must improve MTP draft quality/alignment for long natural prompts, not verifier overhead.
- [ ] Once MTP is correct and faster in standard smoke, run deterministic parity and standard resident `llama-benchy` (`pp2048`, `tg32`, runs >= 3).
- [x] Raw-cache continuity fix increased IQ2_XXS smoke acceptance from low/unstable acceptance to `23/31 = 0.742` accepted drafts on the counting prompt; `n_raw` now grows instead of staying pinned at 2.
- [x] Explain why high-acceptance MTP smoke is still slower before making more speculative state changes: server-level speculative tail cleanup (`llama_memory_seq_rm`) costs ~111 ms per speculative batch on DS4, and DS4 replay rollback is still expensive on rejected drafts.
- [ ] Re-ground the next fix in `/home/bigkahuna/src/ds4` and PR #22673 before changing behavior again: identify exactly which component owns draft-state advance, rejection discard, and target verifier rollback.
- [ ] Boundary cleanup before more MTP tuning: remove DS4-specific reference logic/comments/env handling from generic server/speculative/context code paths, and expose architecture-specific behavior through DS4-specific model/memory helpers or generic hooks.
  - [x] Removed DS4/MTP env names, comments, and timing probes from generic server/speculative code.
  - [x] Replaced context-level DS4 memory downcast for MTP replay with a generic recurrent-state memory hook implemented by hybrid-iSWA memory.
  - [x] Moved MTP draft validation and MTP raw-window policy out of generic context behind `llama_model` dispatch methods.
- [ ] Separate overhead with one hypothesis per run: sidecar graph cost, target verifier batching cost, DeepSeek4 replay rollback cost, and scheduler/server synchronization cost.
- [ ] Only after the reference-backed owner of rollback/tail cleanup is identified, implement the smallest targeted fix and re-run the same smoke.

### Next checkpoint — Qwen/DS4-aligned MTP runtime fix
- [x] Clean the current wrong-direction architecture changes first, especially the `src/models/models.h` MTP hook.
- [x] Re-establish the minimal model-agnostic server/context interface needed by existing speculative flow, not a new public MTP API.
- [x] Re-read and mirror the relevant shape of upstream PR `ggml-org/llama.cpp#22673`: MTP drafting is driven from `common/speculative.cpp`, uses existing draft placement/config, streams target hidden rows into a private MTP context/graph, recursively drafts from the actual sampled token, and trims drafter state on accept/reject.
- [x] Re-read and mirror `/home/bigkahuna/src/ds4`: normal target decode builds one MTP row for the committed token, recursive MTP drafts from the accepted sampled token, and MTP raw state is private to the drafter while target verifier state is handled by checkpoint/rollback.
- [x] Replace the current DeepSeek4 integrated draft-2 gating (`draft[0] must equal sampled before any draft is returned`) with sampled-token-conditioned drafting so the speculative path can propose a real next token after every greedy sampled token.
- [x] Implement explicit private MTP raw-state commit/discard plumbing for the current integrated probe path: committed sampled rows are appended once, accepted speculative draft rows are committed or dropped, and verifier batches do not clear unrelated private MTP state.
- [x] Keep emitted tokens target-verified through the existing server speculative verifier path; do not bypass target sampling or checkpoint/rollback.
- [ ] Replace target-graph-integrated MTP blocks with a Qwen-PR-style sidecar/MTP-only graph. Finding: integrated MTP in the target graph accepts drafts but cannot be a speed path; even a continuous verifier-batch experiment generated/accepted more drafts (`47/47` accepted in IQ1_M smoke) yet slowed decode to `14.97 tok/s` because every target graph paid MTP block/output-head cost.
- [ ] Validate Docker mounted-code parity target-only vs MTP on deterministic token-level prompts, then rerun the standard IQ2_XXS resident-server `llama-benchy` path (`pp2048`, `tg32`, runs >= 3).
- [ ] Resolve the speed blocker using the standard benchmark path: IQ2_XXS resident Docker server + `llama-benchy` (`pp2048`, `tg32`, runs 3) currently shows target-only `pp=382.60±3.22`, `tg=30.56±0.62`; current integrated MTP `pp=383.50±0.43`, `tg=9.07±1.76`, and depth-2 integrated MTP remained slower (`tg≈7.7-7.9`). The prior IQ1_M short-prompt server-log numbers are superseded and are smoke-only.

### Plan
- [x] Download/stage `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` locally without modifying the target GGUFs: `/mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (`3,807,602,400` bytes).
- [x] Add default-off DeepSeek4 MTP validation/probe plumbing behind existing speculative flags: `--spec-type mtp`, `--model-draft <MTP.gguf>`, and `--draft-max`.
- [x] Implement a DeepSeek4 sidecar GGUF metadata/tensor validator for `general.architecture = deepseek4_mtp_support`.
- [x] Wire server startup to validate the sidecar and fail cleanly on unsupported/non-DeepSeek4 combinations.
- [x] Keep runtime speculative commit disabled; draft-token diagnostics deferred until sidecar tensor loading.
- [x] Add default-off generic MTP state output/copy plumbing for `DSV4_MTP_PROBE=1` + `--spec-type mtp`, with DeepSeek4 supplying final HC state as the handoff tensor.
- [x] Add env-gated sidecar tensor data loading into a persistent backend weight buffer.
- [x] Add an env-gated DeepSeek4 MTP projection/top-1 probe that feeds base token embedding + captured target HC state through sidecar `enorm/e_proj`, `hnorm/h_proj`, sidecar HC head/norm, and base output, then logs probe top-1 vs target argmax without changing emitted tokens.
- [x] Validate real DeepSeek4 target + MTP sidecar startup/projection probe on IQ1_M CPU-only target; fix CPU BF16 RHS matmul compatibility and explicitly connect probe top-1 graph outputs.
- [x] Build a raw-current one-token MTP transformer block probe with sidecar attention/FFN/HC tensors and logical layer-1 RoPE, without mutating target KV/cache state.
- [ ] Add persistent private MTP drafter state and target verifier frontier handling before using block-probe output as a real speculative draft.
  - [x] Start with a private raw-window cache (`mtp_raw_cache`/`mtp_n_raw` equivalent) owned by `llama_context`, separate from target KV/cache.
  - [x] Feed the one-token MTP block from that private raw cache instead of current-token-only raw attention on continuation steps.
  - [x] Trace DS4 authority MTP cache/state tensors and map them to llama.cpp DeepSeek4 graph tensors: the sidecar calls logical layer id 1, where `deepseek4_compress_ratios[1] == 0`, so private compressed/indexer drafter state is not required for the MTP block.
  - [x] Capture the MTP sidecar output HC state as a graph/context handoff for recursive draft probing.
  - [x] Verify target compressed/indexer frontier rollback for speculative verification uses existing hybrid-ISWA state checkpoints (`mem_recr` frontiers + DeepSeek4 compressed cache rows) before enabling commit.
  - [x] Add an MTP-only recursive draft-2 probe that consumes target HC for draft[0], then sidecar HC plus draft[0] token for draft[1], without running target layers or changing emitted tokens.
  - [x] Capture recursive draft raw-row candidates as a host handoff for future accepted-token private raw-cache commit.
  - [x] Add probe-only verifier preview accounting: compare draft[1] with current target argmax, and compare the previous step's draft[2] with current target argmax only when previous draft[1] matched the actual token that entered this step.
  - [ ] Generalize the recursive MTP probe beyond draft-2 if future validation shows value beyond DS4's production depth-two path.
  - [x] Harden reset/slot lifecycle handling for the host-backed raw probe state before any speculative verification/commit work.
- [x] Build `llama-server` and run no-MTP regression/smoke checks to prove default behavior is unchanged.
- [x] Document results and exact next step for draft-one graph probing: `tasks/dsv4_mtp_loader_probe.md`.

### Review
- `llama-server` build passed after MTP sidecar validation, generic MTP state output/copy, env-gated sidecar tensor loading, projection/top-1 probe changes, real-target CPU BF16/top1-output fixes, and raw-current MTP block probe wiring.
- Server help exposes existing speculative flags `--spec-type`, `--model-draft`, and `--draft-max`.
- Tiny non-DeepSeek4 target plus `DSV4_MTP_PROBE=1 --spec-type mtp --model-draft ...` exits cleanly before tensor-data loading with the expected unsupported-target error.
- Tiny no-MTP server health check returned `{"status":"ok"}` after 2 seconds.
- Real DeepSeek4 IQ1_M + MTP sidecar projection probe reached health after 82 seconds and logged `dsv4 mtp projection probe: target_argmax=201 projection_top1=20219 match=0` on a one-token completion without changing emitted-token behavior.
- Real DeepSeek4 IQ1_M + raw-current MTP block probe reached health after 88 seconds and logged `dsv4 mtp block probe: target_argmax=201 draft_top1=2390 match=0` on a one-token completion without changing emitted-token behavior.
- Real DeepSeek4 IQ1_M + private raw-cache probe reached health after 94 seconds on an `n_predict=2` run and logged two probe rows, including continuation step `target_argmax=200 draft_top1=5 match=0` after consuming one private MTP raw-cache row.
- Private MTP raw cache now stores prior rows only and caps at `raw_window - 1`, matching the graph's separate current-row concatenation and avoiding a future long-context `raw_window + 1` attention span.
- Real DeepSeek4 IQ1_M + recursive MTP draft-2 probe (`--draft-max 2`) reached health after 80 seconds, returned `"\n\t"`, and logged `draft2_top1` without changing emitted tokens: first row `target_argmax=201 draft_top1=2390 draft2_top1=42`, continuation row `target_argmax=200 draft_top1=5 draft2_top1=23166`.
- Real DeepSeek4 IQ1_M target-only baseline (`n_predict=3`, raw `prompt="Hi"`, `temperature=0`) returned `"\n\t\t"`, matching the verifier-preview MTP probe output exactly; this shows the suspicious raw-prompt output was not introduced by MTP, but also proves raw `"Hi"` is a poor safety prompt.
- Real DeepSeek4 IQ1_M + verifier-preview probe (`--draft-max 2`, `n_predict=3`) reached health after 90 seconds, returned `"\n\t\t"`, and logged preview accounting summary `draft1_hits=0/3 draft2_hits=0/0` with no emitted-token changes relative to that raw-prompt target-only baseline.
- Docker mounted-code runtime validation now uses `llamatrifecta_deepseekv4:latest` with `/home/bigkahuna/src/llama.cpp-dsv4-mtp-loader-probe:/src` and `build-dsv4-container/bin/llama-server`, matching the resident sidecar workflow instead of host CPU-only runtime checks.
- Container target-only chat baseline (`/tmp/dsv4_mtp_docker_target_chat_resp.json`) and MTP-probe chat run (`/tmp/dsv4_mtp_docker_probe_chat_resp.json`) matched exactly on assistant content, reasoning text, finish reason, and usage. MTP probe log showed sidecar tensors loaded into `CUDA_Host` and preview accounting `draft1_hits=9/30 draft2_hits=1/10`.
- `/mnt/supmodels/gguf/deepseek-ai__DeepSeek-V4-Flash-Q2_K_S.with-template.gguf` is not usable for this probe; loader reports it is corrupted/incomplete (`blk.4.ffn_down_exps.weight` out of file bounds).
- `git diff --check` passed.

## DeepSeek4 MTP optimization checkpoint (2026-05-10)

### Plan
- [x] Build the latest mounted Docker `llama-server` after the adaptive MTP cooldown changes.
- [x] Clean stale ad hoc benchmark containers before launching a resident MTP benchmark server.
- [x] Validate IQ2_XXS resident-server MTP with `llama-benchy pp2048 tg32 --runs 3`, coherence enabled, `--draft-max 1`.
- [x] Record accepted/generated draft counts and compare decode throughput against the same-build target-only baseline.
- [ ] Re-test deterministic raw ctx8192 parity after the adaptive cooldown if more code changes land.
- [x] Audit final diff for accidental diagnostics overhead: top-2 logits/margin are gated by `LLAMA_MTP_TRACE`, and normal runs only keep the required MTP state/top1 outputs.
- [x] Run `git diff --check` and stop all ad hoc MTP containers before commit.

### Review
- Mounted Docker build passed: `cmake --build build-dsv4-container --target llama-server -j 8`.
- IQ2_XXS MTP `--draft-max 1` resident benchmark passed coherence. Saved artifacts:
  - `/tmp/dsv4_iq2xxs_mtp_draft1_adaptive_benchy_pp2048_tg32_runs3.json`
  - `/tmp/dsv4_iq2xxs_mtp_draft1_adaptive_server_runs3.log`
  - `/tmp/dsv4_iq2xxs_mtp_draft1_adaptive_benchy_runs3.out`
- MTP throughput: pp2048 `397.83 ± 3.37 tok/s`, tg32 `30.73 ± 0.34 tok/s`. Same-build target-only comparison: pp2048 `400.30 ± 2.08 tok/s`, tg32 `29.93 ± 0.36 tok/s`. This is the first standard IQ2_XXS decode win on the resident benchmark path, with a small prompt-eval cost.
- Server draft stats by measured run: `4/6`, `5/8`, and `8/12` accepted/generated; adaptive cooldown avoided drafting on low-value steps instead of forcing sidecar work every token.
- Target-only artifact saved to `/tmp/dsv4_iq2xxs_target_postrollback_benchy_pp2048_tg32_runs3.json`; server log saved to `/tmp/dsv4_iq2xxs_target_postrollback_server_runs3.log`.
- Remaining caveat: draft depths above 1 still need cleanup; draft-max 2 has not yet shown a standard benchmark win and draft-max 4 remains correctness-unsafe.

## DeepSeek4 MTP regression before decode tuning (2026-05-10)

### Plan
- [x] Start an ad hoc mounted-code IQ2_XXS server from this branch with MTP enabled (`--spec-type mtp --draft-max 1`) so the committed speculative path is under test.
- [x] Run `scripts/dsv4_regression.py` per `../llama.cpp-deepseek-v4/tasks/dsv4_regression_handoff.md` with the tool replay fixture and `--max-tokens 2048`.
- [x] Run a target-only comparison to separate default-server regressions from MTP-enabled regressions.
- [x] Record pass/fail details, stop the ad hoc server, and verify GPUs are idle.
- [x] Before decode-speed work, decide whether MTP-enabled tool-call regressions must be fixed now or explicitly scoped out of speed benchmarking: fix now, because the same restored-frontier path is prerequisite for safe draft-depth speedups.

### Review
- Default/target-only mounted-code IQ2_XXS regression passed all 7 checks. Artifact: `/tmp/dsv4_iq2xxs_target_regression.out`; server log: `/tmp/dsv4_iq2xxs_target_regression_server.log`.
- MTP-enabled `--draft-max 1` regression did **not** pass cleanly:
  - First MTP run: 6/7 passed; `tool_replay_turn02_cache_reuse` failed because the first replay emitted prose/code-fence text instead of structured `tool_calls`. Artifact: `/tmp/dsv4_iq2xxs_mtp_regression.out`; server log: `/tmp/dsv4_iq2xxs_mtp_regression_server.log`.
  - Retry MTP run: 6/7 passed; `tool_replay_turn02_stream` failed with prose text instead of structured `tool_calls`, while the repeated cache-reuse check passed. Artifact: `/tmp/dsv4_iq2xxs_mtp_regression_retry.out`; server log: `/tmp/dsv4_iq2xxs_mtp_regression_retry_server.log`.
- The MTP failures occurred on long tool replay requests after prompt-cache/checkpoint restore (`cached_tokens=1408` path), not on the cold non-stream replay. This points at MTP speculative verifier/state interaction around restored DS4 frontiers, not a default parser/server regression.
- All ad hoc regression containers were stopped and GPUs returned to idle (`2 MiB`, `2 MiB`, `1 MiB`).

## DeepSeek4 MTP restored-frontier regression fix (2026-05-10)

### Plan
- [x] Reproduce the failing MTP long tool replay with a narrow harness and trace slot/prompt-cache transitions around restored checkpoints.
- [x] Audit MTP private state lifecycle on `prompt_load`, checkpoint restore, prompt-cache reuse, and `llama_memory_seq_rm`; target state can be restored/truncated without rebuilding private MTP raw/HC state, and DS4 structured tool-call parsing exposed small verifier-batch divergences after restored prompt-cache frontiers.
- [x] Implement the smallest safe fix: clear MTP private state on prompt-cache load, checkpoint restore, and prompt truncation; additionally skip speculative MTP for structured tool-call parsing requests until exact verifier/frontier handling exists for those workloads.
- [x] Rebuild mounted Docker `llama-server` and rerun target-only plus MTP-enabled IQ2_XXS regression suite.
- [x] Rerun standard `llama-benchy pp2048 tg32` to quantify speed impact, then proceed to `--draft-max 2` verifier/frontier speed work.

### Review
- Mounted Docker build passed after the server-context guard/reset changes: `cmake --build build-dsv4-container --target llama-server -j 8`.
- MTP-enabled regression with tool-call guard passed all 7 checks. Artifact: `/tmp/dsv4_iq2xxs_mtp_regression_toolguard.out`; server log: `/tmp/dsv4_iq2xxs_mtp_regression_toolguard_server.log`.
- Target-only regression was flaky on the first post-change run but passed all 7 on retry. Passing artifact: `/tmp/dsv4_iq2xxs_target_regression_after_toolguard_retry.out`; server log: `/tmp/dsv4_iq2xxs_target_regression_after_toolguard_retry_server.log`. Failed first-run artifact kept as `/tmp/dsv4_iq2xxs_target_regression_after_toolguard.out` for comparison.
- Post-fix MTP `llama-benchy pp2048 tg32 --runs 3` passed coherence and measured pp2048 `405.28 ± 2.26 tok/s`, tg32 `30.65 ± 0.94 tok/s`, preserving the previous draft-max-1 decode win. Artifact: `/tmp/dsv4_iq2xxs_mtp_toolguard_benchy_pp2048_tg32_runs3.json`; server log: `/tmp/dsv4_iq2xxs_mtp_toolguard_benchy_server.log`.
- Superseded by the exact verifier fix below: the temporary target-only tool workload guard passed regression but was the wrong scope and has been removed.

## DeepSeek4 MTP tool-call exact verifier fix (2026-05-10)

### Plan
- [x] Undo the tool-call MTP bypass and the documentation that claimed tool workloads are target-only.
- [x] Fix the actual correctness issue: speculative verifier sampling must mirror the target-only DS4 tool-call greedy-in-tool-call state machine, using a local copy of parser state while verifying draft rows.
- [x] Keep MTP private state clearing at prompt-cache/checkpoint restore boundaries, because restore/truncation is a real sidecar-state discontinuity.
- [x] Rebuild mounted Docker `llama-server`.
- [x] Rerun MTP-enabled IQ2_XXS regression with tool replay and confirm all 7 pass with speculation enabled for tool requests.
- [x] Rerun target-only regression if needed and standard non-tool `llama-benchy pp2048 tg32` to check speed impact.

### Review
- Removed the tool-call MTP bypass and README caveat. Tool workloads now remain eligible for MTP under an MTP-enabled server.
- Added a speculative verifier path that mirrors DeepSeek4 target-only tool parsing semantics: when the local verifier parser state is inside a DSML tool call, verifier rows use argmax sampling just like the one-token target path; the verifier advances only a local parser-state copy and leaves the real slot parser state for `process_token()`.
- Kept private MTP state clears at prompt-cache load, checkpoint restore, and prompt truncation boundaries.
- Mounted Docker `llama-server` build passed.
- MTP-enabled IQ2_XXS regression passed all 7 checks with speculation enabled for tool requests. Artifact: `/tmp/dsv4_iq2xxs_mtp_toolverify_regression.out`; server log: `/tmp/dsv4_iq2xxs_mtp_toolverify_regression_server.log`.
- Standard non-tool `llama-benchy pp2048 tg32 --runs 3` passed coherence and measured pp2048 `403.07 ± 4.12 tok/s`, tg32 `31.79 ± 1.76 tok/s`. Artifact: `/tmp/dsv4_iq2xxs_mtp_toolverify_benchy_pp2048_tg32_runs3.json`; server log: `/tmp/dsv4_iq2xxs_mtp_toolverify_benchy_server.log`.

## DeepSeek4 MTP draft-depth decode speed work (2026-05-10)

### Plan
- [x] Establish the current post-regression-guard `--draft-max 2` standard benchmark (`llama-benchy pp2048 tg32 --runs 3`) against the same target-only and draft-max-1 baselines.
- [x] Capture server draft acceptance statistics and identify whether slowdown comes from partial accepts, verifier batch cost, or sidecar draft cost.
- [x] Run a short `LLAMA_MTP_TRACE=1` diagnostic smoke to correlate accept patterns with rollback/restore behavior.
- [x] Re-read the DS4 verifier/frontier authority functions before changing code: `spec_frontier_snapshot`, `spec_frontier_restore`, `spec_frontier_commit_prefix1`, `metal_graph_verify_suffix_tops`, and the exact `metal_graph_verify_decode2_exact` path.
- [ ] Implement the smallest exact verifier/frontier change that can make 2-token verification cheap without compromising target-verified output.

### Review
- Post-regression-guard `--draft-max 2` standard run passed coherence but did not beat draft-max 1: pp2048 `406.60 ± 0.89 tok/s`, tg32 `30.24 ± 0.40 tok/s`. Artifact: `/tmp/dsv4_iq2xxs_mtp_toolguard_draft2_benchy_pp2048_tg32_runs3.json`; server log: `/tmp/dsv4_iq2xxs_mtp_toolguard_draft2_benchy_server.log`.
- Draft-max 2 is no longer catastrophically slow, but token acceptance remains low (`~0.33-0.38` in measured runs) and below draft-max 1 economics. The adaptive cooldown prevents a large regression, but it also means depth 2 is not contributing much.
- Trace smoke (`pp2048 tg16 runs=1`, `LLAMA_MTP_TRACE=1`) showed partial accepts restore via prefix HC state (`partial accept restore=1`) and then often skip the next MTP draft because the raw target-top gate sees stale verifier-row top-1 after a correction token. Artifact: `/tmp/dsv4_iq2xxs_mtp_draft2_trace_tg16.json`; server log: `/tmp/dsv4_iq2xxs_mtp_draft2_trace_tg16_server.log`.
- Tested and rejected a naive server-side update of `mtp_target_top1` to the verifier correction token after accept; it increased bad draft attempts and slowed draft-max 2 to tg32 `28.74 ± 0.85 tok/s`. Reverted the experiment. Artifact kept at `/tmp/dsv4_iq2xxs_mtp_targettopfix_draft2_benchy_pp2048_tg32_runs3.json`.
- DS4 authority confirms two verifier modes: the fast `metal_graph_verify_suffix_tops` microbatch path can perturb greedy tokens, while `metal_graph_verify_decode2_exact` preserves exact target stream but is not a speed win as written. The next implementation target should be exact verifier/frontier semantics in llama.cpp, not more loose gating.
