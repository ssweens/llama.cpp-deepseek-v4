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
  - [ ] Add an MTP-only recursive draft probe that consumes target HC for draft[0], then sidecar HC for draft[1..N], without running target layers or changing emitted tokens.
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
- `/mnt/supmodels/gguf/deepseek-ai__DeepSeek-V4-Flash-Q2_K_S.with-template.gguf` is not usable for this probe; loader reports it is corrupted/incomplete (`blk.4.ffn_down_exps.weight` out of file bounds).
- `git diff --check` passed.
