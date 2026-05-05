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
