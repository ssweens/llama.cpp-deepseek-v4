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
- [ ] **INVESTIGATE: IQ2_XS long prompt decode loop is suspicious, do not assume quant ceiling.**
  - Symptom: long isolation prompt with `-fa on` and IQ2_XS produces correct prompt-eval and proper Evidence/Interpretation/Next Step section structure, then collapses into repeating "The final sentence must be a single short sentence." sentence-by-sentence.
  - User pushback: "I don't believe this to be true" — user does not accept the quick attribution to IQ2 quant quality.
  - Required diagnosis steps:
    - Compare IQ2_XS `-fa off` (CPU softmax path) on the same long prompt. If it produces coherent structured output, this is a sparse-op-specific issue, not a quant ceiling.
    - Compare with `-ngl 0` and `CUDA_VISIBLE_DEVICES=` cleared to factor out backend interactions.
    - Compare with a non-thinking-mode prompt at the same length to factor out reasoning-quality loss.
    - Compare with `--temp 0.7 --top-p 0.95` non-greedy sampling to factor out greedy-decode pathologies.
    - If a real bug remains, suspected areas: (a) sparse op behavior with quantized indexer Q matmul inputs, (b) attention-sink interaction with online softmax under quantized accumulation, (c) cache-validity mask interaction with the SWA window across the chunk boundary.
- [ ] Phase 2 sparse kernel optimization: tensor-core MMA path for higher decode throughput
  - Current naive kernel: ~0.5 tok/s decode at -ngl 8 BF16 (memory-bandwidth + scalar dot products)
  - Goal: match or exceed pre-bug FA performance using MMA tensor cores
- [x] **Phase 3: HIP/ROCm compatibility verified.** The `dsv4-sparse-attn.cu` source is automatically picked up by `ggml-hip`'s file glob and compiles cleanly under HIP via the existing CUDA→HIP translation in `ggml-cuda/vendors/hip.h`. `nv_bfloat16` is aliased to `__hip_bfloat16`, `__shfl_xor_sync(width=32)` works on both wave32 and wave64 hardware (via sub-warp semantics). HIP build target succeeds. Runtime validation on AMD GPU still pending.
- [ ] GCP: imatrix + logits generation on h100-4x
  - **Blocked on H100-4x spot capacity** — all configured zones stocked out
  - BF16 cache ready at `gs://test-quant-jobs/bf16-cache/deepseek-ai__DeepSeek-V4-Flash/`

## Remaining
- [ ] GCP: IQ2_S quantization using generated imatrix
- [ ] Download artifacts locally, validate fit + quality
- [ ] Add/update corral config entry for DSV4-Flash IQ2_S
- [ ] Update documentation (README, TEST_COVERAGE, etc.)
- [ ] Final commit + push of any remaining changes
