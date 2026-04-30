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
- [x] Add NaN-prevention guard to FA kernels for fully-masked attention columns (CUDA + SYCL/HIP)
- [x] Write detailed custom sparse-gather kernel implementation plan

## In Progress
- [ ] Implement `GGML_OP_DSV4_SPARSE_ATTN` custom kernel (see `tasks/dsv4_sparse_attn_kernel_plan.md`)
  - Phase 1: CUDA kernel with index-gather, online softmax, attention sink support
  - Phase 2: HIP/ROCm portability via shared CUDA/HIP macros
  - Phase 3: Integrate into `src/models/deepseek4.cpp` graph builder, replace dense mask path
- [ ] GCP: imatrix + logits generation on h100-4x
  - **Blocked on H100-4x spot capacity** — all configured zones stocked out
  - BF16 cache ready at `gs://test-quant-jobs/bf16-cache/deepseek-ai__DeepSeek-V4-Flash/`

## Remaining
- [ ] GCP: IQ2_S quantization using generated imatrix
- [ ] Download artifacts locally, validate fit + quality
- [ ] Add/update corral config entry for DSV4-Flash IQ2_S
- [ ] Update documentation (README, TEST_COVERAGE, etc.)
- [ ] Final commit + push of any remaining changes
