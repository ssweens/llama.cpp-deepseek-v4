# DSv4 Sparse Gather-Attention CUDA/HIP Kernel — Implementation Plan

## Executive Summary

Replace the current dense-mask approach for DeepSeek-V4-Flash compressed attention with a dedicated sparse gather-attention CUDA kernel. The HF reference implementation (`kernel.py::sparse_attn_kernel`) gathers only top-k KV positions by index, achieving O(N·topk) compute and memory instead of O(N²). Our current approach builds a dense mask with -inf for non-selected positions and feeds it to generic FA kernels, wasting both memory and compute.

---

## 1. Architecture Overview

### 1.1 Current Dense-Mask Approach (What We Replace)

The current code path in `src/models/deepseek4.cpp` (lines ~1463-1530 for prompt, ~1699-1736 for decode):

1. **Indexer** computes `idx_scores` via matmul of indexer Q/K → `ggml_argsort_top_k` → `idx_topk`
2. **`dsv4_build_compressed_mask_from_topk()`** (line 463) builds a dense mask:
   - Creates `[1, n_comp, n_tokens]` filled with `-inf`
   - Uses `ggml_set_rows` to place score-derived values at topk positions
   - Result: a dense `[n_comp, n_tokens]` mask where most entries are `-inf`
3. **`build_attn_mha()`** (line 1526/1736) concatenates window mask + comp_mask → dense `full_mask`
4. **`ggml_flash_attn_ext()`** receives this dense mask and processes ALL positions, even masked ones

**Problems:**
- Dense mask is `O(n_kv × n_tokens)` memory — for 128K context with ratio=4, that's 32K compressed entries × n_tokens × sizeof(f16)
- FA kernel computes `Q·K^T` for ALL positions, then zeroes out masked ones via exp(-inf)
- Required a NaN-prevention hack (clamping `-inf` max to 0.0f)
- Concatenating window + compressed masks doubles the KV dimension further

### 1.2 New Sparse Gather-Attention Op: `GGML_OP_DSV4_SPARSE_ATTN`

**Concept:** Instead of building a dense mask, pass the top-k indices directly to a custom kernel that gathers only the selected KV positions.

**Inputs:**
- `Q` — query tensor `[head_dim, n_heads, n_tokens, batch]` (F32, same as current FA input)
- `KV_cache` — compressed KV cache `[head_dim, 1, n_kv_total, batch]` (F16/BF16, MLA-style where K=V)
- `topk_idxs` — indices of selected positions `[topk, n_tokens, batch]` (I32)
- `attn_sink` — per-head learned attention sink `[n_heads]` (F32)

**Output:**
- Attention result `[head_dim, n_heads, n_tokens, batch]` (F32)

**Op parameters (in `op_params`):**
- `scale` (float) — attention scale factor, typically `1/sqrt(head_dim)`

### 1.3 Index-Gather vs Dense-Mask Approach

| Aspect | Dense Mask (Current) | Sparse Gather (Proposed) |
|--------|---------------------|--------------------------|
| Mask memory | `O(n_kv × n_tokens)` | `O(topk × n_tokens)` — indices only |
| KV positions computed | All `n_kv` | Only `topk` |
| Memory access pattern | Sequential KV scan | Gathered by index |
| Softmax | Full-width, many `-inf` exp()s | Only topk positions + sink |
| NaN risk | Yes (exp(-inf) edge cases) | None — no masked positions |
| Kernel complexity | Uses generic FA with mask | Dedicated simple kernel |

### 1.4 Relationship to Window Attention

The sparse gather kernel replaces ONLY the compressed-cache attention path. Window/SWA attention (the recent raw tokens) continues to use the existing FA infrastructure via `build_attn`. The final output is a combination of:
1. Window attention output (existing FA)
2. Compressed attention output (new sparse kernel)

These two results must be combined. Options:
- **Option A (Recommended):** The sparse kernel handles only compressed positions. At the graph level, we run window FA and sparse compressed FA separately, then merge with proper softmax rescaling.
- **Option B:** The sparse kernel handles both — window indices + compressed top-k indices are concatenated into a single index tensor.

We recommend **Option A** for Phase 1 (simpler, no changes to window FA), with Option B as a potential Phase 2 optimization.

**Important Note for Option A:** Merging two separate attention outputs requires tracking the softmax normalization statistics (max value and sum of exponentials) from each. The existing FA infrastructure already supports this via `dst_meta` (the `float2` containing max and rowsum per query). The sparse kernel must similarly output these statistics so the merge kernel can properly rescale and combine:

```
merged_output = (scale_window * window_output + scale_comp * comp_output) / (scale_window * window_rowsum + scale_comp * comp_rowsum)
where scale_x = exp(max_x - max_combined)
```

---

## 2. CUDA Kernel Design

### 2.1 HF Reference Kernel Analysis

From `kernel.py` line 277, the HF `sparse_attn_kernel` operates as follows:

```
Grid:        (m, b)         — one block per (query_position, batch)
Threads:     256
Block size:  64 KV positions per iteration
Pipeline:    2 stages
```

**Per-block algorithm:**
1. Load `Q[batch, query_pos, :, :]` into shared memory — shape `[n_heads, head_dim]`
2. For each block of 64 KV positions (pipelined):
   a. Gather indices: `idxs[i] = topk_idxs[batch, query_pos, t*64 + i]` (or -1 if OOB)
   b. Gather KV: `kv_shared[i, j] = kv[batch, idxs[i], j]` (or 0 if invalid)
   c. Initialize `acc_s[i, j] = -inf` for invalid indices, 0 for valid
   d. GEMM: `acc_s += Q @ KV^T` — computes `[n_heads, 64]` scores
   e. Scale: `acc_s *= scale`
   f. **Online softmax update:**
      - Save `scores_max_prev = scores_max`
      - `scores_max = max(scores_max, row_max(acc_s))` (running max)
      - `scores_scale = exp(scores_max_prev - scores_max)` (rescale factor)
      - `acc_s = exp(acc_s - scores_max)` (softmax numerator)
      - `sum_exp = sum_exp * scores_scale + sum(acc_s)` (running denominator)
      - `acc_o *= scores_scale` (rescale previous output)
   g. GEMM: `acc_o += softmax(acc_s) @ KV` — accumulate attention output
3. **Sink integration** (after all KV blocks):
   - `sum_exp += exp(attn_sink - scores_max)` — adds sink to denominator
4. Final normalization: `acc_o /= sum_exp`
5. Write `acc_o` to output

**Key insight:** The HF kernel processes ALL heads for one query position in a single block. This works because DSv4 uses MLA (Multi-Latent Attention) where K=V and `head_dim` is the compressed KV dimension (512), shared across all heads.

### 2.2 Our CUDA Kernel Design

We adapt the HF design to the llama.cpp CUDA ecosystem, matching the existing coding patterns in `fattn-common.cuh` and `dsv4.cu`.

#### Thread Block Organization

```
Threads per block: 256 (8 warps × 32 threads)
Grid: (n_tokens, batch)      — one block per query token per batch
```

**Why 256 threads:** Matches HF kernel, provides good occupancy, and 256 threads can efficiently load 512-dimensional KV vectors (2 elements per thread).

#### Template Parameters

```cpp
template <int HEAD_DIM, int BLOCK_KV, int N_HEADS>
__global__ void dsv4_sparse_attn_kernel(...)
```

- `HEAD_DIM`: 512 (DSv4 MLA compressed dimension). Template for compile-time loop unrolling.
- `BLOCK_KV`: 64 (KV positions per iteration, matching HF). Determines shared memory tile size.
- `N_HEADS`: Number of attention heads to process (padded to 16 in HF for efficiency).

#### Memory Layout and Access Pattern

```
Shared memory allocation:
  q_shared:     [N_HEADS, HEAD_DIM]       — query for this position (loaded once)
  kv_shared:    [BLOCK_KV, HEAD_DIM]      — gathered KV tile (reloaded each iteration)
  acc_s_shared: [N_HEADS, BLOCK_KV]       — attention scores (for GEMM output)

Register allocation (per thread):
  acc_o:        [N_HEADS, HEAD_DIM/threads_per_head] — running output accumulator (FP32)
  scores_max:   [N_HEADS_PER_THREAD]      — running max per head (FP32)
  sum_exp:      [N_HEADS_PER_THREAD]      — running sum of exponentials (FP32)
```

**Shared memory budget:**
- `q_shared`: `N_HEADS × HEAD_DIM × sizeof(half)` = `16 × 512 × 2` = **16 KB**
- `kv_shared`: `BLOCK_KV × HEAD_DIM × sizeof(half)` = `64 × 512 × 2` = **64 KB**
- `acc_s_shared`: `N_HEADS × BLOCK_KV × sizeof(half)` = `16 × 64 × 2` = **2 KB**
- **Total: ~82 KB** — fits within 96/100 KB shared memory on Ampere+

For GPUs with <96 KB shared memory (e.g., Turing at 64 KB), we can reduce `BLOCK_KV` to 32 or tile `HEAD_DIM` loading.

#### Kernel Pseudocode

```cuda
__global__ void dsv4_sparse_attn_kernel(
    const half  * __restrict__ Q,        // [head_dim, n_heads, n_tokens]
    const half  * __restrict__ KV,       // [head_dim, n_kv]
    const int   * __restrict__ topk_idx, // [topk, n_tokens]
    const float * __restrict__ attn_sink,// [n_heads]
    float       * __restrict__ O,        // [head_dim, n_heads, n_tokens]
    float       * __restrict__ O_meta,   // [2, n_heads, n_tokens] — (max, rowsum) for merge
    const float   scale,
    const int     topk,
    const int     n_heads,
    const int     head_dim,
    const int     n_kv,
    // stride info
    const int     stride_Q_token,
    const int     stride_KV_row
) {
    const int token_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;

    extern __shared__ char smem[];
    half * q_shared  = (half *)smem;                              // [n_heads, head_dim]
    half * kv_shared = q_shared + n_heads * head_dim;             // [BLOCK_KV, head_dim]
    half * s_shared  = kv_shared + BLOCK_KV * head_dim;           // [n_heads, BLOCK_KV]

    // Per-thread accumulators
    float acc_o[...];     // FP32 output accumulator
    float scores_max = -FLT_MAX;
    float sum_exp = 0.0f;

    // 1. Load Q for this token into shared memory
    cooperative_load(Q + token_idx * stride_Q_token, q_shared, n_heads * head_dim);
    __syncthreads();

    // 2. Iterate over topk in blocks of BLOCK_KV
    const int n_blocks = (topk + BLOCK_KV - 1) / BLOCK_KV;
    for (int blk = 0; blk < n_blocks; ++blk) {
        // 2a. Load indices
        int local_idx[BLOCK_KV];  // per-thread subset
        for (int i = threadIdx.x; i < BLOCK_KV; i += blockDim.x) {
            int global_i = blk * BLOCK_KV + i;
            local_idx[i] = (global_i < topk) ? topk_idx[token_idx * topk + global_i] : -1;
        }
        __syncthreads();

        // 2b. Gather KV by index into shared memory
        for (int i = ...; i < BLOCK_KV; ...) {
            int idx = local_idx[i];
            if (idx >= 0 && idx < n_kv) {
                // Copy KV[idx, :] into kv_shared[i, :]
                for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                    kv_shared[i * head_dim + j] = KV[idx * stride_KV_row + j];
                }
            } else {
                // Zero-fill for OOB
                for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                    kv_shared[i * head_dim + j] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // 2c. Compute attention scores: S = Q @ KV^T  -> [n_heads, BLOCK_KV]
        // Each warp handles a subset of heads
        compute_qk_gemm(q_shared, kv_shared, s_shared, n_heads, head_dim, BLOCK_KV, scale);
        __syncthreads();

        // 2d. Mask invalid positions with -inf
        for (int i = threadIdx.x; i < n_heads * BLOCK_KV; i += blockDim.x) {
            int kv_pos = i % BLOCK_KV;
            int global_kv = blk * BLOCK_KV + kv_pos;
            if (global_kv >= topk || local_idx[kv_pos] < 0) {
                s_shared[i] = __float2half(-INFINITY);
            }
        }
        __syncthreads();

        // 2e. Online softmax update (FlashAttention-style)
        float scores_max_prev = scores_max;
        float block_max = reduce_max_per_head(s_shared, ...);
        scores_max = fmaxf(scores_max, block_max);
        float rescale = expf(scores_max_prev - scores_max);

        // Rescale previous accumulators
        for (...) acc_o[...] *= rescale;
        sum_exp *= rescale;

        // Exponentiate scores
        for (int i = ...) {
            float s = __half2float(s_shared[...]);
            s = expf(s - scores_max);
            s_shared[...] = __float2half(s);
            sum_exp += s;
        }
        __syncthreads();

        // 2f. Accumulate: O += softmax(S) @ KV
        compute_sv_gemm(s_shared, kv_shared, acc_o, n_heads, head_dim, BLOCK_KV);
        __syncthreads();
    }

    // 3. Attention sink integration
    for (int h = ...) {
        float sink_val = attn_sink[h];
        sum_exp += expf(sink_val - scores_max);
    }

    // 4. Normalize and write output
    for (...) {
        O[...] = acc_o[...] / sum_exp;
    }

    // 5. Write meta (max, rowsum) for merge with window attention
    if (O_meta) {
        for (int h = ...) {
            O_meta[token_idx * n_heads * 2 + h * 2 + 0] = scores_max;
            O_meta[token_idx * n_heads * 2 + h * 2 + 1] = sum_exp;
        }
    }
}
```

### 2.3 GEMM Strategy for head_dim=512

For `head_dim=512`, the Q·K^T computation is substantial. Two approaches:

**Option A: Warp-level dot products (simpler, Phase 1)**
- Each warp handles one (head, kv_position) pair
- 32 threads compute a 512-element dot product via warp-level reduction
- With 8 warps and BLOCK_KV=64: need `n_heads × 64` dot products
- For 16 heads: 1024 dot products, each warp does `1024/8 = 128` dot products
- This is compute-bound but straightforward

**Option B: MMA-based (higher throughput, Phase 2)**
- Use tensor core operations from `mma.cuh`
- Interpret Q·K^T as a matrix multiplication: `[n_heads, head_dim] × [head_dim, BLOCK_KV]^T`
- Requires F16 accumulation with F32 reduction
- Matches the existing MMA infrastructure in `fattn-mma-f16.cuh`

**Recommendation:** Start with Option A for correctness, then optimize to Option B.

### 2.4 Attention Sink Handling

The HF kernel (line 338-339) adds the attention sink AFTER all KV blocks are processed:

```python
for i in T.Parallel(h):
    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
```

This is mathematically equivalent to having an extra "virtual" KV position with a score equal to `attn_sink[h]`. The sink contributes to the softmax denominator but does NOT contribute to the output numerator (no KV value is multiplied). This acts as a learned bias that shifts the softmax temperature.

Our kernel must replicate this exactly:
1. Process all topk KV blocks with online softmax
2. After the loop, for each head: `sum_exp[h] += expf(attn_sink[h] - scores_max[h])`
3. If `attn_sink[h] > scores_max[h]`, we must also rescale `acc_o` and `sum_exp`:
   ```cpp
   float new_max = fmaxf(scores_max[h], attn_sink[h]);
   float scale = expf(scores_max[h] - new_max);
   acc_o[h] *= scale;
   sum_exp[h] = sum_exp[h] * scale + expf(attn_sink[h] - new_max);
   scores_max[h] = new_max;
   ```

### 2.5 Numerical Considerations

- **All intermediate softmax computations in FP32.** The HF kernel uses FP32 for `acc_o`, `scores_max`, `sum_exp`.
- **KV data in F16/BF16** — loaded from cache, kept as half in shared memory.
- **Q data in F16** — pre-scaled by `scale` during load (matching FA convention).
- **Score accumulation in FP32** — dot product of F16 Q and F16 KV, accumulated in FP32.
- **Output in FP32** — matches `ggml_flash_attn_ext` output convention.

---

## 3. HIP/ROCm Portability

### 3.1 Compatibility Strategy

The existing FA codebase already handles CUDA/HIP portability extensively. We reuse the same patterns:

```cpp
#include "common.cuh"    // Provides CUDA/HIP compatibility macros
```

Key portability considerations:

| Feature | CUDA | HIP (ROCm) | Our Approach |
|---------|------|------------|--------------|
| Warp size | 32 | 64 (CDNA) / 32 (RDNA) | Use `ggml_cuda_get_physical_warp_size()` |
| Shared memory | 96-228 KB | 64 KB (CDNA LDS) | Template `BLOCK_KV` to fit |
| `__shfl_xor_sync` | Available | Available via HIP | Use directly |
| `__syncthreads` | Available | Available | Use directly |
| `__half2float` | Available | Available | Use directly |
| cp_async | Ampere+ | Not available | Not needed (gather pattern) |
| MMA/WMMA | Varies | MFMA/WMMA | Phase 2 only |

### 3.2 Warp-Level Primitives

For the Phase 1 dot-product kernel, we only need:
- `__shfl_xor_sync` for warp-level reductions (both CUDA and HIP)
- `__shfl_sync` for broadcasting (both CUDA and HIP)
- `atomicMax` / warp-level max reduction (via shuffle)

These are all portable across CUDA and HIP via the existing `common.cuh` macros.

### 3.3 Shared Memory Sizing

For ROCm CDNA (MI100/MI200/MI300) with 64 KB LDS:
- `q_shared`: 16 KB (unchanged)
- With `BLOCK_KV=32`: `kv_shared` = 32 KB, total ~50 KB — fits
- With `BLOCK_KV=64`: `kv_shared` = 64 KB, total ~82 KB — does NOT fit

**Solution:** Template `BLOCK_KV` and select at launch time:
```cpp
if (amd_mfma_available(cc)) {
    launch<HEAD_DIM, 32, N_HEADS>(...);  // 32 KV per block for CDNA
} else {
    launch<HEAD_DIM, 64, N_HEADS>(...);  // 64 KV per block for NVIDIA
}
```

---

## 4. Integration Points

### 4.1 New ggml Op Registration

**File: `ggml/include/ggml.h`**

Add after the existing DSv4 ops (after line ~2593):

```c
GGML_OP_DSV4_SPARSE_ATTN,
```

Add function declaration (after line ~2593):

```c
// Sparse gather-attention for DeepSeek-V4 compressed cache.
// Q:         [head_dim, n_heads, n_tokens]   (F32)
// kv_cache:  [head_dim, n_kv]                (F16)
// topk_idxs: [topk, n_tokens]               (I32)
// attn_sink: [n_heads]                       (F32)
// Returns:   [head_dim, n_heads, n_tokens]   (F32)
GGML_API struct ggml_tensor * ggml_dsv4_sparse_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * kv_cache,
        struct ggml_tensor  * topk_idxs,
        struct ggml_tensor  * attn_sink,
        float                 scale);
```

**File: `ggml/src/ggml.c`**

Add op name string, implement `ggml_dsv4_sparse_attn()`:

```c
struct ggml_tensor * ggml_dsv4_sparse_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * kv_cache,
        struct ggml_tensor  * topk_idxs,
        struct ggml_tensor  * attn_sink,
        float                 scale) {

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(topk_idxs->type == GGML_TYPE_I32);

    // Output shape matches Q: [head_dim, n_heads, n_tokens]
    struct ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        q->ne[0], q->ne[1], q->ne[2]);

    result->op = GGML_OP_DSV4_SPARSE_ATTN;
    result->src[0] = q;
    result->src[1] = kv_cache;
    result->src[2] = topk_idxs;
    result->src[3] = attn_sink;

    // Store scale in op_params
    memcpy(result->op_params, &scale, sizeof(float));

    return result;
}
```

### 4.2 CUDA Backend Dispatch

**File: `ggml/src/ggml-cuda/ggml-cuda.cu`**

Add to the op dispatch switch (after `GGML_OP_DSV4_ROPE_TAIL`):

```cpp
case GGML_OP_DSV4_SPARSE_ATTN:
    ggml_cuda_op_dsv4_sparse_attn(ctx, dst);
    break;
```

Add to the `ggml_cuda_compute_forward_supports` switch:

```cpp
case GGML_OP_DSV4_SPARSE_ATTN:
```

### 4.3 New Kernel Files

**File: `ggml/src/ggml-cuda/dsv4-sparse-attn.cuh`**

```cpp
#pragma once
#include "common.cuh"

void ggml_cuda_op_dsv4_sparse_attn(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
```

**File: `ggml/src/ggml-cuda/dsv4-sparse-attn.cu`**

Contains the kernel implementation and launch logic. Following the pattern of `dsv4.cu`.

### 4.4 Graph Builder Changes in `src/models/deepseek4.cpp`

The key change is in the compressed attention path. Currently:

```cpp
// Current: build dense mask from topk indices, then use generic FA
ggml_tensor * idx_topk = ggml_argsort_top_k(ctx0, idx_scores, n_idx_topk);
comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, idx_scores, idx_topk);
full_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
cur = build_attn_mha(Qcur, Kall, Vall, nullptr, full_mask, sinks, nullptr, kq_scale, il);
```

**Proposed (sparse path):**

```cpp
// New: pass topk indices directly to sparse kernel
ggml_tensor * idx_topk = ggml_argsort_top_k(ctx0, idx_scores, n_idx_topk);

// Run window attention separately (existing FA on raw K/V only)
ggml_tensor * window_out = build_attn_mha(Qcur, k_raw, k_raw, nullptr,
    raw_mask, sinks, nullptr, kq_scale, il);

// Run sparse compressed attention via new kernel
ggml_tensor * comp_kv = kv_comp_cache;  // [head_dim, n_comp_visible]
ggml_tensor * comp_out = ggml_dsv4_sparse_attn(ctx0,
    Qcur_permuted,   // [head_dim, n_heads, n_tokens]
    comp_kv,          // [head_dim, n_comp_visible]
    idx_topk,         // [n_idx_topk, n_tokens]
    sinks,            // [n_heads]
    kq_scale);

// Merge window and compressed outputs
// (requires meta from both — see merge kernel below)
cur = dsv4_merge_attn_outputs(ctx0, window_out, window_meta, comp_out, comp_meta);
```

### 4.5 How topk_idxs Flows Through the Graph

The index tensor originates from:
1. **Indexer Q/K matmul:** `idx_scores = ggml_mul_mat(k_perm, q_perm)` → `[n_comp, n_tokens, n_heads]`
2. **Score aggregation:** relu → weight multiply → sum over heads → `[n_comp, n_tokens]`
3. **Top-k selection:** `idx_topk = ggml_argsort_top_k(idx_scores, n_idx_topk)` → `[n_idx_topk, n_tokens]` (I32)

The `idx_topk` tensor contains column indices into the compressed KV cache. Each value in `[0, n_comp_visible)` identifies which compressed KV row to gather. This tensor is passed directly to `ggml_dsv4_sparse_attn` as `src[2]`.

### 4.6 Merge Kernel for Window + Compressed Attention

We need a small merge kernel (or reuse the existing `flash_attn_combine_results` pattern):

```cpp
// GGML_OP_DSV4_MERGE_ATTN or implemented as part of the graph
// For each (token, head, dim):
//   combined_max = max(window_max, comp_max)
//   w_scale = exp(window_max - combined_max)
//   c_scale = exp(comp_max - combined_max)
//   output = (w_scale * window_out * window_rowsum + c_scale * comp_out * comp_rowsum)
//          / (w_scale * window_rowsum + c_scale * comp_rowsum)
```

This is structurally identical to the `flash_attn_combine_results` kernel in `fattn-common.cuh` (line ~510) and the stream-k fixup logic. We can either:
- Add a dedicated `GGML_OP_DSV4_MERGE_ATTN` op
- Or expose the sparse kernel output with meta and let the caller merge in the graph

**Recommendation:** Add `GGML_OP_DSV4_MERGE_ATTN` for clarity. It's a simple element-wise operation.

---

## 5. Performance Analysis

### 5.1 Memory Savings

**Current dense-mask approach** for a single layer, single token decode:
- Window KV: `n_window` positions (typically 4096)
- Compressed KV: `n_comp` positions (context/4, e.g., 32K for 128K context)
- Dense mask: `(n_window + n_comp) × n_tokens × sizeof(f16)` = `(4096 + 32768) × 1 × 2` = **~72 KB per token**
- FA processes all `36864` positions, even though only `topk=512` compressed are selected

**New sparse approach:**
- Index tensor: `topk × n_tokens × sizeof(i32)` = `512 × 1 × 4` = **2 KB per token**
- No dense mask needed for compressed portion
- Window mask still needed (but smaller): `n_window × n_tokens × sizeof(f16)` = **8 KB per token**
- **Savings: ~62 KB per token** (~86% reduction in mask memory)

### 5.2 Compute Savings

**Current:** FA computes Q·K^T for all `n_window + n_comp = 36,864` positions
- FLOPs per token per head: `2 × head_dim × 36864` = `2 × 512 × 36864` ≈ **37.7 MFLOP**
- Most of the compressed positions are immediately masked out

**New sparse:**
- Window FA: `2 × 512 × 4096` ≈ **4.2 MFLOP** per token per head
- Sparse compressed: `2 × 512 × 512` ≈ **0.52 MFLOP** per token per head
- **Total: ~4.7 MFLOP** — **~8× compute reduction**

### 5.3 Expected Speedup

The actual speedup depends on whether the current FA is compute-bound or memory-bound:

**For single-token decode (n_tokens=1):**
- Current FA is memory-bandwidth bound (loading 36K+ KV rows × 512 × 2 bytes)
- Memory load: `36864 × 512 × 2 = 37.7 MB` per head per token
- New sparse: `(4096 + 512) × 512 × 2 = 4.7 MB` per head per token
- **Expected speedup: ~4-8×** for the attention computation itself

**For prompt processing (n_tokens > 1):**
- More compute-bound; FA is already reasonably efficient
- But the dense mask overhead is proportionally larger
- **Expected speedup: ~2-4×** for the attention portion

**Caveat:** The gather access pattern for the sparse kernel is less cache-friendly than sequential KV access. This may partially offset the compute savings. The HF kernel mitigates this by using software pipelining (2 stages) to overlap gather latency with compute.

### 5.4 Typical DSv4 Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `head_dim` (KV) | 512 | MLA compressed dimension |
| `head_dim` (Q) | 576 | 512 nope + 64 rope (but only 512 for compressed cache) |
| `n_heads` | 16 | Number of attention heads |
| `n_attn_index_topk` | 512 | Top-k positions to attend |
| `compress_ratio` | 4 | 1 compressed position per 4 tokens |
| `n_window` (SWA) | 4096 | Sliding window size |
| `n_kv_total` | window + context/4 | Combined KV length |

---

## 6. Testing Strategy

### 6.1 Unit Test: Sparse vs Dense Equivalence

**File: `tests/test-dsv4-sparse-attn.cu`** (or `.cpp` with CUDA backend)

```
Test: verify that sparse gather-attention produces identical output to
      dense-masked FA for the same set of selected positions.

Setup:
  1. Create random Q [512, 16, 4], KV [512, 1024], topk=64
  2. Build dense mask: -inf for all positions except topk_idxs; 0 for selected
  3. Run ggml_flash_attn_ext with dense mask → output_dense
  4. Run ggml_dsv4_sparse_attn with topk_idxs → output_sparse
  5. Assert max_abs_diff(output_dense, output_sparse) < 1e-3 (F16 tolerance)
```

### 6.2 Attention Sink Test

```
Test: verify sink handling matches the HF kernel behavior.

Setup:
  1. Create Q, KV, topk_idxs with known values
  2. Set attn_sink to specific values (e.g., [0.5, 1.0, -0.5, ...])
  3. Compute expected output manually:
     - softmax_denom = sum(exp(scores - max)) + exp(sink - max)
     - output = sum(softmax_weights * KV) / softmax_denom
  4. Run kernel and compare
```

### 6.3 Edge Cases

- `topk = 0` — no compressed positions (output should be pure sink normalization)
- `topk > n_kv` — some indices are OOB (should be handled gracefully)
- `topk = n_kv` — all positions selected (equivalent to dense, no mask)
- `n_tokens = 1` — single token decode (primary use case)
- `n_tokens > 1` — batched/chunked decode
- Invalid indices (`-1`) in topk_idxs — should be ignored

### 6.4 Integration Test with Actual DSv4 Model

```
Test: end-to-end inference with DSv4-Flash model, comparing:
  1. Dense mask path (current) — baseline output
  2. Sparse kernel path (new) — should produce near-identical output
  3. Compare perplexity on a reference text corpus
  4. Measure latency improvement
```

### 6.5 Numerical Precision Validation

```
Test: run with various intermediate precisions and compare:
  1. All-FP32 reference implementation (CPU)
  2. FP16 KV + FP32 accumulation (target)
  3. BF16 KV + FP32 accumulation (if BF16 cache is used)

Metrics:
  - Max absolute error per element
  - Mean absolute error
  - Relative error distribution
```

### 6.6 HIP/ROCm Validation

- Run the same unit tests on AMD GPU (MI300X)
- Verify BLOCK_KV=32 path works with CDNA shared memory limits
- Compare output with NVIDIA reference

---

## 7. Migration Path

### Phase 1: Dual Path (Keep Both)

**Goal:** Add sparse kernel alongside existing dense mask path. Select at runtime.

**Changes:**
1. Add `GGML_OP_DSV4_SPARSE_ATTN` op, kernel, and dispatch
2. Add `GGML_OP_DSV4_MERGE_ATTN` op for combining window + compressed attention
3. In `deepseek4.cpp`, add a config flag: `use_sparse_attn` (default: `false`)
4. When enabled, use `ggml_dsv4_sparse_attn` instead of `dsv4_build_compressed_mask_from_topk` + dense FA
5. Window attention continues to use existing FA path
6. Add merge kernel to combine window + compressed outputs

**Testing:** Both paths produce identical output for all test cases.

**Note:** In Phase 1, we do NOT remove the `-inf` max clamping hack. Both paths coexist.

### Phase 2: Default to Sparse

**Goal:** Make sparse kernel the default for DSv4 layers that have indexer + compressed cache.

**Changes:**
1. Set `use_sparse_attn = true` by default when:
   - Model is DSv4 with `compress_ratio > 0`
   - `n_attn_index_topk > 0`
   - CUDA/HIP backend is available
2. Extensive benchmarking across GPU generations (Turing, Ampere, Ada, Hopper, CDNA)
3. Tune kernel parameters (BLOCK_KV, thread count) per architecture
4. Consider MMA-based GEMM for Phase 2 kernel optimization
5. Remove `use_sparse_attn` flag; sparse is always used for eligible layers

### Phase 3: Remove Dense Mask Path for DSv4

**Goal:** Clean up code by removing the dense mask path for DSv4 compressed attention.

**Changes:**
1. Remove `dsv4_build_compressed_mask_from_topk()` function
2. Remove the dense mask concatenation logic in the compressed attention path
3. Remove the `-inf` max clamping hack (no longer needed)
4. Remove unused mask construction helpers
5. Update documentation

**Prerequisite:** Phase 2 has been running in production for multiple releases with no regressions.

---

## 8. File Change Summary

| File | Change | Phase |
|------|--------|-------|
| `ggml/include/ggml.h` | Add `GGML_OP_DSV4_SPARSE_ATTN`, `GGML_OP_DSV4_MERGE_ATTN`, function decls | 1 |
| `ggml/src/ggml.c` | Op name strings, `ggml_dsv4_sparse_attn()`, `ggml_dsv4_merge_attn()` impl | 1 |
| `ggml/src/ggml-cuda/dsv4-sparse-attn.cuh` | New header file | 1 |
| `ggml/src/ggml-cuda/dsv4-sparse-attn.cu` | New kernel file (~400-600 lines) | 1 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Op dispatch + supports | 1 |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Add new .cu file | 1 |
| `src/models/deepseek4.cpp` | Sparse attention path + merge logic | 1 |
| `tests/test-dsv4-sparse-attn.cpp` | Unit tests | 1 |
| `src/models/deepseek4.cpp` | Default to sparse, remove dense path | 2-3 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Remove -inf clamping hack | 3 |

---

## 9. Open Questions / Risks

1. **Merge overhead:** Splitting window + compressed into two separate FA calls requires an extra merge kernel. For single-token decode this is trivial, but for prompt processing with many tokens the overhead may be non-negligible. Profile carefully.

2. **Gather memory bandwidth:** Random-access KV gather is less cache-friendly than sequential access. For large KV caches that don't fit in L2, this could be a bottleneck. The HF kernel uses software pipelining to mitigate this.

3. **head_dim=576 vs 512:** DSv4 uses `head_dim=576` for Q (512 nope + 64 rope) but only 512 for the compressed KV cache (MLA projection drops the rope dims). The sparse kernel operates on the 512-dim compressed cache. The rope dims only participate in window attention (raw K/V). This is already handled correctly by the MLA view in the existing code (`V_is_K_view` for 576→512).

4. **FP8 KV cache:** The compressed KV cache in DSv4 uses FP8 quantization on the nope dims. The sparse gather kernel must handle dequantization during the gather. For Phase 1, we assume the cache is already dequantized to F16 (the existing path does this). Phase 2 could add an FP8 gather path.

5. **Graph scheduling:** The two separate attention calls (window FA + sparse compressed) can potentially run concurrently on different CUDA streams. This is a Phase 2+ optimization.

6. **Alternative: Single unified sparse kernel** — Instead of splitting window + compressed, we could create a single kernel that handles both via a combined index tensor (window indices + compressed top-k indices). This avoids the merge overhead but is more complex. Consider for Phase 2 if merge overhead is significant.
