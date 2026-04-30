#include "dsv4-sparse-attn.cuh"
#include "convert.cuh"
#include <cmath>
#include <cstdint>

// =============================================================================
// DSV4 Unified Sparse Attention (gather-style) CUDA kernel.
//
// One thread block per (token, head, batch). The block iterates over:
//   1. All kv_window positions (contiguous, like SWA)
//   2. The topk_idxs gathered positions in kv_comp
//   3. The per-head learned attention sink (denominator only)
//
// Online softmax (FlashAttention-style) is used to compute the result in
// a single pass with O(head_dim_kv) memory.
//
// Threading model:
//   - BLOCK_THREADS threads per block (default 128)
//   - Each thread owns HEAD_DIM_KV / BLOCK_THREADS contiguous elements of
//     the head_dim_kv axis.
//   - Each KV position is dotted with Q via a warp-level reduction;
//     the resulting score is broadcast through shared memory and used to
//     update the per-thread acc_o slice.
//
// For DSv4: head_dim_kv = 512, n_heads = 16. So we launch
// (n_tokens * n_heads * batch) blocks of 128 threads each.
// =============================================================================

#ifndef DSV4_SPARSE_ATTN_BLOCK_THREADS
#define DSV4_SPARSE_ATTN_BLOCK_THREADS 128
#endif

template <typename KVT, int HEAD_DIM_KV, int BLOCK_THREADS>
static __global__ void dsv4_sparse_attn_kernel(
        const float    * __restrict__ q,            // [head_dim_q, n_heads, n_tokens, batch]
        const KVT      * __restrict__ kv_comp,      // [head_dim_kv, 1, n_kv_comp, batch]
        const KVT      * __restrict__ kv_window,    // [head_dim_kv, 1, n_window, batch] (or NULL)
        const float    * __restrict__ window_mask,  // [n_window, n_tokens, 1, 1] (or NULL)
        const int32_t  * __restrict__ topk_idxs,    // [topk, n_tokens, batch]
        const float    * __restrict__ attn_sink,    // [n_heads] (or NULL)
        float          * __restrict__ out,          // [head_dim_kv, n_heads, n_tokens, batch]
        const float                    scale,
        const int                      head_dim_q,
        const int                      n_heads,
        const int                      n_tokens,
        const int                      n_kv_comp,
        const int                      n_window,
        const int                      topk,
        // strides in elements (not bytes)
        const int64_t                  q_stride_h,
        const int64_t                  q_stride_t,
        const int64_t                  q_stride_b,
        const int64_t                  kc_stride_p,
        const int64_t                  kc_stride_b,
        const int64_t                  kw_stride_p,
        const int64_t                  kw_stride_b,
        const int64_t                  wmask_stride_t,
        const int64_t                  idx_stride_t,
        const int64_t                  idx_stride_b,
        const int64_t                  o_stride_h,
        const int64_t                  o_stride_t,
        const int64_t                  o_stride_b) {
    static_assert(HEAD_DIM_KV % BLOCK_THREADS == 0,
                  "HEAD_DIM_KV must be a multiple of BLOCK_THREADS");
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_KV / BLOCK_THREADS;
    constexpr int WARP_SIZE_LOCAL  = 32;

    const int t_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int tid   = threadIdx.x;

    if (t_idx >= n_tokens || h_idx >= n_heads) return;

    // Q pointer for this (b, t, h).
    const float * q_ptr =
        q + b_idx * q_stride_b + t_idx * q_stride_t + h_idx * q_stride_h;

    // KV cache pointers (KV is shared across heads).
    const KVT * kv_c_base = kv_comp   + b_idx * kc_stride_b;
    const KVT * kv_w_base = kv_window ? (kv_window + b_idx * kw_stride_b) : nullptr;

    // topk indices for this (b, t).
    const int32_t * idx_ptr =
        topk_idxs + b_idx * idx_stride_b + t_idx * idx_stride_t;

    // Output pointer for this (b, t, h).
    float * o_ptr =
        out + b_idx * o_stride_b + t_idx * o_stride_t + h_idx * o_stride_h;

    // Shared memory: query (head_dim_kv floats) + per-block score broadcast.
    __shared__ float q_shared[HEAD_DIM_KV];
    __shared__ float score_shared;

    // Load Q[..head_dim_kv] into shared memory. (We ignore the trailing rope dims
    // of Q -- the caller is responsible for rope-tail handling.)
    #pragma unroll
    for (int i = tid; i < HEAD_DIM_KV; i += BLOCK_THREADS) {
        q_shared[i] = q_ptr[i];
    }
    __syncthreads();

    // Per-thread accumulator slice.
    float acc_o[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) acc_o[i] = 0.0f;

    float scores_max = -INFINITY;
    float sum_exp    = 0.0f;

    // Lambda: process one KV row, with an optional additive mask value.
    // Reads kv_row pointer, computes Q dot K, performs online softmax update.
    auto process_kv = [&](const KVT * kv_row, float mask_add) {
        // Early-out for fully-masked positions to avoid the dot product entirely.
        const bool masked = (mask_add == -INFINITY);
        // Compute partial dot product: each thread accumulates ELEMS_PER_THREAD products.
        float partial = 0.0f;
        if (!masked) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                const int d = tid * ELEMS_PER_THREAD + i;
                partial += q_shared[d] * ggml_cuda_cast<float>(kv_row[d]);
            }
        }

        // Block reduction to compute the full dot product.
        // First: warp-level reduction.
        #pragma unroll
        for (int offset = WARP_SIZE_LOCAL / 2; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset, WARP_SIZE_LOCAL);
        }
        // Cross-warp reduction via shared memory.
        __shared__ float warp_sums[BLOCK_THREADS / WARP_SIZE_LOCAL];
        const int warp_id = tid / WARP_SIZE_LOCAL;
        const int lane_id = tid % WARP_SIZE_LOCAL;
        if (lane_id == 0) warp_sums[warp_id] = partial;
        __syncthreads();
        // First warp reduces the warp_sums.
        if (warp_id == 0) {
            float v = (lane_id < BLOCK_THREADS / WARP_SIZE_LOCAL) ? warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = WARP_SIZE_LOCAL / 2; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, offset, WARP_SIZE_LOCAL);
            }
            if (lane_id == 0) {
                score_shared = masked ? -INFINITY : (v * scale + mask_add);
            }
        }
        __syncthreads();

        const float s = score_shared;
        if (s == -INFINITY) {
            return; // fully masked: no contribution
        }
        const float new_max = fmaxf(scores_max, s);
        const float scale_prev = isinf(scores_max) ? 0.0f : expf(scores_max - new_max);
        const float w = expf(s - new_max);

        // Update accumulators.
        sum_exp = sum_exp * scale_prev + w;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int d = tid * ELEMS_PER_THREAD + i;
            acc_o[i] = acc_o[i] * scale_prev + w * ggml_cuda_cast<float>(kv_row[d]);
        }
        scores_max = new_max;
    };

    // Phase 1: window positions (with optional causal/SWA mask).
    const float * wmask_row = window_mask ? (window_mask + (int64_t) t_idx * wmask_stride_t) : nullptr;
    for (int k = 0; k < n_window; ++k) {
        const KVT * kv_row = kv_w_base + (int64_t) k * kw_stride_p;
        const float m = wmask_row ? wmask_row[k] : 0.0f;
        process_kv(kv_row, m);
    }

    // Phase 2: gathered topk positions in kv_comp.
    for (int i = 0; i < topk; ++i) {
        const int32_t kv_pos = idx_ptr[i];
        if (kv_pos < 0 || kv_pos >= n_kv_comp) continue;
        const KVT * kv_row = kv_c_base + (int64_t) kv_pos * kc_stride_p;
        process_kv(kv_row, 0.0f);
    }

    // Phase 3: attention sink (denominator only).
    if (attn_sink != nullptr) {
        const float sink_val = attn_sink[h_idx];
        const float new_max = fmaxf(scores_max, sink_val);
        const float scale_prev = isinf(scores_max) ? 0.0f : expf(scores_max - new_max);
        sum_exp = sum_exp * scale_prev + expf(sink_val - new_max);
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            acc_o[i] = acc_o[i] * scale_prev;
        }
        scores_max = new_max;
    }

    // Phase 4: normalize and write output.
    const float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int d = tid * ELEMS_PER_THREAD + i;
        o_ptr[d] = acc_o[i] * inv;
    }
}

// -----------------------------------------------------------------------------
// Launch dispatcher
// -----------------------------------------------------------------------------

template <typename KVT>
static void launch_dsv4_sparse_attn(
        const float    * q,
        const KVT      * kv_comp,
        const KVT      * kv_window,
        const float    * window_mask,
        const int32_t  * topk_idxs,
        const float    * attn_sink,
        float          * out,
        const float      scale,
        const int        head_dim_q,
        const int        head_dim_kv,
        const int        n_heads,
        const int        n_tokens,
        const int        n_kv_comp,
        const int        n_window,
        const int        topk,
        const int        batch,
        const int64_t    q_stride_h,
        const int64_t    q_stride_t,
        const int64_t    q_stride_b,
        const int64_t    kc_stride_p,
        const int64_t    kc_stride_b,
        const int64_t    kw_stride_p,
        const int64_t    kw_stride_b,
        const int64_t    wmask_stride_t,
        const int64_t    idx_stride_t,
        const int64_t    idx_stride_b,
        const int64_t    o_stride_h,
        const int64_t    o_stride_t,
        const int64_t    o_stride_b,
        cudaStream_t     stream) {
    const dim3 grid(n_tokens, n_heads, batch);
    constexpr int BLOCK_THREADS = DSV4_SPARSE_ATTN_BLOCK_THREADS;

    // Only head_dim_kv == 512 is currently supported (DSv4 MLA compressed dim).
    // Other sizes are reserved for future model families.
    if (head_dim_kv != 512) {
        GGML_ABORT("dsv4_sparse_attn: unsupported head_dim_kv=%d (only 512 implemented)", head_dim_kv);
    }
    dsv4_sparse_attn_kernel<KVT, 512, BLOCK_THREADS><<<grid, BLOCK_THREADS, 0, stream>>>(
        q, kv_comp, kv_window, window_mask, topk_idxs, attn_sink, out, scale,
        head_dim_q, n_heads, n_tokens, n_kv_comp, n_window, topk,
        q_stride_h, q_stride_t, q_stride_b,
        kc_stride_p, kc_stride_b,
        kw_stride_p, kw_stride_b,
        wmask_stride_t,
        idx_stride_t, idx_stride_b,
        o_stride_h, o_stride_t, o_stride_b);
}

// -----------------------------------------------------------------------------
// ggml dispatcher
// -----------------------------------------------------------------------------

void ggml_cuda_op_dsv4_sparse_attn(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_kv_c  = dst->src[1];
    const ggml_tensor * src_kv_w  = dst->src[2]; // may be NULL
    const ggml_tensor * src_wmask = dst->src[3]; // may be NULL
    const ggml_tensor * src_idx   = dst->src[4];
    const ggml_tensor * src_sink  = dst->src[5]; // may be NULL

    GGML_ASSERT(src_q->type == GGML_TYPE_F32);
    GGML_ASSERT(src_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int head_dim_q  = (int) src_q->ne[0];
    const int n_heads     = (int) src_q->ne[1];
    const int n_tokens    = (int) src_q->ne[2];
    const int batch       = (int) src_q->ne[3];

    const int head_dim_kv = (int) src_kv_c->ne[0];
    const int n_kv_comp   = (int) src_kv_c->ne[2];
    const int n_window    = src_kv_w ? (int) src_kv_w->ne[2] : 0;

    const int topk        = (int) src_idx->ne[0];

    // Strides in elements.
    auto stride_elems = [](const ggml_tensor * t, int dim) -> int64_t {
        return (int64_t) t->nb[dim] / (int64_t) ggml_type_size(t->type);
    };

    const int64_t q_stride_h = stride_elems(src_q, 1);
    const int64_t q_stride_t = stride_elems(src_q, 2);
    const int64_t q_stride_b = stride_elems(src_q, 3);

    const int64_t kc_stride_p = stride_elems(src_kv_c, 2);
    const int64_t kc_stride_b = stride_elems(src_kv_c, 3);

    const int64_t kw_stride_p = src_kv_w ? stride_elems(src_kv_w, 2) : 0;
    const int64_t kw_stride_b = src_kv_w ? stride_elems(src_kv_w, 3) : 0;

    const int64_t wmask_stride_t = src_wmask ? stride_elems(src_wmask, 1) : 0;

    const int64_t idx_stride_t = stride_elems(src_idx, 1);
    const int64_t idx_stride_b = stride_elems(src_idx, 2);

    const int64_t o_stride_h = stride_elems(dst, 1);
    const int64_t o_stride_t = stride_elems(dst, 2);
    const int64_t o_stride_b = stride_elems(dst, 3);

    cudaStream_t stream = ctx.stream();

    const float * q_data        = (const float *)   src_q->data;
    const float * wmask_data    = src_wmask ? (const float *) src_wmask->data : nullptr;
    const int32_t * idx_data    = (const int32_t *) src_idx->data;
    const float * sink_data     = src_sink ? (const float *) src_sink->data : nullptr;
    float * o_data              = (float *)         dst->data;

    switch (src_kv_c->type) {
        case GGML_TYPE_F32: {
            const float * kc = (const float *) src_kv_c->data;
            const float * kw = src_kv_w ? (const float *) src_kv_w->data : nullptr;
            launch_dsv4_sparse_attn<float>(
                q_data, kc, kw, wmask_data, idx_data, sink_data, o_data, scale,
                head_dim_q, head_dim_kv, n_heads, n_tokens, n_kv_comp, n_window, topk, batch,
                q_stride_h, q_stride_t, q_stride_b,
                kc_stride_p, kc_stride_b,
                kw_stride_p, kw_stride_b,
                wmask_stride_t,
                idx_stride_t, idx_stride_b,
                o_stride_h, o_stride_t, o_stride_b,
                stream);
        } break;
        case GGML_TYPE_F16: {
            const half * kc = (const half *) src_kv_c->data;
            const half * kw = src_kv_w ? (const half *) src_kv_w->data : nullptr;
            launch_dsv4_sparse_attn<half>(
                q_data, kc, kw, wmask_data, idx_data, sink_data, o_data, scale,
                head_dim_q, head_dim_kv, n_heads, n_tokens, n_kv_comp, n_window, topk, batch,
                q_stride_h, q_stride_t, q_stride_b,
                kc_stride_p, kc_stride_b,
                kw_stride_p, kw_stride_b,
                wmask_stride_t,
                idx_stride_t, idx_stride_b,
                o_stride_h, o_stride_t, o_stride_b,
                stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * kc = (const nv_bfloat16 *) src_kv_c->data;
            const nv_bfloat16 * kw = src_kv_w ? (const nv_bfloat16 *) src_kv_w->data : nullptr;
            launch_dsv4_sparse_attn<nv_bfloat16>(
                q_data, kc, kw, wmask_data, idx_data, sink_data, o_data, scale,
                head_dim_q, head_dim_kv, n_heads, n_tokens, n_kv_comp, n_window, topk, batch,
                q_stride_h, q_stride_t, q_stride_b,
                kc_stride_p, kc_stride_b,
                kw_stride_p, kw_stride_b,
                wmask_stride_t,
                idx_stride_t, idx_stride_b,
                o_stride_h, o_stride_t, o_stride_b,
                stream);
        } break;
        default:
            GGML_ABORT("dsv4_sparse_attn (CUDA): unsupported KV type %d",
                       (int) src_kv_c->type);
    }

    CUDA_CHECK(cudaGetLastError());
}
