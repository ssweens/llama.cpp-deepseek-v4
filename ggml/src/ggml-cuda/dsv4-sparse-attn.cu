#include "dsv4-sparse-attn.cuh"
#include "convert.cuh"
#include <cmath>
#include <cstdint>

// =============================================================================
// DSV4 Unified Sparse Attention (gather-style) CUDA kernel.
//
// One thread block per (token, head, batch). The block iterates over:
//   1. All kv_window positions (contiguous, like SWA), with an optional
//      causal/SWA mask added to the score.
//   2. The topk_idxs gathered positions in kv_comp
//   3. The per-head learned attention sink (denominator only)
//
// Online softmax (FlashAttention-style) is used to compute the result in
// a single pass with O(head_dim_kv) memory.
//
// Threading model (Phase 2a — warp-parallel KV chunks):
//   - BLOCK_THREADS = 128 threads per block (default), arranged as 4 warps.
//   - Each warp processes its OWN KV position concurrently. The block
//     therefore advances through KV positions in chunks of N_WARPS=4 per
//     iteration, performing a single block sync per chunk instead of one
//     per KV position.
//   - For the dot product within a warp: 32 threads cooperate, each
//     accumulating HEAD_DIM_KV / WARP_SIZE elements, then warp-reducing.
//   - For the acc_o accumulator update: each thread owns
//     HEAD_DIM_KV / BLOCK_THREADS contiguous output elements and combines
//     all CHUNK_KV new KV contributions in one pass per chunk.
//
// Memory layout (shared):
//   q_shared[HEAD_DIM_KV]                    — query loaded once
//   kv_chunk_shared[CHUNK_KV][HEAD_DIM_KV]   — gathered KV for current chunk
//   scores_shared[CHUNK_KV]                  — Q·K dot products for chunk
//   masks_shared[CHUNK_KV]                   — additive masks (-inf to skip)
//
// For DSv4: head_dim_kv = 512, n_heads = 16. So we launch
// (n_tokens * n_heads * batch) blocks of 128 threads each.
//
// Phase 2b (TODO): replace the warp-parallel scalar dot products with
// MMA tensor-core tiles. Process more KV positions per iteration
// (BLOCK_KV = 32 or 64) and run all heads through a single block grid.
// Expected additional ~2-4x speedup on Ampere/Ada/Blackwell.
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
    constexpr int WARP_SIZE_LOCAL  = 32;
    static_assert(BLOCK_THREADS % WARP_SIZE_LOCAL == 0,
                  "BLOCK_THREADS must be a multiple of warp size");
    constexpr int N_WARPS          = BLOCK_THREADS / WARP_SIZE_LOCAL;
    // CHUNK_KV controls per-chunk KV fan-in.
    //
    // CUDA path: keep phase-2a decode-oriented CHUNK_KV=N_WARPS.
    // HIP path: reduce fan-in to lower LDS/register pressure and improve
    // occupancy on gfx1151 iGPU workloads.
#ifdef GGML_USE_HIP
    constexpr int CHUNK_KV         = 2;
#else
    constexpr int CHUNK_KV         = N_WARPS;
#endif
    static_assert(CHUNK_KV >= 1 && CHUNK_KV <= N_WARPS,
                  "CHUNK_KV must be in [1, N_WARPS] so each slot has a dedicated warp");
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_KV / BLOCK_THREADS;

    const int t_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int tid   = threadIdx.x;

    if (t_idx >= n_tokens || h_idx >= n_heads) return;

    const int warp_id = tid / WARP_SIZE_LOCAL;
    const int lane_id = tid % WARP_SIZE_LOCAL;

    // Q pointer for this (b, t, h).
    const float * q_ptr =
        q + b_idx * q_stride_b + t_idx * q_stride_t + h_idx * q_stride_h;

    // KV cache pointers (KV is shared across heads).
    const KVT * kv_c_base = kv_comp   + b_idx * kc_stride_b;
    const KVT * kv_w_base = kv_window ? (kv_window + b_idx * kw_stride_b) : nullptr;

    // topk indices for this (b, t).
    const int32_t * idx_ptr =
        topk_idxs + b_idx * idx_stride_b + t_idx * idx_stride_t;

    // Window mask row for this token (shape [n_window]).
    const float * wmask_row = window_mask
        ? (window_mask + (int64_t) t_idx * wmask_stride_t)
        : nullptr;

    // Output pointer for this (b, t, h).
    float * o_ptr =
        out + b_idx * o_stride_b + t_idx * o_stride_t + h_idx * o_stride_h;

    // Shared memory.
    __shared__ float q_shared[HEAD_DIM_KV];
    __shared__ float scores_shared[CHUNK_KV];
    __shared__ float masks_shared[CHUNK_KV];
    __shared__ KVT   kv_chunk_shared[CHUNK_KV * HEAD_DIM_KV];

    // Load Q[..head_dim_kv] into shared memory (ignore trailing rope dims of Q).
    #pragma unroll
    for (int i = tid; i < HEAD_DIM_KV; i += BLOCK_THREADS) {
        q_shared[i] = q_ptr[i];
    }
    __syncthreads();

    // Per-thread output accumulator slice.
    float acc_o[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) acc_o[i] = 0.0f;

    float scores_max = -INFINITY;
    float sum_exp    = 0.0f;

    // -------------------------------------------------------------------------
    // process_chunk(get_kv_row, n_pos): process up to CHUNK_KV positions.
    //   get_kv_row(int slot, const KVT *& kv_row, float & mask_add) is a
    //   functor that, for slot in [0, n_pos), returns the KV row pointer and
    //   the additive mask. For slot >= n_pos the slot is treated as masked.
    //   Setting mask_add to -INFINITY skips the slot entirely.
    // -------------------------------------------------------------------------
    auto process_chunk = [&](auto get_kv_row, int n_pos) {
        // Step 1: gather KV chunk into shared memory and prepare masks.
        // Each active warp gathers its own slot. Extra warps idle this step.
        if (warp_id < CHUNK_KV) {
            const int slot = warp_id;
            float    m      = -INFINITY;
            const KVT * src = nullptr;
            if (slot < n_pos) {
                get_kv_row(slot, src, m);
            }
            const bool valid = (slot < n_pos) && (src != nullptr) && (m != -INFINITY);
            // Each active warp loads its own slot's KV row.
            for (int d = lane_id; d < HEAD_DIM_KV; d += WARP_SIZE_LOCAL) {
                kv_chunk_shared[slot * HEAD_DIM_KV + d] = valid
                    ? src[d]
                    : (KVT) 0;
            }
            if (lane_id == 0) {
                masks_shared[slot] = valid ? m : -INFINITY;
            }
        }
        __syncthreads();

        // Step 2: warp-parallel Q·K dot products. Each active warp handles one slot.
        // When CHUNK_KV < N_WARPS, the extra warps idle this step.
        if (warp_id < CHUNK_KV) {
            const int slot = warp_id;
            const KVT * kv_row = &kv_chunk_shared[slot * HEAD_DIM_KV];
            float partial = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < HEAD_DIM_KV; d += WARP_SIZE_LOCAL) {
                partial += q_shared[d] * ggml_cuda_cast<float>(kv_row[d]);
            }
            // Warp reduction.
            #pragma unroll
            for (int offset = WARP_SIZE_LOCAL / 2; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset, WARP_SIZE_LOCAL);
            }
            if (lane_id == 0) {
                const float m = masks_shared[slot];
                scores_shared[slot] = (m == -INFINITY) ? -INFINITY : (partial * scale + m);
            }
        }
        __syncthreads();

        // Step 3: chunk-level online softmax update.
        // First, compute the chunk max (small loop, all threads do it).
        float chunk_max = -INFINITY;
        #pragma unroll
        for (int s = 0; s < CHUNK_KV; ++s) {
            chunk_max = fmaxf(chunk_max, scores_shared[s]);
        }
        if (chunk_max == -INFINITY) {
            // Whole chunk is masked or out-of-range; nothing to do.
            return;
        }
        const float new_max    = fmaxf(scores_max, chunk_max);
        const float scale_prev = isinf(scores_max) ? 0.0f : expf(scores_max - new_max);

        // Per-slot softmax weights w[s] = exp(score[s] - new_max), with
        // -inf scores producing zero (skipped).
        float w[CHUNK_KV];
        float chunk_sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < CHUNK_KV; ++s) {
            const float sc = scores_shared[s];
            w[s] = (sc == -INFINITY) ? 0.0f : expf(sc - new_max);
            chunk_sum += w[s];
        }
        sum_exp = sum_exp * scale_prev + chunk_sum;

        // Step 4: update acc_o = acc_o * scale_prev + sum_s(w[s] * kv_chunk_shared[s][d]).
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int d = tid * ELEMS_PER_THREAD + i;
            float v = acc_o[i] * scale_prev;
            #pragma unroll
            for (int s = 0; s < CHUNK_KV; ++s) {
                if (w[s] != 0.0f) {
                    v += w[s] * ggml_cuda_cast<float>(kv_chunk_shared[s * HEAD_DIM_KV + d]);
                }
            }
            acc_o[i] = v;
        }
        scores_max = new_max;
        __syncthreads();
    };

    // ---- Phase 1: window positions (with optional causal/SWA mask). ----
    for (int kv_base = 0; kv_base < n_window; kv_base += CHUNK_KV) {
        const int n_pos = min(CHUNK_KV, n_window - kv_base);
        process_chunk([&](int slot, const KVT *& kv_row, float & mask_add) {
            const int k = kv_base + slot;
            kv_row   = kv_w_base + (int64_t) k * kw_stride_p;
            mask_add = wmask_row ? wmask_row[k] : 0.0f;
        }, n_pos);
    }

    // ---- Phase 2: gathered topk positions in kv_comp. ----
    for (int i_base = 0; i_base < topk; i_base += CHUNK_KV) {
        const int n_pos = min(CHUNK_KV, topk - i_base);
        process_chunk([&](int slot, const KVT *& kv_row, float & mask_add) {
            const int i = i_base + slot;
            const int32_t kv_pos = idx_ptr[i];
            if (kv_pos < 0 || kv_pos >= n_kv_comp) {
                kv_row   = nullptr;
                mask_add = -INFINITY; // mark this slot as masked / skipped
            } else {
                kv_row   = kv_c_base + (int64_t) kv_pos * kc_stride_p;
                mask_add = 0.0f;
            }
        }, n_pos);
    }

    // ---- Phase 3: attention sink (denominator only). ----
    if (attn_sink != nullptr) {
        const float sink_val = attn_sink[h_idx];
        const float new_max  = fmaxf(scores_max, sink_val);
        const float scale_prev = isinf(scores_max) ? 0.0f : expf(scores_max - new_max);
        sum_exp = sum_exp * scale_prev + expf(sink_val - new_max);
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            acc_o[i] = acc_o[i] * scale_prev;
        }
        scores_max = new_max;
    }

    // ---- Phase 4: normalize and write output. ----
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
