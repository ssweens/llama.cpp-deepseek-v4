#include "dsv4.cuh"
#include <cmath>

// ============================================================
// DSV4_HC_SPLIT_SINKHORN
// ============================================================
// One thread per row. Reads mixes, scale, base; writes dst (same shape as mixes).
// Computes pre/post sigmoids and softmax+sinkhorn coupling matrix.

static __global__ void kernel_dsv4_hc_split_sinkhorn(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        int64_t n_rows,
        int64_t mix_hc,
        int64_t nb01,    // byte stride between mixes rows
        int64_t nb1,     // byte stride between dst rows
        int     n_hc,
        int     sinkhorn_iters,
        float   epsv) {

    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rows) return;
    if (n_hc <= 0 || n_hc > 16) return;

    const float * mix = (const float *)((const char *)mixes + tid * nb01);
    float       * out = (float       *)((      char *)dst   + tid * nb1);

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    // pre: sigmoid of first n_hc elements
    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    // post: 2*sigmoid of next n_hc elements
    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    // comb: n_hc x n_hc coupling matrix starting at offset 2*n_hc
    float c[16 * 16];

    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }
        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            c[src_hc + dst_hc * n_hc] = c[src_hc + dst_hc * n_hc] * inv_sum + epsv;
        }
    }

    // column normalise
    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float col_sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            col_sum += c[src_hc + dst_hc * n_hc];
        }
        const float inv_denom = 1.0f / (col_sum + epsv);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        // row normalise
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                row_sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (row_sum + epsv);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
        // col normalise
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float col_sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                col_sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (col_sum + epsv);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * mixes = dst->src[0];
    const struct ggml_tensor * scale = dst->src[1];
    const struct ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    const int     n_hc           = ggml_get_op_params_i32(dst, 0);
    const int     sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float   epsv           = ggml_get_op_params_f32(dst, 2);
    const int64_t mix_hc         = mixes->ne[0];
    const int64_t n_rows         = ggml_nrows(mixes);

    const float * mixes_d = (const float *) mixes->data;
    const float * scale_d = (const float *) scale->data;
    const float * base_d  = (const float *) base->data;
    float       * dst_d   = (float       *) dst->data;

    const int nth = std::min(256, (int)n_rows);
    const int nblocks = ((int)n_rows + nth - 1) / nth;

    kernel_dsv4_hc_split_sinkhorn<<<nblocks, nth, 0, ctx.stream()>>>(
        mixes_d, scale_d, base_d, dst_d,
        n_rows, mix_hc,
        (int64_t)mixes->nb[1], (int64_t)dst->nb[1],
        n_hc, sinkhorn_iters, epsv);
}

// ============================================================
// DSV4_HC_WEIGHTED_SUM
// ============================================================
// Computes sum_{hc} weights[hc, token] * x[embd, hc, token] -> dst[embd, token]

static __global__ void kernel_dsv4_hc_weighted_sum(
        const char * __restrict__ x,
        const char * __restrict__ weights,
        char       * __restrict__ dst,
        int64_t n_embd,
        int64_t n_hc,
        int64_t n_tokens,
        int64_t nb_x0, int64_t nb_x1, int64_t nb_x2,
        int64_t nb_w0, int64_t nb_w1,
        int64_t nb0,   int64_t nb1) {

    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_tokens;
    if (gid >= n_elem) return;

    const int64_t d = gid % n_embd;
    const int64_t t = gid / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        const float xv = *(const float *)(x       + d*nb_x0 + h*nb_x1 + t*nb_x2);
        const float wv = *(const float *)(weights + h*nb_w0 + t*nb_w1);
        acc += xv * wv;
    }

    *(float *)(dst + d*nb0 + t*nb1) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * x       = dst->src[0];
    const struct ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    const int nth    = (int)std::min((int64_t)256, n_elem);
    const int nblocks = (int)((n_elem + nth - 1) / nth);

    kernel_dsv4_hc_weighted_sum<<<nblocks, nth, 0, ctx.stream()>>>(
        (const char *)x->data, (const char *)weights->data, (char *)dst->data,
        n_embd, n_hc, n_tokens,
        (int64_t)x->nb[0], (int64_t)x->nb[1], (int64_t)x->nb[2],
        (int64_t)weights->nb[0], (int64_t)weights->nb[1],
        (int64_t)dst->nb[0], (int64_t)dst->nb[1]);
}

// ============================================================
// DSV4_HC_EXPAND
// ============================================================
// dst[d, dst_hc, t] = block_out[d, t] * post[dst_hc, t]
//                   + sum_{src_hc} comb[dst_hc, src_hc, t] * residual[d, src_hc, t]

static __global__ void kernel_dsv4_hc_expand(
        const char * __restrict__ block_out,
        const char * __restrict__ residual,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char       * __restrict__ dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        int64_t nb_block0, int64_t nb_block1,
        int64_t nb_res0,   int64_t nb_res1, int64_t nb_res2,
        int64_t nb_post0,  int64_t nb_post1,
        int64_t nb_comb0,  int64_t nb_comb1, int64_t nb_comb2,
        int64_t nb0, int64_t nb1, int64_t nb2) {

    const int64_t gid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    if (gid >= n_elem) return;

    const int64_t d      = gid % n_embd;
    const int64_t tmp    = gid / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float block_v = *(const float *)(block_out + d*nb_block0 + t*nb_block1);
    const float post_v  = *(const float *)(post      + dst_hc*nb_post0 + t*nb_post1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *(const float *)(comb     + dst_hc*nb_comb0 + src_hc*nb_comb1 + t*nb_comb2);
        const float res_v  = *(const float *)(residual + d*nb_res0 + src_hc*nb_res1 + t*nb_res2);
        acc += comb_v * res_v;
    }

    *(float *)(dst + d*nb0 + dst_hc*nb1 + t*nb2) = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * block_out = dst->src[0];
    const struct ggml_tensor * residual  = dst->src[1];
    const struct ggml_tensor * post      = dst->src[2];
    const struct ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    const int nth    = (int)std::min((int64_t)256, n_elem);
    const int nblocks = (int)((n_elem + nth - 1) / nth);

    kernel_dsv4_hc_expand<<<nblocks, nth, 0, ctx.stream()>>>(
        (const char *)block_out->data, (const char *)residual->data,
        (const char *)post->data,      (const char *)comb->data,
        (char *)dst->data,
        n_embd, n_hc, n_tokens,
        (int64_t)block_out->nb[0], (int64_t)block_out->nb[1],
        (int64_t)residual->nb[0],  (int64_t)residual->nb[1], (int64_t)residual->nb[2],
        (int64_t)post->nb[0],      (int64_t)post->nb[1],
        (int64_t)comb->nb[0],      (int64_t)comb->nb[1],     (int64_t)comb->nb[2],
        (int64_t)dst->nb[0],       (int64_t)dst->nb[1],      (int64_t)dst->nb[2]);
}

// ============================================================
// DSV4_FP8_KV_QUANTIZE
// ============================================================
// One threadblock per row. 64 threads. Quantises non-rope dims to FP8 E4M3,
// copies rope dims unchanged.

static __device__ float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp2f((float)(exp - 7));
}

static __device__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e4m3fn_value(best);
}

static __global__ void kernel_dsv4_fp8_kv_quantize_f32(
        const char * __restrict__ src0,
        char       * __restrict__ dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t nb0,  int64_t nb1,  int64_t nb2,  int64_t nb3,
        int     n_rot) {

    // one block per row
    const int64_t row   = blockIdx.x;
    const int64_t n_rows = ne01 * ne02 * ne03;
    if (row >= n_rows) return;

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src0 + i1*nb01 + i2*nb02 + i3*nb03;
    char       * dst_base = dst  + i1*nb1  + i2*nb2  + i3*nb3;

    const int64_t n_nope = ne00 - n_rot;

    __shared__ float scratch[64];

    // quantise nope dims in chunks of 64
    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (threadIdx.x < 64) {
            v = *(const float *)(src_base + (off + threadIdx.x) * nb00);
            scratch[threadIdx.x] = fabsf(v);
        }
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            if (threadIdx.x < (unsigned)stride) {
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        const float amax  = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (threadIdx.x < 64) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *(float *)(dst_base + (off + threadIdx.x) * nb0) = q;
        }
        __syncthreads();
    }

    // copy rope dims unchanged
    for (int64_t i = n_nope + threadIdx.x; i < ne00; i += 64) {
        *(float *)(dst_base + i * nb0) = *(const float *)(src_base + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int n_rot  = ggml_get_op_params_i32(dst, 0);
    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    kernel_dsv4_fp8_kv_quantize_f32<<<(int)n_rows, 64, 0, ctx.stream()>>>(
        (const char *)src0->data, (char *)dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        (int64_t)src0->nb[0], (int64_t)src0->nb[1], (int64_t)src0->nb[2], (int64_t)src0->nb[3],
        (int64_t)dst->nb[0],  (int64_t)dst->nb[1],  (int64_t)dst->nb[2],  (int64_t)dst->nb[3],
        n_rot);
}

// ============================================================
// DSV4_ROPE_TAIL
// ============================================================
// Partial RoPE: copies nope dims, applies YaRN RoPE to last n_dims.
// Grid: (ne01, ne02, ne03), threads: min(256, ne00).

static __device__ void rope_yarn_ramp_device(const float low, const float high, const int i0,
                                              float * cos_theta, float * sin_theta,
                                              float theta_scaled, float ext_factor,
                                              float attn_factor) {
    const float ramp = fminf(1.0f, fmaxf(0.0f, (i0 / 2 - low) / fmaxf(0.001f, high - low)));
    const float mix  = (1.0f - ramp) * ext_factor;
    theta_scaled = theta_scaled * (1.0f - mix) + theta_scaled / (mix == 0.0f ? 1.0f : mix) * mix;
    *cos_theta = cosf(theta_scaled) * attn_factor;
    *sin_theta = sinf(theta_scaled) * attn_factor;
}

static __device__ void rope_yarn_device(float theta, float freq_scale, float lo, float hi,
                                        int i0, float ext_factor, float attn_factor,
                                        float * cos_theta, float * sin_theta) {
    const float theta_scaled = theta * freq_scale;
    if (ext_factor == 0.0f) {
        *cos_theta = cosf(theta_scaled) * attn_factor;
        *sin_theta = sinf(theta_scaled) * attn_factor;
    } else {
        rope_yarn_ramp_device(lo, hi, i0, cos_theta, sin_theta, theta_scaled, ext_factor, attn_factor);
    }
}

static __device__ void rope_yarn_corr_dims_device(int n_dims, int n_ctx_orig, float freq_base,
                                                   float beta_fast, float beta_slow,
                                                   float * lo, float * hi) {
    const float start = floorf((float)n_dims * logf(n_ctx_orig / (beta_fast * 2.0f * M_PI)) / (2.0f * logf(freq_base)));
    const float end   = ceilf ((float)n_dims * logf(n_ctx_orig / (beta_slow * 2.0f * M_PI)) / (2.0f * logf(freq_base)));
    *lo = fmaxf(0.0f, fminf((float)(n_dims - 1), start));
    *hi = fmaxf(0.0f, fminf((float)(n_dims - 1), end));
}

static __global__ void kernel_dsv4_rope_tail_f32(
        const char * __restrict__ src0,
        const int  * __restrict__ pos,
        const float* __restrict__ freq_factors,  // may be NULL
        char       * __restrict__ dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t nb0,  int64_t nb1,  int64_t nb2,  int64_t nb3,
        int     n_dims, int mode, int n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {

    // grid: (ne01, ne02, ne03), threads: up to ne00
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    const char * src_base = src0 + i3*nb03 + i2*nb02 + i1*nb01;
    char       * dst_base = dst  + i3*nb3  + i2*nb2  + i1*nb1;

    const int n_nope = (int)ne00 - n_dims;
    const float theta_base = (float)pos[i2];
    const float inv_ndims  = -1.0f / (float)n_dims;
    const bool is_neox     = (mode == 2);

    float lo, hi;
    rope_yarn_corr_dims_device(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, &lo, &hi);

    for (int i0 = threadIdx.x; i0 < (int)ne00; i0 += blockDim.x) {
        if (i0 < n_nope) {
            *(float *)(dst_base + i0*nb0) = *(const float *)(src_base + i0*nb00);
            continue;
        }

        const int r = i0 - n_nope;

        if (is_neox) {
            const int n_half = n_dims / 2;
            if (r >= n_half) continue;

            const int ic  = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base * powf(freq_base, inv_ndims * rel_i0);
            const float ff    = freq_factors ? freq_factors[ic] : 1.0f;

            float cos_t, sin_t;
            rope_yarn_device(theta / ff, freq_scale, lo, hi, rel_i0, ext_factor, attn_factor, &cos_t, &sin_t);
            if (inverse) sin_t = -sin_t;

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *(const float *)(src_base + j0*nb00);
            const float x1 = *(const float *)(src_base + j1*nb00);
            *(float *)(dst_base + j0*nb0) = x0*cos_t - x1*sin_t;
            *(float *)(dst_base + j1*nb0) = x0*sin_t + x1*cos_t;
        } else {
            if (r & 1) continue;

            const int ic = r / 2;
            const float theta = theta_base * powf(freq_base, inv_ndims * r);
            const float ff    = freq_factors ? freq_factors[ic] : 1.0f;

            float cos_t, sin_t;
            rope_yarn_device(theta / ff, freq_scale, lo, hi, r, ext_factor, attn_factor, &cos_t, &sin_t);
            if (inverse) sin_t = -sin_t;

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *(const float *)(src_base + j0*nb00);
            const float x1 = *(const float *)(src_base + j1*nb00);
            *(float *)(dst_base + j0*nb0) = x0*cos_t - x1*sin_t;
            *(float *)(dst_base + j1*nb0) = x0*sin_t + x1*cos_t;
        }
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0         = dst->src[0];
    const struct ggml_tensor * src1_pos     = dst->src[1];
    const struct ggml_tensor * src2_freq    = dst->src[2];  // may be NULL

    GGML_ASSERT(src0->type    == GGML_TYPE_F32);
    GGML_ASSERT(src1_pos->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    int32_t params[16];
    memcpy(params, dst->op_params, sizeof(params));
    const int     n_dims      = params[0];
    const int     mode        = params[1];
    const int     n_ctx_orig  = params[2];
    const int     inverse     = params[3];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   params + 4, sizeof(float));
    memcpy(&freq_scale,  params + 5, sizeof(float));
    memcpy(&ext_factor,  params + 6, sizeof(float));
    memcpy(&attn_factor, params + 7, sizeof(float));
    memcpy(&beta_fast,   params + 8, sizeof(float));
    memcpy(&beta_slow,   params + 9, sizeof(float));

    const dim3 grid(src0->ne[1], src0->ne[2], src0->ne[3]);
    const int  nth = (int)std::min((int64_t)256, src0->ne[0]);

    const float * freq_factors_d = src2_freq ? (const float *)src2_freq->data : nullptr;

    kernel_dsv4_rope_tail_f32<<<grid, nth, 0, ctx.stream()>>>(
        (const char *)src0->data,
        (const int  *)src1_pos->data,
        freq_factors_d,
        (char *)dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        (int64_t)src0->nb[0], (int64_t)src0->nb[1], (int64_t)src0->nb[2], (int64_t)src0->nb[3],
        (int64_t)dst->nb[0],  (int64_t)dst->nb[1],  (int64_t)dst->nb[2],  (int64_t)dst->nb[3],
        n_dims, mode, n_ctx_orig, inverse,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}
