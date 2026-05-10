#include "models.h"
#include "../llama-impl.h"
#include "../llama-kv-cache.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
static bool dsv4_debug_paths_enabled() {
    const char * value = std::getenv("LLAMA_DEBUG_DSV4_PATHS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

class dsv4_position_graph_guard : public llm_graph_input_i {
public:
    explicit dsv4_position_graph_guard(const llama_ubatch & ubatch) :
        n_tokens(ubatch.n_tokens),
        n_pos(ubatch.n_pos) {
        if (ubatch.pos != nullptr) {
            pos.assign(ubatch.pos, ubatch.pos + (size_t) ubatch.n_tokens * ubatch.n_pos);
        }
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        const llama_ubatch & ubatch = params.ubatch;
        if (ubatch.n_tokens != n_tokens || ubatch.n_pos != n_pos) {
            return false;
        }
        if ((ubatch.pos == nullptr) != pos.empty()) {
            return false;
        }
        if (ubatch.pos == nullptr) {
            return true;
        }
        return std::equal(pos.begin(), pos.end(), ubatch.pos);
    }

private:
    uint32_t n_tokens = 0;
    uint32_t n_pos = 0;
    std::vector<llama_pos> pos;
};

static inline float dsv4_e4m3_to_float(uint8_t x) {
    const uint8_t sign = (x >> 7) & 0x1u;
    const uint8_t exp  = (x >> 3) & 0xFu;
    const uint8_t mant = x & 0x7u;

    if (exp == 0xFu && mant == 0x7u) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float val;
    if (exp == 0) {
        val = mant * std::exp2(-9.0f);
    } else {
        val = (1.0f + mant * (1.0f / 8.0f)) * std::exp2(static_cast<float>(exp) - 7.0f);
    }

    return sign ? -val : val;
}

static inline uint8_t dsv4_float_to_e4m3(float f) {
    if (std::isnan(f)) {
        return 0x7F;
    }

    const uint8_t sign = std::signbit(f) ? 0x80u : 0x00u;
    const float a = std::fabs(f);

    static const std::array<float, 0x7F> values = [] {
        std::array<float, 0x7F> arr{};
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = dsv4_e4m3_to_float(static_cast<uint8_t>(i));
        }
        return arr;
    }();

    if (a >= values.back()) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }

    auto it = std::lower_bound(values.begin(), values.end(), a);
    if (it == values.begin()) {
        return sign;
    }

    const int idx_hi = int(it - values.begin());
    const int idx_lo = idx_hi - 1;
    const float lo = values[idx_lo];
    const float hi = values[idx_hi];
    const float dlo = a - lo;
    const float dhi = hi - a;
    const int idx = dlo < dhi ? idx_lo : (dhi < dlo ? idx_hi : ((idx_hi & 1) == 0 ? idx_hi : idx_lo));

    return static_cast<uint8_t>(sign | idx);
}

static inline float dsv4_round_pow2_scale(float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return 1.0f;
    }
    return std::exp2(std::ceil(std::log2(scale)));
}

static void dsv4_fp8_qat_blockwise_n(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, int64_t block_size) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));

    constexpr float fp8_min = -448.0f;
    constexpr float fp8_max =  448.0f;
    constexpr float min_amax = 1.0e-4f;

    const int64_t n0 = a->ne[0];
    const int64_t n_rows = ggml_nelements(a) / n0;

    GGML_ASSERT(n0 % block_size == 0);

    const float * src = static_cast<const float *>(a->data);
    float * out = static_cast<float *>(dst->data);

    const int64_t row0 = (n_rows * ith) / nth;
    const int64_t row1 = (n_rows * (ith + 1)) / nth;

    for (int64_t row = row0; row < row1; ++row) {
        const float * src_row = src + row * n0;
        float * out_row = out + row * n0;

        for (int64_t col = 0; col < n0; col += block_size) {
            float amax = min_amax;
            for (int64_t j = 0; j < block_size; ++j) {
                amax = std::max(amax, std::fabs(src_row[col + j]));
            }

            const float scale = dsv4_round_pow2_scale(amax / fp8_max);

            for (int64_t j = 0; j < block_size; ++j) {
                // Match CUDA DSv4 clamp semantics (fmin/fmax): NaNs are sanitized
                // before E4M3 conversion instead of propagating through std::clamp.
                const float q = std::fmin(std::fmax(src_row[col + j] / scale, fp8_min), fp8_max);
                out_row[col + j] = dsv4_e4m3_to_float(dsv4_float_to_e4m3(q)) * scale;
            }
        }
    }
}

static void dsv4_fp8_qat_blockwise(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);
    dsv4_fp8_qat_blockwise_n(dst, a, ith, nth, 64);
}

static void dsv4_fp8_qat_blockwise_128(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);
    dsv4_fp8_qat_blockwise_n(dst, a, ith, nth, 128);
}

static void dsv4_fp4_qat_blockwise(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));

    constexpr int64_t block_size = 32;
    constexpr float fp4_max = 6.0f;
    constexpr float min_amax = 1.0e-6f;

    const int64_t n0 = a->ne[0];
    const int64_t n_rows = ggml_nelements(a) / n0;

    GGML_ASSERT(n0 % block_size == 0);

    const float * src = static_cast<const float *>(a->data);
    float * out = static_cast<float *>(dst->data);

    const int64_t row0 = (n_rows * ith) / nth;
    const int64_t row1 = (n_rows * (ith + 1)) / nth;

    for (int64_t row = row0; row < row1; ++row) {
        const float * src_row = src + row * n0;
        float * out_row = out + row * n0;

        for (int64_t col = 0; col < n0; col += block_size) {
            float amax = min_amax;
            for (int64_t j = 0; j < block_size; ++j) {
                amax = std::max(amax, std::fabs(src_row[col + j]));
            }

            const float scale = dsv4_round_pow2_scale(amax / fp4_max);

            for (int64_t j = 0; j < block_size; ++j) {
                const float q = std::fmin(std::fmax(src_row[col + j] / scale, -fp4_max), fp4_max);
                out_row[col + j] = q * scale;
            }
        }
    }
}

class dsv4_static_i32_input : public llm_graph_input_i {
public:
    dsv4_static_i32_input(ggml_tensor * tensor, std::vector<int32_t> values) : tensor(tensor), values(std::move(values)) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        GGML_ASSERT(tensor != nullptr);
        GGML_ASSERT(ggml_backend_buffer_is_host(tensor->buffer));
        GGML_ASSERT((int64_t) values.size() == tensor->ne[0]);
        std::memcpy(tensor->data, values.data(), values.size() * sizeof(values[0]));
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        // The data is constant and was captured at construction. As long as the
        // tensor still exists and its shape matches the captured values, the
        // graph can reuse this input.
        return tensor != nullptr && (int64_t) values.size() == tensor->ne[0];
    }

private:
    ggml_tensor * tensor;
    std::vector<int32_t> values;
};

class dsv4_mtp_raw_cache_input : public llm_graph_input_i {
  public:
    dsv4_mtp_raw_cache_input(ggml_tensor * tensor, const std::vector<float> * cache, uint32_t n_raw, int64_t row_size) :
        tensor(tensor),
        cache(cache),
        n_raw(n_raw),
        row_size(row_size) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        GGML_ASSERT(tensor != nullptr);
        GGML_ASSERT(cache != nullptr);
        GGML_ASSERT(tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(tensor->ne[0] == row_size);
        GGML_ASSERT(tensor->ne[2] == (int64_t) n_raw);
        GGML_ASSERT(cache->size() >= (size_t) n_raw * (size_t) row_size);
        ggml_backend_tensor_set(tensor, cache->data(), 0, (size_t) n_raw * (size_t) row_size * sizeof(float));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return tensor != nullptr && cache == params.mtp_raw_cache && n_raw == params.mtp_n_raw &&
               tensor->ne[0] == row_size && tensor->ne[2] == (int64_t) n_raw;
    }

  private:
    ggml_tensor *              tensor;
    const std::vector<float> * cache;
    uint32_t                   n_raw;
    int64_t                    row_size;
};

static bool dsv4_is_contiguous_single_seq(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;
    if (n_tokens <= 0) {
        return false;
    }

    const llama_seq_id seq0 = ubatch->seq_id[0][0];
    const llama_pos pos0 = ubatch->pos[0];
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (ubatch->seq_id[i][0] != seq0 || ubatch->pos[i] != pos0 + i) {
            return false;
        }
    }

    return true;
}

class dsv4_prompt_window_mask_input : public llm_graph_input_i {
public:
    dsv4_prompt_window_mask_input(ggml_tensor * mask, uint32_t window_size, bool use_alibi, bool causal_attn) :
        mask(mask),
        window_size(window_size),
        use_alibi(use_alibi),
        causal_attn(causal_attn) {
    }

    bool can_reuse(const llm_graph_params & params) override {
        // Mask shape is [n_tokens, n_tokens]. Reuse only when the new ubatch
        // has the same n_tokens; set_input below will rewrite the per-position
        // mask values according to the new positions.
        if (mask == nullptr) return false;
        const int64_t n = (int64_t) params.ubatch.n_tokens;
        return mask->ne[0] == n && mask->ne[1] == n;
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(mask != nullptr);
        GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));

        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t n_raw = mask->ne[0];

        float * data = static_cast<float *>(mask->data);
        std::fill(data, data + ggml_nelements(mask), -INFINITY);

        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos p1 = ubatch->pos[i1];
            const llama_pos min_pos = window_size > 0 ? p1 - (llama_pos) window_size + 1 : std::numeric_limits<llama_pos>::min();
            const int64_t row = i1 * n_raw;

            for (int64_t i0 = 0; i0 < n_raw; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0_cur = ubatch->pos[i0];
                if (s0 != s1) {
                    continue;
                }
                if (causal_attn && p0_cur > p1) {
                    continue;
                }
                if (window_size > 0 && p0_cur < min_pos) {
                    continue;
                }
                data[row + i0] = use_alibi ? -std::abs(float(p0_cur - p1)) : 0.0f;
            }
        }
    }

private:
    ggml_tensor * mask;
    uint32_t window_size;
    bool use_alibi;
    bool causal_attn;
};

class dsv4_prompt_comp_mask_input : public llm_graph_input_i {
public:
    dsv4_prompt_comp_mask_input(ggml_tensor * mask, int64_t n_comp, uint32_t ratio, bool use_alibi, bool causal_attn) :
        mask(mask),
        n_comp(n_comp),
        ratio(ratio),
        use_alibi(use_alibi),
        causal_attn(causal_attn) {
    }

    bool can_reuse(const llm_graph_params & params) override {
        // Mask shape is [n_comp, n_tokens]. n_comp is captured at construction;
        // reuse only when the new ubatch n_tokens still matches the second axis.
        if (mask == nullptr) return false;
        return mask->ne[0] == n_comp && mask->ne[1] == (int64_t) params.ubatch.n_tokens;
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(mask != nullptr);
        GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));

        const int64_t n_tokens = ubatch->n_tokens;

        float * data = static_cast<float *>(mask->data);
        std::fill(data, data + ggml_nelements(mask), -INFINITY);

        if (ratio == 0 || !dsv4_is_contiguous_single_seq(ubatch)) {
            return;
        }

        // Match antirez fill_compress_causal: use absolute position p1 for visibility.
        const llama_pos pos0 = ubatch->pos[0];
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            const llama_pos p1 = ubatch->pos[i1];
            const int64_t valid = causal_attn ? std::min<int64_t>(n_comp, (int64_t)(p1 + 1) / (int64_t) ratio) : n_comp;
            const int64_t row = i1 * n_comp;

            for (int64_t c = 0; c < valid; ++c) {
                const llama_pos p_comp = pos0 + c * (llama_pos) ratio;
                data[row + c] = use_alibi ? -std::abs(float(p_comp - p1)) : 0.0f;
            }
        }
    }

private:
    ggml_tensor * mask;
    int64_t n_comp;
    uint32_t ratio;
    bool use_alibi;
    bool causal_attn;
};

class dsv4_abs_comp_mask_input : public llm_graph_input_i {
public:
    dsv4_abs_comp_mask_input(ggml_tensor * mask, int64_t n_comp, uint32_t ratio, bool use_alibi, bool causal_attn) :
        mask(mask),
        n_comp(n_comp),
        ratio(ratio),
        use_alibi(use_alibi),
        causal_attn(causal_attn) {
    }

    bool can_reuse(const llm_graph_params & params) override {
        // Mask shape is [n_comp, n_tokens] (or [n_comp, n_tokens, 1, 1]).
        // n_comp is captured at construction; reuse only when n_tokens matches.
        if (mask == nullptr) return false;
        return mask->ne[0] == n_comp && mask->ne[1] == (int64_t) params.ubatch.n_tokens;
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(mask != nullptr);
        GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));

        float * data = static_cast<float *>(mask->data);
        std::fill(data, data + ggml_nelements(mask), -INFINITY);

        if (ratio == 0) {
            return;
        }

        const int64_t n_tokens = ubatch->n_tokens;
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            const llama_pos p1 = ubatch->pos[i1];
            const int64_t valid = causal_attn ? std::min<int64_t>(n_comp, (p1 + 1) / (int64_t) ratio) : n_comp;
            const int64_t row = i1 * n_comp;
            for (int64_t c = 0; c < valid; ++c) {
                const llama_pos p_comp = c * (llama_pos) ratio;
                data[row + c] = use_alibi ? -std::abs(float(p_comp - p1)) : 0.0f;
            }
        }
    }

private:
    ggml_tensor * mask;
    int64_t n_comp;
    uint32_t ratio;
    bool use_alibi;
    bool causal_attn;
};

struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

struct dsv4_state_pair {
    ggml_tensor * kv;
    ggml_tensor * score;
};

struct dsv4_decode_compressor {
    ggml_tensor * kv_state;
    ggml_tensor * score_state;
    ggml_tensor * kv_comp;
};

static dsv4_state_layout dsv4_make_state_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows = compress_ratio == 4 ? 8 : compress_ratio;
    return { width, rows, width * rows };
}

static ggml_tensor * dsv4_new_filled_2d(ggml_context * ctx, int64_t ne0, int64_t ne1, float val) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1), val);
}

// Graph-build-time check: only emit ggml_cont when the tensor is actually
// non-contiguous. ggml_cont always emits a GGML_OP_CONT (memory copy on every
// backend) regardless of input layout, so a `cont(contig_tensor)` is pure waste.
// Profiling shows CONT is ~9% of compute time on DSv4 decode — most of those
// calls are no-op-equivalent and can be skipped.
static inline ggml_tensor * cont_if_needed(ggml_context * ctx, ggml_tensor * t) {
    return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

static ggml_tensor * dsv4_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    return ggml_cast(ctx, ggml_arange(ctx, (float) begin, (float) end, 1.0f), GGML_TYPE_I32);
}

static ggml_tensor * dsv4_view_cols(ggml_context * ctx, ggml_tensor * x, int64_t ne0, int64_t ne1, int64_t col_off, int64_t row_off) {
    return ggml_view_2d(ctx, x, ne0, ne1, x->nb[1], ggml_row_size(x->type, col_off) + row_off * x->nb[1]);
}

static ggml_tensor * dsv4_view_state_segment(ggml_context * ctx, ggml_tensor * state_all, int64_t state_off, int64_t width, int64_t rows) {
    return ggml_view_2d(ctx, state_all, width, rows, width * state_all->nb[0], state_off * state_all->nb[0]);
}

// Store state for one or more sequences into the recurrent state buffer.
// src shape: {elems} for single-seq, or {elems, n_seqs} for multi-seq.
// Writes to n_seqs consecutive cells starting at head.
static void dsv4_store_state_segment(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * src, ggml_tensor * dst_state, int64_t state_size, uint32_t head, int64_t state_off, int64_t n_seqs = 1) {
    GGML_ASSERT(src != nullptr && dst_state != nullptr);
    src = cont_if_needed(ctx, src);
    const int64_t elems_per_seq = ggml_nelements(src) / n_seqs;

    for (int64_t s = 0; s < n_seqs; ++s) {
        ggml_tensor * src_s = (n_seqs == 1)
            ? ggml_reshape_1d(ctx, src, elems_per_seq)
            : ggml_reshape_1d(ctx, ggml_view_2d(ctx, src, elems_per_seq, 1, src->nb[1], s * src->nb[1]), elems_per_seq);
        ggml_tensor * dst_view = ggml_view_1d(ctx, dst_state, elems_per_seq,
                ((head + s) * state_size + state_off) * ggml_element_size(dst_state));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src_s, dst_view));
    }
}

static void dsv4_store_cache_rows(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * dst_cache, ggml_tensor * src, int64_t row_start, int64_t n_rows) {
    if (src == nullptr || n_rows <= 0) {
        return;
    }

    src = cont_if_needed(ctx, src);
    src = ggml_reshape_2d(ctx, src, dst_cache->ne[0], n_rows);

    ggml_tensor * rows = dsv4_arange_i32(ctx, row_start, row_start + n_rows);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, dst_cache, src, rows));
}

static ggml_tensor * dsv4_cache_view_3d(ggml_context * ctx, ggml_tensor * cache, int64_t n_rows) {
    ggml_tensor * view = ggml_view_2d(ctx, cache, cache->ne[0], n_rows, cache->nb[1], 0);
    return ggml_reshape_3d(ctx, view, cache->ne[0], 1, n_rows);
}

static ggml_tensor * dsv4_softmax_pool_ratio(ggml_context * ctx, ggml_tensor * kv, ggml_tensor * score) {
    score = ggml_soft_max(ctx, score);
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * dsv4_shift_overlap_state(ggml_context * ctx, ggml_tensor * x, float pad_value) {
    const int64_t n_embd = x->ne[0];
    const int64_t ratio = x->ne[1];
    const int64_t n_comp = x->ne[2];

    ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1, x->nb[1], x->nb[2], 0);
    ggml_tensor * pad = ggml_fill(ctx, cont_if_needed(ctx, first), pad_value);
    if (n_comp == 1) {
        return pad;
    }

    ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1, x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

// Build compressor prefill state for one or more sequences.
// x: {n_embd, n_seq_tokens} for single-seq, or {n_embd, n_seq_tokens, n_seqs} for multi-seq.
// Returns state pair where each tensor is {width, rows} for single-seq or {width, rows, n_seqs}.
static dsv4_state_pair dsv4_build_compressor_prefill_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * wkv,
        ggml_tensor  * wgate,
        ggml_tensor  * ape,
        int64_t        head_dim,
        int64_t        n_seq_tokens,
        int64_t        compress_ratio,
        int64_t        n_seqs = 1) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    const int64_t cutoff    = (n_seq_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_seq_tokens - cutoff;

    // ggml_mul_mat naturally batches over dim 2+, so these produce
    // {width, n_seq_tokens} for n_seqs==1 or {width, n_seq_tokens, n_seqs} for n_seqs>1.
    ggml_tensor * kv    = ggml_mul_mat(ctx, wkv, x);
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    // Helper to create filled tensors that broadcast across n_seqs
    auto new_filled = [&](int64_t w, int64_t h, float val) -> ggml_tensor * {
        if (n_seqs <= 1) {
            return dsv4_new_filled_2d(ctx, w, h, val);
        }
        ggml_tensor * t2 = dsv4_new_filled_2d(ctx, w, h, val);
        // Reshape to {w, h, 1} and repeat to {w, h, n_seqs}
        t2 = ggml_reshape_3d(ctx, t2, w, h, 1);
        return ggml_repeat_4d(ctx, t2, w, h, n_seqs, 1);
    };

    // View helper: for 2D/3D tensors, view rows [start, start+count) per sequence
    auto view_rows = [&](ggml_tensor * t, int64_t w, int64_t count, int64_t start) -> ggml_tensor * {
        if (n_seqs <= 1) {
            return ggml_view_2d(ctx, t, w, count, t->nb[1], start * t->nb[1]);
        }
        return ggml_view_3d(ctx, t, w, count, n_seqs, t->nb[1], t->nb[2], start * t->nb[1]);
    };

    if (compress_ratio == 4) {
        ggml_tensor * kv_prev    = new_filled(layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_prev = new_filled(layout.width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            kv_prev = view_rows(kv, layout.width, compress_ratio, cutoff - compress_ratio);
            score_prev = view_rows(score, layout.width, compress_ratio, cutoff - compress_ratio);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        ggml_tensor * kv_curr    = new_filled(layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_curr = new_filled(layout.width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            ggml_tensor * kv_rem = view_rows(kv, layout.width, remainder, cutoff);
            ggml_tensor * sc_rem = view_rows(score, layout.width, remainder, cutoff);
            if (n_seqs <= 1) {
                sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));
            } else {
                // Broadcast ape across n_seqs
                ggml_tensor * ape_slice = ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0);
                ape_slice = ggml_reshape_3d(ctx, ape_slice, layout.width, remainder, 1);
                sc_rem = ggml_add(ctx, sc_rem, ape_slice);
            }

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem, new_filled(layout.width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem, new_filled(layout.width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev, kv_curr, 1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    ggml_tensor * kv_state    = new_filled(layout.width, compress_ratio, 0.0f);
    ggml_tensor * score_state = new_filled(layout.width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        ggml_tensor * kv_rem = view_rows(kv, layout.width, remainder, cutoff);
        ggml_tensor * sc_rem = view_rows(score, layout.width, remainder, cutoff);
        if (n_seqs <= 1) {
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));
        } else {
            ggml_tensor * ape_slice = ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0);
            ape_slice = ggml_reshape_3d(ctx, ape_slice, layout.width, remainder, 1);
            sc_rem = ggml_add(ctx, sc_rem, ape_slice);
        }

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem, new_filled(layout.width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem, new_filled(layout.width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}
} // namespace

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_hc        = hparams.n_hc;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());
    GGML_ASSERT(n_hc > 0);

    ggml_tensor * cur;

    auto as_f32 = [&](ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32);
    };

    auto mul_mat_checked = [&](ggml_tensor * a, ggml_tensor * b, const char * tag) -> ggml_tensor * {
        // CPU MUL_MAT supports non-BF16 weights with F32 (or quantized vec-dot) RHS, not BF16 RHS.
        // Casting preserves the BF16-quantized values while keeping CPU-only DeepSeek4 graphs schedulable.
        if (b->type == GGML_TYPE_BF16 && a->type != GGML_TYPE_BF16) {
            b = ggml_cast(ctx0, b, GGML_TYPE_F32);
        }
        if (!(a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0)) {
            GGML_ABORT("deepseek4 mul_mat mismatch at %s: a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]",
                    tag,
                    (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                    (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
        }
        return ggml_mul_mat(ctx0, a, b);
    };

    // Uses the proper ggml_dsv4_fp8_kv_quantize op (CUDA-accelerated)
    auto apply_fp8_qat_nope_2d = [&](ggml_tensor * t, const char * tag, int il_cur) -> ggml_tensor * {
        t = as_f32(t);
        if (n_rot <= 0 || n_rot >= t->ne[0]) { return t; }
        const int64_t nope_dim = t->ne[0] - n_rot;
        if (nope_dim % 64 != 0) {
            GGML_ABORT("DeepSeek4 FP8 KV QAT requires non-RoPE dim divisible by 64, got ne0=%lld n_rot=%lld nope=%lld",
                    (long long) t->ne[0], (long long) n_rot, (long long) nope_dim);
        }
        t = ggml_dsv4_fp8_kv_quantize(ctx0, t, (int)n_rot);
        cb(t, tag, il_cur);
        return t;
    };

    auto apply_fp8_qat_full_2d = [&](ggml_tensor * t, const char * tag, int il_cur) -> ggml_tensor * {
        t = as_f32(cont_if_needed(ctx0, t));
        GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1);
        if (t->ne[0] % 128 != 0) {
            GGML_ABORT("DeepSeek4 dense FP8 QAT requires innermost dim divisible by 128, got ne0=%lld", (long long) t->ne[0]);
        }
        t = ggml_map_custom1(ctx0, t, dsv4_fp8_qat_blockwise_128, GGML_N_TASKS_MAX, nullptr);
        t = ggml_cast(ctx0, t, GGML_TYPE_BF16);
        cb(t, tag, il_cur);
        return t;
    };

    auto apply_dense_fp8_qat = [&](ggml_tensor * t, bool enabled, const char * tag, int il_cur) -> ggml_tensor * {
        return enabled ? apply_fp8_qat_full_2d(t, tag, il_cur) : t;
    };

    auto cast_dense_fp8_out = [&](ggml_tensor * t, bool enabled) -> ggml_tensor * {
        return enabled ? ggml_cast(ctx0, t, GGML_TYPE_BF16) : t;
    };

    // make_hc_state: matches antirez — reshape3d -> repeat_4d -> reshape3d.
    // Broadcasts the embedding to n_hc HC-copies for each token.
    auto make_hc_state = [&](ggml_tensor * x) -> ggml_tensor * {
        x = as_f32(x);
        const int64_t embd = x->ne[0];
        const int64_t toks = x->ne[1];
        if (n_hc == 1) {
            return ggml_reshape_3d(ctx0, x, embd, 1, toks);
        }
        ggml_tensor * y = ggml_reshape_3d(ctx0, x, embd, 1, toks);
        y = ggml_repeat_4d(ctx0, y, embd, n_hc, toks, 1);
        return ggml_reshape_3d(ctx0, y, embd, n_hc, toks);
    };

    auto flatten_hc_state = [&](ggml_tensor * x_hc) -> ggml_tensor * {
        return ggml_reshape_2d(ctx0, cont_if_needed(ctx0, as_f32(x_hc)), x_hc->ne[0] * x_hc->ne[1], x_hc->ne[2]);
    };

    struct hc_pre_result {
        ggml_tensor * collapsed;
        ggml_tensor * post;
        ggml_tensor * comb;
    };

    // hc_pre: uses ggml_dsv4_hc_split_sinkhorn (proper op, CUDA-accelerated)
    auto hc_pre = [&](ggml_tensor * x_hc, ggml_tensor * hc_fn, ggml_tensor * hc_base, ggml_tensor * hc_scale, const char * tag, int il_cur) -> hc_pre_result {
        GGML_ASSERT(x_hc->ne[0] == n_embd && x_hc->ne[1] == n_hc);
        const int64_t hc_dim = n_embd * n_hc;
        ggml_tensor * flat = cont_if_needed(ctx0, ggml_reshape_2d(ctx0, x_hc, hc_dim, n_tokens));
        flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
        ggml_tensor * mixes = mul_mat_checked(hc_fn, flat, "hc.fn"); // [(2+n_hc)*n_hc, n_tokens]
        cb(mixes, std::string(tag).append("_mixes").c_str(), il_cur);

        // ggml_dsv4_hc_split_sinkhorn: computes pre/post/comb in one fused op
        ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx0, mixes, hc_scale, hc_base, (int)n_hc, (int)hparams.n_hc_sinkhorn_iters, hparams.f_hc_eps);
        ggml_tensor * pre  = ggml_view_2d(ctx0, split, n_hc, n_tokens, split->nb[1], 0);
        ggml_tensor * post = ggml_view_2d(ctx0, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
        ggml_tensor * comb = ggml_view_2d(ctx0, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
        if (n_tokens != 1) {
            pre  = cont_if_needed(ctx0, pre);
            post = cont_if_needed(ctx0, post);
            comb = cont_if_needed(ctx0, comb);
        }
        comb = ggml_reshape_3d(ctx0, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]

        // ggml_dsv4_hc_weighted_sum: sum_{hc} pre[hc,t] * x[embd,hc,t]
        // Result is F32 (matches antirez — do not cast to BF16, the proper op already produces F32)
        ggml_tensor * collapsed = ggml_dsv4_hc_weighted_sum(ctx0, x_hc, pre);
        cb(collapsed, tag, il_cur);
        return { collapsed, post, comb };
    };

    // hc_post: uses ggml_dsv4_hc_expand (proper op, CUDA-accelerated)
    auto hc_post = [&](ggml_tensor * x, ggml_tensor * residual_hc, ggml_tensor * post, ggml_tensor * comb, const char * tag, int il_cur) -> ggml_tensor * {
        x = as_f32(x);
        ggml_tensor * out = ggml_dsv4_hc_expand(ctx0, x, residual_hc, post, comb);
        cb(out, tag, il_cur);
        return out;
    };

    // hc_head: uses ggml_dsv4_hc_weighted_sum (proper op, CUDA-accelerated)
    auto hc_head = [&](ggml_tensor * x_hc, ggml_tensor * hc_fn, ggml_tensor * hc_base, ggml_tensor * hc_scale, const char * tag) -> ggml_tensor * {
        GGML_ASSERT(x_hc->ne[0] == n_embd && x_hc->ne[1] == n_hc);

        if (hc_fn == nullptr || hc_base == nullptr || hc_scale == nullptr) {
            // fallback: simple mean across HC copies
            ggml_tensor * mean = nullptr;
            for (int64_t i = 0; i < n_hc; ++i) {
                ggml_tensor * x_i = as_f32(ggml_view_2d(ctx0, x_hc, n_embd, n_tokens, x_hc->nb[2], i * x_hc->nb[1]));
                mean = mean == nullptr ? x_i : ggml_add(ctx0, mean, x_i);
            }
            return ggml_scale(ctx0, mean, 1.0f / (float)n_hc);
        }

        const int64_t hc_dim = n_embd * n_hc;
        ggml_tensor * flat = cont_if_needed(ctx0, ggml_reshape_2d(ctx0, x_hc, hc_dim, n_tokens));
        flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
        ggml_tensor * pre = mul_mat_checked(hc_fn, flat, "hc.head.fn"); // [n_hc, n_tokens]
        pre = ggml_mul(ctx0, pre, ggml_view_1d(ctx0, hc_scale, 1, 0));
        pre = ggml_add(ctx0, pre, hc_base);
        pre = ggml_add(ctx0, ggml_sigmoid(ctx0, pre), ggml_fill(ctx0, pre, hparams.f_hc_eps));

        ggml_tensor * out = ggml_dsv4_hc_weighted_sum(ctx0, x_hc, pre);
        cb(out, tag, -1);
        return out;
    };

    auto apply_cvec_hc = [&](ggml_tensor * x_hc, int il_cur) -> ggml_tensor * {
        ggml_tensor * flat = flatten_hc_state(x_hc);
        flat = build_cvec(flat, il_cur);
        return ggml_reshape_3d(ctx0, flat, n_embd, n_hc, flat->ne[1]);
    };

    // build_compressed_pool: processes inp tokens in windows of size `ratio`.
    // For multi-seq (n_seqs > 1), inp is {n_embd, n_tokens} where n_tokens = n_seq_tokens * n_seqs.
    // The lambda uses the captured n_seq_tokens variable (set per-layer) to respect sequence boundaries.
    auto build_compressed_pool = [&](ggml_tensor * inp, ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm,
                                     int64_t out_dim, int64_t head_dim, uint32_t ratio, const char * tag, int il_cur) -> ggml_tensor * {
        // Use the per-layer n_seq_tokens and n_seqs captured from the enclosing scope.
        // For single-seq, n_seq_tokens == n_tokens; for multi-seq, n_seq_tokens < n_tokens.
        const int64_t local_n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t local_n_seqs       = ubatch.n_seqs;

        if (wkv == nullptr || wgate == nullptr || ratio == 0 || local_n_seq_tokens < (int64_t) ratio) {
            return nullptr;
        }

        GGML_ASSERT(wkv->ne[0] == n_embd);
        GGML_ASSERT(wkv->ne[1] == out_dim);
        GGML_ASSERT(wgate->ne[0] == n_embd);
        GGML_ASSERT(wgate->ne[1] == out_dim);
        if (ape != nullptr) {
            GGML_ASSERT(ape->ne[0] == out_dim);
            GGML_ASSERT(ape->ne[1] == ratio);
        }
        if (norm != nullptr) {
            GGML_ASSERT(norm->ne[0] == head_dim);
        }

        // Windows are computed per-sequence to avoid crossing sequence boundaries.
        const int64_t n_tok_usable_per_seq = (local_n_seq_tokens / (int64_t) ratio) * (int64_t) ratio;
        const int64_t n_windows_per_seq = n_tok_usable_per_seq / (int64_t) ratio;
        if (n_windows_per_seq == 0) {
            return nullptr;
        }

        // For multi-seq, reshape inp to {n_embd, n_seq_tokens, n_seqs} so mul_mat batches correctly
        ggml_tensor * inp_ms = (local_n_seqs > 1)
            ? ggml_reshape_3d(ctx0, inp, n_embd, local_n_seq_tokens, local_n_seqs)
            : inp;
        ggml_tensor * kv = mul_mat_checked(wkv, inp_ms, "compressor.wkv");
        ggml_tensor * gate = mul_mat_checked(wgate, inp_ms, "compressor.wgate");

        // Trim to usable tokens per sequence
        if (n_tok_usable_per_seq < local_n_seq_tokens) {
            if (local_n_seqs > 1) {
                kv   = ggml_view_3d(ctx0, kv,   out_dim, n_tok_usable_per_seq, local_n_seqs, kv->nb[1], kv->nb[2], 0);
                gate = ggml_view_3d(ctx0, gate, out_dim, n_tok_usable_per_seq, local_n_seqs, gate->nb[1], gate->nb[2], 0);
            } else {
                kv   = ggml_view_2d(ctx0, kv,   out_dim, n_tok_usable_per_seq, kv->nb[1], 0);
                gate = ggml_view_2d(ctx0, gate, out_dim, n_tok_usable_per_seq, gate->nb[1], 0);
            }
        }

        // Reshape into windows. For multi-seq: {out_dim, ratio, n_windows_per_seq * n_seqs}
        // The windows within each sequence are contiguous, so this is safe.
        const int64_t total_windows = n_windows_per_seq * local_n_seqs;
        ggml_tensor * kv_flat = (local_n_seqs > 1) ? ggml_reshape_2d(ctx0, cont_if_needed(ctx0, kv), out_dim, n_tok_usable_per_seq * local_n_seqs) : kv;
        ggml_tensor * gate_flat = (local_n_seqs > 1) ? ggml_reshape_2d(ctx0, cont_if_needed(ctx0, gate), out_dim, n_tok_usable_per_seq * local_n_seqs) : gate;
        ggml_tensor * kv_3d = ggml_reshape_3d(ctx0, kv_flat, out_dim, ratio, total_windows);
        ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate_flat, out_dim, ratio, total_windows);

        if (ape != nullptr) {
            ggml_tensor * ape_f32 = as_f32(ape);
            ggml_tensor * ape_3d = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, ape_f32, out_dim, ratio, 1), gate_3d);
            gate_3d = ggml_add(ctx0, gate_3d, ape_3d);
        }

        ggml_tensor * pool = nullptr;
        if (ratio == 4 && out_dim == 2 * head_dim) {
            ggml_tensor * kv_a = ggml_view_3d(ctx0, kv_3d, head_dim, ratio, total_windows, kv_3d->nb[1], kv_3d->nb[2], 0);
            ggml_tensor * kv_b = ggml_view_3d(ctx0, kv_3d, head_dim, ratio, total_windows, kv_3d->nb[1], kv_3d->nb[2], kv_3d->nb[0] * head_dim);
            ggml_tensor * gate_a = ggml_view_3d(ctx0, gate_3d, head_dim, ratio, total_windows, gate_3d->nb[1], gate_3d->nb[2], 0);
            ggml_tensor * gate_b = ggml_view_3d(ctx0, gate_3d, head_dim, ratio, total_windows, gate_3d->nb[1], gate_3d->nb[2], gate_3d->nb[0] * head_dim);

            // Shifted window: for each sequence, the first window gets zero/neg-inf,
            // subsequent windows get the previous window's values.
            // For multi-seq, reshape to 4D {head_dim, ratio, n_windows_per_seq, n_seqs},
            // shift within dim 2 (per-sequence), then flatten back.
            // Build shifted window: for each sequence, prepend a zero/neg-inf window
            // and take windows [0..n_windows_per_seq-2] as the "previous" values.
            // We loop over sequences at graph-build time (cheap) to avoid 4D reshapes
            // that would require ggml_cont copies of non-contiguous views.
            ggml_tensor * kv_a_shift;
            ggml_tensor * gate_a_shift;
            if (n_windows_per_seq > 1) {
                // Build per-sequence shifted views, then concat across sequences.
                std::vector<ggml_tensor *> kv_parts, gate_parts;
                for (int64_t s = 0; s < local_n_seqs; ++s) {
                    const int64_t win_off = s * n_windows_per_seq;
                    // Previous windows [0..n_windows_per_seq-2] for this sequence
                    ggml_tensor * kv_a_prev_s = ggml_view_3d(ctx0, kv_a, head_dim, ratio, n_windows_per_seq - 1,
                                                              kv_a->nb[1], kv_a->nb[2], win_off * kv_a->nb[2]);
                    ggml_tensor * gate_a_prev_s = ggml_view_3d(ctx0, gate_a, head_dim, ratio, n_windows_per_seq - 1,
                                                               gate_a->nb[1], gate_a->nb[2], win_off * gate_a->nb[2]);
                    // Zero/neg-inf initial window
                    ggml_tensor * kv_zero_s = ggml_fill(ctx0, cont_if_needed(ctx0, ggml_view_3d(ctx0, kv_a, head_dim, ratio, 1,
                                                          kv_a->nb[1], kv_a->nb[2], win_off * kv_a->nb[2])), 0.0f);
                    ggml_tensor * gate_ninf_s = ggml_fill(ctx0, cont_if_needed(ctx0, ggml_view_3d(ctx0, gate_a, head_dim, ratio, 1,
                                                           gate_a->nb[1], gate_a->nb[2], win_off * gate_a->nb[2])), -INFINITY);
                    kv_parts.push_back(ggml_concat(ctx0, kv_zero_s, kv_a_prev_s, 2));
                    gate_parts.push_back(ggml_concat(ctx0, gate_ninf_s, gate_a_prev_s, 2));
                }
                // Concat all sequences' shifted windows along dim 2
                kv_a_shift = kv_parts[0];
                gate_a_shift = gate_parts[0];
                for (int64_t s = 1; s < local_n_seqs; ++s) {
                    kv_a_shift = ggml_concat(ctx0, kv_a_shift, kv_parts[s], 2);
                    gate_a_shift = ggml_concat(ctx0, gate_a_shift, gate_parts[s], 2);
                }
            } else {
                kv_a_shift = ggml_fill(ctx0, cont_if_needed(ctx0, kv_a), 0.0f);
                gate_a_shift = ggml_fill(ctx0, cont_if_needed(ctx0, gate_a), -INFINITY);
            }

            ggml_tensor * kv_ov = ggml_concat(ctx0, kv_a_shift, kv_b, 1);        // [head_dim, 2*ratio, total_windows]
            ggml_tensor * gate_ov = ggml_concat(ctx0, gate_a_shift, gate_b, 1);  // [head_dim, 2*ratio, total_windows]

            ggml_tensor * gate_s = ggml_permute(ctx0, gate_ov, 1, 0, 2, 3);      // [2*ratio, head_dim, total_windows]
            ggml_tensor * w = ggml_soft_max(ctx0, cont_if_needed(ctx0, gate_s));
            w = ggml_permute(ctx0, w, 1, 0, 2, 3);                                // [head_dim, 2*ratio, total_windows]

            ggml_tensor * pooled = ggml_mul(ctx0, kv_ov, w);
            pooled = ggml_permute(ctx0, pooled, 1, 0, 2, 3);                      // [2*ratio, head_dim, total_windows]
            pooled = cont_if_needed(ctx0, ggml_cast(ctx0, pooled, GGML_TYPE_F32));
            pooled = ggml_sum_rows(ctx0, pooled);                                  // [1, head_dim, total_windows]
            pool = ggml_reshape_2d(ctx0, pooled, head_dim, total_windows);          // [head_dim, total_windows]
        } else {
            ggml_tensor * gate_s = ggml_permute(ctx0, gate_3d, 1, 0, 2, 3);       // [ratio, out_dim, total_windows]
            ggml_tensor * w = ggml_soft_max(ctx0, cont_if_needed(ctx0, gate_s));
            w = ggml_permute(ctx0, w, 1, 0, 2, 3);                                 // [out_dim, ratio, total_windows]

            ggml_tensor * pooled = ggml_mul(ctx0, kv_3d, w);
            pooled = ggml_permute(ctx0, pooled, 1, 0, 2, 3);                       // [ratio, out_dim, total_windows]
            pooled = cont_if_needed(ctx0, ggml_cast(ctx0, pooled, GGML_TYPE_F32));
            pooled = ggml_sum_rows(ctx0, pooled);                                   // [1, out_dim, total_windows]
            pool = ggml_reshape_2d(ctx0, pooled, out_dim, total_windows);            // [out_dim, total_windows]

            if (out_dim != head_dim) {
                pool = ggml_view_2d(ctx0, pool, head_dim, total_windows, pool->nb[1], 0);
            }
        }

        if (norm != nullptr) {
            pool = ggml_rms_norm(ctx0, pool, hparams.f_norm_rms_eps);
            pool = ggml_mul(ctx0, pool, norm);
        }

        cb(pool, tag, il_cur);
        return pool;
    };

    res->add_input(std::make_unique<dsv4_position_graph_guard>(ubatch));
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_mem_hybrid_iswa * inp_mem = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    llm_graph_input_attn_kv * inp_attn_kv = nullptr;
    llm_graph_input_rs * inp_rs = nullptr;
    const llama_memory_hybrid_iswa_context * mctx_dsv4 = nullptr;
    const bool has_hybrid_iswa = dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx) != nullptr;

    if (has_hybrid_iswa) {
        inp_mem = build_inp_mem_hybrid_iswa();
        inp_attn_iswa = inp_mem->get_attn();
        inp_rs = inp_mem->get_recr();
        mctx_dsv4 = inp_mem->mctx;
    } else {
        inp_attn_kv = build_attn_inp_kv();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // Per-graph mask tensor cache. Many DSV4 mask tensors depend only on
    // (kind, ratio, n_tokens, n_topk) and are identical across layers that
    // share those parameters. We cache them so the scheduler only sees a
    // small number of GGML_TENSOR_FLAG_INPUT tensors instead of one per
    // layer per mask kind. This is required for multi-GPU pipeline
    // parallelism (GGML_SCHED_MAX_SPLIT_INPUTS=30 in upstream ggml).
    std::unordered_map<std::string, ggml_tensor *> mask_cache;
    auto cache_mask = [&](const std::string & key, auto && create) -> ggml_tensor * {
        auto it = mask_cache.find(key);
        if (it != mask_cache.end()) {
            return it->second;
        }
        ggml_tensor * t = create();
        mask_cache.emplace(key, t);
        return t;
    };

    struct dsv4_rope_cfg {
        int32_t n_ctx_orig;
        float freq_base;
        float freq_scale;
        float ext_factor;
        float attn_factor;
        float beta_fast;
        float beta_slow;
    };

    auto get_rope_cfg = [&](uint32_t compress_ratio) -> dsv4_rope_cfg {
        if (compress_ratio > 0) {
            float compress_attn_factor = 1.0f;
            if (ext_factor != 0.0f && freq_scale > 0.0f) {
                // DeepSeek-V4 compressed RoPE uses YaRN frequency interpolation, but the
                // official runtime does not apply YaRN's magnitude scaling. ggml_rope_ext
                // applies that internal scale when YaRN is enabled, so pass the inverse here.
                compress_attn_factor /= 1.0f + 0.1f * std::log(1.0f / freq_scale);
            }

            return {
                n_ctx_orig,
                hparams.rope_freq_base_compress > 0.0f ? hparams.rope_freq_base_compress : freq_base,
                freq_scale,
                ext_factor,
                compress_attn_factor,
                beta_fast,
                beta_slow,
            };
        }

        // compress_ratio == 0 (dense layers): match antirez — use base rope_theta + cparams beta_fast/slow,
        // disable YaRN scaling.
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            beta_fast,
            beta_slow,
        };
    };

    // apply_partial_rope_with_pos: uses ggml_dsv4_rope_tail (proper fused op)
    auto apply_partial_rope_with_pos = [&](ggml_tensor * x, ggml_tensor * pos, const dsv4_rope_cfg & rope_cfg, bool inverse) -> ggml_tensor * {
        const int64_t head_dim = x->ne[0];
        const int64_t rope_dim = n_rot;
        if (rope_dim <= 0 || rope_dim > head_dim) {
            return x;
        }
        if (head_dim == rope_dim) {
            // full rope, no nope prefix — use standard rope_ext
            return inverse
                ? ggml_rope_ext_back(ctx0, x, pos, nullptr,
                        (int)rope_dim, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                        rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow)
                : ggml_rope_ext(ctx0, x, pos, nullptr,
                        (int)rope_dim, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                        rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow);
        }
        // partial rope: ggml_dsv4_rope_tail handles nope prefix + rope tail in one op
        x = as_f32(cont_if_needed(ctx0, x));
        return ggml_dsv4_rope_tail(ctx0, x, pos, nullptr,
                (int)rope_dim, rope_type, rope_cfg.n_ctx_orig,
                rope_cfg.freq_base, rope_cfg.freq_scale, rope_cfg.ext_factor,
                rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, inverse);
    };

    auto apply_partial_rope = [&](ggml_tensor * x, const dsv4_rope_cfg & rope_cfg, bool inverse) -> ggml_tensor * {
        return apply_partial_rope_with_pos(x, inp_pos, rope_cfg, inverse);
    };

    auto dsv4_pool_decode_state = [&](ggml_tensor * kv, ggml_tensor * score, ggml_tensor * norm, ggml_tensor * pos, int64_t head_dim, const dsv4_rope_cfg & rope_cfg) -> ggml_tensor * {
        const int64_t n_rows = kv->ne[1];
        kv = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, ggml_transpose(ctx0, kv)), n_rows, head_dim, 1);
        score = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, ggml_transpose(ctx0, score)), n_rows, head_dim, 1);

        ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx0, kv, score);
        pooled = ggml_rms_norm(ctx0, pooled, hparams.f_norm_rms_eps);
        pooled = ggml_mul(ctx0, pooled, norm);
        pooled = ggml_reshape_3d(ctx0, pooled, head_dim, 1, 1);

        return apply_partial_rope_with_pos(pooled, pos, rope_cfg, /*inverse=*/false);
    };

    auto dsv4_build_compressor_decode = [&](ggml_tensor * x, ggml_tensor * prev_kv_state, ggml_tensor * prev_score_state,
                                            ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm,
                                            int64_t head_dim, int64_t pos, int64_t compress_ratio, const dsv4_rope_cfg & rope_cfg) -> dsv4_decode_compressor {
        const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
        const int64_t pos_mod = pos % compress_ratio;
        const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
        const bool should_compress = (pos + 1) % compress_ratio == 0;

        ggml_tensor * kv_cur = ggml_mul_mat(ctx0, wkv, x);
        ggml_tensor * sc_cur = ggml_mul_mat(ctx0, wgate, x);
        ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx0, ape, GGML_TYPE_F32);
        sc_cur = ggml_add(ctx0, sc_cur, ggml_view_2d(ctx0, ape_f, layout.width, 1, ape_f->nb[1], pos_mod * ape_f->nb[1]));

        ggml_tensor * row_idx = dsv4_arange_i32(ctx0, row, row + 1);
        ggml_tensor * kv_state = ggml_set_rows(ctx0, prev_kv_state, kv_cur, row_idx);
        ggml_tensor * score_state = ggml_set_rows(ctx0, prev_score_state, sc_cur, row_idx);
        ggml_tensor * kv_comp = nullptr;

        if (should_compress) {
            ggml_tensor * kv_pool;
            ggml_tensor * score_pool;

            if (compress_ratio == 4) {
                ggml_tensor * kv_prev = dsv4_view_cols(ctx0, kv_state, head_dim, compress_ratio, 0, 0);
                ggml_tensor * kv_curr = dsv4_view_cols(ctx0, kv_state, head_dim, compress_ratio, head_dim, compress_ratio);
                ggml_tensor * sc_prev = dsv4_view_cols(ctx0, score_state, head_dim, compress_ratio, 0, 0);
                ggml_tensor * sc_curr = dsv4_view_cols(ctx0, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

                kv_pool = ggml_concat(ctx0, kv_prev, kv_curr, 1);
                score_pool = ggml_concat(ctx0, sc_prev, sc_curr, 1);

                ggml_tensor * shifted_kv = dsv4_view_cols(ctx0, kv_state, layout.width, compress_ratio, 0, compress_ratio);
                ggml_tensor * shifted_score = dsv4_view_cols(ctx0, score_state, layout.width, compress_ratio, 0, compress_ratio);
                kv_state = ggml_concat(ctx0, shifted_kv, shifted_kv, 1);
                score_state = ggml_concat(ctx0, shifted_score, shifted_score, 1);
            } else {
                kv_pool = kv_state;
                score_pool = score_state;
            }

            ggml_tensor * comp_pos = dsv4_arange_i32(ctx0, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
            kv_comp = dsv4_pool_decode_state(kv_pool, score_pool, norm, comp_pos, head_dim, rope_cfg);
        }

        return { kv_state, score_state, kv_comp };
    };

    auto dsv4_build_compressor_decode_chunk = [&](ggml_tensor * x, ggml_tensor * prev_kv_state, ggml_tensor * prev_score_state,
                                                  ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm,
                                                  int64_t head_dim, int64_t compress_ratio, const dsv4_rope_cfg & rope_cfg) -> dsv4_decode_compressor {
        // Match antirez: project the whole decode chunk once, then view each token's
        // projected row. This keeps chunked decode graph semantics identical to the
        // reference path and avoids one matmul subgraph per token.
        const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
        ggml_tensor * kv_all = ggml_mul_mat(ctx0, wkv, x);   // [width, n_tokens]
        ggml_tensor * sc_all = ggml_mul_mat(ctx0, wgate, x);
        ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx0, ape, GGML_TYPE_F32);

        ggml_tensor * kv_state = prev_kv_state;
        ggml_tensor * score_state = prev_score_state;
        ggml_tensor * kv_comp = nullptr;

        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_pos pos_i = ubatch.pos ? ubatch.pos[i] : (llama_pos) i;
            const int64_t pos_mod = pos_i % compress_ratio;

            ggml_tensor * kv_cur = ggml_view_2d(ctx0, kv_all, layout.width, 1, kv_all->nb[1], i * kv_all->nb[1]);
            ggml_tensor * sc_cur = ggml_view_2d(ctx0, sc_all, layout.width, 1, sc_all->nb[1], i * sc_all->nb[1]);
            sc_cur = ggml_add(ctx0, sc_cur, ggml_view_2d(ctx0, ape_f, layout.width, 1, ape_f->nb[1], pos_mod * ape_f->nb[1]));

            const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
            const bool should_compress = (pos_i + 1) % compress_ratio == 0;

            ggml_tensor * row_idx = dsv4_arange_i32(ctx0, row, row + 1);
            kv_state = ggml_set_rows(ctx0, kv_state, kv_cur, row_idx);
            score_state = ggml_set_rows(ctx0, score_state, sc_cur, row_idx);

            if (should_compress) {
                ggml_tensor * kv_pool;
                ggml_tensor * score_pool;

                if (compress_ratio == 4) {
                    ggml_tensor * kv_prev = dsv4_view_cols(ctx0, kv_state, head_dim, compress_ratio, 0, 0);
                    ggml_tensor * kv_curr = dsv4_view_cols(ctx0, kv_state, head_dim, compress_ratio, head_dim, compress_ratio);
                    ggml_tensor * sc_prev = dsv4_view_cols(ctx0, score_state, head_dim, compress_ratio, 0, 0);
                    ggml_tensor * sc_curr = dsv4_view_cols(ctx0, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

                    kv_pool = ggml_concat(ctx0, kv_prev, kv_curr, 1);
                    score_pool = ggml_concat(ctx0, sc_prev, sc_curr, 1);

                    ggml_tensor * shifted_kv = dsv4_view_cols(ctx0, kv_state, layout.width, compress_ratio, 0, compress_ratio);
                    ggml_tensor * shifted_score = dsv4_view_cols(ctx0, score_state, layout.width, compress_ratio, 0, compress_ratio);
                    kv_state = ggml_concat(ctx0, shifted_kv, shifted_kv, 1);
                    score_state = ggml_concat(ctx0, shifted_score, shifted_score, 1);
                } else {
                    kv_pool = kv_state;
                    score_pool = score_state;
                }

                ggml_tensor * comp_pos = dsv4_arange_i32(ctx0, pos_i + 1 - compress_ratio, pos_i + 2 - compress_ratio);
                ggml_tensor * cur_comp = dsv4_pool_decode_state(kv_pool, score_pool, norm, comp_pos, head_dim, rope_cfg);
                kv_comp = kv_comp == nullptr ? cur_comp : ggml_concat(ctx0, kv_comp, cur_comp, 2);
            }
        }

        return { kv_state, score_state, kv_comp };
    };

    auto dsv4_build_compressor_prefill_ratio4_replay = [&](ggml_tensor * x, ggml_tensor * prev_kv_state, ggml_tensor * prev_score_state,
                                                            ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm,
                                                            int64_t head_dim, llama_pos first_pos, const dsv4_rope_cfg & rope_cfg,
                                                            const char * tag, int il_cur) -> dsv4_decode_compressor {
        // Fast path for aligned continuation prefill chunks. This mirrors the ds4
        // Metal `compressor_prefill_ratio4_replay` path: project the whole chunk,
        // seed the first pooled compressed row from the previous compressor state,
        // then pool all compressed rows in parallel instead of emitting one
        // SET_ROWS/softmax/RoPE mini-graph per token.
        constexpr int64_t ratio = 4;
        GGML_ASSERT(n_tokens > 0 && n_tokens % ratio == 0);
        GGML_ASSERT(first_pos > 0 && first_pos % ratio == 0);

        const dsv4_state_layout layout = dsv4_make_state_layout(ratio, head_dim);
        const int64_t n_comp = n_tokens / ratio;

        ggml_tensor * kv_all = ggml_mul_mat(ctx0, wkv, x);   // [2*head_dim, n_tokens]
        ggml_tensor * sc_all = ggml_mul_mat(ctx0, wgate, x);
        ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx0, ape, GGML_TYPE_F32);

        ggml_tensor * kv_3d = ggml_reshape_3d(ctx0, kv_all, layout.width, ratio, n_comp);
        ggml_tensor * sc_3d = ggml_reshape_3d(ctx0, sc_all, layout.width, ratio, n_comp);
        ggml_tensor * ape_3d = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, ape_f, layout.width, ratio, 1), sc_3d);
        sc_3d = ggml_add(ctx0, sc_3d, ape_3d);

        ggml_tensor * kv_a = ggml_view_3d(ctx0, kv_3d, head_dim, ratio, n_comp, kv_3d->nb[1], kv_3d->nb[2], 0);
        ggml_tensor * kv_b = ggml_view_3d(ctx0, kv_3d, head_dim, ratio, n_comp, kv_3d->nb[1], kv_3d->nb[2], kv_3d->nb[0] * head_dim);
        ggml_tensor * sc_a = ggml_view_3d(ctx0, sc_3d, head_dim, ratio, n_comp, sc_3d->nb[1], sc_3d->nb[2], 0);
        ggml_tensor * sc_b = ggml_view_3d(ctx0, sc_3d, head_dim, ratio, n_comp, sc_3d->nb[1], sc_3d->nb[2], sc_3d->nb[0] * head_dim);

        ggml_tensor * kv_seed = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, dsv4_view_cols(ctx0, prev_kv_state, head_dim, ratio, 0, 0)), head_dim, ratio, 1);
        ggml_tensor * sc_seed = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, dsv4_view_cols(ctx0, prev_score_state, head_dim, ratio, 0, 0)), head_dim, ratio, 1);

        ggml_tensor * kv_a_shift = kv_seed;
        ggml_tensor * sc_a_shift = sc_seed;
        if (n_comp > 1) {
            ggml_tensor * kv_a_prev = ggml_view_3d(ctx0, kv_a, head_dim, ratio, n_comp - 1, kv_a->nb[1], kv_a->nb[2], 0);
            ggml_tensor * sc_a_prev = ggml_view_3d(ctx0, sc_a, head_dim, ratio, n_comp - 1, sc_a->nb[1], sc_a->nb[2], 0);
            kv_a_shift = ggml_concat(ctx0, kv_seed, kv_a_prev, 2);
            sc_a_shift = ggml_concat(ctx0, sc_seed, sc_a_prev, 2);
        }

        ggml_tensor * kv_ov = ggml_concat(ctx0, kv_a_shift, kv_b, 1);       // [head_dim, 8, n_comp]
        ggml_tensor * sc_ov = ggml_concat(ctx0, sc_a_shift, sc_b, 1);       // [head_dim, 8, n_comp]

        ggml_tensor * sc_s = ggml_permute(ctx0, sc_ov, 1, 0, 2, 3);         // [8, head_dim, n_comp]
        ggml_tensor * w = ggml_soft_max(ctx0, cont_if_needed(ctx0, sc_s));
        w = ggml_permute(ctx0, w, 1, 0, 2, 3);                              // [head_dim, 8, n_comp]

        ggml_tensor * pooled = ggml_mul(ctx0, kv_ov, w);
        pooled = ggml_permute(ctx0, pooled, 1, 0, 2, 3);                    // [8, head_dim, n_comp]
        pooled = cont_if_needed(ctx0, ggml_cast(ctx0, pooled, GGML_TYPE_F32));
        pooled = ggml_sum_rows(ctx0, pooled);                              // [1, head_dim, n_comp]
        pooled = ggml_reshape_2d(ctx0, pooled, head_dim, n_comp);
        pooled = ggml_rms_norm(ctx0, pooled, hparams.f_norm_rms_eps);
        pooled = ggml_mul(ctx0, pooled, norm);

        ggml_tensor * kv_comp = ggml_reshape_3d(ctx0, pooled, head_dim, 1, n_comp);
        ggml_tensor * comp_pos = ggml_cast(ctx0, ggml_arange(ctx0, (float) first_pos, (float) (first_pos + n_tokens), (float) ratio), GGML_TYPE_I32);
        kv_comp = apply_partial_rope_with_pos(kv_comp, comp_pos, rope_cfg, /*inverse=*/false);
        cb(kv_comp, tag, il_cur);

        // Post-prefill recurrent state: rows 0..3 contain the final full window;
        // rows 4..7 are clear until subsequent decode writes the next current
        // half. This matches ds4's prefill-state finalization.
        ggml_tensor * kv_last = ggml_view_2d(ctx0, kv_3d, layout.width, ratio, kv_3d->nb[1], (n_comp - 1) * kv_3d->nb[2]);
        ggml_tensor * sc_last = ggml_view_2d(ctx0, sc_3d, layout.width, ratio, sc_3d->nb[1], (n_comp - 1) * sc_3d->nb[2]);
        ggml_tensor * kv_empty = ggml_fill(ctx0, cont_if_needed(ctx0, kv_last), 0.0f);
        ggml_tensor * sc_empty = ggml_fill(ctx0, cont_if_needed(ctx0, sc_last), -INFINITY);

        return {
            ggml_concat(ctx0, kv_last, kv_empty, 1),
            ggml_concat(ctx0, sc_last, sc_empty, 1),
            kv_comp,
        };
    };

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * hc_state = make_hc_state(inpL);
    cb(hc_state, "hc_state_init", -1);

    // Grouped-output identity ids: identical across all layers (depend only on
    // n_o_groups and n_tokens). Build once before the loop so the scheduler
    // sees a single tensor instead of one per layer (43 layers × 3 ops = 129
    // redundant graph nodes), reducing graph splits and scheduler bookkeeping.
    ggml_tensor * grouped_out_ids = nullptr;
    {
        const int64_t n_groups = hparams.n_o_groups;
        if (n_groups > 0) {
            ggml_tensor * ids = ggml_arange(ctx0, 0.0f, (float) n_groups, 1.0f);
            ids = ggml_cast(ctx0, ids, GGML_TYPE_I32);
            grouped_out_ids = ggml_repeat_4d(ctx0, ids, n_groups, n_tokens, 1, 1);
            cb(grouped_out_ids, "grouped_out_ids", -1);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        hc_pre_result attn_mix = hc_pre(hc_state, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_base, model.layers[il].hc_attn_scale, "attn_hc_pre", il);

        ggml_tensor * attn_inp = model.layers[il].attn_norm != nullptr
            ? build_norm(as_f32(attn_mix.collapsed), model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il)
            : attn_mix.collapsed;
        attn_inp = as_f32(attn_inp);
        cb(attn_inp, "attn_norm", il);

        const uint32_t compress_ratio = hparams.deepseek4_compress_ratios[il];
        const auto rope_cfg = get_rope_cfg(compress_ratio);
        const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;

        ggml_tensor * attn_inp_wq_a_src = hparams.deepseek4_fp8_wq_a ? ggml_cast(ctx0, attn_inp, GGML_TYPE_BF16) : attn_inp;
        ggml_tensor * attn_inp_wq_a = apply_dense_fp8_qat(attn_inp_wq_a_src, hparams.deepseek4_fp8_wq_a, "attn_inp_wq_a_fp8_qat", il);

        ggml_tensor * q_residual = mul_mat_checked(model.layers[il].wq_a, attn_inp_wq_a, "attn.wq_a");
        q_residual = cast_dense_fp8_out(q_residual, hparams.deepseek4_fp8_wq_a);
        cb(q_residual, "attn_q_a", il);

        q_residual = as_f32(q_residual);
        q_residual = build_norm(q_residual, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q_residual, "attn_q_a_norm", il);

        ggml_tensor * q_residual_fp8_src = hparams.deepseek4_fp8_wq_b ? ggml_cast(ctx0, q_residual, GGML_TYPE_BF16) : q_residual;
        ggml_tensor * q_residual_fp8 = apply_dense_fp8_qat(q_residual_fp8_src, hparams.deepseek4_fp8_wq_b, "attn_q_a_norm_fp8_qat", il);
        ggml_tensor * Qcur = mul_mat_checked(model.layers[il].wq_b, q_residual_fp8, "attn.wq_b");
        Qcur = cast_dense_fp8_out(Qcur, hparams.deepseek4_fp8_wq_b);
        cb(Qcur, "attn_q_b", il);

        Qcur = as_f32(Qcur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
        cb(Qcur, "attn_q_norm_perhead", il);
        Qcur = apply_partial_rope(Qcur, rope_cfg, /*inverse=*/false);
        cb(Qcur, "attn_q_roped", il);

        // KV path: matches antirez — matmul, norm, reshape3d, rope_tail, fp8_kv_quantize.
        // Order: rope BEFORE fp8 quantize because rope only rotates rope-dim region;
        // fp8_kv_quantize must operate on the post-rope rope-dim values unchanged.
        ggml_tensor * attn_inp_wkv_src = hparams.deepseek4_fp8_wkv ? ggml_cast(ctx0, attn_inp, GGML_TYPE_BF16) : attn_inp;
        ggml_tensor * attn_inp_wkv = apply_dense_fp8_qat(attn_inp_wkv_src, hparams.deepseek4_fp8_wkv, "attn_inp_wkv_fp8_qat", il);
        ggml_tensor * KVcur = mul_mat_checked(model.layers[il].wkv_a_mqa, attn_inp_wkv, "attn.wkv");
        KVcur = cast_dense_fp8_out(KVcur, hparams.deepseek4_fp8_wkv);
        cb(KVcur, "attn_wkv", il);

        KVcur = as_f32(KVcur);
        KVcur = build_norm(KVcur, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(KVcur, "attn_kv_norm", il);

        ggml_tensor * Kcur = ggml_reshape_3d(ctx0, KVcur, n_embd_head, 1, n_tokens);
        Kcur = apply_partial_rope(Kcur, rope_cfg, /*inverse=*/false);
        cb(Kcur, "attn_k_roped", il);
        // FP8 KV cache simulation matching antirez: quantizes nope dims, leaves rope dims alone
        Kcur = as_f32(Kcur);
        Kcur = apply_fp8_qat_nope_2d(Kcur, "attn_kv_qat", il);

        ggml_tensor * Vcur = Kcur;

        ggml_tensor * idx_prior = nullptr;
        ggml_tensor * idx_sel = nullptr;
        ggml_tensor * c_pool = nullptr;
        ggml_tensor * idx_pool = nullptr;

        ggml_tensor * prev_kv_state_all = nullptr;
        ggml_tensor * prev_sc_state_all = nullptr;
        dsv4_state_layout attn_state_layout = { 0, 0, 0 };
        dsv4_state_layout index_state_layout = { 0, 0, 0 };

        auto store_attn_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
            if (!has_hybrid_iswa || mctx_dsv4 == nullptr) {
                return;
            }
            for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
            }
        };

        auto store_index_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
            if (!has_hybrid_iswa || mctx_dsv4 == nullptr) {
                return;
            }
            for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
            }
        };

        const int64_t n_seqs       = ubatch.n_seqs;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const llama_pos first_pos_batch = ubatch.pos ? ubatch.pos[0] : 0;
        const bool use_local_prompt_attn = is_prefill && n_tokens > 1 && first_pos_batch == 0;

        if (compress_ratio > 0 && model.layers[il].attn_compressor_wkv != nullptr) {
            if (has_hybrid_iswa) {
                const int64_t state_size = hparams.n_embd_r();
                attn_state_layout = dsv4_make_state_layout(compress_ratio, n_embd_head);
                prev_kv_state_all = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), state_size, ubatch.n_seqs);
                prev_sc_state_all = build_rs(inp_rs, inp_rs->mctx->get_s_l(il), state_size, ubatch.n_seqs);

                if (is_prefill) {
                    // For multi-seq, reshape attn_inp to {n_embd, n_seq_tokens, n_seqs}
                    // so the compressor processes each sequence independently.
                    ggml_tensor * attn_inp_3d = (n_seqs > 1)
                        ? ggml_reshape_3d(ctx0, attn_inp, n_embd, n_seq_tokens, n_seqs)
                        : attn_inp;
                    dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, attn_inp_3d,
                            model.layers[il].attn_compressor_wkv,
                            model.layers[il].attn_compressor_wgate,
                            model.layers[il].attn_compressor_ape,
                            n_embd_head,
                            n_seq_tokens,
                            compress_ratio,
                            n_seqs);
                    dsv4_store_state_segment(ctx0, gf, state.kv, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0, n_seqs);
                    dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0, n_seqs);
                }
            }

            if (use_local_prompt_attn) {
                const int64_t compressor_dim = n_embd_head * (compress_ratio == 4 ? 2 : 1);
                c_pool = build_compressed_pool(
                        attn_inp,
                        model.layers[il].attn_compressor_wkv,
                        model.layers[il].attn_compressor_wgate,
                        model.layers[il].attn_compressor_ape,
                        model.layers[il].attn_compressor_norm,
                        compressor_dim,
                        n_embd_head,
                        compress_ratio,
                        "attn_compressor_pool",
                        il);

                if (c_pool != nullptr) {
                    cb(c_pool, "attn_compressor_pool_ready", il);
                }
            }

            if (compress_ratio == 4 &&
                model.layers[il].attn_indexer_q_b != nullptr &&
                model.layers[il].attn_indexer_weights_proj != nullptr &&
                model.layers[il].attn_indexer_compressor_wkv != nullptr) {
                if (has_hybrid_iswa) {
                    index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.n_embd_head_index);
                    if (is_prefill) {
                        const int64_t state_size = hparams.n_embd_r();
                        ggml_tensor * attn_inp_3d = (n_seqs > 1)
                            ? ggml_reshape_3d(ctx0, attn_inp, n_embd, n_seq_tokens, n_seqs)
                            : attn_inp;
                        dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, attn_inp_3d,
                                model.layers[il].attn_indexer_compressor_wkv,
                                model.layers[il].attn_indexer_compressor_wgate,
                                model.layers[il].attn_indexer_compressor_ape,
                                hparams.n_embd_head_index,
                                n_seq_tokens,
                                compress_ratio,
                                n_seqs);
                        dsv4_store_state_segment(ctx0, gf, index_state.kv, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);
                        dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);
                    }
                }
                GGML_ASSERT(model.layers[il].attn_indexer_q_b->ne[0] == hparams.n_lora_q);
                GGML_ASSERT(model.layers[il].attn_indexer_q_b->ne[1] == (int64_t) hparams.n_head_index * hparams.n_embd_head_index);
                GGML_ASSERT(model.layers[il].attn_indexer_weights_proj->ne[0] == n_embd);
                GGML_ASSERT(model.layers[il].attn_indexer_weights_proj->ne[1] == hparams.n_head_index);

                if (use_local_prompt_attn) {
                    const int64_t idx_dim = hparams.n_embd_head_index;
                    const int64_t idx_comp_dim = idx_dim * 2;
                    idx_pool = build_compressed_pool(
                            attn_inp,
                            model.layers[il].attn_indexer_compressor_wkv,
                            model.layers[il].attn_indexer_compressor_wgate,
                            model.layers[il].attn_indexer_compressor_ape,
                            model.layers[il].attn_indexer_compressor_norm,
                            idx_comp_dim,
                            idx_dim,
                            compress_ratio,
                            "attn_indexer_pool",
                            il);
                    if (idx_pool != nullptr) {
                        cb(idx_pool, "attn_indexer_pool_ready", il);
                    }
                }
            }
        }

        const uint32_t prompt_window_size = hparams.deepseek4_sliding_window > 0 ? hparams.deepseek4_sliding_window : (uint32_t) n_tokens;
        bool use_prompt_sparse_attn = false;
        // Unified sparse-gather attention path for prompt eval.
        // Active iff this is the first prompt chunk, the layer has a compressor +
        // indexer, and there is at least one compressed position to gather from.
        // Later prompt chunks must use the cache-aware decode/replay path below;
        // otherwise they lose all context from previous chunks.
        const bool can_use_sparse_prompt =
            use_local_prompt_attn &&
            c_pool != nullptr && c_pool->ne[1] > 0 &&
            idx_pool != nullptr &&
            model.layers[il].attn_indexer_q_b != nullptr &&
            model.layers[il].attn_indexer_weights_proj != nullptr &&
            hparams.n_attn_index_topk > 0;

        ggml_tensor * prompt_comp_pos = nullptr;
        ggml_tensor * prompt_c_attn = nullptr;
        const int64_t n_comp_prompt = (use_local_prompt_attn && !can_use_sparse_prompt && c_pool != nullptr) ? c_pool->ne[1] : 0;
        if (n_comp_prompt > 0) {
            ggml_tensor * comp_pos_idx = cache_mask(
                "comp_pos_idx:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp_prompt),
                [&]() {
                    std::vector<int32_t> comp_rows(n_comp_prompt);
                    for (int64_t i = 0; i < n_comp_prompt; ++i) {
                        comp_rows[i] = int32_t(i * compress_ratio);
                    }
                    ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_comp_prompt);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_comp_pos_idx");
                    res->add_input(std::make_unique<dsv4_static_i32_input>(t, comp_rows));
                    return t;
                });

            ggml_tensor * inp_pos_2d = ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens);
            prompt_comp_pos = ggml_get_rows(ctx0, inp_pos_2d, comp_pos_idx);
            prompt_comp_pos = ggml_reshape_1d(ctx0, prompt_comp_pos, n_comp_prompt);
            cb(prompt_comp_pos, "attn_comp_pos", il);

            prompt_c_attn = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(c_pool)), n_embd_head, 1, n_comp_prompt);
            prompt_c_attn = apply_partial_rope_with_pos(prompt_c_attn, prompt_comp_pos, rope_cfg, /*inverse=*/false);
            prompt_c_attn = apply_fp8_qat_nope_2d(prompt_c_attn, "attn_compressor_pool_qat", il);
            cb(prompt_c_attn, "attn_compressor_pool_roped", il);
            store_attn_cache_rows(prompt_c_attn, 0, n_comp_prompt);
        }

        if (can_use_sparse_prompt) {
            if (dsv4_debug_paths_enabled()) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                LLAMA_LOG_WARN("deepseek4/path: layer=%d path=prompt-sparse n_tokens=%lld n_seq_tokens=%lld first_pos=%d last_pos=%d is_prefill=%d compress_ratio=%u\n",
                        il, (long long) n_tokens, (long long) n_seq_tokens, first_pos, last_pos, (int) is_prefill, compress_ratio);
            }
            const int64_t n_comp = c_pool->ne[1];
            const int64_t idx_dim = hparams.n_embd_head_index;
            const int64_t n_index_heads = hparams.n_head_index;
            const int32_t n_idx_topk = std::min<int32_t>((int32_t) hparams.n_attn_index_topk, (int32_t) n_comp);

            // ----- compressed positions table (i*compress_ratio for i in [0, n_comp)) -----
            ggml_tensor * comp_pos_idx = cache_mask(
                "comp_pos_idx:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp),
                [&]() {
                    std::vector<int32_t> comp_rows(n_comp);
                    for (int64_t i = 0; i < n_comp; ++i) {
                        comp_rows[i] = int32_t(i * compress_ratio);
                    }
                    ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_comp);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_comp_pos_idx");
                    res->add_input(std::make_unique<dsv4_static_i32_input>(t, comp_rows));
                    return t;
                });

            ggml_tensor * inp_pos_2d = ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens);
            ggml_tensor * comp_pos = ggml_get_rows(ctx0, inp_pos_2d, comp_pos_idx);
            comp_pos = ggml_reshape_1d(ctx0, comp_pos, n_comp);
            cb(comp_pos, "attn_comp_pos", il);

            // ----- compressed attn pool: rope FIRST, THEN fp8 quantize, store to DSv4 attn_k cache -----
            ggml_tensor * c_attn = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(c_pool)), n_embd_head, 1, n_comp);
            c_attn = apply_partial_rope_with_pos(c_attn, comp_pos, rope_cfg, /*inverse=*/false);
            c_attn = apply_fp8_qat_nope_2d(c_attn, "attn_compressor_pool_qat", il);
            cb(c_attn, "attn_compressor_pool_roped", il);
            store_attn_cache_rows(c_attn, 0, n_comp);

            // ----- indexer compressor pool: rope only, no fp8 quantize, store to DSv4 index_k cache -----
            ggml_tensor * idx_comp = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(idx_pool)), idx_dim, 1, n_comp);
            idx_comp = apply_partial_rope_with_pos(idx_comp, comp_pos, rope_cfg, /*inverse=*/false);
            idx_comp = ggml_reshape_2d(ctx0, cont_if_needed(ctx0, idx_comp), idx_dim, n_comp);
            cb(idx_comp, "attn_indexer_pool_roped", il);
            store_index_cache_rows(idx_comp, 0, n_comp);

            // ----- indexer scores: Q (idx_q_b @ residual + rope) dot K (idx_comp), summed over heads -----
            ggml_tensor * idx_kv_3d = ggml_reshape_3d(ctx0, idx_comp, idx_dim, 1, n_comp);

            ggml_tensor * idx_q_inp_src = hparams.deepseek4_fp8_indexer_q ? ggml_cast(ctx0, q_residual, GGML_TYPE_BF16) : q_residual;
            ggml_tensor * idx_q_inp = apply_dense_fp8_qat(idx_q_inp_src, hparams.deepseek4_fp8_indexer_q, "attn_indexer_q_inp_fp8_qat", il);
            ggml_tensor * idx_q = mul_mat_checked(model.layers[il].attn_indexer_q_b, idx_q_inp, "indexer.q_b");
            idx_q = cast_dense_fp8_out(idx_q, hparams.deepseek4_fp8_indexer_q);
            idx_q = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(idx_q)), idx_dim, n_index_heads, n_tokens);
            idx_q = apply_partial_rope(idx_q, rope_cfg, /*inverse=*/false);
            cb(idx_q, "attn_indexer_q_roped", il);

            ggml_tensor * k_perm = ggml_permute(ctx0, idx_kv_3d, 0, 2, 1, 3);  // [idx_dim, n_comp, 1]
            ggml_tensor * q_perm = ggml_permute(ctx0, idx_q,     0, 2, 1, 3);  // [idx_dim, n_tokens, n_index_heads]
            ggml_tensor * idx_scores = ggml_mul_mat(ctx0, k_perm, q_perm);     // [n_comp, n_tokens, n_index_heads]
            idx_scores = ggml_relu(ctx0, idx_scores);

            ggml_tensor * idx_weights_inp = ggml_cast(ctx0, attn_inp, GGML_TYPE_BF16);
            ggml_tensor * idx_weights = mul_mat_checked(model.layers[il].attn_indexer_weights_proj, idx_weights_inp, "indexer.weights_proj");
            idx_weights = as_f32(idx_weights);
            const float idx_scale_f = 1.0f / sqrtf((float)idx_dim * (float)n_index_heads);
            idx_weights = ggml_scale(ctx0, idx_weights, idx_scale_f);
            idx_weights = ggml_reshape_3d(ctx0, idx_weights, 1, n_index_heads, n_tokens);
            idx_weights = ggml_permute(ctx0, idx_weights, 0, 2, 1, 3);
            cb(idx_weights, "attn_indexer_weights", il);

            idx_scores = ggml_mul(ctx0, idx_scores, idx_weights);
            idx_scores = cont_if_needed(ctx0, ggml_permute(ctx0, idx_scores, 1, 2, 0, 3));
            idx_scores = ggml_sum_rows(ctx0, idx_scores);
            idx_scores = ggml_reshape_2d(ctx0, idx_scores, n_comp, n_tokens);
            cb(idx_scores, "attn_indexer_scores", il);

            // Add a per-(comp,token) -inf mask that hides positions ahead of the current token.
            // This is causal-only (no interspersed -inf), so it is safe even before argsort.
            ggml_tensor * idx_valid_mask = cache_mask(
                "prompt_idx_valid:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp) + ":" + std::to_string((int)cparams.causal_attn),
                [&]() {
                    ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_comp, n_tokens);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_prompt_comp_valid_mask");
                    res->add_input(std::make_unique<dsv4_prompt_comp_mask_input>(t, n_comp, compress_ratio, false, cparams.causal_attn));
                    return t;
                });
            idx_scores = ggml_add(ctx0, idx_scores, idx_valid_mask);
            idx_prior = idx_scores;
            cb(idx_prior, "attn_indexer_prior", il);

            idx_sel = ggml_argsort_top_k(ctx0, idx_scores, n_idx_topk);
            cb(idx_sel, "attn_indexer_topk", il);

            // ----- ensure the standard SWA mctx writes for Kcur/Vcur happen so future
            //       decode steps can read this prompt's window from the cache. -----
            if (has_hybrid_iswa) {
                ggml_build_forward_expand(gf, inp_attn_iswa->self_kq_mask_swa);
                const auto * mctx_swa = inp_attn_iswa->mctx->get_swa();
                ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, Kcur, inp_attn_iswa->get_k_idxs_swa(), il));
                ggml_build_forward_expand(gf, mctx_swa->cpy_v(ctx0, Vcur, inp_attn_iswa->get_v_idxs_swa(), il));
            } else {
                ggml_build_forward_expand(gf, inp_attn_kv->get_kq_mask());
                ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_k(ctx0, Kcur, inp_attn_kv->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_v(ctx0, Vcur, inp_attn_kv->get_v_idxs(), il));
            }

            // ----- causal window mask for prompt eval -----
            // Without this, query token t would attend to future tokens t+1..n_tokens-1
            // in the same prompt batch (i.e. broken causality).
            ggml_tensor * window_mask = cache_mask(
                "prompt_window_causal:" + std::to_string(prompt_window_size) + ":" + std::to_string((int)cparams.causal_attn),
                [&]() {
                    ggml_tensor * t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_prompt_window_mask");
                    res->add_input(std::make_unique<dsv4_prompt_window_mask_input>(t, prompt_window_size, hparams.use_alibi, cparams.causal_attn));
                    return t;
                });

            // ----- unified sparse-gather attention -----
            ggml_tensor * sink = model.layers[il].attn_sinks;
            if (sink && sink->type != GGML_TYPE_F32) {
                sink = ggml_cast(ctx0, sink, GGML_TYPE_F32);
            }
            cur = ggml_dsv4_sparse_attn(
                ctx0,
                Qcur,
                c_attn,                 // [head_dim, 1, n_comp] (just-built compressed pool)
                Kcur,                   // [head_dim, 1, n_tokens] (window = current prompt tokens)
                window_mask,            // [n_tokens, n_tokens] causal+SWA mask
                idx_sel,                // [n_idx_topk, n_tokens]
                sink,                   // [n_heads]
                kq_scale,
                cparams.causal_attn ? (int32_t) prompt_window_size : 0);
            cb(cur, "attn_sparse_prompt", il);
            use_prompt_sparse_attn = true;
        } else if (prompt_c_attn != nullptr) {
            if (dsv4_debug_paths_enabled()) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                LLAMA_LOG_WARN("deepseek4/path: layer=%d path=prompt-compressed n_tokens=%lld n_seq_tokens=%lld first_pos=%d last_pos=%d is_prefill=%d compress_ratio=%u\n",
                        il, (long long) n_tokens, (long long) n_seq_tokens, first_pos, last_pos, (int) is_prefill, compress_ratio);
            }

            if (has_hybrid_iswa) {
                ggml_build_forward_expand(gf, inp_attn_iswa->self_kq_mask_swa);
                const auto * mctx_swa = inp_attn_iswa->mctx->get_swa();
                ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, Kcur, inp_attn_iswa->get_k_idxs_swa(), il));
                ggml_build_forward_expand(gf, mctx_swa->cpy_v(ctx0, Vcur, inp_attn_iswa->get_v_idxs_swa(), il));
            } else {
                ggml_build_forward_expand(gf, inp_attn_kv->get_kq_mask());
                ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_k(ctx0, Kcur, inp_attn_kv->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_v(ctx0, Vcur, inp_attn_kv->get_v_idxs(), il));
            }

            ggml_tensor * raw_mask = cache_mask(
                "prompt_window_causal:" + std::to_string(prompt_window_size) + ":" + std::to_string((int)cparams.causal_attn),
                [&]() {
                    ggml_tensor * t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_prompt_window_mask");
                    res->add_input(std::make_unique<dsv4_prompt_window_mask_input>(t, prompt_window_size, hparams.use_alibi, cparams.causal_attn));
                    return t;
                });
            ggml_tensor * comp_mask = cache_mask(
                "prompt_comp_causal:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp_prompt) + ":" + std::to_string((int)cparams.causal_attn),
                [&]() {
                    ggml_tensor * t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_comp_prompt, n_tokens, 1, 1);
                    ggml_set_input(t);
                    ggml_set_name(t, "dsv4_prompt_comp_mask");
                    res->add_input(std::make_unique<dsv4_prompt_comp_mask_input>(t, n_comp_prompt, compress_ratio, hparams.use_alibi, cparams.causal_attn));
                    return t;
                });
            ggml_tensor * attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
            // DeepSeek4 raw+compressed prompt attention uses a concatenated
            // F32 mask. The FlashAttention path produced non-finite logits for
            // long prompts here; keep this path on the standard F32 softmax MHA.
            ggml_tensor * attn_mask_cnv = attn_mask;
            ggml_tensor * k_all = ggml_concat(ctx0, Kcur, prompt_c_attn, 2);

            ggml_tensor * sink = model.layers[il].attn_sinks;
            if (sink && sink->type != GGML_TYPE_F32) {
                sink = ggml_cast(ctx0, sink, GGML_TYPE_F32);
            }

            cur = build_attn_mha(Qcur, k_all, k_all, nullptr, attn_mask_cnv, sink, nullptr, kq_scale, il, true);
            cb(cur, "attn_compressed_prompt", il);
            use_prompt_sparse_attn = true;
        }

        // -----------------------------------------------------------------------------
        // Decode-time compressor + indexer cache updates.
        //
        // These run ALWAYS when the layer has a compressor (regardless of whether the
        // sparse-attention path is used this step) so that future decode steps can read
        // newly-compressed positions from the DSv4 attn_k / index_k caches.
        //
        // We then conditionally route attention through ggml_dsv4_sparse_attn when there
        // is at least one compressed position to gather from. Otherwise we leave
        // use_prompt_sparse_attn=false and let the standard build_attn() path handle the
        // window-only attention (no -inf mask, no FA bug).
        // -----------------------------------------------------------------------------
        if (!use_prompt_sparse_attn && has_hybrid_iswa && compress_ratio > 0 && model.layers[il].attn_compressor_wkv != nullptr) {
            // For multi-seq decode, each token belongs to a different sequence.
            // We use seq_id from token 0; for n_seqs > 1 with equal_seqs,
            // all tokens within a sub-batch share the same seq_id.
            const llama_seq_id seq_id = ubatch.seq_id[0][0];
            const int64_t state_size = hparams.n_embd_r();

            ggml_tensor * prev_attn_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows);
            ggml_tensor * prev_attn_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows);

            const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
            const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
            const int64_t n_comp_before  = first_pos / (llama_pos) compress_ratio;
            const int64_t n_comp_visible = (last_pos + 1) / (llama_pos) compress_ratio;

            bool can_use_vectorized_replay =
                compress_ratio == 4 &&
                n_tokens > 1 &&
                n_seqs == 1 &&
                ubatch.n_seq_id[0] == 1 &&
                ubatch.pos != nullptr &&
                first_pos > 0 &&
                first_pos % (llama_pos) compress_ratio == 0 &&
                n_tokens % (int64_t) compress_ratio == 0;
            if (can_use_vectorized_replay) {
                for (int64_t i = 1; i < n_tokens; ++i) {
                    if (ubatch.pos[i] != first_pos + (llama_pos) i) {
                        can_use_vectorized_replay = false;
                        break;
                    }
                }
            }
            if (dsv4_debug_paths_enabled()) {
                LLAMA_LOG_WARN("deepseek4/path: layer=%d path=%s n_tokens=%lld n_seq_tokens=%lld first_pos=%d last_pos=%d is_prefill=%d compress_ratio=%u\n",
                        il, can_use_vectorized_replay ? "prefill-replay" : "decode-replay",
                        (long long) n_tokens, (long long) n_seq_tokens, first_pos, last_pos, (int) is_prefill, compress_ratio);
            }

            dsv4_decode_compressor dec = n_tokens == 1
                    ? dsv4_build_compressor_decode(attn_inp, prev_attn_kv_state, prev_attn_sc_state,
                            model.layers[il].attn_compressor_wkv,
                            model.layers[il].attn_compressor_wgate,
                            model.layers[il].attn_compressor_ape,
                            model.layers[il].attn_compressor_norm,
                            n_embd_head, first_pos, compress_ratio, rope_cfg)
                    : can_use_vectorized_replay
                        ? dsv4_build_compressor_prefill_ratio4_replay(attn_inp, prev_attn_kv_state, prev_attn_sc_state,
                                model.layers[il].attn_compressor_wkv,
                                model.layers[il].attn_compressor_wgate,
                                model.layers[il].attn_compressor_ape,
                                model.layers[il].attn_compressor_norm,
                                n_embd_head, first_pos, rope_cfg, "attn_compressor_replay_prefill", il)
                        : dsv4_build_compressor_decode_chunk(attn_inp, prev_attn_kv_state, prev_attn_sc_state,
                                model.layers[il].attn_compressor_wkv,
                                model.layers[il].attn_compressor_wgate,
                                model.layers[il].attn_compressor_ape,
                                model.layers[il].attn_compressor_norm,
                                n_embd_head, compress_ratio, rope_cfg);

            dsv4_store_state_segment(ctx0, gf, dec.kv_state, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0, n_seqs);
            dsv4_store_state_segment(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0, n_seqs);

            ggml_tensor * kv_comp_new_for_attn = nullptr;
            const int64_t n_comp_new = n_comp_visible - n_comp_before;
            if (dec.kv_comp != nullptr && n_comp_new > 0) {
                // Match antirez and the prompt path: compressed attention KV rows are
                // partial-RoPE'd first, then FP8-quantized on the nope prefix before
                // entering the compressed cache. The indexer cache below intentionally
                // remains BF16/F32 and must not use this quantization step.
                kv_comp_new_for_attn = apply_fp8_qat_nope_2d(dec.kv_comp, "attn_compressor_decode_qat", il);
                dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), kv_comp_new_for_attn, n_comp_before, n_comp_new);
            }

            if (compress_ratio == 4 && model.layers[il].attn_indexer_compressor_wkv != nullptr && index_state_layout.elems > 0) {
                ggml_tensor * prev_index_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                        attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                ggml_tensor * prev_index_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                        attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                dsv4_decode_compressor idx_dec = n_tokens == 1
                        ? dsv4_build_compressor_decode(attn_inp, prev_index_kv_state, prev_index_sc_state,
                                model.layers[il].attn_indexer_compressor_wkv,
                                model.layers[il].attn_indexer_compressor_wgate,
                                model.layers[il].attn_indexer_compressor_ape,
                                model.layers[il].attn_indexer_compressor_norm,
                                hparams.n_embd_head_index, first_pos, compress_ratio, rope_cfg)
                        : can_use_vectorized_replay
                            ? dsv4_build_compressor_prefill_ratio4_replay(attn_inp, prev_index_kv_state, prev_index_sc_state,
                                    model.layers[il].attn_indexer_compressor_wkv,
                                    model.layers[il].attn_indexer_compressor_wgate,
                                    model.layers[il].attn_indexer_compressor_ape,
                                    model.layers[il].attn_indexer_compressor_norm,
                                    hparams.n_embd_head_index, first_pos, rope_cfg, "attn_indexer_replay_prefill", il)
                            : dsv4_build_compressor_decode_chunk(attn_inp, prev_index_kv_state, prev_index_sc_state,
                                    model.layers[il].attn_indexer_compressor_wkv,
                                    model.layers[il].attn_indexer_compressor_wgate,
                                    model.layers[il].attn_indexer_compressor_ape,
                                    model.layers[il].attn_indexer_compressor_norm,
                                    hparams.n_embd_head_index, compress_ratio, rope_cfg);

                dsv4_store_state_segment(ctx0, gf, idx_dec.kv_state, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);
                dsv4_store_state_segment(ctx0, gf, idx_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);

                if (idx_dec.kv_comp != nullptr && n_comp_visible > n_comp_before) {
                    ggml_tensor * idx_comp_new = idx_dec.kv_comp;
                    idx_comp_new = ggml_reshape_2d(ctx0, cont_if_needed(ctx0, idx_comp_new), hparams.n_embd_head_index, n_comp_visible - n_comp_before);

                    dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), idx_comp_new, n_comp_before, n_comp_visible - n_comp_before);
                }
            }

            // Sparse-attention path eligibility for decode:
            //   - compressor and indexer must both be present
            //   - there must be at least one compressed position to gather from
            const bool can_use_sparse_decode =
                n_comp_visible > 0 &&
                compress_ratio == 4 &&
                model.layers[il].attn_indexer_q_b != nullptr &&
                model.layers[il].attn_indexer_weights_proj != nullptr &&
                hparams.n_attn_index_topk > 0;

            if (can_use_sparse_decode) {
                // Manual SWA window write (mirrors what build_attn would do internally).
                const auto * mctx_swa = inp_attn_iswa->mctx->get_swa();
                ggml_tensor * k_store = mctx_swa->cpy_k(ctx0, Kcur, inp_attn_iswa->get_k_idxs_swa(), il);
                ggml_build_forward_expand(gf, k_store);

                ggml_tensor * k_raw_ref = mctx_swa->get_k(ctx0, il);
                ggml_tensor * k_raw = ggml_view_4d(ctx0, k_store,
                        k_raw_ref->ne[0], k_raw_ref->ne[1], k_raw_ref->ne[2], k_raw_ref->ne[3],
                        k_raw_ref->nb[1], k_raw_ref->nb[2], k_raw_ref->nb[3], k_raw_ref->view_offs);
                k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head, 1, k_raw->ne[2]);
                k_raw = as_f32(k_raw);

                // Build kv_comp_cache: concat any cached (older) compressed rows with the just-computed new ones.
                ggml_tensor * kv_comp_cache = nullptr;
                if (kv_comp_new_for_attn != nullptr && n_comp_new > 0) {
                    ggml_tensor * kv_new = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, kv_comp_new_for_attn), n_embd_head, 1, n_comp_new);
                    if (n_comp_before > 0) {
                        ggml_tensor * kv_old = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_before);
                        kv_comp_cache = ggml_concat(ctx0, as_f32(kv_old), as_f32(kv_new), 2);
                    } else {
                        kv_comp_cache = kv_new;
                    }
                } else {
                    kv_comp_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_visible);
                }
                kv_comp_cache = as_f32(kv_comp_cache);

                // ----- indexer Q-side: same prefill flow as prompt (handles n_tokens >= 1). -----
                const int64_t idx_dim = hparams.n_embd_head_index;
                const int64_t n_index_heads = hparams.n_head_index;
                const int32_t n_idx_topk = std::min<int32_t>((int32_t) hparams.n_attn_index_topk, (int32_t) n_comp_visible);

                ggml_tensor * index_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_visible);
                index_cache = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(index_cache)), idx_dim, 1, n_comp_visible);

                ggml_tensor * idx_q_inp_src = hparams.deepseek4_fp8_indexer_q ? ggml_cast(ctx0, q_residual, GGML_TYPE_BF16) : q_residual;
                ggml_tensor * idx_q_inp = apply_dense_fp8_qat(idx_q_inp_src, hparams.deepseek4_fp8_indexer_q, "attn_indexer_q_decode_inp_fp8_qat", il);
                ggml_tensor * idx_q = mul_mat_checked(model.layers[il].attn_indexer_q_b, idx_q_inp, "indexer.decode.q_b");
                idx_q = cast_dense_fp8_out(idx_q, hparams.deepseek4_fp8_indexer_q);
                idx_q = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, as_f32(idx_q)), idx_dim, n_index_heads, n_tokens);
                idx_q = apply_partial_rope(idx_q, rope_cfg, /*inverse=*/false);

                ggml_tensor * k_perm = ggml_permute(ctx0, index_cache, 0, 2, 1, 3);
                ggml_tensor * q_perm = ggml_permute(ctx0, idx_q,       0, 2, 1, 3);
                ggml_tensor * idx_scores = ggml_mul_mat(ctx0, k_perm, q_perm);
                idx_scores = ggml_relu(ctx0, idx_scores);

                ggml_tensor * idx_weights_inp = ggml_cast(ctx0, attn_inp, GGML_TYPE_BF16);
                ggml_tensor * idx_weights = mul_mat_checked(model.layers[il].attn_indexer_weights_proj, idx_weights_inp, "indexer.decode.weights_proj");
                idx_weights = as_f32(idx_weights);
                const float idx_scale_f = 1.0f / sqrtf((float)idx_dim * (float)n_index_heads);
                idx_weights = ggml_scale(ctx0, idx_weights, idx_scale_f);
                idx_weights = ggml_reshape_3d(ctx0, idx_weights, 1, n_index_heads, n_tokens);
                idx_weights = ggml_permute(ctx0, idx_weights, 0, 2, 1, 3);

                idx_scores = ggml_mul(ctx0, idx_scores, idx_weights);
                idx_scores = cont_if_needed(ctx0, ggml_permute(ctx0, idx_scores, 1, 2, 0, 3));
                idx_scores = ggml_sum_rows(ctx0, idx_scores);
                idx_scores = ggml_reshape_2d(ctx0, idx_scores, n_comp_visible, n_tokens);

                // Causal-only valid mask (no interspersed -inf): hides positions ahead of
                // the current token.
                ggml_tensor * idx_valid_mask = cache_mask(
                    "decode_idx_valid:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp_visible) + ":" + std::to_string((int)cparams.causal_attn),
                    [&]() {
                        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_comp_visible, n_tokens);
                        ggml_set_input(t);
                        ggml_set_name(t, "dsv4_decode_comp_valid_mask");
                        res->add_input(std::make_unique<dsv4_abs_comp_mask_input>(t, n_comp_visible, compress_ratio, false, cparams.causal_attn));
                        return t;
                    });
                idx_scores = ggml_add(ctx0, idx_scores, idx_valid_mask);

                ggml_tensor * idx_topk = ggml_argsort_top_k(ctx0, idx_scores, n_idx_topk);
                cb(idx_topk, "attn_indexer_topk", il);

                // ----- unified sparse-gather attention -----
                ggml_tensor * sink = model.layers[il].attn_sinks;
                if (sink && sink->type != GGML_TYPE_F32) {
                    sink = ggml_cast(ctx0, sink, GGML_TYPE_F32);
                }
                // SWA mask for the window phase. This handles both cache-validity
                // (mask out unfilled slots) and within-chunk causality (for chunked decode
                // with n_tokens > 1). Shape: [n_kv, n_tokens/n_stream, 1, n_stream] F32.
                ggml_tensor * window_mask = inp_attn_iswa->self_kq_mask_swa;
                cur = ggml_dsv4_sparse_attn(
                    ctx0,
                    Qcur,
                    kv_comp_cache,    // [head_dim, 1, n_comp_visible]
                    k_raw,            // [head_dim, 1, n_window]
                    window_mask,      // [n_window, n_tokens] SWA validity + causal
                    idx_topk,         // [n_idx_topk, n_tokens]
                    sink,             // [n_heads]
                    kq_scale,
                    0);
                cb(cur, "attn_sparse_decode", il);
                use_prompt_sparse_attn = true;
            } else if (n_comp_visible > 0) {
                if (dsv4_debug_paths_enabled()) {
                    LLAMA_LOG_WARN("deepseek4/path: layer=%d path=decode-compressed n_tokens=%lld n_seq_tokens=%lld first_pos=%d last_pos=%d is_prefill=%d compress_ratio=%u\n",
                            il, (long long) n_tokens, (long long) n_seq_tokens, first_pos, last_pos, (int) is_prefill, compress_ratio);
                }

                const auto * mctx_swa = inp_attn_iswa->mctx->get_swa();
                ggml_tensor * k_store = mctx_swa->cpy_k(ctx0, Kcur, inp_attn_iswa->get_k_idxs_swa(), il);
                ggml_build_forward_expand(gf, k_store);

                ggml_tensor * k_raw_ref = mctx_swa->get_k(ctx0, il);
                ggml_tensor * k_raw = ggml_view_4d(ctx0, k_store,
                        k_raw_ref->ne[0], k_raw_ref->ne[1], k_raw_ref->ne[2], k_raw_ref->ne[3],
                        k_raw_ref->nb[1], k_raw_ref->nb[2], k_raw_ref->nb[3], k_raw_ref->view_offs);
                k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head, 1, k_raw->ne[2]);
                k_raw = as_f32(k_raw);

                ggml_tensor * kv_comp_cache = nullptr;
                if (kv_comp_new_for_attn != nullptr && n_comp_new > 0) {
                    ggml_tensor * kv_new = ggml_reshape_3d(ctx0, cont_if_needed(ctx0, kv_comp_new_for_attn), n_embd_head, 1, n_comp_new);
                    if (n_comp_before > 0) {
                        ggml_tensor * kv_old = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_before);
                        kv_comp_cache = ggml_concat(ctx0, as_f32(kv_old), as_f32(kv_new), 2);
                    } else {
                        kv_comp_cache = kv_new;
                    }
                } else {
                    kv_comp_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_visible);
                }
                kv_comp_cache = as_f32(kv_comp_cache);

                ggml_tensor * comp_mask = cache_mask(
                    "decode_comp_causal:" + std::to_string((uint32_t)compress_ratio) + ":" + std::to_string(n_comp_visible) + ":" + std::to_string((int)cparams.causal_attn),
                    [&]() {
                        ggml_tensor * t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_comp_visible, n_tokens, 1, 1);
                        ggml_set_input(t);
                        ggml_set_name(t, "dsv4_decode_comp_mask");
                        res->add_input(std::make_unique<dsv4_abs_comp_mask_input>(t, n_comp_visible, compress_ratio, hparams.use_alibi, cparams.causal_attn));
                        return t;
                    });
                ggml_tensor * attn_mask = ggml_concat(ctx0, inp_attn_iswa->self_kq_mask_swa, comp_mask, 0);
                // DeepSeek4 raw+compressed continuation attention uses the same
                // mixed mask/KV shape as prompt compressed attention; force the
                // standard F32 softmax MHA instead of FlashAttention.
                ggml_tensor * attn_mask_cnv = attn_mask;
                ggml_tensor * k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);

                ggml_tensor * sink = model.layers[il].attn_sinks;
                if (sink && sink->type != GGML_TYPE_F32) {
                    sink = ggml_cast(ctx0, sink, GGML_TYPE_F32);
                }

                cur = build_attn_mha(Qcur, k_all, k_all, nullptr, attn_mask_cnv, sink, nullptr, kq_scale, il, true);
                cb(cur, "attn_compressed_decode", il);
                use_prompt_sparse_attn = true;
            }
            // else: fall through to standard build_attn below (window-only attention).
            // The dsv4 attn_k / index_k cache writes above still happened so that future
            // decode steps have the data they need.
        }

        if (!use_prompt_sparse_attn) {
            if (has_hybrid_iswa) {
                cur = build_attn(inp_attn_iswa,
                        nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur,
                        nullptr, model.layers[il].attn_sinks, nullptr, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_kv,
                        nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur,
                        nullptr, model.layers[il].attn_sinks, nullptr, kq_scale, il);
            }
        }
        cur = as_f32(cur);
        cb(cur, "attn_out_raw", il);

        // Inverse partial RoPE on the attention output (matches antirez)
        cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
        cur = apply_partial_rope(cur, rope_cfg, /*inverse=*/true);
        cb(cur, "attn_out_unrope", il);

        (void) idx_prior;

        // Grouped output projection: exactly matches antirez/dsv4_grouped_out.
        // Uses ggml_mul_mat_id with per-group ids — keeps tensor 3D throughout,
        // avoids precision loss from intermediate concats/reshapes.
        {
            const int64_t n_groups = hparams.n_o_groups;
            const int64_t group_dim = n_embd_head * (n_head / n_groups);
            const int64_t o_lora_rank = model.layers[il].wq->ne[1] / n_groups;

            GGML_ASSERT(n_head % n_groups == 0);
            GGML_ASSERT(model.layers[il].wq->ne[0] == group_dim);
            GGML_ASSERT(model.layers[il].wq->ne[1] == o_lora_rank * n_groups);

            // cur: [n_embd_head, n_head, n_tokens] -> [group_dim, n_groups, n_tokens]
            ggml_tensor * o = cont_if_needed(ctx0, cur);
            o = ggml_reshape_3d(ctx0, o, group_dim, n_groups, n_tokens);

            // wq is [group_dim, n_groups*o_lora_rank] reshaped to [group_dim, o_lora_rank, n_groups]
            ggml_tensor * wo_a_g = ggml_reshape_3d(ctx0, model.layers[il].wq, group_dim, o_lora_rank, n_groups);

            // Per-group identity ids: [n_groups, n_tokens] — built once before
            // the layer loop and reused across all layers.
            GGML_ASSERT(grouped_out_ids != nullptr);
            GGML_ASSERT(grouped_out_ids->ne[0] == n_groups);
            GGML_ASSERT(grouped_out_ids->ne[1] == n_tokens);

            cur = ggml_mul_mat_id(ctx0, wo_a_g, o, grouped_out_ids); // [o_lora_rank, n_groups, n_tokens]
            cur = ggml_reshape_2d(ctx0, cur, o_lora_rank * n_groups, n_tokens);
        }
        cur = as_f32(cur);
        cb(cur, "attn_o_a", il);

        ggml_tensor * attn_o_b_inp_src = hparams.deepseek4_fp8_attn_out ? ggml_cast(ctx0, cur, GGML_TYPE_BF16) : cur;
        ggml_tensor * attn_o_b_inp = apply_dense_fp8_qat(attn_o_b_inp_src, hparams.deepseek4_fp8_attn_out, "attn_o_a_fp8_qat", il);
        cur = mul_mat_checked(model.layers[il].wo, attn_o_b_inp, "attn.o_b");
        cur = cast_dense_fp8_out(cur, hparams.deepseek4_fp8_attn_out);
        cur = as_f32(cur);
        cb(cur, "attn_o_b", il);

        ggml_tensor * attn_state = hc_post(cur, hc_state, attn_mix.post, attn_mix.comb, "attn_hc_post", il);

        hc_pre_result ffn_mix = hc_pre(attn_state, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_base, model.layers[il].hc_ffn_scale, "ffn_hc_pre", il);
        ggml_tensor * ffn_act_inp = model.layers[il].ffn_norm != nullptr
            ? build_norm(as_f32(ffn_mix.collapsed), model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il)
            : ffn_mix.collapsed;
        ffn_act_inp = as_f32(ffn_act_inp);
        cb(ffn_act_inp, "ffn_norm", il);

        ggml_tensor * selected_experts_in = nullptr;
        if ((uint32_t) il < hparams.n_hash_layers &&
            model.layers[il].ffn_gate_inp_b != nullptr &&
            res->t_inp_tokens != nullptr &&
            !cparams.warmup) {
            // Hash-MoE path: pick experts directly from a per-token lookup table.
            // Skipped during warmup because warmup runs an empty graph reservation
            // where t_inp_tokens may exist but is not a real token stream and the
            // shape relationship between ffn_gate_inp_b and n_expert_used cannot
            // be relied on (matches antirez's `!cparams.warmup` guard).
            GGML_ASSERT(model.layers[il].ffn_gate_inp_b->ne[0] == n_expert_used);
            selected_experts_in = ggml_get_rows(ctx0, model.layers[il].ffn_gate_inp_b, res->t_inp_tokens);
            cb(selected_experts_in, "ffn_hash_selected_experts", il);
        }

        ggml_tensor * moe_out = build_moe_ffn(ffn_act_inp,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                /* probs_in */ nullptr,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s,
                selected_experts_in);
        moe_out = as_f32(moe_out);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * shexp_out = nullptr;
        if (hparams.deepseek4_fp8_shared_expert) {
            // DeepSeek-V4 official inference runs shared expert Linears in FP8 mode,
            // which applies activation quantization per Linear input. Mirror that
            // QAT flow here: input -> (w1,w3), then SwiGLU hidden -> w2.
            ggml_tensor * shexp_inp_src = ggml_cast(ctx0, ffn_act_inp, GGML_TYPE_BF16);
            ggml_tensor * shexp_inp = apply_dense_fp8_qat(shexp_inp_src, true, "ffn_shexp_inp_fp8_qat", il);

            ggml_tensor * shexp_up = build_lora_mm(model.layers[il].ffn_up_shexp, shexp_inp);
            shexp_up = cast_dense_fp8_out(shexp_up, true);
            cb(shexp_up, "ffn_shexp_up", il);

            ggml_tensor * shexp_gate = build_lora_mm(model.layers[il].ffn_gate_shexp, shexp_inp);
            shexp_gate = cast_dense_fp8_out(shexp_gate, true);
            cb(shexp_gate, "ffn_shexp_gate", il);

            // ds4 parity: shared expert uses plain SwiGLU (no clamp).
            // Cast to f32 first — FP8 path carries BF16 activations but
            // CUDA/CPU swiglu kernels require f32 or f16.
            ggml_tensor * shexp_mid = ggml_swiglu_split(ctx0, as_f32(shexp_gate), as_f32(shexp_up));
            cb(shexp_mid, "ffn_shexp_swiglu", il);

            ggml_tensor * shexp_mid_src = ggml_cast(ctx0, shexp_mid, GGML_TYPE_BF16);
            ggml_tensor * shexp_mid_fp8 = apply_dense_fp8_qat(shexp_mid_src, true, "ffn_shexp_mid_fp8_qat", il);

            shexp_out = build_lora_mm(model.layers[il].ffn_down_shexp, shexp_mid_fp8);
            shexp_out = cast_dense_fp8_out(shexp_out, true);
        } else {
            shexp_out = build_ffn(ffn_act_inp,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        shexp_out = as_f32(shexp_out);
        cb(shexp_out, "ffn_shexp_out", il);

        cur = ggml_add(ctx0, moe_out, shexp_out);
        cur = as_f32(cur);
        cb(cur, "ffn_out", il);

        hc_state = hc_post(cur, attn_state, ffn_mix.post, ffn_mix.comb, "ffn_hc_post", il);
        hc_state = apply_cvec_hc(hc_state, il);
        cb(hc_state, "l_out_hc", il);
    }

    if (mtp_probe && n_tokens == 1 && n_outputs == 1) {
        ggml_tensor * mtp_hc_state = as_f32(hc_state);
        cb(mtp_hc_state, "dsv4_mtp_hc_state", -1);
        res->t_mtp_state = mtp_hc_state;

        auto mtp_tensor = [&](const char * name) -> ggml_tensor * {
            if (mtp_tensors == nullptr) {
                return nullptr;
            }
            const auto it = mtp_tensors->find(name);
            return it == mtp_tensors->end() ? nullptr : it->second;
        };

        auto mtp_required = [&](const char * name) -> ggml_tensor * {
            ggml_tensor * t = mtp_tensor(name);
            GGML_ASSERT(t != nullptr);
            return t;
        };

        if (res->t_inp_tokens != nullptr && mtp_tensors != nullptr) {
            ggml_tensor * mtp_e_proj        = mtp_required("mtp.0.e_proj.weight");
            ggml_tensor * mtp_h_proj        = mtp_required("mtp.0.h_proj.weight");
            ggml_tensor * mtp_enorm         = mtp_required("mtp.0.enorm.weight");
            ggml_tensor * mtp_hnorm         = mtp_required("mtp.0.hnorm.weight");
            ggml_tensor * mtp_norm          = mtp_required("mtp.0.norm.weight");
            ggml_tensor * mtp_hc_head_fn    = mtp_required("mtp.0.hc_head_fn.weight");
            ggml_tensor * mtp_hc_head_base  = mtp_required("mtp.0.hc_head_base.weight");
            ggml_tensor * mtp_hc_head_scale = mtp_required("mtp.0.hc_head_scale.weight");

            ggml_tensor * mtp_hc_attn_fn    = mtp_required("mtp.0.hc_attn_fn.weight");
            ggml_tensor * mtp_hc_attn_scale = mtp_required("mtp.0.hc_attn_scale.weight");
            ggml_tensor * mtp_hc_attn_base  = mtp_required("mtp.0.hc_attn_base.weight");
            ggml_tensor * mtp_attn_norm     = mtp_required("mtp.0.attn_norm.weight");
            ggml_tensor * mtp_attn_q_a      = mtp_required("mtp.0.attn_q_a.weight");
            ggml_tensor * mtp_attn_q_a_norm = mtp_required("mtp.0.attn_q_a_norm.weight");
            ggml_tensor * mtp_attn_q_b      = mtp_required("mtp.0.attn_q_b.weight");
            ggml_tensor * mtp_attn_kv       = mtp_required("mtp.0.attn_kv.weight");
            ggml_tensor * mtp_attn_kv_norm  = mtp_required("mtp.0.attn_kv_a_norm.weight");
            ggml_tensor * mtp_attn_sinks    = mtp_required("mtp.0.attn_sinks.weight");
            ggml_tensor * mtp_attn_out_a    = mtp_required("mtp.0.attn_output_a.weight");
            ggml_tensor * mtp_attn_out_b    = mtp_required("mtp.0.attn_output_b.weight");

            ggml_tensor * mtp_hc_ffn_fn      = mtp_required("mtp.0.hc_ffn_fn.weight");
            ggml_tensor * mtp_hc_ffn_scale   = mtp_required("mtp.0.hc_ffn_scale.weight");
            ggml_tensor * mtp_hc_ffn_base    = mtp_required("mtp.0.hc_ffn_base.weight");
            ggml_tensor * mtp_ffn_norm       = mtp_required("mtp.0.ffn_norm.weight");
            ggml_tensor * mtp_ffn_gate_inp   = mtp_required("mtp.0.ffn_gate_inp.weight");
            ggml_tensor * mtp_exp_probs_b    = mtp_required("mtp.0.exp_probs_b.bias");
            ggml_tensor * mtp_ffn_gate_exps  = mtp_required("mtp.0.ffn_gate_exps.weight");
            ggml_tensor * mtp_ffn_up_exps    = mtp_required("mtp.0.ffn_up_exps.weight");
            ggml_tensor * mtp_ffn_down_exps  = mtp_required("mtp.0.ffn_down_exps.weight");
            ggml_tensor * mtp_ffn_gate_shexp = mtp_required("mtp.0.ffn_gate_shexp.weight");
            ggml_tensor * mtp_ffn_up_shexp   = mtp_required("mtp.0.ffn_up_shexp.weight");
            ggml_tensor * mtp_ffn_down_shexp = mtp_required("mtp.0.ffn_down_shexp.weight");

            ggml_tensor * mtp_embed = ggml_get_rows(ctx0, model.tok_embd, res->t_inp_tokens);
            mtp_embed               = as_f32(mtp_embed);
            cb(mtp_embed, "dsv4_mtp_embed", -1);

            ggml_tensor * mtp_e = build_norm(mtp_embed, mtp_enorm, nullptr, LLM_NORM_RMS, -1);
            mtp_e               = mul_mat_checked(mtp_e_proj, mtp_e, "mtp.e_proj");
            mtp_e               = make_hc_state(mtp_e);
            cb(mtp_e, "dsv4_mtp_e_proj_hc", -1);

            ggml_tensor * mtp_h = build_norm(mtp_hc_state, mtp_hnorm, nullptr, LLM_NORM_RMS, -1);
            mtp_h               = mul_mat_checked(mtp_h_proj, mtp_h, "mtp.h_proj");
            cb(mtp_h, "dsv4_mtp_h_proj_hc", -1);

            ggml_tensor * mtp_block_hc = ggml_add(ctx0, mtp_e, mtp_h);
            cb(mtp_block_hc, "dsv4_mtp_input_hc", -1);

            // DS4 calls the single MTP sidecar block with logical layer id 1. For DeepSeek4
            // this is a dense/raw-only attention layer (compress_ratio == 0), so the sidecar
            // drafter needs private raw-window state but no private compressed/indexer state.
            // Target verifier compressed/indexer frontier rollback is a separate commit path.
            const uint32_t mtp_logical_il     = 1;
            const uint32_t mtp_compress_ratio = hparams.deepseek4_compress_ratios[mtp_logical_il];
            GGML_ASSERT(mtp_compress_ratio == 0);
            const auto mtp_rope_cfg = get_rope_cfg(mtp_compress_ratio);

            hc_pre_result mtp_attn_mix =
                hc_pre(mtp_block_hc, mtp_hc_attn_fn, mtp_hc_attn_base, mtp_hc_attn_scale, "dsv4_mtp_attn_hc_pre", -1);
            ggml_tensor * mtp_attn_inp =
                build_norm(as_f32(mtp_attn_mix.collapsed), mtp_attn_norm, nullptr, LLM_NORM_RMS, -1);
            mtp_attn_inp = as_f32(mtp_attn_inp);
            cb(mtp_attn_inp, "dsv4_mtp_attn_norm", -1);

            ggml_tensor * mtp_q_residual = mul_mat_checked(mtp_attn_q_a, mtp_attn_inp, "mtp.attn_q_a");
            mtp_q_residual               = as_f32(mtp_q_residual);
            mtp_q_residual               = build_norm(mtp_q_residual, mtp_attn_q_a_norm, nullptr, LLM_NORM_RMS, -1);
            cb(mtp_q_residual, "dsv4_mtp_attn_q_a_norm", -1);

            ggml_tensor * mtp_Qcur = mul_mat_checked(mtp_attn_q_b, mtp_q_residual, "mtp.attn_q_b");
            mtp_Qcur               = as_f32(mtp_Qcur);
            mtp_Qcur               = ggml_reshape_3d(ctx0, mtp_Qcur, n_embd_head, n_head, n_tokens);
            mtp_Qcur               = ggml_rms_norm(ctx0, mtp_Qcur, hparams.f_norm_rms_eps);
            mtp_Qcur               = apply_partial_rope(mtp_Qcur, mtp_rope_cfg, /*inverse=*/false);
            cb(mtp_Qcur, "dsv4_mtp_attn_q_roped", -1);

            ggml_tensor * mtp_Kcur = mul_mat_checked(mtp_attn_kv, mtp_attn_inp, "mtp.attn_kv");
            mtp_Kcur               = as_f32(mtp_Kcur);
            mtp_Kcur               = build_norm(mtp_Kcur, mtp_attn_kv_norm, nullptr, LLM_NORM_RMS, -1);
            mtp_Kcur               = ggml_reshape_3d(ctx0, mtp_Kcur, n_embd_head, 1, n_tokens);
            mtp_Kcur               = apply_partial_rope(mtp_Kcur, mtp_rope_cfg, /*inverse=*/false);
            mtp_Kcur               = as_f32(mtp_Kcur);
            mtp_Kcur               = apply_fp8_qat_nope_2d(mtp_Kcur, "dsv4_mtp_attn_kv_qat", -1);
            cb(mtp_Kcur, "dsv4_mtp_attn_kv", -1);
            res->t_mtp_raw_current = mtp_Kcur;

            ggml_tensor * mtp_raw_kv = mtp_Kcur;
            if (mtp_n_raw > 0) {
                GGML_ASSERT(mtp_raw_cache != nullptr);
                ggml_tensor * mtp_raw_prev = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, 1, mtp_n_raw);
                ggml_set_input(mtp_raw_prev);
                ggml_set_name(mtp_raw_prev, "dsv4_mtp_raw_cache");
                res->add_input(
                    std::make_unique<dsv4_mtp_raw_cache_input>(mtp_raw_prev, mtp_raw_cache, mtp_n_raw, n_embd_head));
                mtp_raw_kv = ggml_concat(ctx0, mtp_raw_prev, mtp_Kcur, 2);
                cb(mtp_raw_kv, "dsv4_mtp_raw_kv", -1);
            }

            ggml_tensor * mtp_attn_out = build_attn_mha(mtp_Qcur, mtp_raw_kv, mtp_raw_kv, nullptr, nullptr,
                                                        mtp_attn_sinks, nullptr, kq_scale, -1,
                                                        /*force_no_flash_attn=*/true);
            mtp_attn_out = as_f32(mtp_attn_out);
            cb(mtp_attn_out, "dsv4_mtp_attn_out_raw", -1);

            mtp_attn_out = ggml_reshape_3d(ctx0, mtp_attn_out, n_embd_head, n_head, n_tokens);
            mtp_attn_out = apply_partial_rope(mtp_attn_out, mtp_rope_cfg, /*inverse=*/true);
            cb(mtp_attn_out, "dsv4_mtp_attn_out_unrope", -1);

            {
                const int64_t n_groups    = hparams.n_o_groups;
                const int64_t group_dim   = n_embd_head * (n_head / n_groups);
                const int64_t o_lora_rank = mtp_attn_out_a->ne[1] / n_groups;
                GGML_ASSERT(n_head % n_groups == 0);
                GGML_ASSERT(mtp_attn_out_a->ne[0] == group_dim);
                GGML_ASSERT(mtp_attn_out_a->ne[1] == o_lora_rank * n_groups);
                GGML_ASSERT(grouped_out_ids != nullptr);

                ggml_tensor * o      = cont_if_needed(ctx0, mtp_attn_out);
                o                    = ggml_reshape_3d(ctx0, o, group_dim, n_groups, n_tokens);
                ggml_tensor * wo_a_g = ggml_reshape_3d(ctx0, mtp_attn_out_a, group_dim, o_lora_rank, n_groups);
                mtp_attn_out         = ggml_mul_mat_id(ctx0, wo_a_g, o, grouped_out_ids);
                mtp_attn_out         = ggml_reshape_2d(ctx0, mtp_attn_out, o_lora_rank * n_groups, n_tokens);
            }
            mtp_attn_out = as_f32(mtp_attn_out);
            cb(mtp_attn_out, "dsv4_mtp_attn_o_a", -1);

            mtp_attn_out = mul_mat_checked(mtp_attn_out_b, mtp_attn_out, "mtp.attn_o_b");
            mtp_attn_out = as_f32(mtp_attn_out);
            cb(mtp_attn_out, "dsv4_mtp_attn_o_b", -1);

            ggml_tensor * mtp_attn_state =
                hc_post(mtp_attn_out, mtp_block_hc, mtp_attn_mix.post, mtp_attn_mix.comb, "dsv4_mtp_attn_hc_post", -1);

            hc_pre_result mtp_ffn_mix =
                hc_pre(mtp_attn_state, mtp_hc_ffn_fn, mtp_hc_ffn_base, mtp_hc_ffn_scale, "dsv4_mtp_ffn_hc_pre", -1);
            ggml_tensor * mtp_ffn_inp =
                build_norm(as_f32(mtp_ffn_mix.collapsed), mtp_ffn_norm, nullptr, LLM_NORM_RMS, -1);
            mtp_ffn_inp = as_f32(mtp_ffn_inp);
            cb(mtp_ffn_inp, "dsv4_mtp_ffn_norm", -1);

            ggml_tensor * mtp_moe_out = build_moe_ffn(
                mtp_ffn_inp, mtp_ffn_gate_inp, mtp_ffn_up_exps, mtp_ffn_gate_exps, mtp_ffn_down_exps, mtp_exp_probs_b,
                n_expert, n_expert_used, LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, -1);
            mtp_moe_out = as_f32(mtp_moe_out);
            cb(mtp_moe_out, "dsv4_mtp_ffn_moe_out", -1);

            ggml_tensor * mtp_shexp_out =
                build_ffn(mtp_ffn_inp, mtp_ffn_up_shexp, nullptr, nullptr, mtp_ffn_gate_shexp, nullptr, nullptr,
                          mtp_ffn_down_shexp, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, -1);
            mtp_shexp_out = as_f32(mtp_shexp_out);
            cb(mtp_shexp_out, "dsv4_mtp_ffn_shexp_out", -1);

            ggml_tensor * mtp_ffn_out = ggml_add(ctx0, mtp_moe_out, mtp_shexp_out);
            mtp_ffn_out               = as_f32(mtp_ffn_out);
            cb(mtp_ffn_out, "dsv4_mtp_ffn_out", -1);

            ggml_tensor * mtp_next_hc =
                hc_post(mtp_ffn_out, mtp_attn_state, mtp_ffn_mix.post, mtp_ffn_mix.comb, "dsv4_mtp_ffn_hc_post", -1);
            cb(mtp_next_hc, "dsv4_mtp_block_hc", -1);
            res->t_mtp_next_state = mtp_next_hc;

            ggml_tensor * mtp_embd =
                hc_head(mtp_next_hc, mtp_hc_head_fn, mtp_hc_head_base, mtp_hc_head_scale, "dsv4_mtp_block_hc_head");
            mtp_embd               = build_norm(mtp_embd, mtp_norm, nullptr, LLM_NORM_RMS, -1);
            mtp_embd               = as_f32(mtp_embd);
            cb(mtp_embd, "dsv4_mtp_block_norm", -1);

            ggml_tensor * mtp_logits = build_lora_mm(model.output, mtp_embd);
            cb(mtp_logits, "dsv4_mtp_block_logits", -1);

            ggml_tensor * mtp_top1 = ggml_argsort_top_k(ctx0, mtp_logits, 1);
            cb(mtp_top1, "dsv4_mtp_block_top1", -1);
            res->t_mtp_probe_top1 = mtp_top1;
            ggml_build_forward_expand(gf, mtp_top1);
        }
    }

    cur = hc_head(hc_state, model.hc_head_fn, model.hc_head_base, model.hc_head_scale, "hc_head_out");
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cur = as_f32(cur);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
