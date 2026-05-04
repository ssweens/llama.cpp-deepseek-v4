#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

// ----------------------------------------------------------------------------
// K-quant GET_ROWS support
//
// The 2-elements-per-call dequantize kernels in dequantize.cuh do not work for
// K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) because their super-block layout
// stores per-sub-block scales/mins that must be read in chunks aligned to a
// full 256-element super-block. Without K-quant GET_ROWS, every token-embedding
// lookup on a K-quant token_embd weight forces a CPU fallback in the scheduler,
// inserting a backend split per forward pass. For DSv4 (token_embd is Q2_K in
// Q2_K_S builds, Q6_K in IQ-quant builds via the non-expert protect commit),
// this fired on every forward pass.
//
// Strategy: for each (output_row, super_block_within_row) we launch one CUDA
// block whose threads dequantize that super-block directly into the output,
// using the same per-block dequant logic as dequantize_block_qX_K in convert.cu.
// Grid: (ne00 / QK_K, ne10, ne11*ne12). Block: 64 threads (matches the existing
// convert kernels' assumption).
// ----------------------------------------------------------------------------

// Per-sub-block scale/min decoder for Q4_K and Q5_K. Bit-identical copy of
// the helper in convert.cu (which is `static inline __device__` so not visible
// across translation units). Kept in sync with that file.
static inline __device__ void get_rows_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Helper: resolve source super-block pointer for an output (i10, sb) using
// the index in src1 and the per-axis strides of the destination.
static __device__ __forceinline__ const char * kquant_src_super_block_ptr(
        const void * __restrict__ src0,
        const int32_t * __restrict__ src1,
        size_t sb_size_bytes,
        int i10, int i11, int i12,
        size_t s10, size_t s11, size_t s12,
        size_t nb01, size_t nb02, size_t nb03,
        int sb) {
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    return (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03 + sb*sb_size_bytes;
}

template<typename dst_t>
static __global__ void k_get_rows_q2_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t /*ne00*/,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int sb  = blockIdx.x;        // super-block index within the row
    const int i10 = blockIdx.y;        // which output row
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const block_q2_K * sx = (const block_q2_K *) kquant_src_super_block_ptr(
            src0, src1, sizeof(block_q2_K),
            i10, i11, i12, s10, s11, s12, nb01, nb02, nb03, sb);

        const int64_t tid = threadIdx.x;
        const int64_t n   = tid/32;
        const int64_t l   = tid - 32*n;
        const int64_t is  = 8*n + l/16;

        const uint8_t q = sx->qs[32*n + l];
        dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + sb*QK_K + 128*n;

        const float dall = __low2half(sx->dm);
        const float dmin = __high2half(sx->dm);
        y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (sx->scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (sx->scales[is+0] >> 4));
        y[l+32] = ggml_cuda_cast<dst_t>(dall * (sx->scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (sx->scales[is+2] >> 4));
        y[l+64] = ggml_cuda_cast<dst_t>(dall * (sx->scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (sx->scales[is+4] >> 4));
        y[l+96] = ggml_cuda_cast<dst_t>(dall * (sx->scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (sx->scales[is+6] >> 4));
    }
}

template<typename dst_t>
static __global__ void k_get_rows_q3_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t /*ne00*/,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int sb  = blockIdx.x;
    const int i10 = blockIdx.y;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const block_q3_K * sx = (const block_q3_K *) kquant_src_super_block_ptr(
            src0, src1, sizeof(block_q3_K),
            i10, i11, i12, s10, s11, s12, nb01, nb02, nb03, sb);

        const int64_t r = threadIdx.x/4;
        const int64_t tid = r/2;
        const int64_t is0 = r%2;
        const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
        const int64_t n = tid / 4;
        const int64_t j = tid - 4*n;

        const uint8_t m = 1 << (4*n + j);
        const int64_t is = 8*n + 2*j + is0;
        const int shift = 2*j;

        const int8_t us = is <  4 ? (sx->scales[is-0] & 0xF) | (((sx->scales[is+8] >> 0) & 3) << 4) :
                          is <  8 ? (sx->scales[is-0] & 0xF) | (((sx->scales[is+4] >> 2) & 3) << 4) :
                          is < 12 ? (sx->scales[is-8] >>  4) | (((sx->scales[is+0] >> 4) & 3) << 4) :
                                    (sx->scales[is-8] >>  4) | (((sx->scales[is-4] >> 6) & 3) << 4);
        const float d_all = sx->d;
        const float dl = d_all * (us - 32);

        dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + sb*QK_K + 128*n + 32*j;
        const uint8_t * q  = sx->qs + 32*n;
        const uint8_t * hm = sx->hmask;
        for (int l = l0; l < l0+4; ++l) {
            y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_q4_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t /*ne00*/,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int sb  = blockIdx.x;
    const int i10 = blockIdx.y;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const block_q4_K * sx = (const block_q4_K *) kquant_src_super_block_ptr(
            src0, src1, sizeof(block_q4_K),
            i10, i11, i12, s10, s11, s12, nb01, nb02, nb03, sb);

        // assume 32 threads (matches dequantize_block_q4_K in convert.cu)
        const int64_t tid = threadIdx.x;
        const int64_t il  = tid/8;
        const int64_t ir  = tid%8;
        const int64_t is  = 2*il;
        const int64_t n   = 4;

        dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + sb*QK_K + 64*il + n*ir;

        const float dall = __low2half(sx->dm);
        const float dmin = __high2half(sx->dm);
        const uint8_t * q = sx->qs + 32*il + n*ir;

        uint8_t sc, mm;
        get_rows_get_scale_min_k4(is + 0, sx->scales, sc, mm);
        const float d1 = dall * sc; const float m1 = dmin * mm;
        get_rows_get_scale_min_k4(is + 1, sx->scales, sc, mm);
        const float d2 = dall * sc; const float m2 = dmin * mm;
        for (int l = 0; l < n; ++l) {
            y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
            y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_q5_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t /*ne00*/,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int sb  = blockIdx.x;
    const int i10 = blockIdx.y;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const block_q5_K * sx = (const block_q5_K *) kquant_src_super_block_ptr(
            src0, src1, sizeof(block_q5_K),
            i10, i11, i12, s10, s11, s12, nb01, nb02, nb03, sb);

        const int64_t tid = threadIdx.x;
        const int64_t il  = tid/16;
        const int64_t ir  = tid%16;
        const int64_t is  = 2*il;

        dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + sb*QK_K + 64*il + 2*ir;

        const float dall = __low2half(sx->dm);
        const float dmin = __high2half(sx->dm);
        const uint8_t * ql = sx->qs + 32*il + 2*ir;
        const uint8_t * qh = sx->qh + 2*ir;

        uint8_t sc, mm;
        get_rows_get_scale_min_k4(is + 0, sx->scales, sc, mm);
        const float d1 = dall * sc; const float m1 = dmin * mm;
        get_rows_get_scale_min_k4(is + 1, sx->scales, sc, mm);
        const float d2 = dall * sc; const float m2 = dmin * mm;

        const uint8_t hm = 1 << (2*il);
        y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm        ? 16 : 0)) - m1);
        y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm        ? 16 : 0)) - m1);
        y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[0] >>  4) + (qh[0] & (hm << 1) ? 16 : 0)) - m2);
        y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[1] >>  4) + (qh[1] & (hm << 1) ? 16 : 0)) - m2);
    }
}

template<typename dst_t>
static __global__ void k_get_rows_q6_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t /*ne00*/,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int sb  = blockIdx.x;
    const int i10 = blockIdx.y;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const block_q6_K * sx = (const block_q6_K *) kquant_src_super_block_ptr(
            src0, src1, sizeof(block_q6_K),
            i10, i11, i12, s10, s11, s12, nb01, nb02, nb03, sb);

        const int64_t tid = threadIdx.x;
        const int64_t ip  = tid/32;
        const int64_t il  = tid - 32*ip;
        const int64_t is  = 8*ip + il/16;

        dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + sb*QK_K + 128*ip + il;

        const float d = sx->d;

        const uint8_t * ql = sx->ql + 64*ip + il;
        const uint8_t   qh = sx->qh[32*ip + il];
        const int8_t  * sc = sx->scales + is;

        y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const int i11 =  z / ne12; // TODO fastdiv
            const int i12 =  z % ne12;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const int i11 = z / ne12; // TODO fastdiv
            const int i12 = z % ne12;

            if (i00 >= ne00) {
                return;
            }

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst, const int64_t ncols, const int64_t nrows_grad) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int dst_row = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i*ncols + col];
    }

    dst[dst_row*ncols + col] = sum;
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename kernel_t, typename dst_t>
static void get_rows_cuda_kquant(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        kernel_t kernel, int n_threads,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int super_blocks_per_row = ne00 / QK_K;
    // n_threads matches the per-K-quant kernel's launch in convert.cu (Q2_K/Q3_K/Q5_K/Q6_K = 64; Q4_K = 32).
    const dim3 block_dims((unsigned) n_threads, 1, 1);
    const dim3 block_nums((unsigned) super_blocks_per_row,
                          (unsigned) std::min<int64_t>(ne10, UINT16_MAX),
                          (unsigned) std::min<int64_t>(ne11*ne12, UINT16_MAX));

    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    kernel<<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        s1, s2, s3,
        nb01, nb02, nb03,
        s10, s11, s12);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kquant(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3,
                k_get_rows_q2_K<dst_t>, /*n_threads=*/64, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kquant(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3,
                k_get_rows_q3_K<dst_t>, /*n_threads=*/64, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kquant(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3,
                k_get_rows_q4_K<dst_t>, /*n_threads=*/32, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kquant(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3,
                k_get_rows_q5_K<dst_t>, /*n_threads=*/64, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kquant(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3,
                k_get_rows_q6_K<dst_t>, /*n_threads=*/64, stream);
            break;
        default:
            // TODO: remaining K-quants and IQ-quants.
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne1, 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10);
}
