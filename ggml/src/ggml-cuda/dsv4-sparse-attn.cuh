#pragma once
#include "common.cuh"

void ggml_cuda_op_dsv4_sparse_attn(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
