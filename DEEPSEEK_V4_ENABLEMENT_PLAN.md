# DeepSeek-V4 Enablement Plan for llama.cpp

## Status (2026-04-30)

**End-to-end DeepSeek-V4-Flash inference is functional and validated** on the `feat/deepseek4-foundation` branch. Both BF16 and IQ2_XS produce coherent output with `-fa on` (Flash Attention enabled).

Major milestones completed:

- [x] Architecture / hparams / converter (FP8 dequant, partial RoPE, indexer, hybrid-ISWA cache layout)
- [x] CUDA YaRN parity, decode KV-quant, graph-dependency cleanups
- [x] IQ2/Q2/IQ2_XS quantization with full multi-GPU offload
- [x] Multi-sequence support for compressed attention path
- [x] Imatrix capture (multi-seq + single-seq regression validated)
- [x] **`GGML_OP_DSV4_SPARSE_ATTN` custom kernel** (CPU + CUDA naive)
- [x] **Unified sparse-gather attention path in deepseek4.cpp** (eliminated dense -inf mask incompatibility with FA tile architecture)
- [x] BF16 long-prompt validation: structured Evidence/Interpretation/Next Step output, correct termination
- [x] IQ2_XS short-prompt validation: 40 tok/s decode at full GPU offload, correct math

Still open:

- [ ] Phase 2 sparse kernel: tensor-core MMA path (current naive kernel is correctness-first; perf is fine for IQ2 full-offload)
- [ ] HIP/ROCm port of sparse kernel
- [ ] GCP imatrix run on H100-4x (blocked on spot capacity)
- [ ] Final corral config + production rollout

See `tasks/todo.md` for the granular state and `tasks/lessons.md` for the engineering lessons accumulated during this work, especially the Flash Attention sparse-mask incompatibility that drove the custom kernel design.

## Context

Primary target model: [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

Secondary/generalization target: [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

Reference implementation and design homework:

- MLX PR: <https://github.com/ml-explore/mlx-lm/pull/1192>
- Related MLX PR with more robust safetensors fallback: <https://github.com/ml-explore/mlx-lm/pull/1190>
- Official DeepSeek inference code in the HF repo: `inference/model.py`, `inference/kernel.py`, `inference/convert.py`
- DeepSeek-V4 technical report: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>

Local repo baseline when this plan was written:

- Repository: `/home/bigkahuna/src/llama.cpp`
- Branch: `master`
- Remote: `origin/master`
- Local state: clean
- Head observed: `380b4c98 common: support negated args (#17919)`

## Key Findings

DeepSeek-V4-Flash is not a small extension of the current `LLM_ARCH_DEEPSEEK2` path. It reuses some DeepSeek-V2/V3 ideas, but adds several architecture features that require model-specific support.

### Similar to DeepSeek-V2/V3

- MoE with routed experts and shared experts.
- YaRN-style RoPE scaling.
- Low-rank query projection.
- FP8 non-expert weights with block scales.
- MTP tensors exist, but should be skipped initially.

### New or materially different in V4

- Hyper-Connections (`hc_mult = 4`) replace ordinary residual connections.
- Sinkhorn-normalized `comb` matrices are used in every attention and FFN sublayer.
- Sliding-window attention plus learned compressed KV rows.
- Per-layer `compress_ratios` with values `0`, `4`, or `128`.
- `compress_ratio == 4` layers have an `Indexer` for top-k compressed KV row selection.
- First `num_hash_layers = 3` MoE layers use token-id hash routing via `tid2eid`.
- Router scoring uses `sqrtsoftplus`, i.e. `sqrt(softplus(x))`.
- Routed expert weights are stored as MXFP4 / FP4 payload plus E8M0 scales.
- Attention has learned `attn_sink` values.
- Attention output has a grouped low-rank path: `wo_a` + `wo_b`, with `o_groups = 8`, `o_lora_rank = 1024`.

## DeepSeek-V4 Technical Report Findings

The `DeepSeek_V4.pdf` report confirms the implementation should be treated as support for a new hybrid-attention architecture, not just another DeepSeek2/MLA variant.

### CSA and HCA terminology

The paper names the two compressed attention modes:

- **CSA — Compressed Sparse Attention**: compression ratio `m = 4`, overlapped compression, Lightning Indexer, and sparse top-k selection of compressed KV entries.
- **HCA — Heavily Compressed Attention**: compression ratio `m' = 128`, no sparse top-k selector; the model attends over heavily compressed KV entries plus the sliding-window branch.
- Both CSA and HCA use **Shared Key-Value MQA**: each compressed entry is both key and value.
- Both add a local **sliding-window attention branch** with `nwin = 128` so tokens can attend to recent uncompressed KV entries inside the current compression block.

This maps directly to `compress_ratios`:

- `0`: pure sliding-window layer.
- `4`: CSA layer with Indexer and sparse compressed KV selection.
- `128`: HCA layer with dense attention over heavily compressed KV rows.

### Flash vs Pro model setup

DeepSeek-V4-Flash:

- 43 layers, hidden size 4096.
- First two layers use pure sliding-window attention.
- Later layers interleave CSA and HCA.
- CSA top-k is 512.
- Query heads 64, head dimension 512, query compression dimension 1024.
- Output projection groups 8, per-group intermediate output dimension 1024.
- MoE: 1 shared expert, 256 routed experts, expert intermediate size 2048, 6 active experts/token.
- Hash routing for the first 3 MoE layers.
- mHC expansion factor 4, Sinkhorn iterations 20.
- 284B total parameters, 13B active per token.

DeepSeek-V4-Pro:

- 61 layers, hidden size 7168.
- First two layers use HCA, not pure sliding-window attention.
- Later layers interleave CSA and HCA.
- CSA top-k is 1024.
- Query heads 128, head dimension 512, query compression dimension 1536.
- Output projection groups 16, per-group intermediate output dimension 1024.
- MoE: 1 shared expert, 384 routed experts, expert intermediate size 3072, 6 active experts/token.
- Hash routing for the first 3 MoE layers.
- mHC expansion factor 4, Sinkhorn iterations 20.
- 1.6T total parameters, 49B active per token.

The initial converter/loader should therefore be parameterized from `config.json`, not hard-coded to Flash dimensions.

### mHC details relevant to implementation

Manifold-Constrained Hyper-Connections constrain the residual mixing matrix to the Birkhoff polytope, i.e. a non-negative doubly-stochastic matrix. The dynamic `pre`, `post`, and `comb` parameters are generated from RMS-normalized flattened HC residual state. The report confirms:

- `pre = sigmoid(raw_pre)`
- `post = 2 * sigmoid(raw_post)`
- `comb = Sinkhorn(exp(raw_comb))`
- Sinkhorn iteration count is `tmax = 20`.

This reinforces that a fused Sinkhorn implementation is not an optional nicety for performance; it is a central repeated operation in every layer.

### Attention details relevant to implementation

The paper clarifies several details that need to stay aligned with the official inference code:

- Queries and compressed KV entries are RMS-normalized before core attention to avoid exploding logits.
- RoPE is applied only to the final 64 dimensions of query/KV entries.
- Because the KV entry is used as both key and value, the core attention output carries absolute positional information; the output must receive inverse RoPE on its final 64 dimensions.
- Attention sink adds `exp(sink_logit)` to each head's softmax denominator, allowing the total attention mass over real KV rows to be less than 1.
- For CSA, the first compressed block pads the previous-overlap side with `-inf` gate logits and zero KV values.

### KV cache structure guidance

The report explicitly says PagedAttention-style assumptions do not fit V4 well because cache policies and block sizes differ by layer and by compressed mode. It splits serving state into:

- **State cache**: fixed-size per-sequence state for SWA KV plus uncompressed tail tokens not ready for compression.
- **Classical KV cache**: compressed CSA/HCA KV blocks.
- Cache block sizes should align around `lcm(m, m')`; with Flash/Pro values this is `lcm(4, 128) = 128` original tokens.

For llama.cpp, this supports the plan to start with model-specific V4 cache state instead of forcing the architecture into the existing global KV cache abstraction too early.

### Precision and storage guidance

The report confirms two deployment-relevant precision decisions:

- Routed MoE expert weights use FP4/MXFP4 in inference and rollout, not merely simulated quantization.
- The indexer QK path is also quantization-aware in FP4 in the official system, but this can be deferred for llama.cpp; correctness should come before indexer FP4 acceleration.
- KV cache storage is mixed precision in the official system: BF16 for RoPE dimensions and FP8 for non-RoPE dimensions. llama.cpp can start simpler, but this is a later memory optimization target.

### Tokenization / serving note

The paper confirms V4 adds special tokens and response formats for:

- `<think>` / `</think>` reasoning modes.
- DSML tool-call blocks.
- Quick Instruction task tokens such as action/query/title/authority/domain/read-url.

This is not required for core inference enablement, but tokenizer/chat-template support should be tracked separately after model execution works.

## Important MLX Lessons

### Preserve MXFP4 experts

MLX PR #1192 defaults routed experts to:

```python
{"group_size": 32, "bits": 4, "mode": "mxfp4"}
```

The PR comment also notes that keeping experts in MXFP4 significantly reduces memory. llama.cpp current master already has:

- `GGML_TYPE_MXFP4`
- `LLAMA_FTYPE_MOSTLY_MXFP4_MOE`
- existing MXFP4 quant/dequant logic
- existing MoE tensor routing paths via `GGML_OP_MUL_MAT_ID`

Therefore the converter should preserve routed expert tensors as `GGML_TYPE_MXFP4` rather than dequantizing them.

HF safetensors layout observed for an expert tensor:

```text
layers.0.ffn.experts.0.w1.weight I8       [2048, 2048]
layers.0.ffn.experts.0.w1.scale  F8_E8M0  [2048, 128]
```

This maps naturally to GGML MXFP4 blocks: 32 values per block, 1 E8M0 scale byte, and 16 packed FP4 bytes.

### Dequantize non-expert FP8 initially

Non-expert tensors are stored as FP8 with E8M0 block scales, e.g.:

```text
layers.0.attn.wq_a.weight F8_E4M3
layers.0.attn.wq_a.scale  F8_E8M0
```

llama.cpp does not currently have a general `GGML_TYPE_F8_E4M3` tensor path. The first implementation should dequantize these non-expert weights to BF16 or F16 during conversion, while keeping routed experts as MXFP4.

### Safetensors dtype handling is required

`F8_E8M0` support is not universal. The converter should explicitly parse unsupported safetensors dtypes:

- `F8_E8M0`: raw `uint8`, value is `2^(byte - 127)` when dequantizing scales.
- `F8_E4M3`: raw FP8 payload, dequantize with an explicit E4M3 decoder or a Torch path.
- `BF16`: uint16 view if direct loading fails.
- `I8`: expert FP4 payload, do not reinterpret as signed values except as raw packed bytes for MXFP4 assembly.

Prefer a manual parser approach like MLX PR #1190 over temporarily patching safetensors headers.

### Sinkhorn is an early performance priority

A PR comment reports that a custom Sinkhorn kernel improved generation speed from roughly `17 tok/s` to `26 tok/s`.

The unfused operation is tiny but called constantly:

```text
softmax over 4x4 comb logits
column normalize
repeat 19 times:
  row normalize
  column normalize
```

Correctness can start with generic ggml ops, but a fused Sinkhorn op or backend-specific kernel should be prioritized soon after first decode works.

### MTP should be skipped first

MLX drops `mtp.*` weights initially. llama.cpp should do the same until base next-token generation works.

## High-Level Milestones

### Milestone 1: Convert and load a DeepSeek-V4 GGUF

Goal: produce a GGUF that preserves MXFP4 routed experts and can be loaded by llama.cpp.

Scope:

- Add architecture and metadata plumbing.
- Add tensor name definitions.
- Add converter class.
- Convert non-expert FP8 to BF16/F16.
- Preserve expert FP4 as `GGML_TYPE_MXFP4`.
- Skip MTP tensors.
- Load tensors and print sane metadata.

Non-goal:

- No graph execution yet.

### Milestone 2: Tiny synthetic V4 graph runs

Goal: run a small randomly initialized V4-shaped model through prefill + decode.

Scope:

- Implement Hyper-Connections.
- Implement V4 MoE behavior.
- Implement simplified V4 attention with local sliding KV plus compressed KV rows.
- Validate against a Python/MLX reference for small dimensions.

### Milestone 3: Real V4 checkpoints decode slowly

Goal: real Flash checkpoint loads and generates with a slow but correct generic implementation, with Pro covered by the same config-driven code path.

Scope:

- Complete cache state for local sliding window, compressor state, indexer state, and pooled compressed rows.
- Use existing attention machinery with concatenated/gathered local+compressed KV.
- Verify prompt/decode behavior.

### Milestone 4: Performance work

Goal: make real decode usable.

Priority order:

1. Fused Sinkhorn op/kernel.
2. Verify and optimize MXFP4 MoE fast path for `mul_mat_id` on target backend.
3. True sparse attention kernel for selected compressed KV rows.
4. Optional native FP8 non-expert tensor path.

## Detailed Work Breakdown

## Phase 1: GGUF + converter groundwork

### 1.1 Add DeepSeek-V4 architecture plumbing

Files likely involved:

- `src/llama-arch.h`
- `src/llama-arch.cpp`
- `gguf-py/gguf/constants.py`
- `gguf-py/gguf/tensor_mapping.py`
- `convert_hf_to_gguf.py`

Add:

- `LLM_ARCH_DEEPSEEK4`
- GGUF arch name, likely `"deepseek4"`
- Python `MODEL_ARCH.DEEPSEEK4`
- converter registration for `DeepseekV4ForCausalLM`

### 1.2 Add metadata keys

V4-specific metadata needed:

- `compress_ratios` as per-layer array
- `compress_rope_theta`
- `sliding_window`
- `hc_mult`
- `hc_sinkhorn_iters`
- `hc_eps`
- `num_hash_layers`
- `swiglu_limit`
- `o_groups`
- `o_lora_rank`
- `index_n_heads`
- `index_head_dim`
- `index_topk`
- `num_nextn_predict_layers` for future MTP, though MTP execution should be skipped first

Some existing keys can be reused:

- vocab size
- block count
- embedding length
- attention head count
- key/value lengths where applicable
- expert count
- expert used count
- shared expert count
- expert weight scale
- expert norm flag if useful
- RoPE scaling metadata

### 1.3 Add tensor names

Top-level tensors:

- `embed.weight`
- `norm.weight`
- `head.weight`
- `hc_head_fn`
- `hc_head_base`
- `hc_head_scale`

Per-layer attention tensors:

- `layers.N.attn.wq_a.{weight,scale}`
- `layers.N.attn.q_norm.weight`
- `layers.N.attn.wq_b.{weight,scale}`
- `layers.N.attn.wkv.{weight,scale}`
- `layers.N.attn.kv_norm.weight`
- `layers.N.attn.wo_a.{weight,scale}`
- `layers.N.attn.wo_b.{weight,scale}`
- `layers.N.attn.attn_sink`

Per-layer compressor tensors, for layers with `compress_ratio != 0`:

- `layers.N.attn.compressor.wkv.weight`
- `layers.N.attn.compressor.wgate.weight`
- `layers.N.attn.compressor.ape`
- `layers.N.attn.compressor.norm.weight`

Per-layer indexer tensors, for layers with `compress_ratio == 4`:

- `layers.N.attn.indexer.wq_b.{weight,scale}`
- `layers.N.attn.indexer.weights_proj.weight`
- `layers.N.attn.indexer.compressor.wkv.weight`
- `layers.N.attn.indexer.compressor.wgate.weight`
- `layers.N.attn.indexer.compressor.ape`
- `layers.N.attn.indexer.compressor.norm.weight`

Per-layer HC tensors:

- `layers.N.hc_attn_fn`
- `layers.N.hc_attn_base`
- `layers.N.hc_attn_scale`
- `layers.N.hc_ffn_fn`
- `layers.N.hc_ffn_base`
- `layers.N.hc_ffn_scale`

Per-layer MoE tensors:

- `layers.N.ffn.gate.weight`
- `layers.N.ffn.gate.bias` for non-hash layers
- `layers.N.ffn.gate.tid2eid` for hash layers
- `layers.N.ffn.shared_experts.w1.{weight,scale}`
- `layers.N.ffn.shared_experts.w2.{weight,scale}`
- `layers.N.ffn.shared_experts.w3.{weight,scale}`
- `layers.N.ffn.experts.E.w1.{weight,scale}`
- `layers.N.ffn.experts.E.w2.{weight,scale}`
- `layers.N.ffn.experts.E.w3.{weight,scale}`

### 1.4 Converter tensor handling

Implement in `DeepseekV4Model.modify_tensors()` or equivalent:

- Skip all `mtp.*` keys initially.
- Rename HF-style names to GGUF tensor names.
- Stack per-expert routed tensors into llama.cpp expert tensors.
- Convert `w1`, `w2`, `w3` naming to llama.cpp gate/down/up equivalents.
- Convert non-expert FP8 tensors to BF16/F16 using E8M0 scales.
- Assemble expert MXFP4 tensors from raw packed I8/uint8 weight payload plus E8M0 scale bytes.
- Keep `attn_sink`, HC tensors, and router bias/ids as F32/I32 as appropriate.

Validation commands:

```bash
python3 convert_hf_to_gguf.py /path/to/DeepSeek-V4-Flash --outfile /tmp/dsv4-flash.gguf --outtype bf16
./build/bin/gguf-dump /tmp/dsv4-flash.gguf | head -200

python3 convert_hf_to_gguf.py /path/to/DeepSeek-V4-Pro --outfile /tmp/dsv4-pro.gguf --outtype bf16
./build/bin/gguf-dump /tmp/dsv4-pro.gguf | head -200
```

## Phase 2: C++ loader support

Files likely involved:

- `src/llama-hparams.h`
- `src/llama-hparams.cpp`
- `src/llama-model.h`
- `src/llama-model.cpp`
- `src/models/models.h`

Add hparams and tensor pointers. Add `case LLM_ARCH_DEEPSEEK4` for metadata read and tensor creation.

Validation:

```bash
./build/bin/llama-cli -m /tmp/dsv4.gguf -p "hello" -n 1
```

Expected at the end of this phase: load succeeds or reaches a deliberate “graph not implemented” error, not missing tensor/shape failures.

## Phase 3: Generic correctness graph

New file likely:

- `src/models/deepseek4.cpp`

Initial graph behavior should follow MLX PR #1192 rather than the official highly optimized TileLang kernels:

- Use existing attention path where possible.
- Use sliding-window local KV.
- Append compressed KV rows.
- Use top-k gather for `compress_ratio == 4` if feasible.
- For `compress_ratio == 128`, attend over compressed pool directly.
- Pad compressed attention mask with zeros to allow compressed rows.
- Include `attn_sink` in attention if the current attention helper supports it; otherwise add support.

## Phase 4: V4 cache state

Introduce per-layer V4 cache state analogous to MLX:

```text
local sliding-window KV
compressor buffer_kv
compressor buffer_gate
compressor pooled rows
indexer buffer_kv
indexer buffer_gate
indexer pooled rows
```

Do not generalize the global KV cache before correctness requires it. Prefer model-specific state first.

## Phase 5: Optimization

1. Add a fused Sinkhorn op/backend kernel.
2. Verify `GGML_TYPE_MXFP4` routed expert performance through `GGML_OP_MUL_MAT_ID` on target backend.
3. Add true sparse attention kernel.
4. Consider native FP8 tensor support later.

## First Chunk Handoff

### Goal

Start Milestone 1: add the DeepSeek-V4 family conversion and loading foundation without implementing graph execution. The implementation should support both `DeepSeek-V4-Flash` and `DeepSeek-V4-Pro` via `config.json` values from the beginning. Flash remains the lighter first full-conversion target, but Pro shape/metadata support is in scope for the same chunk.

### Recommended first chunk scope

1. Add architecture and metadata plumbing for `DEEPSEEK4`.
2. Add a converter skeleton registered for `DeepseekV4ForCausalLM`.
3. Make converter and loader dimensions fully config-driven so Flash and Pro both parse correctly.
4. Implement safetensors dtype helpers for `F8_E8M0`, `F8_E4M3`, `BF16`, and raw `I8` expert data.
5. Implement non-expert FP8 dequantization to BF16/F16.
6. Implement MXFP4 expert block assembly and write routed expert tensors as `GGML_TYPE_MXFP4`.
7. Skip all `mtp.*` weights.
8. Add a small converter unit/probe path that can process one or two tensors from Flash shard 2 and Pro's equivalent early shard to validate:
   - FP8 non-expert dequantization shape and values are sane.
   - Expert MXFP4 packing matches GGML’s block layout.
   - Flash and Pro config deltas do not require separate architecture code.

### Known source facts to use

Flash config highlights:

```json
{
  "model_type": "deepseek_v4",
  "architectures": ["DeepseekV4ForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 43,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "index_topk": 512,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "sliding_window": 128,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "num_hash_layers": 3,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "num_experts_per_tok": 6,
  "scoring_func": "sqrtsoftplus",
  "swiglu_limit": 10.0,
  "compress_ratios": [0, 0, 4, 128, 4, 128, ...]
}
```

Pro config highlights:

```json
{
  "model_type": "deepseek_v4",
  "architectures": ["DeepseekV4ForCausalLM"],
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 128,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "q_lora_rank": 1536,
  "qk_rope_head_dim": 64,
  "index_topk": 1024,
  "o_groups": 16,
  "o_lora_rank": 1024,
  "sliding_window": 128,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "num_hash_layers": 3,
  "n_routed_experts": 384,
  "n_shared_experts": 1,
  "moe_intermediate_size": 3072,
  "num_experts_per_tok": 6,
  "scoring_func": "sqrtsoftplus",
  "swiglu_limit": 10.0,
  "compress_ratios": [128, 128, 4, 128, 4, 128, ...]
}
```

Observed HF weight patterns:

```text
embed.weight
hc_head_base
hc_head_fn
hc_head_scale
head.weight
layers.N.attn.*
layers.N.attn.compressor.*
layers.N.attn.indexer.*
layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
layers.N.ffn.gate.{weight,bias,tid2eid}
layers.N.hc_attn_*
layers.N.hc_ffn_*
mtp.0.*
norm.weight
```

Shard sample observed from `model-00002-of-00046.safetensors`:

```text
layers.0.ffn.gate.tid2eid       I64      [129280, 6]
layers.0.ffn.gate.weight        BF16     [256, 4096]
layers.0.attn.wo_a.scale        F8_E8M0  [64, 32]
layers.0.attn.wq_a.scale        F8_E8M0  [8, 32]
layers.0.ffn.experts.0.w1.scale F8_E8M0  [2048, 128]
layers.0.attn.wo_a.weight       F8_E4M3  [8192, 4096]
layers.0.attn.wq_a.weight       F8_E4M3  [1024, 4096]
layers.0.ffn.experts.0.w1.weight I8      [2048, 2048]
```

### Files to inspect first

```bash
rg -n "DEEPSEEK2|DeepseekV2Model|MXFP4|MOSTLY_MXFP4_MOE" \
  src gguf-py/gguf convert_hf_to_gguf.py include/llama.h

sed -n '640,720p' gguf-py/gguf/quants.py   # MXFP4 block layout
sed -n '7170,7340p' convert_hf_to_gguf.py  # DeepseekV2Model converter
sed -n '1,220p' src/models/deepseek2.cpp    # existing DeepSeek2 graph pattern
```

### Suggested first validation path

1. Build or reuse existing build.
2. Convert a minimal synthetic/local subset first if possible.
3. Then run full conversion only after tensor mapping and MXFP4 assembly are validated.

Potential quick probes:

```bash
python3 - <<'PY'
# probe safetensors header only; no full download required if using HTTP range manually
PY
```

Then:

```bash
python3 convert_hf_to_gguf.py /models/DeepSeek-V4-Flash --outfile /tmp/dsv4-flash.gguf --outtype bf16
./build/bin/gguf-dump /tmp/dsv4-flash.gguf | rg -n "deepseek4|mxfp4|hc_|compress|indexer|attn_sink" | head -100

python3 convert_hf_to_gguf.py /models/DeepSeek-V4-Pro --outfile /tmp/dsv4-pro.gguf --outtype bf16
./build/bin/gguf-dump /tmp/dsv4-pro.gguf | rg -n "deepseek4|mxfp4|hc_|compress|indexer|attn_sink" | head -100
```

### Completion criteria for first chunk

- `convert_hf_to_gguf.py --print-supported-models` lists `DeepseekV4ForCausalLM`.
- Converter can parse both Flash and Pro V4 configs and emit intended metadata.
- Expert tensors are emitted as `GGML_TYPE_MXFP4` or a clearly documented equivalent path.
- Non-expert FP8 tensors are dequantized to BF16/F16.
- MTP tensors are skipped.
- llama.cpp can at least begin loading Flash and Pro GGUFs and report DeepSeek-V4 metadata.

