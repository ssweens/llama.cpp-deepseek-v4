# DeepSeek-V4 Enablement Todo

## SURGICAL FIXES MADE (replacing approximations with proper ops)

### Custom op infrastructure
- Added `GGML_OP_DSV4_HC_SPLIT_SINKHORN`, `_HC_WEIGHTED_SUM`, `_HC_EXPAND`, `_FP8_KV_QUANTIZE`, `_ROPE_TAIL` to `ggml.h` enum
- Added op-name strings to `ggml.c` GGML_OP_NAME / GGML_OP_SYMBOL tables
- Bumped `GGML_OP_COUNT` from 96 to 101 in `ggml.c` static_asserts
- Added `ggml_dsv4_*` builder functions to `ggml.c` (matches antirez exactly)
- Added CPU forward implementations to `ggml-cpu/ops.cpp` (ported verbatim from antirez)
- Wired CPU dispatch in `ggml-cpu/ggml-cpu.c` (compute switch + n_tasks = n_threads)
- Added CUDA kernels in `ggml-cuda/dsv4.cu` + `dsv4.cuh`
- Wired CUDA dispatch in `ggml-cuda/ggml-cuda.cu` (compute switch + supports_op)
- Wired backend metadata split in `ggml-backend-meta.cpp`
- Added template instantiation `get_arr<vector<float>>` in `llama-model-loader.cpp`

### Graph-level approximation removals (`src/models/deepseek4.cpp`)
- `apply_partial_rope_with_pos`: replaced manual nope/rope split + concat with `ggml_dsv4_rope_tail` (full rope dim path keeps `ggml_rope_ext`)
- `hc_pre`: replaced manual sigmoid+softmax+sinkhorn loop with `ggml_dsv4_hc_split_sinkhorn` + `ggml_dsv4_hc_weighted_sum`. Result is F32 (no spurious BF16 cast)
- `hc_post`: replaced nested per-HC matmul loop with `ggml_dsv4_hc_expand`
- `hc_head`: replaced manual loop with `ggml_dsv4_hc_weighted_sum`
- `apply_fp8_qat_nope_2d`: replaced manual nope-view + map_custom + concat with `ggml_dsv4_fp8_kv_quantize`
- Removed `normalize_comb_rows`/`normalize_comb_cols` lambdas (no longer needed)

### Graph wiring fixes
- KV path: rope BEFORE fp8_kv_quantize (was reversed). Order matters — quantize only touches nope dims, rope only touches rope dims; reversing produced wrong values for `KVcur`.
- Compressed attn pool path: rope BEFORE fp8 quantize (was reversed). Same issue.
- Indexer compressor pool path: removed FP8 quantize (was incorrect; antirez stores indexer cache in BF16, no QAT).
- Initial HC state: matched antirez `repeat_4d` pattern (semantically equivalent, cleaner graph).
- Q path: removed redundant BF16<->F32 round-trip after rope.
- Output path: kept tensor 3D through inverse-rope and grouped output projection (avoids extra reshape2d roundtrips).
- Grouped output projection: replaced manual concat-loop with `ggml_mul_mat_id`+`ggml_arange`+`ggml_repeat_4d` ids (matches antirez `dsv4_grouped_out`).

### Converter / GGUF format alignment (matches antirez wire format)
- Metadata key renames in `gguf-py/gguf/constants.py`:
  - `o_lora_rank` -> `output_lora_rank`
  - new `hash_layer_count` (replacing `num_hash_layers`)
  - `hyperconnection.{mult,sinkhorn_iters,eps}` -> `hyper_connection.{count,sinkhorn_iterations,epsilon}`
  - `compress_rope_theta` -> `compress_rope_freq_base`
  - `index_head_count` -> `indexer.head_count`
  - `index_head_dim` -> `indexer.key_length`
  - `index_topk` -> `indexer.top_k`
- Tensor name renames:
  - `attn_o_a/b` -> `attn_output_a/b`
  - `attn_compressor_wkv/wgate` -> `attn_compressor_kv/gate`
  - `attn_indexer_*` -> `indexer.*` and `indexer_compressor_*`
  - `hc_head_*` -> `output_hc_*` (with `.weight` suffix)
  - `attn_wkv` -> `attn_kv`
- `swiglu_limit` (scalar) -> `swiglu_clamp_exp` (per-layer F32 array)
- `tn(LLM_TENSOR_HC_*, "weight", i)` everywhere it was `tn(LLM_TENSOR_HC_*, i)` in `llama-model.cpp`
- `ffn_gate_inp_b` (tid2eid) loaded with `"weight"` suffix
- C++ enum -> string mapping in `llama-arch.cpp` updated to match antirez wire format
- `LLM_KV_SWIGLU_LIMIT` reads now `LLM_KV_SWIGLU_CLAMP_EXP` array first, falls back to scalar (forward-compatible)

### Improvements over antirez (preserved)
- Hybrid-iswa compressed cache + recurrent decode-state machinery (antirez fork is CPU/Metal only and may have different cache management)
- Indexer top-k decode mask path (in our hybrid-iswa decode flow)
- Compressed RoPE attn_factor: cancels YaRN magnitude scale for compressed layers (matches HF inference exactly)
- DeepSeek-V4-specific resumed-prompt buffer reservation in `llama-context.cpp` `sched_reserve`

## Scope

Complete DeepSeek-V4 support end-to-end in staged milestones:
1. Conversion/loading foundation (Flash + Pro metadata/tensor coverage)
2. Generic correctness graph (synthetic + real checkpoint decoding)
3. Model-specific cache and correctness hardening
4. Performance optimization (Sinkhorn, sparse compressed attention paths)

## Tasks

- [x] Add `DEEPSEEK4` architecture plumbing in C++ and gguf-py.
- [x] Add DeepSeek-V4 metadata keys and tensor mappings needed by converter/loader.
- [x] Add a `DeepseekV4Model` converter registered for `DeepseekV4ForCausalLM`.
- [x] Make converter/loader dimensions config-driven so Flash and Pro both parse.
- [x] Add safetensors helpers for `F8_E8M0`, `F8_E4M3`, `BF16`, and raw `I8` expert data.
- [x] Dequantize non-expert FP8 tensors to BF16/F16 during conversion.
- [x] Preserve routed expert weights as `GGML_TYPE_MXFP4` where possible.
- [x] Skip `mtp.*` tensors for the first chunk.
- [x] Add/adjust loader tensor creation so a converted GGUF reaches an intentional graph-not-implemented path instead of metadata/tensor failures.
- [x] Run targeted converter/static checks.
- [x] Run relevant build/test/format quality gates.

## Immediate Follow-up

- [ ] Trace DeepSeek-V4 official chat/output encoding end-to-end before further graph changes.
  - [ ] Use the official HF `encoding/encoding_dsv4.py` contract as the source of truth.
  - [ ] Stop using legacy `deepseek3` assumptions for validation; verify with DSML / `<think>` handling and raw-output parsing.
  - [ ] Either embed the official V4 template/encoding in GGUF/llama.cpp or use the official prompt+parse flow during validation.
  - [ ] Re-run smoke validation with the official V4 format, then separate parser issues from model-graph issues.

## Current Debug Cycle

### Active Plan: Real-checkpoint semantic parity

- [ ] Use all references directly for the remaining semantic gap, not just the existing port:
  - [x] Paper: CSA/HCA require SWA + compressed KV concat, head RMSNorm, partial RoPE on q/kv/out, attention sink, CSA indexer top-k, HCA dense compressed visibility, FP8-simulated non-RoPE KV, FP4-simulated indexer q/cache path.
  - [x] Official HF inference: first-token/prompt path uses sparse-attn over gathered SWA indices plus compressed indices; decode path uses ring SWA cache plus compressed cache; indexer applies q Hadamard+FP4 and rotated compressed index KV.
  - [x] MLX reference: fetched current `ml-explore/mlx-lm` PR 1192 (`/tmp/mlx-lm`, diff saved to `/tmp/mlx_pr1192_latest.diff`) for latest DeepSeek-V4 semantics instead of relying on stale local snapshots.
  - [x] Inspect `0xSero/deepseek-v4-flash-sm120` for additional DeepSeek-V4 Flash/SM120 implementation clues and separate kernel/perf ideas from graph/semantic fixes.
    - [x] Relevant semantic/validation notes: use official `encoding_dsv4.py`/SGLang built-in `deepseekv4` parser, do not pass a custom Jinja template, use compressed attention backend, use FP8 KV cache layout, and their correctness-first launch disables native FP4 experts (`SGLANG_DSV4_FP4_EXPERTS=0`) in favor of the FP8 checkpoint.
    - [x] Kernel-specific SM120 sparse decode code is not directly portable to llama.cpp correctness; it is useful later for CUDA perf/layout validation, especially FP8 paged MQA cache semantics.
  - [ ] Antirez fork: diff graph/runtime implementation against this branch for remaining first-token differences.
- [ ] Fix or isolate real Flash BF16 semantic failure under deterministic short-generation.
  - [x] Current failure: model now exits cleanly but BF16 first generated text after `The capital of France is` is nonsensical (`Kadaghan...`).
  - [~] Compare current graph against official/MLX/antirez on attention sink, SWA index layout, compressed index offset, RoPE base/scaling, FP8/FP4 activation simulation, and MoE routing precision.
    - [x] Found a concrete antirez/reference mismatch in compressed RoPE magnitude scaling: DeepSeek-V4 uses YaRN frequency interpolation for compressed layers but does not apply YaRN magnitude scaling; local code was passing the generic context `attn_factor`, so ggml's internal YaRN scale was not canceled for compressed layers.
    - [x] Updated compressed-layer RoPE config to pass the inverse YaRN magnitude factor, matching antirez and official runtime behavior.
    - [x] Found another antirez/reference mismatch in hash-routed MoE: local code emulated fixed `tid2eid` routing by masking logits and re-running top-k, while antirez/official pass the hash-selected expert ids directly and gather weights from the original router scores. Added an optional selected-expert override to `build_moe_ffn` and use it for DeepSeek-V4 hash layers.
    - [x] Tested official-runtime routed-expert activation QAT for MXFP4 experts, but parked/reverted it because it did not improve real-output sanity and `0xSero`'s correctness recipe explicitly disables native FP4 experts in SGLang for the FP8 checkpoint.
    - [x] Found a high-impact MoE semantic mismatch in current graph vs paper/HF/MLX/antirez: DeepSeek-V4 multiplies routed weights after SwiGLU and before expert down projection (`w2`), while local generic MoE applied weights after down projection. Added `weight_before_down` for `LLM_ARCH_DEEPSEEK4`.
  - [ ] Validate any candidate fix on tiny controls and real Flash BF16/MXFP4 short-generation.

- [x] Re-read the official DeepSeek-V4 `Attention` / `Indexer` / `Compressor` prompt-time path and mirror its exact top-k index construction semantics in `src/models/deepseek4.cpp`.
- [x] Replace the current prompt-time compressed-attention approximation with official-style sparse index selection:
  - [x] build the sliding-window top-k indices exactly like `get_window_topk_idxs(...)`
  - [x] build compressed top-k indices exactly like `get_compress_topk_idxs(...)` / `Indexer.forward(...)`
  - [x] concatenate raw-window + compressed indices and gather from a unified prompt KV tensor
- [x] If the first token is still wrong, create a small random DeepSeek-V4 differential harness (official PyTorch vs GGUF/llama.cpp) to isolate the next semantic mismatch without relying on the 142B checkpoint.
- [ ] Use the new tiny official-format differential harness to isolate remaining real-checkpoint-only behavior:
  - [x] validate indexed sparse prompt attention (`compress_ratio = 4`) against official PyTorch
  - [x] validate generic compressed attention (`compress_ratio > 4`, no indexer) against official PyTorch
  - [x] validate hash-routing parity for `num_hash_layers > 0`, including `num_experts_per_tok > 1`
  - [x] validate grouped output projection parity for `o_groups > 1`
  - [x] validate that expert-only `MXFP4` quantization on the tiny model preserves the top token
  - [ ] investigate real-checkpoint-only tensor-format / quantization behavior that the tiny BF16 controls still do not exercise
    - [ ] read the DeepSeek-V4 paper sections relevant to architecture / precision / routing before further FP8-path changes
    - [x] build a tiny control that exercises the converter's raw non-expert FP8 path end-to-end
    - [~] compare converter FP8 reconstruction against the official/expected dequantized tensor values
      - [x] verify from the paper + official inference code that dense FP8 layers perform runtime activation quantization (`act_quant`) and FP8 GEMM, not just static weight dequantization
      - [x] verify that the real checkpoint contains a non-MTP `attn.wo_a.scale` artifact that the official inference module does not consume
      - [ ] decide whether DeepSeek-V4 parity requires preserving native dense FP8 tensors into GGUF/runtime rather than dequantizing them to BF16 at conversion time
      - [~] first test the lower-risk parity step: simulate official blockwise activation FP8 quantization on every dense FP8 DeepSeek-V4 linear path while keeping dequantized BF16 weights
        - [x] add a DeepSeek-V4 GGUF metadata flag for dense-FP8-origin checkpoints and route the graph through activation-QAT branches
        - [x] fix the local E4M3 helper so clamped scalar round-trips match `torch.float8_e4m3fn`
        - [x] add a legal 128-aligned raw-FP8 differential control instead of relying on invalid tiny dims
        - [x] refine the metadata to per-subpath flags (`attn_qkv`, `attn_out`, `indexer_q`, `shared_expert`) so synthetic controls can isolate specific FP8 branches
        - [~] current status: the main qkv dense-FP8 interaction bug was fixed by BF16-rounding the KV norm output before the DeepSeek-V4 non-RoPE KV QAT step, but multi-layer parity is still incomplete
          - [x] isolate the earlier `wq_a + wkv` mismatch and fix it so `qmain_only` now matches the Python reference on the legal control
          - [x] verify the one-layer controls for both the first plain layer and the renamed compressed second layer each match the Python reference on `hello`
          - [x] remaining two-layer mismatch was confirmed to be FP8 activation QAT precision accumulation, NOT a graph correctness bug:
              - all three failing controls (`qmain_wo_b`, `attn_only`, `legal`) match Python perfectly when FP8 flags are disabled via `--override-kv deepseek4.fp8.*=bool:false`
              - Python top-10 logits show marginal gaps (0.026 between `advant` and `Staying`)
              - confirmed by cross-checking against MLX DeepSeek-V4 implementation
          - [x] graph math verified correct for multi-layer HC state propagation, Sinkhorn comb normalization, compressed attention, hash routing, and grouped output projection
          - [~] new finding from longer tiny-generation tests: decode-time parity is still incomplete even in no-FP8 mode after several generated tokens (first tokens match, later tokens drift), which points to missing/approximate decode compressed-cache state handling rather than prompt-only graph math
          - [x] external reference identified: `antirez/llama.cpp-deepseek-v4-flash` implements explicit DeepSeek-V4 decode compressor/indexer state (`dsv4_build_compressor_decode*`, per-layer compressed cache writes, decode-time top-k indexer masks) that is currently absent in this branch
          - [x] added DeepSeek-V4 recurrent-state groundwork needed for decode-cache porting:
              - `llama_hparams.deepseek4_state_size`
              - `n_embd_r()` / `n_embd_s()` now return `deepseek4_state_size` when set
              - DeepSeek-V4 loader computes `deepseek4_state_size` from per-layer `compress_ratios`, head dims, and indexer state requirements
          - [x] ported DeepSeek-V4 compressed-cache state persistence + APIs into `llama-memory-hybrid-iswa` (attn/index compressed cache buffers, seq ops, state read/write)
          - [~] wired DeepSeek-V4 graph decode path to recurrent compressor state + compressed cache updates:
              - [x] prefill now stores compressor/indexer tail state in recurrent `r/s` cache segments
              - [x] prompt sparse path now writes compressed attn/index caches into hybrid-iswa DSV4 cache tensors
              - [x] decode path now updates compressor/indexer recurrent states and appends newly compressed rows to DSV4 cache
              - [x] decode indexer top-k compressed mask path now wired in hybrid-iswa decode:
                  - builds decode indexer scores from cached compressed index KV + decode q/weights projections
                  - applies decode-time causal valid mask before top-k selection
                  - constructs sparse compressed mask from top-k (`ggml_argsort_top_k` + `ggml_set_rows` mask build)
                  - retains absolute-causal fallback for non-indexed compressed layers (`ratio != 4`)
              - [x] hybrid-iswa activation stabilized for DeepSeek-V4:
                  - fixed prompt-sparse KV writes to use SWA cache indices in hybrid-iswa mode (base cache can be empty)
                  - aligned DSV4 state/cache store helpers with antirez semantics (`set_rows` + flat state segments)
                  - fixed decode-time mask/KV concat dtype issues for CUDA flash-attn path
                  - DeepSeek-V4 now initializes and runs under hybrid-iswa on tiny controls and real Flash smoke (`-ngl 8`, short decode)
              - [x] decode-time indexer top-k path implemented and smoke-tested; retained BF16 cache/indexer precision (no local Hadamard/FP4 approximation) because applying the approximate prompt-time FP4/Hadamard transform to cached index rows introduced input-buffer allocation failures and is not proven against the official runtime path
    - [ ] re-run first-token validation on real Flash BF16 and MXFP4 after any FP8-path fix
- [ ] Re-validate whether exact prompt-time sparse selection changes the first generated token on BF16/MXFP4 Flash at `-ngl 8` after each remaining real-checkpoint-specific fix.

## Remaining Milestones

- [x] Milestone 1: conversion/loading foundation complete.
- [~] Milestone 2: generic correctness graph started.
  - [x] Replace `deepseek4` graph stub with a runnable baseline graph path.
  - [x] Add runtime support for DeepSeek-V4 router gating func `sqrtsoftplus`.
  - [x] Synthetic model now loads and decodes (baseline graph path).
  - [~] Implement proper hyper-connection (mHC) residual mixing in graph path (HC tensors now load and Sinkhorn-based graph path runs on the real checkpoint; output quality is still incorrect, so parity work remains).
  - [~] Implement token-id hash routing behavior for first `num_hash_layers` (enabled real `ffn_gate_tid2eid` masking with selected-expert router logits; needs further real-checkpoint quality verification).
  - [~] Implement full V4 attention path (CSA/HCA compressor/indexer semantics, not placeholder shared-KV path).
    - [x] Loader now wires DeepSeek-V4 compressor/indexer tensors into runtime layer structs.
    - [x] Added compressor pooling scaffold (wkv/wgate + APE + softmax window reduction + overlap handling for ratio=4).
    - [x] Added indexer scoring/top-k sparse prior scaffold (`attn_indexer_q_b`, compressor pool, `argsort_top_k`).
    - [ ] Feed compressed/indexed KV paths into the real attention computation with correct causal masking/selection semantics.
    - [ ] Replace scaffolds with exact sparse attention masking/selection semantics.
- [ ] Milestone 3: real Flash/Pro checkpoints decode (slow but correct).
- [ ] Milestone 4: performance optimization (fused Sinkhorn, sparse compressed attention kernels, cache memory work).

## Review

- Added DeepSeek-V4 tokenizer fallback to `tokenizer.json` in converter when `AutoTokenizer` fails on unknown `deepseek_v4` config type.
- Verified conversion of synthetic DeepSeek-V4 model succeeds and writes GGUF with DeepSeek-V4 metadata/tensors.
- Verified routed experts are emitted as `MXFP4` in GGUF.
- Added initial DeepSeek-V4 graph builder path in `src/models/deepseek4.cpp`.
- Added `sqrtsoftplus` support to MoE gating in `src/llama-graph.cpp`.
- Added mHC residual-mixing scaffold that uses loaded HC tensors to modulate pre/post residual contributions.
- Added hash-routing scaffold for first `num_hash_layers` using `ffn_gate_tid2eid` + input token IDs to produce sparse expert prior logits.
- Added DeepSeek-V4 compressor/indexer tensor pointer plumbing (`llama_model.h`/`llama-model.cpp`) and graph-side pooling/indexer scaffolding based on `compress_ratio`.
- Extended the graph path to consume `attn_compressor_*` and `attn_indexer_*` tensors (including APE/norm parameters) so real checkpoints exercise full tensor loading/plumbing paths, though those tensors are not yet wired into exact sparse attention selection.
- Corrected DeepSeek-V4 loader dimensions for HC/compressor/indexer tensors to match reference architecture shape rules (`hc_fn` state size = `hc_mult * hidden_size`, compressor out dims = `head_dim` or `2*head_dim` for ratio 4, indexer compressor out dims = `2*index_head_dim`).
- Added DeepSeek-V4 per-layer norm support (`attn_norm` / `ffn_norm`) in converter+loader+graph path (with optional fallback for older synthetic artifacts).
- Fixed loader tensor-op metadata for DeepSeek-V4 HC tensors (`hc_head_*`, `hc_attn_*`, `hc_ffn_*`) and `ffn_gate_tid2eid` so the loader no longer discards them as `GGML_OP_NONE`/unused.
- Increased DeepSeek-V4 graph node reservation sizing so the new HC graph path can build on the real checkpoint without `ggml_new_tensor`/`obj_new` reservation aborts.
- Improved hash-routing scaffold so sparse MoE priors now keep router logits for selected experts instead of writing constant values.
- Added shape assertions and debug checkpoints for HC, compressor/indexer, and hash-routing tensors in DeepSeek-V4 graph build path.
- Cross-checked partial real Flash safetensors shape metadata against loader assumptions (e.g. `hc_attn_fn` 24x16384, compressor 1024x4096 / indexer compressor 256x4096, `ffn.gate.tid2eid` vocab x topk) and aligned loader dimension formulas accordingly.
- Verified synthetic decode reaches runtime generation path (no longer fails at graph-not-implemented boundary).
- Researched the official DeepSeek-V4 chat/output contract from the HF release: the release explicitly does **not** ship a Jinja chat template, and instead defines prompt+parse behavior in `encoding/encoding_dsv4.py` / `encoding/README.md` using `<｜Assistant｜>`, `<think>...</think>`, and DSML tool-call blocks.
- Verified the downloaded HF `tokenizer_config.json` contains no `chat_template`, so the current GGUF lacks embedded chat-template metadata; this makes prior `--chat-template deepseek3` validation unreliable.
- Re-ran validation with the official V4-style DSML prompt contract (`/tmp/dsv4_prompt_chat.txt`) plus `--special`; after the HC loader fix and hash-routing restoration, the real model now emits `<｜image｜>` as the first generated token instead of repeated raw `"<"`, confirming progress but also confirming the remaining correctness issue is still in model semantics rather than chat parsing alone.
- Verified the same `<｜image｜>` first-token failure on the BF16 GGUF (`DeepSeek-V4-Flash-bf16.gguf`), which rules out MXFP4/Q8 quantization as the primary cause.
- Fixed a DeepSeek-V4 converter/runtime metadata gap for MoE routing: the converter now writes `expert_weights_scale` / `expert_weights_norm`, and the loader now defaults DeepSeek-V4 to the official `routed_scaling_factor=1.5` and `norm_topk_prob=true` when older GGUFs omit those keys.
- Added DeepSeek-V4 KV-path FP8 QAT simulation for the main non-RoPE KV slice, matching the official inference path more closely; this alone was not sufficient to fix generation.
- Corrected DeepSeek-V4 layer-dependent RoPE behavior in the graph: compressed-attention layers now use `compress_rope_theta` + YaRN, while non-compressed sliding-window layers use base `rope_theta` without YaRN, matching the official reference architecture.
- Added an intermediate prompt-time compressed-KV attention approximation (concatenated completed compressed windows with a causal mask) to test whether missing CSA participation was the dominant issue; output quality did not materially change, so exact sparse/indexed attention remained the next major gap at that stage.
- Replaced the earlier prompt-only approximation with an exact prompt-time sparse path for DeepSeek-V4:
  - raw sliding-window masking now mirrors `get_window_topk_idxs(...)`
  - compressed selection now mirrors `get_compress_topk_idxs(...)` / `Indexer.forward(...)`
  - indexed layers now apply the official Hadamard rotation before sparse top-k scoring
  - added an approximate FP4 QAT clamp path for indexed q/kv scoring activations
- Added a dedicated DeepSeek-V4 loader hparam for `attention.sliding_window` so the graph can use the official prompt-time window size without re-enabling generic SWA/ISWA cache plumbing.
- Built a tiny official-format DeepSeek-V4 differential harness and validated llama.cpp top-token parity against the official PyTorch model for all of the following branches:
  - indexed sparse attention (`compress_ratio = 4`)
  - generic compressed attention without an indexer (`compress_ratio = 8` control for the non-overlap path)
  - hash routing with `num_hash_layers > 0`
  - hash routing with `num_experts_per_tok > 1`
  - grouped output projection (`o_groups > 1`) after fixing the loader formula for `attn_o_a`
- Fixed a generic DeepSeek-V4 loader bug for grouped output projection: `attn_o_a` input width should be `(n_head * head_dim) / o_groups`, which equals `n_embd` for real Flash/Pro but did not hold for synthetic grouped controls.
- Built a tiny expert-only `MXFP4` quantization control by quantizing just the routed expert tensors in a small official-format model; the top generated token matched the BF16 tiny model, which reduces the likelihood that llama.cpp's generic MXFP4 expert runtime path is the primary root cause of the real Flash `<｜image｜>` failure.
- Despite those graph-side parity checks, the real Flash checkpoint still emits `<｜image｜>` as the first token at `-ngl 8`, so the remaining problem space is now narrowed primarily to real-checkpoint-specific tensor-format / quantization behavior rather than the generic DeepSeek-V4 graph math exercised by the tiny controls.
- Read the DeepSeek-V4 paper sections covering inherited MoE/hash routing, CSA/HCA, FP4 QAT, and the inference KV-cache design before further precision-path changes. Relevant paper-backed constraints for the implementation are:
  - routed experts use FP4/MXFP4 QAT and the CSA indexer QK path uses FP4 QAT
  - queries/compressed KV entries get extra RMSNorm before core attention
  - RoPE is applied to the last 64 dims of q/kv and inversely applied on core attention outputs
  - SWA plus unready-for-compression tail states belong to a separate state-cache path, distinct from the classical compressed KV cache
- Built a tiny official-format raw-FP8 control that mimics the real checkpoint's dense-weight precision path more closely than the earlier BF16 controls. That control currently **does not** match the official reference after conversion to a BF16 GGUF, which is a strong sign that the current converter strategy of dequantizing dense FP8 weights to BF16 is not sufficient for exact DeepSeek-V4 parity.
- Audited the official inference module structure against the real checkpoint tensor names and confirmed one concrete scale-tensor artifact: the checkpoint ships `layers.*.attn.wo_a.scale`, but the official grouped output pre-projection `wo_a` path is instantiated as BF16 and does not consume a scale parameter. Added a targeted converter exception so this extra scale tensor is ignored instead of being folded into `attn_o_a`.
- Added DeepSeek-V4 GGUF metadata to preserve whether the source checkpoint used dense FP8 weights, and then refined that to per-subpath flags for attention qkv, attention output projection, indexer q-projection, and shared experts. This allows the runtime to enable FP8 activation-QAT only for the specific branches that were actually scaled in the source checkpoint.
- Fixed a serious bug in the local DeepSeek-V4 E4M3 helper: the earlier encode/decode logic did not match `torch.float8_e4m3fn` for clamped finite values. Replaced it with a table-driven finite-value implementation that matches PyTorch on scalar round-trips.
- Ported antirez DeepSeek-V4 memory-hybrid compressed cache infrastructure into this branch (`src/llama-memory-hybrid-iswa.h/.cpp`) and adapted it to local hparams field names.
- Wired DeepSeek-V4 graph code to use `build_inp_mem_hybrid_iswa()` when available and added decode-time recurrent compressor state update + compressed-cache append flow for both attn and indexer compressors.
- Added guarded fallback to legacy KV graph input when hybrid-iswa context is not active, and stabilized full hybrid-iswa activation for DeepSeek-V4.
- Fixed hybrid-iswa prompt-time KV writes to target SWA cache indices (`get_k_idxs_swa` / `get_v_idxs_swa`) instead of base-cache indices, preventing cache map misses when base cache has zero layers.
- Aligned DeepSeek-V4 state/cache update helpers with the antirez implementation (`dsv4_store_state_segment`, `dsv4_store_cache_rows`, state-segment views), which removed a decode-time memory corruption path.
- Fixed decode-path dtype handling for CUDA (`concat` in F32 then optional cast for flash-attn), allowing real Flash short-decodes to run without concat-type asserts.
- Implemented decode-time indexer top-k mask construction (previously absolute fallback) using cached compressed index KV and decode q/weight projections; retained BF16 index cache precision rather than the earlier approximate Hadamard/FP4 transform because that transform caused allocation failures and is not yet proven against the official runtime path.
- Built a legal 128-aligned raw-FP8 differential control and used it to refine the remaining mismatch:
  - baseline dequantized-BF16 conversion still matches the Python no-act reference on multiple prompts
  - shared-expert FP8 activation-QAT now matches the Python act reference on the legal control
  - added per-subpath GGUF metadata so the runtime can isolate `wq_a`, `wq_b`, `wkv`, `wo_b`, `indexer.q_b`, and shared-expert FP8 branches independently
  - fixed the earlier `wq_a + wkv` dense-FP8 interaction by BF16-rounding the KV norm output before the non-RoPE KV QAT step
  - after that fix, `qmain_only` now matches the Python reference, but the remaining mismatch only appears once multiple layers interact (`qmain + wo_b`, `attn_only`, and the full legal control)
  - both corresponding one-layer controls now match the Python reference:
    - first plain layer only: Python and llama.cpp both emit `' factorial'`
    - renamed compressed second layer only: Python and llama.cpp both emit `'劳'`
  - this strongly suggests the remaining bug is in two-layer interaction / state propagation precision, not in either single layer by itself
