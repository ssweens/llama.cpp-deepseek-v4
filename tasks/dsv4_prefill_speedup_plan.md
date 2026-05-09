# DeepSeek4 Prefill Speedup Evaluation

Branch: `work/dsv4-prefill-speedup`
Date: 2026-05-09

## Current evidence

Systemctl-managed Corral logs for `deepseek-ai/DeepSeek-V4-Flash-IQ1_M` show prompt prefill is being fragmented into small chunks by the checkpoint-tail policy:

- 858-token prompts are processed as `128 + 128 + ... + 90` token batches, creating checkpoints at 128-token boundaries.
- A longer 7963-token request after prompt-cache checkpoint restore processed a large `4096`-token continuation chunk, then split the remaining tail into `128`-token chunks before a final `27`-token batch.
- The crash at `GGML_SCHED_MAX_SPLIT_INPUTS` happened after this tail prefill/checkpoint sequence; it is a separate scheduler split-overflow bug already fixed locally in `aafa0ad91`.

The 128-token fragmentation is not model math; it comes from `tools/server/server-context.cpp` tail-checkpoint logic:

```cpp
for (int offset = 4 + ((n_ubatch / n_swa) * n_swa); offset > 4; offset -= n_swa) {
    if (should_checkpoint_tail(offset)) {
        should_break = true;
        break;
    }
}
```

For DSv4 `n_swa = 128`, so even prompts smaller than `n_ubatch` can be forced into 128-token prefill batches just to manufacture rollback checkpoints.

## Why this hurts DSv4 prefill disproportionately

DSv4 has two very different prompt/compressor paths:

1. **First prompt chunk (`ubatch.pos[0] == 0`)**
   - Uses vectorized prefill compressor/pool path (`build_compressed_pool`).
   - Processes many prompt tokens in parallel.

2. **Continuation chunks (`ubatch.pos[0] > 0`)**
   - Uses `dsv4_build_compressor_decode_chunk()`.
   - It projects the chunk once, but then emits per-token graph nodes for recurrent compressor state updates and compressed outputs.
   - Artificially splitting a prompt into 128-token chunks moves most of the prompt off the vectorized prefill path and onto this heavier continuation path.

So checkpoint-tail splitting has a double cost:

- smaller batches reduce GPU utilization and graph/CUDA-launch amortization;
- later chunks use the slower continuation compressor graph instead of the vectorized prefill graph.

## Candidate 1 — low risk: stop making SWA-spaced tail checkpoints by default for DSv4 prefill

Status: implemented locally on this branch.

Change the tail checkpoint policy so DSv4 does **not** force every prompt tail through 128-token chunks by default.

Possible policy:

- Keep a checkpoint at `4 + n_ubatch` from the end if useful.
- Keep the final `4` token replay guard if required for logits/checkpoint correctness.
- Disable the SWA-spaced `... 1028, 900, 772, ...` checkpoint series by default for DSv4.
- Optionally expose this as a server parameter if upstream compatibility requires keeping old behavior available.

Implemented policy:

- For contexts where `server_dsv4_cache_alignment(...) > 1`, skip the SWA-spaced loop:
  `4 + n_ubatch`, `4 + n_ubatch - n_swa`, ..., `4 + n_swa`.
- Keep the existing `4 + n_ubatch` guard and final aligned `4` guard.
- Non-DSv4 SWA/hybrid models keep the old checkpoint-tail behavior.

Expected effect:

- Short/medium prompts under `n_ubatch` should usually run as one or two prompt graphs instead of many 128-token graphs.
- Long prompts avoid many 128-token tail graphs.
- Prompt-cache rollback may replay up to roughly one ubatch more in later turns, but that is likely cheaper than slowing every prefill to create dense checkpoint coverage.

Risk:

- More cache-reuse reprocessing after prompt divergence near the tail.
- Should not affect generation correctness if checkpoint restore still falls back safely.

Validation:

- Run systemctl-managed Corral prompt-cache replay and verify no crash/corruption.
- Compare prompt eval time for the same 858-token and ~8k-token Pi histories.
- Verify `cached_tokens` / checkpoint restore still work for repeated-turn history.

## Candidate 1.5 — immediate continuation-prefill graph waste

Status: implemented locally on this branch.

During the Candidate 2 audit, we found continuation prompt chunks were doing redundant compressor work:

- `c_pool = build_compressed_pool(...)`
- `idx_pool = build_compressed_pool(...)`

These pools are prompt-only inputs for the first-chunk sparse prompt path. For nonzero-position chunks, `use_local_prompt_attn` is false, so those tensors are not consumed by attention; the graph falls through to the cache-aware decode/replay compressor path and computes compressor outputs again.

Fix:

- Move `use_local_prompt_attn` before pool construction.
- Build `attn_compressor_pool` / `attn_indexer_pool` only when `use_local_prompt_attn` is true.
- Keep recurrent state construction/cache-aware decode path unchanged.

Expected effect:

- Large continuation prefill chunks after prompt-cache restore skip two unused compressor projection/pooling subgraphs per compressed layer.
- This should improve long-history prefill even before implementing a fully vectorized continuation compressor.

Risk:

- Low: the gated tensors were only used in the first-chunk prompt-sparse path. Nonzero-position chunks already require the decode/replay path to preserve prior context.

## Candidate 2 — larger architecture win: vectorized continuation prefill compressor

Current `is_prefill` is effectively only true for chunks starting at position 0:

```cpp
const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;
```

Continuation prompt chunks after cache restore or after the first `n_batch` chunk use decode-style compressor logic. For long prompts, this means most prompt tokens can miss the vectorized prefill path.

Implement a vectorized continuation prefill path for contiguous single-sequence chunks that start on a safe compression/checkpoint boundary:

- Preconditions:
  - `has_hybrid_iswa`
  - single contiguous sequence
  - `n_tokens > 1`
  - `first_pos % compress_ratio == 0` or stronger DSv4 alignment (`first_pos % 128 == 0`) for current cache-safety rules
- For ratio 4, seed the first shifted/previous window from `prev_kv_state` / `prev_score_state` instead of zero/`-inf`, then use the same vectorized pooling style as `build_compressed_pool` for the chunk.
- Store the generated compressed rows and final recurrent state exactly as the decode-chunk path does.

Expected effect:

- Large continuation prefill chunks after prompt-cache restore should move from per-token recurrent graph construction to parallel compressor/pool math.
- This is the more plausible path to a real 2x prefill improvement on multi-thousand-token prompts.

Risk:

- Higher correctness risk than Candidate 1: ratio-4 overlap state must match decode semantics exactly.
- Needs parity against existing decode-chunk path for several `(first_pos, n_tokens, ratio)` cases.

Validation:

- Build a small backend/unit harness or add debug compare mode that constructs both old decode-chunk and new vectorized-continuation outputs for the same chunk and checks final `kv_state`, `score_state`, and `kv_comp`.
- Then validate full model with systemctl-managed Corral histories.

## Candidate 3 — profiling-guided backend work

After chunking/path fixes, use `GGML_CUDA_PROFILE_OPS=1` on a controlled standalone harness or systemctl-managed service config to identify remaining prefill op costs.

Likely cost centers to confirm:

- MoE `MUL_MAT_ID`
- dense `MUL_MAT` / cuBLAS projections
- compressor/indexer projections
- KV/cache writes (`CPY`, `SET_ROWS`)
- scheduler split copies on multi-GPU layer split
- host-filled mask inputs for prompt/decode masks

Do not optimize sparse attention first unless profiling shows it is actually material.

## Initial ranking

1. Candidate 1: checkpoint-tail policy — easiest, likely immediate win, low correctness risk.
2. Candidate 2: vectorized continuation prefill compressor — larger engineering effort, likely biggest prefill win.
3. Candidate 3: profiler-led backend kernels/fusion — do after 1/2 because current chunking may be distorting profiles.
