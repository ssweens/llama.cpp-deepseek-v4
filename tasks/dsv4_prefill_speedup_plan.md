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

Status: implemented for guarded ratio-4 continuation chunks on `work/dsv4-prefill-speedup` after the main merge.

Current `is_prefill` is effectively only true for chunks starting at position 0:

```cpp
const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;
```

Continuation prompt chunks after cache restore or after the first `n_batch` chunk used decode-style compressor logic. For long prompts, this meant most prompt tokens could miss the vectorized prefill path.

Implemented a vectorized continuation prefill path for the common safe case:

- `compress_ratio == 4`
- single sequence / one seq id
- `n_tokens > 1`
- `first_pos > 0`
- `first_pos % 4 == 0`
- `n_tokens % 4 == 0`
- `ubatch.pos[i] == first_pos + i`

The helper mirrors `/home/bigkahuna/src/ds4` Metal `compressor_prefill_ratio4_replay` semantics:

- Project `kv` and `score` for the whole chunk once.
- Add APE in vectorized `{width, ratio, n_comp}` layout.
- Seed the first shifted previous half from rows `0..3` of the previous compressor state.
- Use the current chunk's previous half for later compressed rows.
- Pool all compressed rows in parallel, apply RMS/norm and compressed RoPE positions.
- Leave recurrent state in post-prefill layout: rows `0..3` hold the final full window; rows `4..7` are cleared until decode writes the next current half.

Fallback remains the existing per-token decode/replay path for decode, multi-seq, non-contiguous positions, non-ratio-4 layers, or unaligned chunks.

Validation:

- Host build: `cmake --build build-vulkan-linux-release --target llama-server -j 8` passed.
- Mounted CUDA/HIP build: `cmake --build /src/build-dsv4-container --target llama-server -j 8` passed inside `llamatrifecta_deepseekv4:latest`.
- First runtime attempt caught a real bug: reshaping non-contiguous previous-state seed views. Fixed by applying `cont_if_needed()` before reshape.
- IQ2_XXS endpoint regression passed all 7 cases after the fix.
- Debug run confirmed `path=prefill-replay` on aligned continuation chunks.

Clean no-debug IQ2_XXS timing from the Mina regression:

| Case | Before vectorized replay | After vectorized replay |
|------|--------------------------|-------------------------|
| Fresh 2486-token Mina prefill | ~220.83 tok/s | 265.04 tok/s |
| Replayed 1078-token Mina continuation | ~151-155 tok/s | 189.85-194.73 tok/s |

Result: useful ~20-28% prefill improvement, but not 2x. Further general-prefill gains likely need backend/profile-led work in prompt indexer/top-k, compressor pooling fusion, cache writes, or MoE/expert matmul throughput.

## Candidate 3 — profiling-guided backend work

After chunking/path fixes, `GGML_PROFILE_OPS=1` on controlled `llama-bench` runs showed sparse attention was material for CUDA prefill:

- IQ2_XXS, p2048/ub2048 before bounded raw-window scan:
  - non-profiled: `345.66 tok/s`
  - profiled `DSV4_SPARSE_ATTN`: `3866.68 ms` / `69.82%`
- Root cause: prompt sparse attention passed `Kcur` as `n_window=n_tokens` plus a dense `[n_tokens,n_tokens]` causal/SWA mask. The CUDA/HIP kernel scanned the full prompt chunk even though DSv4 logical raw SWA is 128 tokens.

Implemented bounded prompt raw-window scanning in `GGML_OP_DSV4_SPARSE_ATTN`:

- New op param `raw_window_limit`; `0` preserves prior full-window behavior.
- Prompt sparse path passes `prompt_window_size` when causal attention is active.
- Decode path passes `0` unchanged.
- CPU and shared CUDA/HIP reference paths honor the bound; Vulkan clone preserves the param but Vulkan execution remains conservative.
- The dense mask is still applied to all scanned rows as a correctness guard.

Results:

| Case | Before | After |
|------|--------|-------|
| CUDA IQ2_XXS p2048/ub2048 non-profiled | 345.66 tok/s | 612.91 tok/s |
| CUDA IQ2_XXS p2048/ub2048 `DSV4_SPARSE_ATTN` | 3866.68 ms / 69.82% | 1315.55 ms / 44.56% |
| CUDA IQ2_XXS p8192/ub512 non-profiled | 263.05 tok/s | 263.42 tok/s |
| ROCm0 IQ2_XXS p64 smoke | n/a | 49.49 tok/s |

Interpretation:

- The bounded raw-window code fix removes a major large-ubatch prompt sparse-attention waste.
- Follow-up mask-skip inside `DSV4_SPARSE_ATTN` avoids loading/dotting raw-window slots that the cache-backed SWA mask marks `-INFINITY`. This preserves behavior because masked slots contribute zero softmax weight.
- CUDA IQ2_XXS p8192/ub512 improved from ~`263 tok/s` after bounded raw-window to `285.11 tok/s` in standalone bench; the same shape's profiled sparse-attn time dropped from `7601.75 ms / 21.75%` to `5055.68 ms / 15.42%`.
- Resident-server API baseline via `llama-benchy` with coherence enabled and `-ub 1024` measured pp2048 `356.36 ± 3.12 tok/s` and pp8192 `276.37 ± 1.18 tok/s` over three runs. Server logs agreed: pp2048 around `366 tok/s`, pp8192 around `277-280 tok/s`.
- Detailed op+tag profiling now shows the remaining wall is routed expert MMQ matmul: `MUL_MAT_ID:moe.ffn_gate`, `MUL_MAT_ID:moe.ffn_up`, and `MUL_MAT_ID:moe.ffn_down` are each ~`4.2-4.3 s` in p8192/ub512. Component diagnostics showed ID compaction and activation quantization are tiny; the expert matmul kernel itself is the wall.
- Reasonable low-risk sparse-attn/prefill cleanup is now mostly squeezed. Further wins require a separate MoE MMQ kernel/tile project or debugging the p8192/ub2048 graph/cuBLAS crash, not small graph plumbing changes.

## Initial ranking

1. Candidate 1: checkpoint-tail policy — easiest, likely immediate win, low correctness risk.
2. Candidate 2: vectorized continuation prefill compressor — larger engineering effort, likely biggest prefill win.
3. Candidate 3: profiler-led backend kernels/fusion — do after 1/2 because current chunking may be distorting profiles.
