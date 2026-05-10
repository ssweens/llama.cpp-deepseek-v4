# DeepSeek4 MTP Loader/Probe Notes

Branch: `work/dsv4-mtp-loader-probe`
Date: 2026-05-09

## Implemented in this branch

- Added server-only `--mtp-model FNAME` and `--mtp-draft N` flags.
- Added default-off fields to `common_params_speculative`:
  - `mtp_model`
  - `mtp_draft`
  - `has_mtp()`
- Added server startup validation for DeepSeek4 MTP sidecar GGUFs.
- Added a default-off DeepSeek4 final-HC output hook for one-token decode graphs.
- Added a host-side copy buffer for that final HC state in `llama_context` for the next draft-one probe step.
- Added env-gated sidecar tensor data loading into a persistent backend weight buffer.
- Kept runtime MTP drafting/speculative commit disabled. Passing `--mtp-model` validates only and logs that drafting is not enabled yet; `DSV4_MTP_PROBE=1` additionally loads sidecar tensor data and enables HC-state capture plumbing.
- Documented the new server flags in `tools/server/README.md`.

## Staged sidecar

Downloaded sidecar:

```text
/mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Verified local header:

```text
size = 3,807,602,400 bytes
general.architecture = deepseek4_mtp_support
general.name = DeepSeek V4 Flash MTP support
deepseek4.expert_count = 256
deepseek4.mtp_layer_count = 1
deepseek4.nextn_predict_layers = 1
tensors = 32
```

## Validation behavior

The server validator is metadata/tensor-directory only in default startup mode. It does not load sidecar tensor data unless `DSV4_MTP_PROBE=1` is also set. Probe mode loads the already-validated sidecar tensors into a persistent backend weight buffer, but still does not run the MTP graph or alter emitted tokens.

Validation requires:

- target model architecture is `LLM_ARCH_DEEPSEEK4`;
- target model has token embedding and output tensors for MTP sharing;
- sidecar file exists;
- sidecar `general.architecture == deepseek4_mtp_support`;
- sidecar `deepseek4.mtp_layer_count == 1`;
- sidecar `deepseek4.nextn_predict_layers == 1`;
- sidecar `deepseek4.expert_count` matches target `n_expert`;
- all 32 expected `mtp.0.*` tensors exist with exact hparams-derived shapes and accepted types.

Accepted routed expert sidecar types mirror DS4 authority: `IQ2_XXS`, `Q2_K`, or `Q4_K`; the current downloaded sidecar uses `Q4_K`.

## Quality gates run

Build:

```text
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release --target llama-server -j 8
```

Help check:

```text
llama-server --help | rg -- '--mtp-model|--mtp-draft'
```

Result: both flags are present in server help.

Unsupported-target startup check:

```text
DSV4_MTP_PROBE=1 llama-server -m /tmp/stories260K.gguf \
  --mtp-model /mnt/models/gguf/deepseek-ai__DeepSeek-V4-Flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --ctx-size 32 --no-warmup --port 18191
```

Result: server exits with rc `1`, reports no sidecar tensor-data load, and emits clean error:

```text
failed to validate DeepSeek4 MTP sidecar '...DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf': --mtp-model is only supported with DeepSeek4 target models
```

Default no-MTP startup check:

```text
llama-server -m /tmp/stories260K.gguf --ctx-size 32 --no-warmup --port 18192
curl http://127.0.0.1:18192/health
```

Result: health returned `{"status":"ok"}` after 2 seconds.

Whitespaces:

```text
git diff --check
```

Result: passed.

Note: whole-file `clang-format --dry-run` on `tools/server/server-context.cpp` reports many pre-existing formatting deviations outside touched lines. `git-clang-format --force origin/main -- ...` was applied to changed lines only.

## Not implemented yet

- No MTP graph.
- No draft token generation.
- No speculative verification/commit.

## HC-state probe plumbing

When a valid DeepSeek4 target is loaded with `--mtp-model` and `DSV4_MTP_PROBE=1`:

- server startup loads the MTP sidecar GGUF tensor data into a persistent backend weight buffer, using the target model HC/output tensor buffer type;
- server startup sets a private context probe flag;
- graph reuse accounts for that flag;
- DeepSeek4 marks the final per-token `hc_state` as an output only for `n_tokens == 1 && n_outputs == 1` graphs, avoiding prompt-prefill HC output blowups;
- decode copies that F32 HC state into `llama_context::get_dsv4_mtp_hc_state()`.

This is intentionally only a handoff/probe surface. It does not alter emitted tokens or run the MTP sidecar graph.

## Next exact step

Create a follow-up branch for draft-one probing, e.g.:

```text
work/dsv4-mtp-hc-probe
```

Recommended scope:

1. Build a one-token MTP graph that consumes:
   - base token embedding;
   - target final HC state;
   - sidecar `enorm/e_proj`, `hnorm/h_proj`, one MTP block, sidecar `hc_head_*`, sidecar `norm`;
   - base output projection.
2. Under `DSV4_MTP_PROBE=1`, log draft top-1 and compare it against the next target argmax without changing emitted tokens.
3. Do not implement speculative commit until deterministic no-MTP vs probe token streams match exactly.
