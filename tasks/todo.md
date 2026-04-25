# DeepSeek-V4 First Chunk Todo

## Scope

Implement the conversion/loading foundation for the DeepSeek-V4 family (`DeepSeek-V4-Flash` and `DeepSeek-V4-Pro`) without graph execution.

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

## Review

- Added DeepSeek-V4 tokenizer fallback to `tokenizer.json` in converter when `AutoTokenizer` fails on unknown `deepseek_v4` config type.
- Verified conversion of synthetic DeepSeek-V4 model now succeeds and writes GGUF with DeepSeek-V4 metadata and tensors.
- Verified `MXFP4` routed expert tensors are emitted (`blk.*.ffn_*_exps.weight`).
- Verified loader path reaches intentional graph boundary:
  - model metadata/tensors load successfully
  - runtime fails with expected message: `DeepSeek-V4 graph execution is not implemented yet`
- Quality gates run:
  - `python3 -m py_compile convert_hf_to_gguf.py gguf-py/gguf/constants.py gguf-py/gguf/gguf_writer.py`
  - `cmake --build build --target llama-cli llama-gguf -j 4`
  - `python3 convert_hf_to_gguf.py /tmp/dsv4-synth --outfile /tmp/dsv4-synth.gguf --outtype bf16`
  - `./build/bin/llama-cli -m /tmp/dsv4-synth.gguf -p 'hello' -n 1`
