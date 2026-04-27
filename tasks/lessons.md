# Lessons

- When validating a new model family, do not guess the chat/output format from nearby models.
  - Trace the official source of truth end-to-end first: HF README/docs, encoding or chat-template files, converter metadata injection, GGUF metadata, and llama.cpp parser selection.
  - If the upstream release does not ship a Jinja chat template, do not treat a nearby built-in template as equivalent without proof.
  - Separate parser-format bugs from model-graph bugs by validating both:
    1. structured parsing with the correct official format, and
    2. raw token output with chat parsing disabled.
- Do not present response-quality conclusions from runs that used an unverified template or parser path.
- When the repository already includes an additional implementation reference (for example MLX for the same model family), do not tunnel on a single upstream implementation.
  - Cross-check PyTorch/HF behavior against the MLX implementation before concluding a parity bug is isolated.
  - Especially for precision, cache/state propagation, and attention-path semantics, compare all available reference implementations before changing runtime math.
- When debugging a new architecture, do not infer precision/layout semantics from checkpoint artifacts alone before reading the model paper / official architecture writeup.
  - Read the paper first, then reconcile the paper with HF inference code, converter behavior, and checkpoint tensors.
  - Treat surprising checkpoint details (extra scale tensors, mixed dtypes, etc.) as hypotheses until the paper and official implementation confirm their role.
- When the user asks to "not stop until all gaps are covered", do not pause at intermediate stabilization points.
  - Close every currently identified functional gap in the active path before proposing commit/push.
  - Treat crash-fix + partial parity as "in progress" until the remaining known semantic gaps are either implemented or explicitly disproven with tests.
- When the user points out that a model implementation "should be close" and says to use references more, immediately re-diff against every available source of truth.
  - Use the paper for required semantics, the official HF inference for exact runtime behavior, MLX for an independent implementation, and external forks for llama.cpp integration details.
  - Do not accept runtime stability as semantic progress unless deterministic outputs or logits are checked against a reference.
  - Pull the live branch/PR references, not stale local snapshots; for DeepSeek-V4 MLX specifically, fetch `ml-explore/mlx-lm` PR 1192 before drawing conclusions.

## DSv4 BF16 garbage output (2026-04-26)
- **Symptom:** Multilingual gibberish from `llama-completion` despite all graph code, op kernels, GGUF metadata keys, and tensor names matching antirez fork verbatim. Both our binary and antirez's CPU build produced the same garbage on our v2 GGUF.
- **Investigation that almost missed it:** Spent hours auditing graph-construction divergences. Confirmed tensor shapes matched HF, then "verified" tensor data against HF using `f.get_tensor(name).float()` — but for `torch.float8_e4m3fn` tensors, `.float()` reinterprets each FP8 byte as its FP8 numeric value (no scale applied). Both sides looked equal, but BOTH were wrong.
- **Actual root cause:** Our DSv4 `prepare_tensors()` overrode the base method to handle 3-tuple yields from `modify_tensors` (for MXFP4 expert packing) — and forgot to call `self.dequant_model()`. That meant FP8 dense weights (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, indexer `wq_b`, shared expert MLP) had their raw FP8 byte values cast directly to BF16 with **no E8M0 block scale applied**. Magnitudes were ~3000× too large per layer × 43 layers = pure noise logits.
- **Diagnostic that found it:** Compared GGUF tensor data byte-for-byte against HF `torch.float8_e4m3fn` weight × E8M0 scale (proper dequantization), instead of against the raw FP8-as-FP32 reinterpretation. Match failed for every FP8 tensor.
- **Lesson 1 — verification on quantized formats:** When validating quantized weight conversion, dequantize using the official scale formula. `torch.float8_e4m3fn → .float()` returns the FP8 numeric value, not the dequantized real-world value. Always multiply by the block scale.
- **Lesson 2 — overriding `prepare_tensors`:** If you override `prepare_tensors` in a `TextModel` subclass and don't call `super().prepare_tensors()`, you skip `self.dequant_model()` (which dequantizes FP8/FP4/GPTQ/etc.). Either call `super()` or explicitly invoke `self.dequant_model()` at the start.
- **Lesson 3 — when both forks fail identically, the bug is in shared inputs (GGUF), not graph code.** Antirez's CPU binary produced ~identical garbage to ours on our GGUF. That should have been the first signal to inspect the GGUF data path, not graph divergence — instead of spending hours auditing graph code that was actually correct.
- **Lesson 4 — magnitude sanity check:** Before deep audits, dump a few weight values from the GGUF and compare to plausible NN weight magnitudes (typically O(0.01–1)). Values like `-128.0`, `-80.0` for an attention projection are an immediate red flag for missing dequant.

## DSv4 CUDA decode degeneration after first compressed slot (2026-04-27)
- **Symptom:** Built `build-cuda` produced coherent first ~3 generated tokens then drifted into repetition (`Let's break it step-by-step: ... 3 apples: 3 apples: ...`). True CPU-only build (`build-cuda` with `CUDA_VISIBLE_DEVICES=` cleared, or a fresh `build-cpu` with `GGML_CUDA=OFF`) produced fully coherent output for the same GGUF and prompt.
- **First narrowing:** Token-by-token n=2/3/4 comparison showed first divergence at exactly the decode step where the first ratio-4 compressed slot is read after being written. Initial hypothesis was a read-after-write scheduling issue between `cpy_k` and the compressed-cache read.
- **Confirming hypothesis was wrong:** Adding explicit graph dependencies (concat of old cache rows + just-computed compressed row, viewing `cpy_k`’s output tensor for the raw SWA read) did not change CUDA output. CPU output already matched antirez. So the bug had to be inside the CUDA backend.
- **Localizing CUDA:** Running the same `build-cuda` binary with `CUDA_VISIBLE_DEVICES=` produced perfect output. With CUDA visible, even `-ngl 0 -fa 0` showed `CUDA0 compute buffer size = 1270.50 MiB` because the scheduler offloads ops to CUDA when possible. So the corruption came from CUDA kernels participating in DSv4 graph nodes, not from `-ngl` placement.
- **Root cause:** `ggml/src/ggml-cuda/dsv4.cu` `kernel_dsv4_rope_tail_f32`’s YaRN math was wrong:
  - `rope_yarn_ramp_device` returned `clamp((i0/2 - low)/(high - low), 0, 1)`. CPU `rope_yarn_ramp` returns `1 - clamp(...)`. Sign of the ramp was inverted.
  - `rope_yarn_device` set `theta_scaled = theta * freq_scale` and then “blended” it against `theta_scaled / mix * mix`, which algebraically equals `theta_scaled`. Result: zero YaRN extrapolation. CPU instead blends `theta_interp = freq_scale * theta_extrap` against `theta_extrap` using `ramp_mix = ramp * ext_factor`.
  - CPU also applies `mscale *= 1 + 0.1 * log(1 / freq_scale)` when `ext_factor != 0`. CUDA did not. Our `get_rope_cfg` divides `attn_factor` by that exact factor in advance assuming the runtime would multiply it back. With CUDA never multiplying, the on-CUDA magnitude was systematically too small.
- **Fix:** Rewrite the CUDA `rope_yarn_ramp_device` and `rope_yarn_device` to mirror `ggml-cpu/ops.cpp`’s `rope_yarn_ramp` and `rope_yarn` exactly: use `1 - clamp(...)`, blend `theta_interp` against `theta_extrap`, and apply the magnitude correction when `ext_factor != 0`.
- **Lesson 1 — kernel parity:** When porting a math kernel (especially RoPE, YaRN, softmax_ext, sinks) from CPU reference to CUDA, mirror the CPU function _line for line_ first. Optimize only after byte-level/numerical parity is verified by `test-backend-ops` or by full-model decode equality.
- **Lesson 2 — scheduler reach:** `-ngl 0` is not equivalent to “CPU only” in a `GGML_CUDA=ON` build. Ops without explicit placement still get scheduled to CUDA if available. To validate “CPU semantics,” either build with `GGML_CUDA=OFF` or run with `CUDA_VISIBLE_DEVICES=` cleared. Don’t conflate `-ngl 0` with “CUDA out of the picture.”
- **Lesson 3 — magnitude correction contracts:** When the call site pre-cancels a YaRN magnitude factor (`attn_factor /= 1 + 0.1 * log(1/freq_scale)`), the kernel _must_ apply that factor internally for the cancellation to work. Document this contract in code at both ends. We added a comment block above the CUDA YaRN helpers stating this contract.
