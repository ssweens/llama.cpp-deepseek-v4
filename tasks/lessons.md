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
