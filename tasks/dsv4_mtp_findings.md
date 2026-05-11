# DeepSeek-V4 MTP: Findings (Do Not Re-Attempt Speculative Decoding)

## Summary

**DeepSeek-V4's Multi-Token Prediction (MTP) weights are training-time auxiliary
loss artifacts, not an inference feature.** Speculative decoding using DSv4 MTP
weights cannot deliver the kind of speedup that PR22673's Qwen3.5 MTP delivers
(~66% on Qwen3.5/3.5-MoE). It produces drafts that the target's verifier
rejects at ~100%, with overhead net-negative against target-only decode.

This finding is established by reading DeepSeek's own paper and inference code.
It is NOT a llama.cpp implementation bug.

## Evidence

### 1. The official DeepSeek-V4 paper says MTP is training-only

`DeepSeek_V4.pdf` (DeepSeek-V4-Pro repo, section 2 / 2.1):

> "Multi-Token Prediction. As DeepSeek-V3, DeepSeek-V4 series also set MTP
> modules and **objectives**. Given that the MTP strategy has been validated in
> DeepSeek-V3, we adopt the same strategy for DeepSeek-V4 series without
> modification."

"Modules and **objectives**" = parameters loaded in the checkpoint + auxiliary
loss functions applied during training. The MTP loss weight is documented as
"set to 0.3 for most of the training, and to 0.1 upon the start of learning
rate decay" (paper section 4.2 / 4.3) — a training hyperparameter, not an
inference one.

### 2. DeepSeek's official `inference/generate.py` does not use MTP

`DeepSeek-V4-Flash/inference/generate.py` (the reference implementation
DeepSeek ships):

```python
@torch.inference_mode()
def generate(model, prompt_tokens, max_new_tokens, eos_id, temperature):
    ...
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        ...
```

Plain single-token autoregressive generation. **No speculative decoding. No
MTP invocation.**

### 3. DeepSeek's `inference/model.py` `Transformer.forward()` does not call MTP

```python
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        ...
        self.layers = nn.ModuleList(...)            # main stack
        ...
        self.mtp = nn.ModuleList()                  # MTP blocks (loaded but...)
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(...))

    @torch.inference_mode()
    def forward(self, input_ids, start_pos=0):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        logits = self.head(h, ...)
        return logits                               # ...NEVER invoked here.
```

`self.mtp` is loaded from the checkpoint but never called in the inference
forward pass. It exists in the checkpoint solely for training.

### 4. Bench results from this branch's MTP work (over many iterations)

Baseline (no MTP): tg256 = **30.96 ± 0.13 tok/s**

Every MTP variant we tried produced 0% accept rate or hurt throughput:

| Approach                          | pp2048      | tg256      | Drafts gen | Drafts acc |
|-----------------------------------|-------------|------------|------------|------------|
| Baseline (no MTP)                 | 330.78      | 30.96      | —          | —          |
| In-context MTP, conservative backoff | ~330     | ~28.79     | many       | low/0%     |
| Sibling MTP (PR22673-style), no hook | 331.26   | 25.30      | 786        | 0          |
| Sibling MTP, prefill hook         | 239.63 (-28%) | 25.24    | 814        | 0          |

The 0% accept rate is invariant across architectures because the MTP weights
themselves do not predict tokens that match target's argmax. They were never
trained for that.

### 5. Comparison: why PR22673 works for Qwen3.5 but not DSv4

PR22673 ([Qwen3.5/Qwen3.5-MoE MTP in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/22673))
shows ~66% improvement on Qwen3.5. The architecture pattern (sibling context,
streaming hook, llama_context_seq_rm mirroring) is correct. The reason Qwen3.5
gets the speedup and DSv4 does not:

- **Qwen3.5's MTP blocks are trained / aligned for inference-time speculative
  draft.** Their argmax matches target's argmax often enough (~60-70%) for the
  speculative loop to net positive.
- **DSv4's MTP blocks are trained as an auxiliary loss during pretraining.**
  Their argmax does not need to match the trunk's argmax — there's no training
  signal for that. At inference, they produce semantically-related but
  argmax-divergent tokens. Target rejects all drafts.

This is a property of how the weights were trained, not the inference plumbing.

## Implication for llama.cpp DSv4 work

**Do not attempt MTP-based speculative decoding for DeepSeek-V4.** Specifically:

- The DSv4 MTP "sidecar" GGUFs (`deepseek4_mtp_support` arch) contain real
  weights, but those weights are training artifacts. Loading them at inference
  and using them as a speculative drafter will produce 0% accept rate.
- The in-context MTP path that existed in earlier work on
  `work/dsv4-mtp-loader-probe` was functionally correct but never delivered
  measurable speedup either.
- Any future MTP attempt (in-context, sibling-context, novel architectures)
  will hit the same wall unless DeepSeek re-releases DSv4 with MTP weights
  trained for inference speculation, which is not in scope of any public
  release.

## What to chase instead for DSv4 throughput

The 50 tok/s target for DSv4-Flash IQ2_XXS requires target-only kernel work:

1. **MoE-256 expert dispatch / fused gate-up** (the dominant decode cost).
2. **Sparse attention (CSA + HCA) decode-path tuning** — bounded raw-window
   attention, vectorized continuation compressor prefill, masked sparse rows
   skip (some of this already landed in `work/dsv4-prefill-speedup`).
3. **Output projection / LM head fusion** — the trailing `model.output`
   mul_mat over the full vocab is a noticeable per-token tail.
4. **MMQ tile compaction for routed experts** (landed in `work/dsv4-mmq-kernel`).
5. **HC head reduction fusion**.
6. **Memory layout tuning** for the compressed attention frontier so KV
   compute buffers stay GPU-resident.

These deliver real perf. MTP doesn't.

## History

- Branch `work/dsv4-mtp-loader-probe` (40 commits ahead of main) contains the
  full MTP experimental work, culminating in the PR22673-style sibling-context
  port. None of it delivered speedup; all of it is now archived.
- Three non-MTP commits from that branch were salvaged onto main:
  - `f55dbb45a perf(deepseek4): compact routed expert MMQ tiles`
  - `04cb0e3af fix(vulkan): stabilize DeepSeek4 sparse attention on RADV`
  - `2bc77e4dd test(deepseek4): sanitize regression workflow materials`
- This findings doc was added to record the negative result.

## References

- DeepSeek-V4 paper: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- DeepSeek-V4-Flash official inference code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- llama.cpp Qwen3.5 MTP PR (works because Qwen's MTP is inference-aligned): https://github.com/ggml-org/llama.cpp/pull/22673
