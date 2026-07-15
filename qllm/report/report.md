# Real vs Quaternion LM: Five-Seed Pilot Report

## Executive conclusion

At this scale, **quaternion is not proven better under the experiment's predefined rules**.

- **Equal width (H2): not supported.** Quaternion perplexity was 33.1% higher,
  far outside the allowed 5%, although it used 43.7% fewer total parameters.
- **Equal parameters (H1): promising but formally inconclusive.** Quaternion perplexity was
  3.5% lower on average. Its paired loss advantage was 0.03546, while
  the predefined threshold was `2 × seed SD = 0.03785`; therefore H1
  **does not pass**.
- A conventional paired 95% confidence interval for `quaternion loss − real loss` is
  [-0.05895, -0.01196], which excludes zero. That is evidence for this tiny setup, but it does
  not override the stricter decision rule or establish scaling behavior.

## How many tokens would establish the answer?

There is **no guaranteed token count**. More tokens reduce under-training bias, but they do not remove variation
between seeds, and the relative ranking can change during training. In this pilot, the equal-parameter quaternion
advantage was largest around 200K–400K tokens and then shrank by 1M tokens. Extrapolating the early win would be
misleading.

The smallest credible next experiment from the original protocol is:

| Scope | Tokens per model per seed | Seeds | Models | Total training tokens |
|---|---:|---:|---:|---:|
| Direct H1 answer: real vs quaternion-same-parameters | 50M | 3 | 2 | **300M** |
| Real plus both quaternion controls | 50M | 3 | 3 | **450M** |
| Full real/complex/quaternion grid | 50M | 3 | 5 | **750M** |

At the measured rates, the two-arm 300M-token test would take approximately **4.3 hours** on this
Apple MPS machine; the three-arm test approximately **7.1 hours**, assuming throughput remains
linear. These are estimates, not guarantees. The stronger original target of 200M per run would cost four times
as much.

The budget must be selected before seeing the result. Claim H1 only if the final fixed-budget quaternion loss
advantage exceeds twice the paired across-seed SD. To establish practical equivalence or a null result, predefine
an acceptable effect (for example ±2% perplexity) and require the confidence interval—not merely the point
estimate—to fit inside it.

## Experimental setup

- Dataset: bounded 8 MiB prefix of `roneneldan/TinyStories`; separate 1 MiB validation prefix
- Tokenizer: byte-level BPE, vocabulary 1,024, trained on training text only
- Budget: 1,000,448 tokens per run (1M requested, rounded to complete batches)
- Seeds: 1, 2, 3, 4, 5; paired batches and schedules within each seed
- Architecture: 2 decoder blocks, width 64 baseline, 4 heads, context 64, MLP ratio 4
- Equal-parameter quaternion width: 100; mismatch from real total count: 0.73%
- Optimizer: AdamW, LR 3e-4, cosine decay, 2% warmup, weight decay 0.1, gradient clip 1.0
- Shared components: real tied embedding/head, real LayerNorm, real attention scores, split GELU
- Changed components: Q/K/V/output and both MLP projections
- Hardware: Apple MPS; PyTorch 2.8.0

## Aggregate results (mean ± sample SD, five seeds)

| Model | Width | Parameters | Validation loss | Perplexity | Tokens/s |
|---|---:|---:|---:|---:|---:|
| real-base | 64 | 168,576 | 4.4235 ± 0.0253 | 83.41 ± 2.13 | 29049 ± 1931 |
| quat-samewidth | 64 | 94,848 | 4.7094 ± 0.0262 | 111.01 ± 2.93 | 14778 ± 848 |
| quat-sameparams | 100 | 169,800 | 4.3880 ± 0.0199 | 80.49 ± 1.62 | 14784 ± 700 |

Quaternion throughput was about 49.1% lower than real at equal parameters.
The structured block matrix saves stored parameters, not dense multiply work, and incurs construction overhead.

## Paired seed detail

`Δ loss` is quaternion-same-parameters minus real; negative favors quaternion.

| Seed | Real loss | Quaternion same-width | Quaternion same-parameters | Δ loss |
|---:|---:|---:|---:|---:|
| 1 | 4.4205 | 4.7509 | 4.3822 | -0.0383 |
| 2 | 4.4634 | 4.6945 | 4.4200 | -0.0435 |
| 3 | 4.3937 | 4.6842 | 4.3846 | -0.0091 |
| 4 | 4.4149 | 4.7176 | 4.3883 | -0.0266 |
| 5 | 4.4249 | 4.6997 | 4.3651 | -0.0597 |

- Equal-width paired difference: +0.28588 ± 0.03679; 95% CI [0.24021, 0.33155]
- Equal-parameter paired difference: -0.03546 ± 0.01893; 95% CI [-0.05895, -0.01196]

![Five-seed validation loss against tokens](aggregate-loss-vs-tokens.png)

![Five-seed validation loss against wall time](aggregate-loss-vs-time.png)

## Interpretation against H1 and H2

**H1 (same parameters): inconclusive under the predefined rule.** All five paired seeds favored quaternion, and
the ordinary paired interval excludes zero. However, the advantage narrowly missed the stricter `> 2 × SD`
threshold. More importantly, its lead decreased late in training. A longer preregistered run is needed to learn
whether this is durable or merely faster early optimization.

**H2 (same width): rejected at this tested scale.** Quaternion used fewer parameters but remained much worse than
the real baseline after 1M tokens. It was not within 5% relative perplexity.

## Limitations and validity threats

1. This is a tiny model and a bounded dataset prefix, not evidence about large language models.
2. One million tokens is an optimization pilot, below the protocol's 50M minimum evidence budget.
3. Validation uses sampled fixed batches per seed rather than a full 2M-token validation sweep; pairing protects
   arm comparisons within a seed, but a larger final evaluation would reduce measurement noise.
4. The tokenizer vocabulary is 1,024 for the pilot rather than the full configuration's 8,192.
5. Quaternion uses split activations, real LayerNorm, and real dot-product attention; this tests structured
   projections, not a fully quaternion-valued network.
6. MPS timing does not predict optimized CUDA kernels or large-model throughput.
7. No complex-valued controls were trained in this direct real-vs-quaternion pilot, though their code and full
   configurations are present.

## Reproduction

```bash
source .venv/bin/activate
python -m unittest qllm.tests.test_layers
for seed in 1 2 3 4 5; do
  python -m qllm.compare --token-budget 1000000 --seed "$seed" --eval-batches 8 --device mps
done
python -m qllm.report.aggregate
```
