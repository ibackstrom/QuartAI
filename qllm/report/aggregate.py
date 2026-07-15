"""Aggregate paired seeds into the final evidence report and mean loss curves."""

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "qllm" / "runs"
OUTPUT = ROOT / "qllm" / "report"
ARMS = ("real-base", "quat-samewidth", "quat-sameparams")


def load_results(seeds):
    return {arm: [json.loads((RUNS / f"{arm}-seed{seed}" / "result.json").read_text())
                  for seed in seeds] for arm in ARMS}


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def fmt_mean_sd(values, digits=4):
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def paired_stats(results, arm):
    differences = [q["best_val_loss"] - r["best_val_loss"]
                   for r, q in zip(results["real-base"], results[arm])]
    mean, sd = mean_sd(differences)
    # Student-t 97.5th percentile for df=4 (five paired seeds).
    margin = 2.776 * sd / math.sqrt(len(differences))
    return differences, mean, sd, (mean - margin, mean + margin)


def plot_curves(results):
    colors = {"real-base": "#333333", "quat-samewidth": "#d95f02", "quat-sameparams": "#1b9e77"}
    for x_key, filename, xlabel in (("tokens", "aggregate-loss-vs-tokens.png", "Training tokens"),
                                    ("seconds", "aggregate-loss-vs-time.png", "Wall-clock seconds")):
        plt.figure(figsize=(7.5, 4.5))
        for arm in ARMS:
            histories = [run["history"] for run in results[arm]]
            x = [statistics.mean(h[i][x_key] for h in histories) for i in range(len(histories[0]))]
            means = [statistics.mean(h[i]["val_loss"] for h in histories) for i in range(len(histories[0]))]
            sds = [statistics.stdev(h[i]["val_loss"] for h in histories) for i in range(len(histories[0]))]
            plt.plot(x, means, marker="o", label=arm, color=colors[arm])
            plt.fill_between(x, [m-s for m, s in zip(means, sds)], [m+s for m, s in zip(means, sds)],
                             color=colors[arm], alpha=0.15)
        plt.xlabel(xlabel); plt.ylabel("Validation loss (mean ± seed SD)")
        plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT / filename, dpi=150); plt.close()


def make_report(seeds=(1, 2, 3, 4, 5)):
    results = load_results(seeds)
    plot_curves(results)
    real = results["real-base"]
    same_width = results["quat-samewidth"]
    same_params = results["quat-sameparams"]
    _, width_diff, width_diff_sd, width_ci = paired_stats(results, "quat-samewidth")
    param_diffs, param_diff, param_diff_sd, param_ci = paired_stats(results, "quat-sameparams")
    real_ppl = statistics.mean(r["best_perplexity"] for r in real)
    width_ppl = statistics.mean(r["best_perplexity"] for r in same_width)
    param_ppl = statistics.mean(r["best_perplexity"] for r in same_params)
    strict_threshold = 2 * param_diff_sd
    strict_pass = -param_diff > strict_threshold
    real_speed = statistics.mean(r["tokens_per_second"] for r in real)
    quat_speed = statistics.mean(r["tokens_per_second"] for r in same_params)
    # Linear estimate from measured end-to-end rates; excludes setup between runs.
    two_arm_hours = 3 * 50_000_000 * (1 / real_speed + 1 / quat_speed) / 3600
    three_arm_hours = two_arm_hours + 3 * 50_000_000 / statistics.mean(
        r["tokens_per_second"] for r in same_width) / 3600

    per_seed = []
    for i, seed in enumerate(seeds):
        per_seed.append(
            f"| {seed} | {real[i]['best_val_loss']:.4f} | {same_width[i]['best_val_loss']:.4f} | "
            f"{same_params[i]['best_val_loss']:.4f} | {param_diffs[i]:+.4f} |"
        )

    report = f"""# Real vs Quaternion LM: Five-Seed Pilot Report

## Executive conclusion

At this scale, **quaternion is not proven better under the experiment's predefined rules**.

- **Equal width (H2): not supported.** Quaternion perplexity was {(width_ppl / real_ppl - 1) * 100:.1f}% higher,
  far outside the allowed 5%, although it used 43.7% fewer total parameters.
- **Equal parameters (H1): promising but formally inconclusive.** Quaternion perplexity was
  {(1 - param_ppl / real_ppl) * 100:.1f}% lower on average. Its paired loss advantage was {-param_diff:.5f}, while
  the predefined threshold was `2 × seed SD = {strict_threshold:.5f}`; therefore H1
  **{'passes' if strict_pass else 'does not pass'}**.
- A conventional paired 95% confidence interval for `quaternion loss − real loss` is
  [{param_ci[0]:.5f}, {param_ci[1]:.5f}], which excludes zero. That is evidence for this tiny setup, but it does
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

At the measured rates, the two-arm 300M-token test would take approximately **{two_arm_hours:.1f} hours** on this
Apple MPS machine; the three-arm test approximately **{three_arm_hours:.1f} hours**, assuming throughput remains
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
- Seeds: {', '.join(map(str, seeds))}; paired batches and schedules within each seed
- Architecture: 2 decoder blocks, width 64 baseline, 4 heads, context 64, MLP ratio 4
- Equal-parameter quaternion width: 100; mismatch from real total count: 0.73%
- Optimizer: AdamW, LR 3e-4, cosine decay, 2% warmup, weight decay 0.1, gradient clip 1.0
- Shared components: real tied embedding/head, real LayerNorm, real attention scores, split GELU
- Changed components: Q/K/V/output and both MLP projections
- Hardware: Apple MPS; PyTorch {real[0]['torch_version']}

## Aggregate results (mean ± sample SD, five seeds)

| Model | Width | Parameters | Validation loss | Perplexity | Tokens/s |
|---|---:|---:|---:|---:|---:|
| real-base | 64 | {real[0]['parameters']:,} | {fmt_mean_sd([r['best_val_loss'] for r in real])} | {fmt_mean_sd([r['best_perplexity'] for r in real], 2)} | {fmt_mean_sd([r['tokens_per_second'] for r in real], 0)} |
| quat-samewidth | 64 | {same_width[0]['parameters']:,} | {fmt_mean_sd([r['best_val_loss'] for r in same_width])} | {fmt_mean_sd([r['best_perplexity'] for r in same_width], 2)} | {fmt_mean_sd([r['tokens_per_second'] for r in same_width], 0)} |
| quat-sameparams | 100 | {same_params[0]['parameters']:,} | {fmt_mean_sd([r['best_val_loss'] for r in same_params])} | {fmt_mean_sd([r['best_perplexity'] for r in same_params], 2)} | {fmt_mean_sd([r['tokens_per_second'] for r in same_params], 0)} |

Quaternion throughput was about {(1 - quat_speed / real_speed) * 100:.1f}% lower than real at equal parameters.
The structured block matrix saves stored parameters, not dense multiply work, and incurs construction overhead.

## Paired seed detail

`Δ loss` is quaternion-same-parameters minus real; negative favors quaternion.

| Seed | Real loss | Quaternion same-width | Quaternion same-parameters | Δ loss |
|---:|---:|---:|---:|---:|
{chr(10).join(per_seed)}

- Equal-width paired difference: {width_diff:+.5f} ± {width_diff_sd:.5f}; 95% CI [{width_ci[0]:.5f}, {width_ci[1]:.5f}]
- Equal-parameter paired difference: {param_diff:+.5f} ± {param_diff_sd:.5f}; 95% CI [{param_ci[0]:.5f}, {param_ci[1]:.5f}]

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
"""
    (OUTPUT / "report.md").write_text(report)
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    make_report()
