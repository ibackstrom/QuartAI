"""Create tables and loss curves from completed runs."""

from pathlib import Path
from typing import Dict, List


def make_report(results: List[Dict], output: Path, smoke: bool) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for x_key, filename, xlabel in (("tokens", "loss-vs-tokens.png", "Training tokens"),
                                    ("seconds", "loss-vs-time.png", "Wall-clock seconds")):
        plt.figure(figsize=(7, 4))
        for result in results:
            plt.plot([p[x_key] for p in result["history"]], [p["val_loss"] for p in result["history"]],
                     marker="o", label=result["name"])
        plt.xlabel(xlabel); plt.ylabel("Validation loss"); plt.legend(); plt.tight_layout()
        plt.savefig(output / filename, dpi=140); plt.close()

    first = results[0]
    rows = ["| Model | Width | Parameters | Best val loss | Perplexity | tok/s | Seconds |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        rows.append(f"| {result['name']} | {result['width']} | {result['parameters']:,} | "
                    f"{result['best_val_loss']:.4f} | {result['best_perplexity']:.2f} | "
                    f"{result['tokens_per_second']:.0f} | {result['wall_seconds']:.1f} |")
    real, same_width, same_params = results
    width_param_change = (same_width["parameters"] / real["parameters"] - 1) * 100
    width_ppl_change = (same_width["best_perplexity"] / real["best_perplexity"] - 1) * 100
    matched_ppl_change = (same_params["best_perplexity"] / real["best_perplexity"] - 1) * 100
    observed = (f"At equal width, quaternion used {abs(width_param_change):.1f}% fewer total parameters and its "
                f"perplexity was {abs(width_ppl_change):.1f}% "
                f"{'higher' if width_ppl_change > 0 else 'lower'}. At approximately equal total parameters, "
                f"quaternion perplexity was {abs(matched_ppl_change):.1f}% "
                f"{'higher' if matched_ppl_change > 0 else 'lower'}.")
    warning = ("**This was a pipeline smoke test, not a scientific result.** The token budget is below the "
               "50M minimum and only one seed was run; H1/H2 cannot be accepted or rejected from these numbers."
               if smoke else "Results should be aggregated over seeds before applying the H1/H2 decision rules.")
    report = f"""# Real vs Quaternion Small-LM Experiment

{warning}

## Setup

- Dataset: {first['dataset']}; tokenizer trained on training text only
- Shared token budget per arm: {first['token_budget']:,}
- Seed: {first['seed']}; device: {first['device']}; PyTorch: {first['torch_version']}
- Decoder: tied real embedding/head, real LayerNorm and attention scores, split GELU
- Only Q/K/V/output and MLP projections differ between arms

## Results

{chr(10).join(rows)}

![Validation loss against tokens](loss-vs-tokens.png)

![Validation loss against wall time](loss-vs-time.png)

## Interpretation

{observed} Lower loss/perplexity is better. `quat-samewidth` tests whether structured weights preserve quality with fewer
projection parameters. `quat-sameparams` widens the quaternion model to within 2% of the real model's total
parameter count. Total same-width reduction is less than 4x because embeddings, norms, and the tied language
head intentionally remain real-valued.

## Limitations

Small scale, one dataset prefix, one seed in quick mode, split rather than analytic quaternion activations,
real-valued attention scores, and block-matrix construction overhead. A claim about LLM-scale quality requires
the full identical-token-budget grid (at least 50M tokens), three seeds, generation review, and mean ± std.
"""
    (output / "report.md").write_text(report)
