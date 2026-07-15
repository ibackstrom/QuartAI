"""Run the preregistered 50M-token real-vs-quaternion experiment overnight."""

import argparse
import json
import math
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from qllm.train import matched_width, model_parameter_count, train_one


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def completed_result(path: Path, expected: dict):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    config = result.get("config", {})
    if all(config.get(key) == value for key, value in expected.items()):
        return result
    raise RuntimeError(
        f"{path} belongs to a different configuration. Use a different --output directory."
    )


def write_summary(results: dict, output: Path) -> None:
    real = results["real-base"]
    quat = results["quat-sameparams"]
    deltas = [q["best_val_loss"] - r["best_val_loss"] for r, q in zip(real, quat)]
    mean_delta = statistics.mean(deltas)
    delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    margin = 4.303 * delta_sd / math.sqrt(len(deltas)) if len(deltas) == 3 else None
    real_ppl = statistics.mean(run["best_perplexity"] for run in real)
    quat_ppl = statistics.mean(run["best_perplexity"] for run in quat)
    strict_pass = -mean_delta > 2 * delta_sd

    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [run["seed"] for run in real],
        "tokens_per_run": [run["token_budget"] for run in real],
        "total_training_tokens": sum(run["token_budget"] for runs in results.values() for run in runs),
        "real": {
            "parameters": real[0]["parameters"],
            "width": real[0]["width"],
            "mean_validation_loss": statistics.mean(run["best_val_loss"] for run in real),
            "mean_perplexity": real_ppl,
            "mean_tokens_per_second": statistics.mean(run["tokens_per_second"] for run in real),
        },
        "quaternion_same_parameters": {
            "parameters": quat[0]["parameters"],
            "width": quat[0]["width"],
            "mean_validation_loss": statistics.mean(run["best_val_loss"] for run in quat),
            "mean_perplexity": quat_ppl,
            "mean_tokens_per_second": statistics.mean(run["tokens_per_second"] for run in quat),
        },
        "paired_quaternion_minus_real_loss": deltas,
        "paired_mean_loss_difference": mean_delta,
        "paired_loss_difference_sd": delta_sd,
        "paired_95_percent_ci": (
            [mean_delta - margin, mean_delta + margin] if margin is not None else None
        ),
        "quaternion_perplexity_change_percent": (quat_ppl / real_ppl - 1) * 100,
        "strict_rule": "quaternion loss advantage > 2 * paired seed SD",
        "strict_rule_passed": strict_pass,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for r, q, delta in zip(real, quat, deltas):
        rows.append(
            f"| {r['seed']} | {r['best_val_loss']:.5f} | {q['best_val_loss']:.5f} | {delta:+.5f} |"
        )
    ci_text = (
        f"[{mean_delta - margin:.5f}, {mean_delta + margin:.5f}]" if margin is not None else "not calculated"
    )
    report = f"""# Overnight real vs quaternion result

Lower validation loss, perplexity, and paired difference favor quaternion.

| Seed | Real loss | Quaternion loss | Quaternion − real |
|---:|---:|---:|---:|
{chr(10).join(rows)}

- Real mean perplexity: **{real_ppl:.3f}**
- Quaternion mean perplexity: **{quat_ppl:.3f}** ({(quat_ppl / real_ppl - 1) * 100:+.2f}%)
- Mean paired loss difference: **{mean_delta:+.5f} ± {delta_sd:.5f} SD**
- Paired 95% CI: **{ci_text}**
- Strict predefined rule passed: **{strict_pass}**
- Total training tokens: **{summary['total_training_tokens']:,}**

See `manifest.json` for the complete experimental configuration and each run's `result.json`
for training curves, speed, environment, and final metrics.
"""
    (output / "summary.md").write_text(report)


def main(args) -> None:
    data_dir = Path("qllm/data/cache")
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Dataset cache is missing. First run: python -m qllm.data.prepare --vocab-size 1024 --train-mb 8 --val-mb 1"
        )
    metadata = json.loads(metadata_path.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    base = {
        "context_length": args.context,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "width": args.width,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "warmup_fraction": 0.02,
        "eval_batches": args.eval_batches,
    }
    real_params = model_parameter_count("real", args.width, base, metadata["vocab_size"])
    quat_width, match_error = matched_width("quaternion", real_params, base, metadata["vocab_size"])
    quat_params = model_parameter_count("quaternion", quat_width, base, metadata["vocab_size"])
    arms = (("real-base", "real", args.width), ("quat-sameparams", "quaternion", quat_width))

    manifest = {
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed-budget equal-parameter real vs quaternion comparison",
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": args.device,
        "dataset": metadata,
        "seeds": args.seeds,
        "arms": [
            {"name": "real-base", "kind": "real", "width": args.width, "parameters": real_params},
            {"name": "quat-sameparams", "kind": "quaternion", "width": quat_width,
             "parameters": quat_params, "parameter_mismatch_percent": match_error * 100},
        ],
        "training": base,
        "planned_total_tokens": args.token_budget * len(args.seeds) * len(arms),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nOVERNIGHT EXPERIMENT", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    results = {name: [] for name, _, _ in arms}
    for seed in args.seeds:
        for name, kind, width in arms:
            config = dict(base, name=name, layer_kind=kind, width=width, seed=seed)
            run_dir = output / f"{name}-seed{seed}"
            result = completed_result(run_dir / "result.json", config)
            if result is None:
                print(f"\nStarting {name}, seed {seed}", flush=True)
                result = train_one(config, data_dir, run_dir, args.device)
            else:
                print(f"\nSkipping completed {name}, seed {seed}", flush=True)
            results[name].append(result)
            print(
                f"Completed {name} seed {seed}: loss={result['best_val_loss']:.5f}, "
                f"ppl={result['best_perplexity']:.3f}, tok/s={result['tokens_per_second']:.0f}",
                flush=True,
            )

    write_summary(results, output)
    print(f"\nDone. Analyze {output / 'summary.md'} and {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-budget", type=int, default=50_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("qllm/runs/overnight-50m"))
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=32)
    main(parser.parse_args())
