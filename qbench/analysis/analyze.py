"""Regenerate paired statistics, plots, and REPORT from machine-readable CSVs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Exact two-sided 95% Student-t quantiles for protocol seed counts (and conservative fallback).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 20: 2.086, 30: 2.042}


def markdown_table(frame):
    """Render a small DataFrame without Pandas' optional tabulate dependency."""
    columns = list(frame.columns)

    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def paired(results):
    primary = results[results.model.isin(["real", "quat"])]
    wide = primary.pivot(index=["token_budget", "seed"], columns="model", values="final_val_loss")
    if "real" not in wide or "quat" not in wide:
        return pd.DataFrame(columns=["token_budget", "n", "mean_delta", "ci_low", "ci_high", "signs"])
    wide = wide.dropna(); wide["delta"] = wide.quat - wide.real
    rows = []
    for budget, group in wide.groupby(level=0):
        values = group.delta; n = len(values); mean = values.mean()
        margin = float("nan") if n < 2 else T975.get(n - 1, 1.96) * values.std(ddof=1) / math.sqrt(n)
        rows.append({"token_budget": budget, "n": n, "mean_delta": mean, "ci_low": mean-margin,
                     "ci_high": mean+margin, "signs": ",".join("+" if x > 0 else "-" for x in values)})
    return pd.DataFrame(rows)


def paired_ablations(results):
    rows = []
    baseline = results[results.model == "quat"][["token_budget", "seed", "final_val_loss"]]
    baseline = baseline.rename(columns={"final_val_loss": "baseline"})
    for model in sorted(set(results.model) - {"real", "quat"}):
        arm = results[results.model == model][["token_budget", "seed", "final_val_loss"]]
        merged = arm.merge(baseline, on=["token_budget", "seed"])
        for budget, group in merged.groupby("token_budget"):
            values = group.final_val_loss - group.baseline
            n = len(values); mean = values.mean()
            margin = float("nan") if n < 2 else T975.get(n - 1, 1.96) * values.std(ddof=1) / math.sqrt(n)
            rows.append({"model": model, "token_budget": budget, "n": n,
                         "mean_delta_vs_quat": mean, "ci_low": mean - margin,
                         "ci_high": mean + margin})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=Path("qbench/results/results.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("qbench/benchmarks/throughput.csv"))
    parser.add_argument("--output", type=Path, default=Path("qbench/analysis/plots")); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results) if args.results.exists() else pd.DataFrame(columns=["model", "token_budget", "seed", "final_val_loss"])
    identities = []
    for path in sorted((args.results.parent / "metadata").glob("*.json")):
        item = json.loads(path.read_text()); identities.append(item)
    if len(identities) != len(results):
        raise ValueError("every results row must have exactly one metadata file")
    identity_by_run = {item["run_id"]: item for item in identities}
    if len(identity_by_run) != len(identities) or set(results.run_id) != set(identity_by_run):
        raise ValueError("duplicate or unmatched run metadata")
    protocol_ids = {(item["source_hash"], item["dataset_hash"]) for item in identities}
    if len(protocol_ids) > 1: raise ValueError("mixed protocol/source or dataset digests")
    duplicate_keys = [(item["source_hash"], item["nominal_tokens"], item["seed"], item["model"]) for item in identities]
    if len(duplicate_keys) != len(set(duplicate_keys)): raise ValueError("duplicate protocol/budget/seed/model")
    if not results.empty and not np.isfinite(results.final_val_loss).all(): raise ValueError("nonfinite final runs")
    stats = paired(results)
    if not stats.empty:
        stats["outcome"] = ["quaternion win" if row.ci_high < 0 else
                            "real win" if row.ci_low > 0 else "inconclusive"
                            for _, row in stats.iterrows()]
    stats.to_csv(args.output / "paired_statistics.csv", index=False)
    if not stats.empty:
        plt.errorbar(stats.token_budget, stats.mean_delta,
                     yerr=[stats.mean_delta-stats.ci_low, stats.ci_high-stats.mean_delta], marker="o")
        plt.axhline(0, color="black"); plt.xscale("log"); plt.xlabel("training tokens"); plt.ylabel("quat - real loss")
        plt.tight_layout(); plt.savefig(args.output / "delta_vs_budget.png"); plt.close()
    if args.benchmark.exists():
        bench = pd.read_csv(args.benchmark); plt.bar(bench.implementation, bench.tokens_per_sec)
        plt.xticks(rotation=20); plt.ylabel("tokens/s"); plt.tight_layout(); plt.savefig(args.output / "throughput.png"); plt.close()
    curves_dir = args.results.parent / "curves"
    curve_files = list(curves_dir.glob("*.csv")) if curves_dir.exists() else []
    for curve_file in curve_files:
        curve = pd.read_csv(curve_file)
        plt.plot(curve.tokens, curve.val_loss, label=curve_file.stem)
    if curve_files:
        plt.xlabel("training tokens"); plt.ylabel("validation loss"); plt.legend(fontsize=6)
        plt.tight_layout(); plt.savefig(args.output / "validation_curves.png"); plt.close()
    ablations = results[~results.model.isin(["real", "quat"])]
    ablation_stats = paired_ablations(results)
    ablation_stats.to_csv(args.output / "ablations.csv", index=False)
    registered = {(b, s, m) for b in (500000,1000000,2000000,5000000,10000000,25000000,50000000)
                  for s in (range(1, 6) if b <= 10000000 else range(1, 4)) for m in ("real", "quat")}
    present = set(zip(results.token_budget, results.seed, results.model)) if not results.empty else set()
    complete = registered <= present
    supported = complete and any(r.token_budget <= 10000000 and r.ci_high < 0 for _, r in stats.iterrows()) and any(r.token_budget == 50000000 and r.ci_low > 0 for _, r in stats.iterrows())
    rejected = complete and all(r.ci_high >= 0 for _, r in stats.iterrows())
    decision = "H1 supported." if supported else "H1 rejected." if rejected else "Inconclusive; progress only (registered matrix incomplete)." if not complete else "H1 inconclusive."
    brackets = []
    ordered = stats.sort_values("token_budget")
    for (_, left), (_, right) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
        if left.mean_delta == 0 or right.mean_delta == 0 or left.mean_delta * right.mean_delta < 0:
            brackets.append("%s to %s tokens" % (int(left.token_budget), int(right.token_budget)))
    crossover = ", ".join(brackets) if brackets else "No tested adjacent-budget sign change."
    setup = "No run metadata yet."
    if identities:
        first = identities[0]
        cfg = first["model_config"]
        setup = ("TinyStories, data hash `%s`. %s layers, %s heads, context %s, batch 8, "
                 "vocab %s. Final evaluation uses %s sequential tokens with batch %s; curve "
                 "evaluation uses %s tokens." % (first["dataset_hash"], cfg["n_layers"],
                 cfg["n_heads"], cfg["context_length"], cfg["vocab_size"],
                 first["evaluation"]["final_tokens"], first["evaluation"]["batch"],
                 first["evaluation"]["curve_tokens"]))
    report = ["# Quaternion Phase 2 Report", "", "_Generated entirely from CSV and per-run metadata by `qbench.analysis.analyze`._", "",
              "## Setup", setup, "",
              "The awkward width-100 parameter-matched real control is intentionally omitted: matching it would require a dubious low-rank architecture and change more than algebra.", "", "## Paired crossover statistics", ""]
    report.append(markdown_table(stats) if not stats.empty else "No complete real/quaternion pairs yet.")
    report += ["", "## Ablations", markdown_table(ablation_stats) if not ablation_stats.empty else "No complete ablation pairs yet.", "",
               "## Decision rule", decision, "", "Crossover is reported only as a bracket between adjacent tested budgets whose mean/outcome changes sign; no interpolation.", "",
               "Crossover bracket: " + crossover, "", "## Limitations", "Single dataset, single small scale, and device-dependent throughput. Generation samples require completed 50M checkpoints."]
    samples_file = args.results.parent / "samples.json"
    if samples_file.exists():
        samples = json.loads(samples_file.read_text())
        report += ["", "## Generation samples", ""]
        for sample in samples:
            report += ["### %s, training seed %s" % (sample["model"], sample["run_id"]),
                       "Prompt: `%s`" % sample["prompt"], "", sample["text"], ""]
    Path("qbench/REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__": main()
