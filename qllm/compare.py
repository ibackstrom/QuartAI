"""Run an understandable apples-to-apples real-vs-quaternion comparison.

The three arms answer two different questions:
  1. Same width: does quaternion weight sharing retain quality with fewer weights?
  2. Same parameters: does a wider quaternion model use the same budget better?

All arms see the same tokenizer, batches, token budget, optimizer, and schedule.
"""

import argparse
import json
from pathlib import Path

from qllm.data.prepare import prepare
from qllm.report.make_report import make_report
from qllm.train import matched_width, model_parameter_count, train_one


def run_comparison(args) -> None:
    data_dir = Path("qllm/data/cache")
    if not (data_dir / "metadata.json").exists():
        print("Preparing a small TinyStories BPE dataset...")
        prepare(data_dir, args.vocab_size, args.train_mb, args.val_mb)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    base = {"context_length": args.context, "n_layers": args.layers, "n_heads": args.heads,
            "width": args.width, "mlp_ratio": 4, "dropout": 0.1, "batch_size": args.batch_size,
            "token_budget": args.token_budget, "learning_rate": 3e-4, "weight_decay": 0.1,
            "warmup_fraction": 0.02, "eval_batches": args.eval_batches, "seed": args.seed}
    real_params = model_parameter_count("real", args.width, base, metadata["vocab_size"])
    quat_width, match_error = matched_width("quaternion", real_params, base, metadata["vocab_size"])
    arms = [("real-base", "real", args.width), ("quat-samewidth", "quaternion", args.width),
            ("quat-sameparams", "quaternion", quat_width)]

    print("\nFAIR COMPARISON")
    print("Same width isolates weight sharing; same parameters isolates parameter efficiency.")
    print(f"Matched quaternion width: {quat_width} (parameter mismatch {match_error:.2%}).")
    results = []
    for name, kind, width in arms:
        config = dict(base, name=name, layer_kind=kind, width=width)
        results.append(train_one(config, data_dir, Path("qllm/runs") / f"{name}-seed{args.seed}", args.device))
    make_report(results, Path("qllm/report"), smoke=args.token_budget < 50_000_000)

    real, same_width, same_params = results
    print("\nPLAIN-ENGLISH RESULT")
    print(f"Real baseline:          {real['parameters']:,} params, perplexity {real['best_perplexity']:.2f}")
    print(f"Quaternion same width:  {same_width['parameters']:,} params, perplexity {same_width['best_perplexity']:.2f}")
    print(f"Quaternion same params: {same_params['parameters']:,} params, perplexity {same_params['best_perplexity']:.2f}")
    width_param_change = (same_width["parameters"] / real["parameters"] - 1) * 100
    width_ppl_change = (same_width["best_perplexity"] / real["best_perplexity"] - 1) * 100
    matched_ppl_change = (same_params["best_perplexity"] / real["best_perplexity"] - 1) * 100
    print(f"At equal width, quaternion used {abs(width_param_change):.1f}% fewer total parameters and perplexity "
          f"was {abs(width_ppl_change):.1f}% {'higher' if width_ppl_change > 0 else 'lower'}.")
    print(f"At equal parameters, quaternion perplexity was {abs(matched_ppl_change):.1f}% "
          f"{'higher' if matched_ppl_change > 0 else 'lower'}.")
    if args.token_budget < 50_000_000:
        print("These are smoke-test directions only, not evidence for or against the hypotheses.")
    print("Lower perplexity is better. See qllm/report/report.md for caveats and plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-budget", type=int, default=32768, help="use >=50M for evidence, small values are smoke tests")
    parser.add_argument("--train-mb", type=int, default=8)
    parser.add_argument("--val-mb", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    run_comparison(parser.parse_args())
