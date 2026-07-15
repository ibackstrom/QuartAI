"""Run one idempotent Phase-2 experiment and append its protocol row."""

import argparse
import csv
import json
import math
import os
import subprocess
import time
import hashlib
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from qbench.data import dataset_hash
from qllm.model.transformer import ModelConfig, TransformerLM, parameter_count
from qllm.train import choose_device, evaluate_sequential, make_batch, set_seed

CURVE_TOKENS = 131_072
FINAL_TOKENS = 2_000_000
EVAL_BATCH = 64


def source_hash(root=Path(".")):
    digest = hashlib.sha256()
    paths = [root / "quaternion_phase2_protocol.md"] + sorted((root / "qbench").rglob("*.py")) + sorted((root / "qllm").rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode()); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def initialize_model(config, seed, gaussian_ablation=False):
    """Construct an arm and reset training RNG; Gaussian changes quaternion params only."""
    if gaussian_ablation:
        baseline_config = ModelConfig(**dict(asdict(config), quaternion_init="parcollet"))
        gaussian_config = ModelConfig(**dict(asdict(config), quaternion_init="gaussian"))
        set_seed(seed); baseline = TransformerLM(baseline_config)
        set_seed(seed); model = TransformerLM(gaussian_config)
        with torch.no_grad():
            for (name, target), (_, source) in zip(model.named_parameters(), baseline.named_parameters()):
                if not any(part in name for part in ("weight_r", "weight_x", "weight_y", "weight_z")):
                    target.copy_(source)
    else:
        set_seed(seed)
        model = TransformerLM(config)
    set_seed(seed)  # constructor draw counts must never select the dropout stream
    return model


def canonical_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"): torch.mps.synchronize()


def _atomic_torch(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def _atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)

SCHEMA = ["run_id", "model", "width", "params", "token_budget", "seed", "data_hash",
          "final_val_loss", "final_val_ppl", "best_val_loss", "tokens_per_sec", "peak_mem_mb",
          "wall_clock_s", "git_commit", "device", "timestamp"]
MODELS = {
    "real": (64, "real", None, None, "parcollet"),
    "quat": (100, "quaternion", None, None, "parcollet"),
    "quat_w64": (64, "quaternion", None, None, "parcollet"),
    "quat_attn": (100, "real", "quaternion", "real", "parcollet"),
    "quat_ffn": (100, "real", "real", "quaternion", "parcollet"),
    "quat_gaussian": (100, "quaternion", None, None, "gaussian"),
}


def _write_unique(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    rows = [old for old in rows if old["run_id"] != row["run_id"]] + [row]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(args) -> dict:
    if args.eval_tokens != FINAL_TOKENS or args.eval_batch_size != EVAL_BATCH:
        raise ValueError("protocol runs require exactly 2,000,000 final tokens and eval batch 64")
    metadata = json.loads((args.data / "metadata.json").read_text())
    dtype = np.dtype(metadata["dtype"])
    train = np.memmap(args.data / "train.bin", dtype=dtype, mode="r")
    valid = np.memmap(args.data / "validation.bin", dtype=dtype, mode="r")
    if args.eval_tokens >= 2_000_000 and len(valid) <= 2_000_000:
        raise ValueError("Phase-2 validation cache must contain more than 2M tokens")
    width, kind, attn, ffn, init = MODELS[args.model]
    source = source_hash()
    digest_config = {"source_hash": source, "dataset_hash": dataset_hash(args.data), "model": args.model,
                     "model_config": asdict(ModelConfig(vocab_size=int(metadata["vocab_size"]), context_length=64,
                     n_layers=2, n_heads=4, width=width, dropout=.1, layer_kind=kind,
                     attention_layer_kind=attn, ffn_layer_kind=ffn, quaternion_init=init)),
                     "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": .1, "clip": 1.0,
                     "schedule": "2%-warmup-cosine"}, "nominal_tokens": args.tokens, "seed": args.seed,
                     "evaluation": {"curve_tokens": CURVE_TOKENS, "final_tokens": FINAL_TOKENS, "batch": EVAL_BATCH},
                     "torch_version": torch.__version__, "tokenizers_version": __import__("tokenizers").__version__}
    digest = canonical_digest(digest_config)
    run_id = f"{args.model}-t{args.tokens}-s{args.seed}-{digest[:12]}"
    results_file = args.output / "results.csv"
    metadata_file = args.output / "metadata" / (run_id + ".json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    if results_file.exists():
        existing = {r["run_id"]: r for r in csv.DictReader(results_file.open())}
        if run_id in existing and not args.force:
            if not metadata_file.exists() or json.loads(metadata_file.read_text()).get("run_digest") != digest:
                raise ValueError("run-id collision or missing metadata")
            return existing[run_id]
    set_seed(args.seed); device = choose_device(args.device)
    config = ModelConfig(vocab_size=int(metadata["vocab_size"]), context_length=64, n_layers=2,
                         n_heads=4, width=width, dropout=0.1, layer_kind=kind,
                         attention_layer_kind=attn, ffn_layer_kind=ffn, quaternion_init=init)
    model = initialize_model(config, args.seed, args.model == "quat_gaussian").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    per_step = 8 * 64; steps = max(1, math.ceil(args.tokens / per_step)); warmup = max(1, int(.02 * steps))
    generator = torch.Generator().manual_seed(args.seed + 1000)
    eval_steps = set([0, steps - 1] + [max(0, math.ceil(steps * p / 20) - 1) for p in range(1, 21)])
    curves, curve_best, processed, training_s, first_step = [], float("inf"), 0, 0.0, 0
    checkpoint_dir = args.output / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = checkpoint_dir / (run_id + "-latest.pt"), checkpoint_dir / (run_id + "-best.pt")
    if latest.exists() and not args.force:
        state = torch.load(latest, map_location=device)
        if state["run_digest"] != digest: raise ValueError("checkpoint digest mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        first_step, processed, curves, curve_best, training_s = state["next_step"], state["processed_tokens"], state["curves"], state["curve_best_loss"], state["training_s"]
        generator.set_state(state["batch_generator"]); torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda" and state.get("device_rng") is not None: torch.cuda.set_rng_state(state["device_rng"], device)
        if device.type == "mps" and state.get("device_rng") is not None and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(state["device_rng"])
    start = time.perf_counter()
    for step in range(first_step, steps):
        scale = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        optimizer.param_groups[0]["lr"] = 3e-4 * scale
        x, y = make_batch(train, 8, 64, generator, device)
        _sync(device); train_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True); loss = model(x, y)[1]; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); processed += per_step
        _sync(device); training_s += time.perf_counter() - train_start
        if step in eval_steps:
            val_loss = evaluate_sequential(model, valid, CURVE_TOKENS, EVAL_BATCH, 64, device)
            curves.append({"step": step + 1, "tokens": processed, "train_loss": loss.item(), "val_loss": val_loss, "eval_tokens": CURVE_TOKENS})
            if val_loss < curve_best:
                curve_best = val_loss
                _atomic_torch({"model": model.state_dict(), "model_config": vars(config), "run_digest": digest}, best_path)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "next_step": step + 1,
                     "processed_tokens": processed, "batch_generator": generator.get_state(), "cpu_rng": torch.get_rng_state(),
                     "device_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else
                         torch.mps.get_rng_state() if device.type == "mps" and hasattr(torch.mps, "get_rng_state") else None),
                     "curves": curves, "curve_best_loss": curve_best, "training_s": training_s, "run_digest": digest}
            _atomic_torch(state, latest)
    final = evaluate_sequential(model, valid, FINAL_TOKENS, EVAL_BATCH, 64, device)
    elapsed = time.perf_counter() - start
    _atomic_torch({"model": model.state_dict(), "model_config": vars(config), "run_digest": digest}, checkpoint_dir / (run_id + "-final.pt"))
    curve_file = args.output / "curves" / (run_id + ".csv"); curve_file.parent.mkdir(parents=True, exist_ok=True)
    with curve_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "tokens", "train_loss", "val_loss", "eval_tokens"]); writer.writeheader(); writer.writerows(curves)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL): commit = "source-sha256:" + source
    except (OSError, subprocess.CalledProcessError): commit = "source-sha256:" + source
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else ""
    row = dict(zip(SCHEMA, [run_id, args.model, width, parameter_count(model), args.tokens, args.seed,
        dataset_hash(args.data), final, math.exp(min(final, 80)), final, processed / training_s, peak,
        elapsed, commit, str(device), datetime.now(timezone.utc).isoformat()]))
    _atomic_json(dict(digest_config, run_id=run_id, run_digest=digest, requested_training_tokens=args.tokens,
                      actual_processed_training_tokens=processed, synchronized_training_s=training_s,
                      curve_best_val_loss=curve_best, wall_clock_s=elapsed, result=row), metadata_file)
    _write_unique(results_file, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True); parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--data", type=Path, default=Path("qbench/data"))
    parser.add_argument("--output", type=Path, default=Path("qbench/results")); parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-tokens", type=int, default=2_000_000); parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true"); args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
