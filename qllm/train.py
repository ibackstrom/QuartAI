"""Training loop used unchanged by real, complex, and quaternion models."""

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from qllm.model.transformer import ModelConfig, TransformerLM, parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_parameter_count(kind: str, width: int, base: Dict, vocab_size: int) -> int:
    # Exact formula for this architecture (tied embedding/head and bias-free projections).
    multiplier = {"real": 12, "complex": 6, "quaternion": 3}[kind]
    layers = int(base["n_layers"])
    fixed_per_width = vocab_size + int(base["context_length"]) + 4 * layers + 2
    return multiplier * layers * width * width + fixed_per_width * width


def matched_width(kind: str, target: int, base: Dict, vocab_size: int) -> Tuple[int, float]:
    component = {"real": 1, "complex": 2, "quaternion": 4}[kind]
    stride = math.lcm(int(base["n_heads"]), component)
    candidates = range(stride, int(base["width"]) * 4 + stride, stride)
    width = min(candidates, key=lambda w: abs(model_parameter_count(kind, w, base, vocab_size) - target))
    error = abs(model_parameter_count(kind, width, base, vocab_size) - target) / target
    if error > 0.02:
        raise ValueError(f"closest {kind} width {width} misses parameter target by {error:.2%}")
    return width, error


def make_batch(data: np.memmap, batch_size: int, context: int, generator: torch.Generator,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - context - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i + context], dtype=np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + context + 1], dtype=np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: TransformerLM, data: np.memmap, batches: int, batch_size: int,
             context: int, device: torch.device, seed: int) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = make_batch(data, batch_size, context, generator, device)
        losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_sequential(model: TransformerLM, data: np.memmap, token_count: int,
                        batch_size: int, context: int, device: torch.device) -> float:
    """Evaluate one fixed prefix with non-overlapping sequential windows."""
    usable = min(token_count, len(data) - 1)
    windows = usable // context
    if windows < 1:
        raise ValueError("validation split is shorter than one context window")
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for first in range(0, windows, batch_size):
        starts = [i * context for i in range(first, min(first + batch_size, windows))]
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i + context], dtype=np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + context + 1], dtype=np.int64)) for i in starts])
        loss = model(x.to(device), y.to(device))[1]
        count = len(starts) * context
        total_loss += loss.item() * count
        total_tokens += count
    model.train()
    return total_loss / total_tokens


def train_one(config: Dict, data_dir: Path, run_dir: Path, device_name: str = "auto") -> Dict:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    dtype = np.dtype(metadata["dtype"])
    train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(data_dir / "validation.bin", dtype=dtype, mode="r")
    vocab_size = int(metadata["vocab_size"])
    seed = int(config.get("seed", 1))
    set_seed(seed)
    device = choose_device(device_name)

    model_config = ModelConfig(vocab_size=vocab_size, context_length=int(config["context_length"]),
                               n_layers=int(config["n_layers"]), n_heads=int(config["n_heads"]),
                               width=int(config["width"]), mlp_ratio=int(config.get("mlp_ratio", 4)),
                               dropout=float(config.get("dropout", 0.1)), layer_kind=config["layer_kind"],
                               attention_layer_kind=config.get("attention_layer_kind"),
                               ffn_layer_kind=config.get("ffn_layer_kind"),
                               quaternion_init=config.get("quaternion_init", "parcollet"))
    model = TransformerLM(model_config).to(device)
    params = parameter_count(model)
    if not config.get("attention_layer_kind") and not config.get("ffn_layer_kind"):
        expected = model_parameter_count(config["layer_kind"], config["width"], config, vocab_size)
        assert params == expected, (params, expected)

    batch_size, context = int(config["batch_size"]), int(config["context_length"])
    tokens_per_step = batch_size * context
    steps = max(1, math.ceil(int(config["token_budget"]) / tokens_per_step))
    warmup = max(1, int(steps * float(config.get("warmup_fraction", 0.02))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config.get("weight_decay", 0.1)))
    generator = torch.Generator().manual_seed(seed + 1000)
    eval_interval = max(1, int(config.get("eval_interval", max(steps // 5, 1))))
    eval_batches = int(config.get("eval_batches", 8))
    history: List[Dict] = []
    best_loss = float("inf")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    processed = 0

    for step in range(steps):
        if step < warmup:
            scale = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(config["learning_rate"]) * scale
        x, y = make_batch(train_data, batch_size, context, generator, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        processed += tokens_per_step

        if step == 0 or (step + 1) % eval_interval == 0 or step + 1 == steps:
            val_loss = evaluate(model, val_data, eval_batches, batch_size, context, device, seed + 2000)
            elapsed = time.perf_counter() - start
            point = {"step": step + 1, "tokens": processed, "train_loss": loss.item(),
                     "val_loss": val_loss, "seconds": elapsed, "tokens_per_second": processed / elapsed}
            history.append(point)
            print(f"{config['name']:>16} step {step+1:4}/{steps} val {val_loss:.4f} "
                  f"ppl {math.exp(min(val_loss, 80)):.2f} tok/s {point['tokens_per_second']:.0f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model.state_dict(), "model_config": vars(model_config)}, run_dir / "best.pt")

    peak_memory = (torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    result = {"name": config["name"], "layer_kind": config["layer_kind"], "width": config["width"],
              "parameters": params, "seed": seed, "token_budget": processed, "best_val_loss": best_loss,
              "best_perplexity": math.exp(min(best_loss, 80)), "wall_seconds": time.perf_counter() - start,
              "tokens_per_second": processed / (time.perf_counter() - start), "peak_memory_bytes": peak_memory,
              "device": str(device), "torch_version": torch.__version__, "python": platform.python_version(),
              "dataset": metadata["dataset"], "history": history, "config": config}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def load_config(path: Path) -> Dict:
    base_path = path.parent / "real-base.yaml"
    base = yaml.safe_load(base_path.read_text()) if path.name != "real-base.yaml" else {}
    base.update(yaml.safe_load(path.read_text()))
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("qllm/data/cache"))
    parser.add_argument("--runs", type=Path, default=Path("qllm/runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--token-budget", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("match") == "same_params":
        metadata = json.loads((args.data / "metadata.json").read_text())
        target = model_parameter_count("real", int(cfg["width"]), cfg, int(metadata["vocab_size"]))
        cfg["width"], _ = matched_width(cfg["layer_kind"], target, cfg, int(metadata["vocab_size"]))
    if args.token_budget:
        cfg["token_budget"] = args.token_budget
    train_one(cfg, args.data, args.runs / f"{cfg['name']}-seed{cfg['seed']}", args.device)
