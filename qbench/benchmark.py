"""Forward throughput benchmark for real, fused, and reference projections."""

import argparse
import csv
import time
from pathlib import Path

import torch

from qllm.layers import QuaternionLinear, RealLinear
from qllm.train import choose_device


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"): torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=50); parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64); parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--width", type=int, default=100); parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("qbench/benchmarks/throughput.csv")); args = parser.parse_args()
    device = choose_device(args.device); x = torch.randn(args.batch, args.seq, args.width, device=device)
    quat = QuaternionLinear(args.width, args.width, bias=False).to(device)
    cases = [("real", RealLinear(args.width, args.width, bias=False).to(device), None),
             ("quaternion_fused", quat, None), ("quaternion_naive", quat, quat.naive_forward)]
    rows = []
    with torch.inference_mode():
      for name, layer, function in cases:
        call = function or layer
        for _ in range(args.warmups): call(x)
        sync(device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iterations): call(x)
        sync(device)
        elapsed = time.perf_counter() - start
        rows.append({"implementation": name, "batch": args.batch, "seq": args.seq, "width": args.width,
                     "warmups": args.warmups, "iterations": args.iterations,
                     "tokens_per_sec": args.batch * args.seq * args.iterations / elapsed,
                     "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else
                         torch.mps.current_allocated_memory() / 2 ** 20 if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory") else "unavailable"),
                     "device": str(device)})
    speeds = {row["implementation"]: row["tokens_per_sec"] for row in rows}
    if speeds["quaternion_fused"] <= speeds["quaternion_naive"]:
        raise RuntimeError("fused quaternion projection is not faster than naive on %s" % device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
