"""Stage-A deterministic verification on the prepared, real memmaps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qllm.model.transformer import ModelConfig
from qllm.train import choose_device, evaluate_sequential, make_batch, set_seed
from qbench.run import initialize_model


def execution(kind, data, device):
    metadata = json.loads((data / "metadata.json").read_text())
    train = np.memmap(data / "train.bin", dtype=np.dtype(metadata["dtype"]), mode="r")
    valid = np.memmap(data / "validation.bin", dtype=np.dtype(metadata["dtype"]), mode="r")
    width = 64 if kind == "real" else 100
    config = ModelConfig(int(metadata["vocab_size"]), 64, 2, 4, width, dropout=.1, layer_kind=kind)
    set_seed(1); torch.use_deterministic_algorithms(True)
    model = initialize_model(config, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.1)
    batches = torch.Generator().manual_seed(1001); train_curve, val_curve = [], []
    for step in range(200):
        x, y = make_batch(train, 8, 64, batches, device)
        optimizer.zero_grad(set_to_none=True); loss = model(x, y)[1]; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_curve.append(loss.item())
        if (step + 1) % 10 == 0:
            val_curve.append(evaluate_sequential(model, valid, 131072, 64, 64, device))
    return train_curve, val_curve


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=Path("qbench/data"))
    parser.add_argument("--device", default="auto"); args = parser.parse_args()
    if not (args.data / "metadata.json").exists(): raise FileNotFoundError("prepare qbench data first")
    device = choose_device(args.device)
    for kind in ("real", "quaternion"):
        first, second = execution(kind, args.data, device), execution(kind, args.data, device)
        differences = [abs(left - right) for left_curve, right_curve in zip(first, second)
                       for left, right in zip(left_curve, right_curve)]
        maximum = max(differences)
        if device.type == "mps":
            # MPS reports no forbidden nondeterministic operation but accumulates
            # sub-ULP scheduling differences in reductions.
            if maximum > 1e-6:
                raise AssertionError("%s MPS reproducibility drift: %.9g" % (kind, maximum))
            print("%s: 200-step MPS curves agree within 1e-6 (max %.9g)" % (kind, maximum))
        else:
            if maximum != 0:
                raise AssertionError("%s curves are not bit-identical: %.9g" % (kind, maximum))
            print("%s: bit-identical 200-step train/validation curves on %s" % (kind, device))


if __name__ == "__main__": main()
