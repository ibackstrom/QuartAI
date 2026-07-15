"""Evaluate a saved checkpoint and print a short generation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from qllm.model import ModelConfig, TransformerLM
from qllm.train import choose_device, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("qllm/data/cache"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--new-tokens", type=int, default=80)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    device = choose_device()
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    metadata = json.loads((args.data / "metadata.json").read_text())
    validation = np.memmap(args.data / "validation.bin", dtype=np.dtype(metadata["dtype"]), mode="r")
    loss = evaluate(model, validation, 20, 8, config.context_length, device, 1234)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    prompt = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    model.eval()
    output = model.generate(prompt, args.new_tokens)[0].tolist()
    print(f"validation loss={loss:.4f}, perplexity={math.exp(loss):.2f}")
    print(tokenizer.decode(output))
