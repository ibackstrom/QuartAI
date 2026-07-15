"""Download a bounded TinyStories sample, train BPE on train only, and encode it."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"


def download_prefix(url: str, destination: Path, max_bytes: int) -> None:
    if destination.exists() and destination.stat().st_size >= max_bytes:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        remaining = max_bytes
        while remaining:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            output.write(chunk)
            remaining -= len(chunk)


def prepare(output_dir: Path, vocab_size: int, train_mb: int, val_mb: int,
            download: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt, val_txt = output_dir / "train.txt", output_dir / "validation.txt"
    if download:
        download_prefix(BASE + "TinyStoriesV2-GPT4-train.txt", train_txt, train_mb * 1024 * 1024)
        download_prefix(BASE + "TinyStoriesV2-GPT4-valid.txt", val_txt, val_mb * 1024 * 1024)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train([str(train_txt)], trainer)  # validation never influences vocabulary
    tokenizer.save(str(output_dir / "tokenizer.json"))

    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    counts = {}
    for split, path in (("train", train_txt), ("validation", val_txt)):
        ids = tokenizer.encode(path.read_text(errors="ignore")).ids
        np.asarray(ids, dtype=dtype).tofile(output_dir / f"{split}.bin")
        counts[split] = len(ids)
    metadata = {"dataset": "roneneldan/TinyStories (bounded prefix)", "vocab_size": tokenizer.get_vocab_size(),
                "dtype": np.dtype(dtype).name, "tokens": counts, "train_mb": train_mb, "val_mb": val_mb}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("qllm/data/cache"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--train-mb", type=int, default=64)
    parser.add_argument("--val-mb", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.vocab_size, args.train_mb, args.val_mb), indent=2))
