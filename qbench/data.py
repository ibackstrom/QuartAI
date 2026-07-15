"""Prepare Phase 2 without changing any Phase-1 training artifact."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from qllm.data.prepare import BASE, download_prefix

SEPARATOR = b"\n<|endoftext|>\n"
TOKENIZER_SHA256 = "8c67350fd8f8b52071cf792950c4d06087bf3f0579f20548b26d949c4dee603b"
TRAIN_SHA256 = "34b660826d8e10827e1b210720f6d1538aeec41c46823ff6f82e8e45797babec"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "train.bin", "validation.bin"):
        digest.update((directory / name).read_bytes())
        digest.update(SEPARATOR)
    return digest.hexdigest()


def story_sets(directory: Path):
    """Return complete stories only; downloaded prefix edge fragments are ignored."""
    def complete(path: Path):
        parts = path.read_bytes().split(SEPARATOR)
        # Both first and last pieces can be partial prefixes/suffixes.
        return set(part.strip() for part in parts[1:-1] if part.strip())
    return complete(directory / "train.txt"), complete(directory / "validation.txt")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def prepare_phase2(output: Path, phase1: Path = Path("qllm/data/cache")) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    expected = {"tokenizer.json": TOKENIZER_SHA256, "train.bin": TRAIN_SHA256}
    for name, pinned in expected.items():
        if sha256(phase1 / name) != pinned:
            raise ValueError("Phase-1 %s does not match pinned SHA256" % name)
    for name in ("tokenizer.json", "train.bin", "train.txt"):
        shutil.copyfile(str(phase1 / name), str(output / name))
        if sha256(output / name) != sha256(phase1 / name):
            raise IOError("byte copy verification failed: %s" % name)

    validation_txt = output / "validation.txt"
    download_prefix(BASE + "TinyStoriesV2-GPT4-valid.txt", validation_txt, 8 * 1024 * 1024)
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    ids = tokenizer.encode(validation_txt.read_text(errors="ignore")).ids
    np.asarray(ids, dtype=np.uint16).tofile(str(output / "validation.bin"))
    if len(ids) <= 2_000_000:
        raise ValueError("validation must contain more than 2,000,000 tokens")
    train, valid = story_sets(output)
    if train & valid:
        raise ValueError("train and validation contain a shared complete story")
    components = {name: sha256(output / name) for name in
                  ("tokenizer.json", "train.bin", "train.txt", "validation.bin", "validation.txt")}
    metadata = {"dataset": "roneneldan/TinyStories", "vocab_size": tokenizer.get_vocab_size(),
                "dtype": "uint16", "tokens": {"train": (output / "train.bin").stat().st_size // 2,
                "validation": len(ids)}, "data_hash": dataset_hash(output),
                "component_hashes": components, "story_disjoint": True}
    _atomic_json(output / "metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("qbench/data"))
    parser.add_argument("--phase1", type=Path, default=Path("qllm/data/cache"))
    args = parser.parse_args()
    print(json.dumps(prepare_phase2(args.output, args.phase1), indent=2))
