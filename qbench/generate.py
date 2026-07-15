"""Generate fixed qualitative samples from completed final 50M checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from qllm.model.transformer import ModelConfig, TransformerLM
from qllm.train import choose_device, set_seed

PROMPTS = ["Once upon a time", "The little robot", "One sunny morning", "Mia found a strange"]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=Path("qbench/results"))
    parser.add_argument("--data", type=Path, default=Path("qbench/data")); parser.add_argument("--device", default="auto")
    parser.add_argument("--new-tokens", type=int, default=128); args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json")); device = choose_device(args.device); outputs = []
    for metadata_path in sorted((args.results / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata["nominal_tokens"] != 50_000_000 or metadata["model"] not in ("real", "quat"): continue
        checkpoint = args.results / "checkpoints" / (metadata["run_id"] + "-final.pt")
        if not checkpoint.exists(): continue
        state = torch.load(checkpoint, map_location=device); model = TransformerLM(ModelConfig(**state["model_config"])).to(device)
        model.load_state_dict(state["model"]); model.eval()
        for index, prompt in enumerate(PROMPTS):
            set_seed(10000 + index); tokens = torch.tensor([tokenizer.encode(prompt).ids], device=device)
            with torch.inference_mode(): generated = model.generate(tokens, args.new_tokens, .8)
            outputs.append({"run_id": metadata["run_id"], "model": metadata["model"], "seed": 10000 + index,
                            "temperature": .8, "prompt": prompt, "text": tokenizer.decode(generated[0].tolist())})
    destination = args.results / "samples.json"; destination.write_text(json.dumps(outputs, indent=2) + "\n")
    print("wrote %d samples to %s" % (len(outputs), destination))


if __name__ == "__main__": main()
