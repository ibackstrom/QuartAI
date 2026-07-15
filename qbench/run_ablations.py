import subprocess
import sys

for budget in (1_000_000, 50_000_000):
    for seed in range(1, 4):
        for model in ("quat_w64", "quat_attn", "quat_ffn", "quat_gaussian"):
            subprocess.run([sys.executable, "-m", "qbench.run", "--model", model, "--tokens", str(budget), "--seed", str(seed)], check=True)
