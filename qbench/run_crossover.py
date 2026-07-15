import subprocess
import sys

for budget in (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000):
    seeds = range(1, 6) if budget <= 10_000_000 else range(1, 4)
    for seed in seeds:
        for model in ("real", "quat"):
            subprocess.run([sys.executable, "-m", "qbench.run", "--model", model, "--tokens", str(budget), "--seed", str(seed)], check=True)
