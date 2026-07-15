# Why I ran this test

I had made a claim that quaternion projections might outperform ordinary real-valued matrix projections at a similar parameter count. The first experiment gave me a reason to say it. It did not give me enough evidence to keep saying it.

The awkward part was the contradiction. A 1M-token pilot favored the quaternion model. A later 50M-token, three-seed test favored the real model in every seed. Either the pilot was noise, or the answer changed with the amount of training data.

I wanted the second possibility tested without moving the target after seeing the numbers. The token budgets, seed counts, metric, confidence intervals, and decision rule were fixed first. The tokenizer and training stream were copied byte for byte from Phase 1. The evaluation set was enlarged and then held fixed for every run.

I started the sweep before a business trip and left the workstation running. It processed 1.247 billion training tokens across 86 runs in a little over 20 hours. This is still a small experiment, not a claim about every model or dataset. But it is large enough to replace the guess with a measured boundary.

The result was not a clean victory for either architecture. That is the useful part.

[Read the technical article](quaternion-transformers-crossover.md)
