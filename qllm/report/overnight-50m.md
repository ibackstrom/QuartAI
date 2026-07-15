# Real and quaternion language models after 50 million tokens

## Short article

We trained a real-valued Transformer and an equal-parameter quaternion variant on TinyStories. Each model ran for 50 million tokens with seeds 1, 2, and 3. The experiment processed 300,002,304 tokens in total and took 4 hours 15 minutes on Apple MPS.

The real model had 168,576 parameters and width 64. The quaternion model had 169,800 parameters and width 100, a parameter difference of 0.73 percent.

The real model won all three paired runs. Its mean validation loss was 2.66070, compared with 2.70390 for quaternion. Mean perplexity was 14.309 for real and 14.941 for quaternion, so quaternion perplexity was 4.41 percent higher. The paired loss difference, quaternion minus real, was +0.04320. Its 95 percent confidence interval was [+0.01049, +0.07592].

Quaternion was also slower in this implementation. It processed 14,758 tokens per second, compared with 29,289 for real.

The earlier 1 million-token pilot favored quaternion at equal parameter count. That result did not persist at 50 million tokens. For this model, dataset, and implementation, quaternion projections reduced final quality and training throughput.

## Brief post

Our 300 million-token comparison is complete. The real Transformer beat the equal-parameter quaternion model in all three seeds. Real reached 14.309 mean perplexity, quaternion reached 14.941, 4.41 percent higher. Quaternion also trained at about half the throughput. The advantage seen in the 1 million-token pilot disappeared with longer training.
