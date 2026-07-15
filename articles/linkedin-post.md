I do not like empty claims or guesses. That includes my own.

I previously claimed that quaternion projections could beat ordinary real-valued matrix projections at the same parameter count when training data is limited. A short pilot supported it. One short run is not evidence I would trust, so I tested the claim properly.

Yesterday I had a business trip. Before leaving, I gave my workstation a job. It ran for a little over 20 hours.

The test covered 86 training runs and 1.247 billion training tokens. I compared a 168,576-parameter real Transformer with a 169,800-parameter quaternion Transformer on the same TinyStories token stream. Seeds were paired, validation was fixed, and the decision rule was written before the sweep.

Short result:

* Quaternion won at 0.5M, 1M, and 2M tokens.
* At 5M, the difference was inconclusive.
* Real matrices won at 10M, 25M, and 50M tokens.
* The measured crossover sits between 5M and 10M tokens.
* At 50M, real won in all three seeds. Mean validation loss was 2.7047 for real and 2.7419 for quaternion.

So the original claim survives, but only in a narrower form. In this small language model, quaternion structure helped when data was scarce. It became a constraint after enough training. It was also slower.

That is more useful than saying quaternions are better. They are not, as a general replacement for matrices. Here they bought early sample efficiency and gave it back later.

Protocol, code, results, graphs, and the full write-up:
https://github.com/ibackstrom/QuartAI
