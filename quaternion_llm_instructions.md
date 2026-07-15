# Instructions for AI Agent: Quaternion & Complex-Valued Language Model Experiment

## Mission

Implement and train small autoregressive language models using **hypercomplex-valued weights** (complex numbers and quaternions) and rigorously compare them against a standard **real-valued transformer baseline** to measure whether hypercomplex parameterization improves parameter efficiency, sample efficiency, or final quality.

You are expected to work autonomously: set up the environment, implement all code, run all experiments, and produce a final report with tables and plots. Do not skip the baseline or the controls — the entire point is a fair comparison.

---

## 1. Scientific framing (read before coding)

### 1.1 The hypothesis being tested

**H1 (parameter efficiency):** At an equal number of *trainable parameters*, a quaternion-valued transformer achieves lower validation perplexity than a real-valued transformer, because the Hamilton product enforces weight sharing across the 4 quaternion components.

**H2 (width efficiency):** At equal *hidden width* (same activation dimensionality), the quaternion model reaches comparable perplexity with ~4× fewer parameters (complex: ~2× fewer).

**Null result is acceptable.** If the real baseline wins, report it honestly. The deliverable is evidence, not confirmation.

### 1.2 Key mathematical background

A quaternion is `q = r + xi + yj + zk` with `i² = j² = k² = ijk = −1`.

The **Hamilton product** of two quaternions `q = (r₁,x₁,y₁,z₁)` and `p = (r₂,x₂,y₂,z₂)`:

```
r = r₁r₂ − x₁x₂ − y₁y₂ − z₁z₂
x = r₁x₂ + x₁r₂ + y₁z₂ − z₁y₂
y = r₁y₂ − x₁z₂ + y₁r₂ + z₁x₂
z = r₁z₂ + x₁y₂ − y₁x₂ + z₁r₂
```

A **quaternion linear layer** mapping `n` quaternion units to `m` quaternion units has `4·n·m` real parameters, whereas a real linear layer over the equivalent `4n → 4m` real dimensions has `16·n·m` parameters. That is the 4× reduction. For complex layers the reduction is 2×.

Implementation note: implement the quaternion linear layer as a single real matmul against a **structured block matrix** built from the four weight components:

```
W_hamilton = [[ Wr, -Wx, -Wy, -Wz],
              [ Wx,  Wr, -Wz,  Wy],
              [ Wy,  Wz,  Wr, -Wx],
              [ Wz, -Wy,  Wx,  Wr]]
```

Then `output = input_real_view @ W_hamilton.T`. This runs on GPU as one dense matmul and is autograd-friendly. Do NOT loop over quaternion components in Python.

### 1.3 Prior art to be consistent with (naming, not required reading)

- Trabelsi et al., "Deep Complex Networks" (2018)
- Parcollet et al., "Quaternion Recurrent Neural Networks" / "Quaternion Convolutional Neural Networks" (2018–2019)
- Zhang et al., "Beyond Fully-Connected Layers with Quaternions" / PHM layers (2021)

Use the same conventions where possible (split activations, quaternion-aware init).

---

## 2. Environment setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy datasets tokenizers matplotlib pandas tqdm
```

- Target PyTorch ≥ 2.1 with CUDA if available; fall back to CPU with reduced model sizes.
- Fix seeds: `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and set `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Create the repo layout:

```
qllm/
  layers/quaternion.py      # QuaternionLinear, ComplexLinear + init
  layers/norms.py           # LayerNorm variants
  model/transformer.py      # shared decoder skeleton, parameterized by layer type
  data/prepare.py           # dataset download + tokenization
  train.py                  # training loop, logging, checkpoints
  eval.py                   # perplexity, generation samples
  experiments/configs/*.yaml
  report/                   # plots, tables, final report.md
```

---

## 3. Implementation requirements

### 3.1 Layers (`layers/quaternion.py`)

Implement three interchangeable linear layers with identical interfaces (`in_features`, `out_features` in REAL dimensions, both divisible by 4):

1. **`RealLinear`** — thin wrapper over `nn.Linear` (baseline).
2. **`ComplexLinear`** — weights stored as `(W_re, W_im)`, applied via the 2×2 block structure `[[W_re, -W_im],[W_im, W_re]]`. Parameter count = half of real.
3. **`QuaternionLinear`** — four weight tensors `(Wr, Wx, Wy, Wz)`, applied via the Hamilton block matrix above. Parameter count = quarter of real.

**Initialization:** implement quaternion-aware init (Parcollet-style): sample magnitude from a Chi distribution with 4 DOF scaled to preserve variance `σ = 1/√(2·fan_in_quaternion)`, sample a random unit pure quaternion for direction, and a uniform phase. For ComplexLinear use the analogous Rayleigh-magnitude complex init from Trabelsi et al. If this proves unstable, fall back to component-wise Xavier scaled by `1/√4` (quaternion) or `1/√2` (complex), and record which init was used.

**Unit tests (mandatory before training):**
- Hamilton product of unit quaternions has unit norm (tolerance 1e-5).
- `QuaternionLinear` output matches a naive per-component reference implementation on random input.
- Parameter counts are exactly 1/4 and 1/2 of `RealLinear` at the same real dimensions.
- Gradients flow (no NaN, nonzero) through a 3-layer stack after one backward pass.

### 3.2 Activations and norms

- Use **split activations**: apply GELU independently to every real component. (Fully quaternion-analytic activations are a known open problem; split activations are the standard choice — state this in the report.)
- LayerNorm: use standard real LayerNorm over the full real-valued view. Optionally implement quaternion layer norm (normalize by quaternion magnitude) as an ablation, not the default.

### 3.3 Model (`model/transformer.py`)

One decoder-only transformer skeleton, parameterized by `layer_cls ∈ {RealLinear, ComplexLinear, QuaternionLinear}`. The hypercomplex layer replaces:

- Q, K, V projections and the attention output projection
- both MLP projections

Keep **real-valued**: token embedding table, final LM head (tied to embeddings), LayerNorms, and the attention softmax itself (attention scores are computed on the real view via standard scaled dot-product; do NOT invent quaternion attention scores in v1 — that is an ablation, see §6).

Architecture defaults (small, single-GPU friendly):

- Context length: 256
- Layers: 6, Heads: 8
- Baseline hidden width: 384 (real), MLP ratio 4
- Dropout 0.1, weight tying on/off recorded in config

### 3.4 Matched configurations (critical)

Build these five configs. Parameter counts must be within ±2% between matched pairs — print and assert this at startup.

| Config | Layer type | Width | Matching target |
|---|---|---|---|
| `real-base` | Real | 384 | reference |
| `quat-samewidth` | Quaternion | 384 | same width, ~4× fewer params (tests H2) |
| `quat-sameparams` | Quaternion | ~768 | same params as real-base (tests H1) |
| `cplx-samewidth` | Complex | 384 | same width, ~2× fewer params |
| `cplx-sameparams` | Complex | ~544 | same params as real-base |

Compute exact widths programmatically to hit the param target; widths must stay divisible by 4 (quaternion) / 2 (complex) and by head count.

---

## 4. Data

- **Primary dataset: TinyStories** (`roneneldan/TinyStories` on Hugging Face). It is small enough to see convergence quickly and produces qualitatively judgeable generations. If the network blocks Hugging Face, fall back to WikiText-2 raw, or as last resort a public-domain Gutenberg text bundle — record which was used.
- Tokenizer: train a BPE tokenizer, vocab 8192, on the training split only.
- Fixed token budget per run: **200M training tokens** (reduce to 50M if compute-limited, but keep it IDENTICAL across all configs).
- Held-out validation split, identical for all runs.

---

## 5. Training protocol

- Optimizer: AdamW, lr 3e-4 with cosine decay, warmup 2% of steps, weight decay 0.1, grad clip 1.0, batch size chosen to fit memory but identical token throughput per step across configs.
- Mixed precision (bf16) if GPU supports it; verify the quaternion block-matmul is numerically stable in bf16 first (compare fp32 vs bf16 outputs on random input; if relative error > 1e-2, keep quaternion layers in fp32).
- **3 seeds per config** (seeds 1, 2, 3). If compute-limited, 3 seeds for `real-base` and `quat-sameparams`, 1 seed for the rest — report which.
- Log every 100 steps: train loss, val loss (on a fixed 2M-token val slice), tokens/sec, peak memory, wall-clock.
- Checkpoint the best-val model per run.
- Abort-and-diagnose rule: if any run NaNs, halve the lr for that config once, restart, and record it. Do not silently tune one arm and not the others.

---

## 6. Evaluation and ablations

### 6.1 Primary metrics (report all, per config, mean ± std across seeds)

1. Final validation perplexity at the fixed token budget
2. Validation perplexity vs tokens curve (sample efficiency)
3. Trainable parameter count
4. Wall-clock time to reach real-base's final perplexity (or "not reached")
5. Tokens/sec and peak GPU memory
6. 10 generation samples per model (temperature 0.8, fixed prompts) for qualitative eyeballing

### 6.2 Decision rules

- **H1 supported** if `quat-sameparams` beats `real-base` perplexity by more than 2× the across-seed std.
- **H2 supported** if `quat-samewidth` is within 5% relative perplexity of `real-base` despite ~4× fewer parameters.
- Otherwise report the null result with the curves.

### 6.3 Ablations (run only after the main grid completes)

1. Quaternion init vs component-wise Xavier (does init matter?)
2. Quaternion layers in MLP only vs attention only vs both
3. Optional stretch: quaternion attention scores via Hamilton-product Q·K̄ (conjugate), real part as score — flag as experimental

---

## 7. Reporting

Produce `report/report.md` containing:

1. Exact configs, seeds, dataset, tokenizer, hardware
2. A results table for all five configs (all §6.1 metrics)
3. Two plots: val-loss vs tokens, and val-loss vs wall-clock
4. Honest interpretation against H1/H2 decision rules, including failure modes observed (instability, throughput penalty from the block-matrix construction, etc.)
5. Limitations section: small scale, one dataset, split activations, real-valued attention scores — and what would be needed to claim anything at LLM scale

## 8. Honesty constraints (non-negotiable)

- Never compare runs with different token budgets, tokenizers, or lr schedules.
- Never cherry-pick the best seed; always report mean ± std.
- If the quaternion model is slower per token (the block-matrix construction adds overhead), report time-matched curves too, not just token-matched.
- If results are null or negative, the report says so plainly.
