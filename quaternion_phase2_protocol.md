# Phase 2 Protocol: Real vs Quaternion Transformers — Finding Where Quaternions Win

**Instructions for an AI coding agent.** Execute in order. Do not skip the verification stage. Do not change hypotheses or decision rules after seeing results.

---

## 0. Context (read first)

Phase 1 result: at ~170K params on TinyStories, 50M tokens/run, 3 paired seeds, the real model beat the quaternion model (mean val loss 2.6607 vs 2.7039, paired Δ = +0.0432, 95% CI [+0.0105, +0.0759]). A 1M-token pilot had favored quaternion. Quaternion throughput was ~2× lower (14,758 vs 29,289 tok/s) on Apple MPS.

Phase 2 has three goals:
1. **Characterize the crossover** — is quaternion a sample-efficiency advantage that disappears with data?
2. **Remove confounds** — width mismatch (100 vs 64) and slow Hamilton-product implementation.
3. **Produce a real benchmark** — a reproducible harness with fixed data, fixed evals, machine-readable results.

---

## 1. Stage A — Verification harness (do this before any training)

Build `tests/test_models.py` covering both models (`smallgpt_real`, `smallgpt_quat`). All tests must pass before Stage B.

1. **Parameter count test.** Assert exact parameter counts and log them. Fail if the real/quat difference exceeds 1.0%.
2. **Quaternion algebra test.** Verify the Hamilton product implementation against a naive reference on random inputs (atol 1e-5). Check the four identities: i²=j²=k²=−1, ij=k, jk=i, ki=j.
3. **Equivalence test.** A quaternion linear layer with weights set to a real scalar embedding (i, j, k components zeroed appropriately) must reproduce a real linear layer's output.
4. **Gradient test.** `torch.autograd.gradcheck` on the quaternion linear layer in float64.
5. **Determinism test.** Two runs with seed 1 for 200 steps must produce bit-identical loss curves (set all seeds, `torch.use_deterministic_algorithms(True)` where the backend allows).
6. **Overfit test.** Both models must reach < 0.1 train loss on a single batch of 64 sequences within 2,000 steps. If either fails, stop and debug.
7. **Data hygiene.** Assert train/val split is disjoint; hash the tokenized dataset and record the hash in every results row.

## 2. Stage B — Fast quaternion implementation

The 2× slowdown is implementation, not physics. Replace per-component Hamilton products with a **single dense matmul**:

- For quaternion weight W = (Wr, Wi, Wj, Wk) with real shapes [d/4, d/4] each, construct once per forward pass the real block matrix H(W) of shape [d, d]:

```
[ Wr  -Wi  -Wj  -Wk ]
[ Wi   Wr  -Wk   Wj ]
[ Wj   Wk   Wr  -Wi ]
[ Wk  -Wj   Wi   Wr ]
```

- Then `y = x @ H(W).T`. Verify numerically against the naive implementation (test A2 applies).
- Benchmark both implementations: 50 warmup iters, 200 timed iters, batch 64, seq 256, report tok/s and peak memory. Record in `benchmarks/throughput.csv`.
- Use the faster implementation for all Stage C/D runs. Report the new tok/s ratio in the final write-up.

## 3. Stage C — Experiment 1: Crossover sweep (primary experiment)

**Hypothesis H1 (pre-registered):** quaternion structure acts as an inductive bias that helps in the data-limited regime and hurts in the data-rich regime; there exists a crossover token budget T* below which quaternion val loss ≤ real val loss.

- **Models:** exactly the Phase 1 pair — real (168,576 params, width 64) and quaternion (169,800 params, width 100). Same tokenizer, data order per seed, LR schedule (cosine, scaled to each budget), batch size, and eval protocol.
- **Token budgets:** 0.5M, 1M, 2M, 5M, 10M, 25M, 50M.
- **Seeds:** 5 seeds (1–5) for budgets ≤ 10M; 3 seeds (1–3) for 25M and 50M. Seeds are paired: seed k of real and seed k of quat see identical data order.
- **Metric:** final val loss on a fixed held-out set of ≥ 2M tokens, evaluated with the same code path for both models. Also log val loss every 5% of the budget to get curves.
- **Compute estimate:** ~935M total training tokens; at Phase 1 throughputs roughly 10–14 h on MPS, less with the Stage B speedup. If needed, drop the 25M budget, never the small budgets.

**Analysis (fixed in advance):**
- Per budget: paired Δ = quat − real per seed; report mean Δ, 95% CI (t-distribution, df = n_seeds − 1), and per-seed sign.
- Plot mean Δ vs log(token budget) with CI band. The crossover T* is where the CI band crosses zero.
- Decision rule: H1 supported if quaternion wins (CI excludes 0, Δ < 0) at ≥ 1 small budget AND loses at 50M. H1 rejected if real wins or ties at every budget.

## 4. Stage D — Experiment 2: Confound ablations

Run only at two budgets (1M and 50M), 3 seeds each, to explain *why* the Phase 1 result holds.

1. **Width-matched quaternion:** width 64 (≈ 4× fewer projection params than real). Separates "algebra effect" from "shape effect." If width-64 quat ≈ width-100 quat, width wasn't doing anything; if it's much worse, the extra width was propping quaternion up.
2. **Param-matched real at width 100:** if feasible via factorized/low-rank projections, this is the mirror control. If not feasible cleanly, skip and note it.
3. **Placement ablation:** quaternion in FFN only, attention projections only, both. Determines whether harm is localized.
4. **Init ablation:** standard quaternion init (Parcollet-style, chi-distributed magnitude + random unit axis) vs naive Gaussian per component. Bad init can fully explain late-training losses.

## 5. Stage E — The benchmark deliverable

Package everything as `qbench/` so anyone can rerun it:

- `qbench/run.py --model {real,quat,quat_w64,...} --tokens N --seed K` → one results row.
- `results/results.csv` schema: `run_id, model, width, params, token_budget, seed, data_hash, final_val_loss, final_val_ppl, best_val_loss, tokens_per_sec, peak_mem_mb, wall_clock_s, git_commit, device, timestamp`.
- `results/curves/*.csv`: step-level val-loss curves.
- `analysis/analyze.py`: regenerates all statistics and plots from `results.csv` alone. No hand-computed numbers in the report.
- Plots: (1) Δ vs token budget with CI band, (2) val-loss curves overlaid per budget, (3) throughput bar chart naive vs fused vs real, (4) ablation table.
- `REPORT.md`: filled from a template — setup, exact configs, results tables, CI values, decision-rule outcomes, limitations (single dataset, single small scale, single device), and 3–5 generation samples per model at 50M for qualitative sanity.

## 6. Interpretation & next-direction rules (pre-committed)

- **If a crossover exists:** frame quaternions as a *sample-efficiency / low-data method*, not a general replacement. Next step: test on a genuinely small dataset (e.g., 5–10M-token corpus trained to convergence) and one scale-up point (~2.8M params) at the crossover ratio of tokens/param.
- **If quaternion loses everywhere:** write it up as a clean negative result (these are valuable — keep the CIs). Then pivot to domains where the quaternion algebra matches the data's structure instead of being an arbitrary constraint: 3D rotation/pose data (robotics, molecular conformations, IMU signals), RGB(A) image channels, or complex/hypercomplex-valued signals (audio, RF). Language token embeddings have no natural 4D structure, which is the most likely root cause of the Phase 1 result.
- **If results are mixed/noisy:** increase to 7 seeds at the two most ambiguous budgets before concluding anything. Never add seeds only where quaternion is winning.

## 7. Reporting standards

- Every claim in `REPORT.md` must trace to a row in `results.csv`.
- Report means with CIs, per-seed signs, and effect sizes; never a single-seed comparison.
- Report negative results with the same prominence as positive ones.
- State the scope honestly: "for this dataset, scale, and implementation."
