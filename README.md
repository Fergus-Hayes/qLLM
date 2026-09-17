# qLLM

Utilities for benchmarking (quantized) language models.

## LLM benchmark: perplexity, inference time & memory

The `qllm` package fetches any Hugging Face **causal LM** and reports:

- **Perplexity** on a text corpus (WikiText-2 by default), computed with a
  sliding window so every target token is scored with maximal left-context and
  counted exactly once. The window is clamped per model to its native context
  length, so the same setting works for short- and long-context models.
- **Inference time** — prefill latency (single forward pass over the prompt)
  and greedy-decoding throughput in tokens/second.
- **Memory footprint** — parameter, buffer, and total memory needed to hold the
  model, measured from the actual tensors so it reflects the loaded dtype
  (e.g. `float16` halves the footprint versus `float32`).

It is model-agnostic: SmolLM2-135M is only the default target — pass any
number of model ids (or a list file) to benchmark them in one sweep.

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Default target (HuggingFaceTB/SmolLM2-135M), CPU, full WikiText-2 test set
python benchmark_llm.py                 # or:  python -m qllm

# An explicit model
python benchmark_llm.py HuggingFaceTB/SmolLM2-360M

# Sweep several models in one run
python benchmark_llm.py HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B gpt2

# Read the model list from a file (see models.txt)
python benchmark_llm.py --models-file models.txt

# GPU, half precision, quick perplexity subset
python benchmark_llm.py gpt2 --device cuda --dtype float16 --max-eval-tokens 20000
```

Run `python benchmark_llm.py --help` for the full list of options (dataset,
window size, stride, prompt, warm-up runs, `--trust-remote-code`, `--revision`,
and flags to skip either the perplexity or the timing stage).

### Speeding up the perplexity evaluation

The perplexity loop scores overlapping sliding windows; those windows are
independent, so the biggest win is **running several per forward pass** instead
of one at a time. This is controlled by `--ppl-batch-size` (default 8) and is
numerically identical to the one-at-a-time result (verified across batch sizes)
— it just keeps the matmul units saturated.

```bash
# GPU: large batch + half precision is by far the fastest
python benchmark_llm.py --device cuda --dtype float16 --ppl-batch-size 32

# CPU: batch a few windows and use all cores
python benchmark_llm.py --ppl-batch-size 8 --threads 8

# Fewer FLOPs (accuracy trade-off): non-overlapping windows halve the work
python benchmark_llm.py --stride 1024            # stride == window
```

Levers, roughly in order of impact:

1. **`--device cuda --dtype float16`/`bfloat16`** — a GPU is 10–100× a CPU here;
   half precision adds ~2× and halves memory.
2. **`--ppl-batch-size N`** — near-linear speedup until memory-bound (raise it on
   GPU; keep it modest for large models or low-RAM CPUs; `1` restores the old
   behavior).
3. **`--threads N`** — CPU intra-op parallelism (BLAS already threads, but this
   pins the count).
4. **`--stride` (↑) / `--max-eval-tokens` (↓)** — do less work: a larger stride
   reduces window overlap, `--max-eval-tokens` caps the corpus for a quick run.
5. **Across models**, the sweep is embarrassingly parallel — run separate
   processes (with `--resume`) or, on multi-GPU, one model per device.

The same `--ppl-batch-size` / `--threads` flags apply to `analyze_layers.py`,
where they accelerate the per-layer perplexity-sensitivity probe.

### Results / CSV output

Each run is appended (with a header on first creation) to a **per-model** CSV at
`results/llms/<model-name>/benchmark.csv`, and to a **combined** CSV at
`results/llms/summary.csv` for cross-model comparison. So as more models are
analyzed, `summary.csv` accumulates one comparable row per run:

| override | effect |
| --- | --- |
| `--results-dir DIR` | base directory for per-model CSVs |
| `--summary-csv PATH` | combined CSV path (`''` disables it) |
| `--no-csv` | write no CSV at all |

CSV columns: `timestamp, model_id, device, dtype, num_parameters,
param_memory_mb, buffer_memory_mb, total_memory_mb, context_window, perplexity,
eval_tokens, dataset, perplexity_seconds, prompt_tokens, prefill_latency_ms,
generation_tokens_per_second, generated_tokens, error`.

### Example output

```
============================================================
Model:               HuggingFaceTB/SmolLM2-135M
Device / dtype:      cpu / float32
Parameters:          134,515,008 (134.5M)
Parameter memory:    513.13 MB
Buffer memory:       0.00 MB
Total model memory:  513.13 MB (0.501 GB)
Context window:      1024 tokens
Perplexity:          14.8021 (304985 tokens on Salesforce/wikitext/wikitext-2-raw-v1:test)
Perplexity eval:     2514.85 s
Prefill latency:     57.90 ms (6 prompt tokens)
Generation speed:    16.18 tokens/s (64 new tokens)
============================================================

Saved results to results/llms/SmolLM2-135M/benchmark.csv
Appended to summary  results/llms/summary.csv
```

> Absolute numbers depend on hardware, dtype, and evaluation settings; the
> perplexity figure is deterministic for a given window/stride/dataset.

## Perplexity error vs. number of tokens

`perplexity_error.py` (module `qllm.ppl_error_cli`) measures how the **sampling
error** of a perplexity estimate shrinks as more tokens are scored — the question
"how many tokens do I need for a trustworthy PPL?".

Perplexity is `PPL = exp(mean NLL)`, so estimating it on `N` tokens is estimating
a mean; by the delta method its *relative* error equals the standard error of the
mean NLL, which for text falls as `~ C · N^(-1/2)` (quadruple the tokens → halve
the error). The constant `C` and the **effective** sample size (smaller than `N`,
because tokens in a window share context) are model- and corpus-specific, so the
tool measures them rather than assuming them.

It scores a long corpus **once**, keeping the per-window `(nll_sum, n_tokens)`
sufficient statistics (`qllm.window_nlls`), then reports:

- the **running** perplexity vs. tokens (a convergence trace);
- the **relative error vs. `N`**, from a block bootstrap over windows *and* from
  direct disjoint `N`-token chunks (two independent standard-error estimates),
  against the i.i.d.-token ideal `∝ N^(-1/2)`;
- a fit `rel_err ≈ C · N^p` (with `p ≈ -0.5`), the **variance-inflation** over
  i.i.d. (the cost of within-window correlation), and the **tokens required** for
  target errors (e.g. 1%, 0.5%, 0.1%).

```bash
# Dense-model PPL error vs. tokens (score up to 131072 tokens once) + plots + CSV
python perplexity_error.py HuggingFaceTB/SmolLM2-135M --max-eval-tokens 131072

# Also the PAIRED ratio error for one compressed layer — the quantity a
# compression sweep actually compares (its shared corpus noise cancels)
python perplexity_error.py HuggingFaceTB/SmolLM2-135M --max-eval-tokens 131072 \
    --compare-chi 8 --block 10 --layer-type v_proj --tensorization qubit
```

The paired mode scores the model with one layer MPO-compressed on the **same**
windows and reports the error of the perplexity *ratio* (compressed / dense).
Because a sweep compares that ratio, and the dense and compressed NLLs are
strongly correlated per token, the ratio's error is what should be compared
against the differences you care about (e.g. the ~0.001 gaps between healed and
cold `ppl_ratio`) when choosing `--per-layer-eval-tokens` / `--max-eval-tokens`.
Output is `results/llms/<model>/perplexity_error/` (`perplexity_error.csv` +
`perplexity_error.png`).

## Per-layer analysis: entropy, spectrum & sensitivity vs. depth

`analyze_layers.py` (module `qllm.layer_analysis`) inspects **every 2-D weight
matrix** in a model and computes:

| metric | definition |
| --- | --- |
| **Shannon entropy** | `H = -Σ pᵢ log pᵢ` of the normalized singular-value spectrum `pᵢ = σᵢ/Σσ` (nats); `exp(H)` is the *effective rank* |
| **Rényi-2 (collision) entropy** | `H₂ = -log Σ pᵢ²` |
| **Spectral gap** | `σ₁ − σ₂` |
| **Condition number** | `σ_max / σ_min` |
| **Perplexity sensitivity** | relative PPL change per unit relative weight perturbation: perturb `W` so `‖ΔW‖/‖W‖ = ε`, re-measure PPL, restore — `(PPL' − PPL)/PPL/ε` |

Each weight matrix is tagged with its **layer type** (module role, e.g.
`self_attn.q_proj`) and **depth** (transformer block index), and the tool prints
and plots how every metric varies over depth for each layer type.

```bash
python analyze_layers.py                                  # default: SmolLM2-135M
python analyze_layers.py HuggingFaceTB/SmolLM2-360M
python analyze_layers.py gpt2 --skip-sensitivity          # spectral metrics only (fast)
python analyze_layers.py --sensitivity-eval-tokens 2048 --epsilon 0.02
```

### Comparing across layers of different shapes

Weight matrices differ in shape (`q_proj` 576×576, `k/v_proj` 192×576,
`mlp` 1536×576, `embed_tokens` 49152×576, …), and raw spectral metrics are
confounded by two things that have nothing to do with the layer's "character":

1. **Scale** — the overall magnitude of the weights (hence of the singular
   values) varies by initialization, learning rate, and normalization.
2. **Size/shape** — even statistically identical random matrices give larger
   entropy, wider spectral gaps, and larger condition numbers as the number of
   singular values `k = min(rows, cols)` (and the aspect ratio) grows.

So the analysis reports **shape-normalized, scale-free** variants alongside the
raw ones, and the depth summary/plots use the normalized set:

| raw metric | normalized / comparable form | why it's comparable |
| --- | --- | --- |
| Shannon / Rényi-2 entropy | `÷ log k` → `shannon_entropy_normalized`, `renyi2_entropy_normalized` ∈ [0,1] | `log k` is the max possible entropy for `k` singular values |
| effective rank `exp(H)` | `effective_rank_ratio = exp(H)/k` ∈ (0,1] | fraction of directions "active" |
| — | `stable_rank = ‖W‖_F²/σ₁²`, `stable_rank_ratio = /k` | scale-free, robust (no σ_min blow-up) |
| spectral gap `σ₁−σ₂` | `relative_spectral_gap = (σ₁−σ₂)/σ₁` ∈ [0,1] | dividing by σ₁ removes the weight scale |
| condition number `σ₁/σ_min` | `log_condition_number = log₁₀κ`; `condition_number_rmt_ratio = κ / E[κ_random(shape)]`; `condition_number_mp_ratio = κ / κ_MP` | κ grows with size, so it is divided by a random-matrix baseline for the *same shape*. `condition_number_rmt_ratio` uses the **empirical** median κ of iid Gaussian matrices of that shape — defined for **every** shape including square. `condition_number_mp_ratio` uses the analytic Marchenko–Pastur bulk value `κ_MP = (1+√γ)/(1−√γ)`, `γ = k/max(rows,cols)`, which is finite only for rectangular matrices (NaN for square, where γ=1 makes it diverge — random square matrices are genuinely asymptotically ill-conditioned, `κ ~ n`). |
| perplexity sensitivity | already `(ΔPPL/PPL)/ε` with `‖ΔW‖/‖W‖ = ε` | relative response to a relative perturbation — scale-free and shape-comparable by construction |

The guiding rule: express every quantity as a **ratio** (dividing out the weight
scale, e.g. by σ₁ or ‖W‖_F) and against its **shape ceiling or random-matrix
expectation** (dividing out `k`/aspect ratio, e.g. by `log k`, `κ_MP`, or the
empirical `E[κ_random(shape)]`). The empirical condition-number baseline is a
small Monte-Carlo (`--rmt-samples`, default 5, cached per shape) — the one
normalization that is well-defined for square projections (`q_proj`, `o_proj`)
too. Raw columns are kept in the CSV for reference. For cross-*model* comparison,
z-scoring each normalized metric within `(model, layer_type)` groups removes any
residual family-specific offset.

Outputs go to `results/llms/<model-name>/layer_analysis/` — the
`layer_analysis.csv` checkpoint and the `depth_<metric>.png` plots. Files from
the older flat layout are migrated automatically on the next run, so existing
checkpoints (and the expensive sensitivity values in them) are preserved.

**Printouts & checkpointing.** Progress is printed as each layer is processed
(spectral line, sensitivity line, running count + ETA). Results are written to
`results/llms/<model-name>/layer_analysis.csv`, which **is** the checkpoint:
the CSV is rewritten atomically after each layer, and re-running skips a layer
only when the metrics you asked for are already present. Checkpointing is
**per-metric**, so a fast spectral-only pass (`--skip-sensitivity`) followed by a
full run correctly fills in the missing sensitivity column in place (no
duplicate rows). Pass `--recompute` to ignore the checkpoint entirely. A console
summary of "metric over depth, by layer type" is printed at the end, and (if
`matplotlib` is installed) one `depth_<metric>.png` per metric is saved next to
the CSV. The perplexity-sensitivity probe deliberately uses a small token budget
(it runs once per layer); tune it with `--sensitivity-eval-tokens`,
`--sensitivity-window`, `--epsilon`, and `--sensitivity-samples`.

The benchmark sweep (`benchmark_llm.py`) similarly prints perplexity progress
(running PPL + ETA) and supports `--resume` to skip models already recorded in
their per-model CSV.

## CompactifAI: MPO compression vs. bond dimension

`compactify.py` (module `qllm.compactifai`) implements the compression method of
*CompactifAI: Extreme Compression of Large Language Models using
Quantum-Inspired Tensor Networks* (Tomut et al.) and sweeps its truncation
parameter.

**Method.** Each Self-Attention / MLP weight matrix in a decoder block is
replaced by a **Matrix Product Operator (MPO)**: the matrix indices are
reshaped (`d_out = o₁…o_N`, `d_in = i₁…i_N`), permuted to interleave the pairs
`(o_k, i_k)`, and decomposed by **N−1 sequential SVDs keeping only the largest χ
singular values** at each step. The bond dimension **χ** controls how much of the
layer's correlation structure is retained. Parameters stored are the sum of the
MPO tensor sizes — for the paper's 216×216 / 3-site example this reproduces
exactly `2·36χ + 36χ²` (verified in the test suite).

Embedding and head layers are excluded, as in the paper. The truncated MPO is
contracted back into the layer so perplexity reflects the compressed operator
exactly, while the *reported* parameter count is that of the stored MPO tensors.

**Two modes** (`--mode`):

| mode | what is compressed | result |
| --- | --- | --- |
| `global` (default) | **all** eligible layers together at each χ | one perplexity per χ — the whole-model compression curve |
| `per-layer` | **one layer at a time**, every other layer left dense | a perplexity-vs-χ curve *per layer* — isolates each layer's tolerance to truncation |

Use `--mode both` to run each in turn.

>
> The paper's **"healing"** (brief retraining) stage is available in per-layer
> mode via `--heal` (see below). Without it, reported perplexities are
> pre-healing — the raw cost of truncation.

```bash
# 12 log-spaced bond dimensions on SmolLM2-135M
python compactify.py

# Explicit bond dimensions, faster eval, threaded SVD build
python compactify.py --chi 2 4 8 16 32 64 128 --ppl-batch-size 16 --svd-workers 4

# Follow the paper's sensitivity advice
python compactify.py --exclude-down-proj      # leave block-output MLP dense
python compactify.py --min-depth 4            # don't compress the earliest blocks

# 3-site MPO (two sequential SVDs, as in the paper's figure)
python compactify.py --mpo-sites 3
```

χ values are **logarithmically spaced** between `--chi-min` and `--chi-max`
(default: auto — the largest χ that still stores fewer parameters than the dense
matrix, computed per model from the layer shapes).

**Speed.** The sweep is built to avoid repeated work:

| technique | effect |
| --- | --- |
| **Cached SVDs** | a 2-site MPO needs exactly one SVD per layer, so each layer is decomposed **once** and every χ is produced by truncation alone — `n_χ × n_layers` SVDs become `n_layers` |
| **Corpus tokenized once** | the dataset is loaded/tokenized a single time and re-scored for each χ |
| **Batched scoring** | `--ppl-batch-size` scores many sliding windows per forward pass |
| `--svd-workers N` | builds the SVD cache across threads (LAPACK releases the GIL) |
| `--threads`, `--device cuda`, `--dtype float16` | the usual device/precision levers |
| `--max-eval-tokens` | caps tokens per evaluation (default 20000, since the corpus is re-scored once per χ) |

### Per-layer profiling: a perplexity curve for every layer

`--mode per-layer` compresses **one layer at a time** (all others left dense) and
measures perplexity at each χ, giving an independent curve per layer — the
paper's layer-sensitivity profiling. This costs `n_layers x n_chi` evaluations,
so it uses its own smaller token budget (`--per-layer-eval-tokens`, default 4096)
and is checkpointed per (layer, χ) pair.

**Recommended invocation** — a representative subset of layers across the full
depth range, at a token budget that is meaningful without being prohibitive:

```bash
python compactify.py --mode per-layer --preset standard --ppl-batch-size 16 --threads 8
```

`--preset` picks the subset and budget for you; `--num-depths` chooses that many
**evenly spaced decoder blocks automatically from the model's actual depth**
(always including the first and last), so it works for any model without you
knowing its block count. All 7 layer types are profiled at each depth.

| preset | blocks | χ points | tokens/eval | evaluations (7 types) |
| --- | --- | --- | --- | --- |
| `quick` | 4 | 6 | 4 096 | ~168 |
| `standard` | 6 | 8 | 8 192 | ~336 |
| `thorough` | 10 | 10 | 16 384 | ~700 |

For SmolLM2-135M (30 blocks), `standard` profiles blocks
`[0, 6, 12, 17, 23, 29]` — early, middle and late — which is where the paper's
sensitivity gradient shows up. Any explicit flag overrides the preset
(`--preset standard --num-depths 4`), and `--layer-types q_proj down_proj`
narrows the set further.

Two things keep the budget affordable: the probe uses **non-overlapping**
windows (`--per-layer-stride` defaults to the window size, halving the forward
passes versus the overlapping default), and each layer's χ list is clamped to its
exact-reconstruction rank so redundant lossless points are skipped. Because every
condition is scored on the *same* tokens, corpus-sampling error largely cancels in
`ppl_ratio`, so a moderate budget still resolves differences between layers well.

```bash
# Paper-style explicit blocks (it uses 0, 5, 15, 31 for a 32-block model)
python compactify.py --mode per-layer --profile-depths 0 5 15 29

# Every eligible layer (slow: ~n_layers x n_chi evaluations)
python compactify.py --mode per-layer

# Whole-model sweep and per-layer profile in one go
python compactify.py --mode both

# Redraw the figures from existing CSVs, without touching the model
python compactify.py --plot-only
```

**Outputs** — everything lands in `results/llms/<model>/compactifai/`
(checkpointed; re-running skips work already recorded, `--recompute` forces a
redo; files from the older flat layout are migrated automatically):

- `compactifai_sweep.csv` — one row per χ (all layers compressed): parameter and
  memory reduction (layer-level and model-level), mean relative reconstruction
  error, perplexity, and ratio to the uncompressed baseline.
- `compactifai_layers.csv` — per (χ, layer): the index factorization, bond
  dimensions, parameter counts, compression ratio, and Frobenius error.
- `compactifai_per_layer.csv` — per (layer, χ) from `--mode per-layer`:
  perplexity with only that layer compressed.
- `perplexity_vs_bond_dimension_per_layer.png` — **the per-layer curves**: one
  panel per layer type, one log-log curve per decoder block (coloured by depth),
  with the dense baseline as a dashed line.
- `perplexity_vs_bond_dimension_all_layers.png` — every profiled layer on a
  single axes, coloured by layer type.
- `perplexity_vs_bond_dimension_global.png` — the whole-model sweep: perplexity
  and percentage of parameters removed vs. χ.

A summary table is printed at the end, one row per bond dimension (columns
shown here; values are produced by the run):

```
   chi         params  model -%  layer -%  rel.err  perplexity  x base
   ...   (compressed   (model    (layer    (Frobenius  (measured  (vs
          param count)  size      size      recon.      ppl)       baseline)
                        saved)    saved)    error)
```

Perplexity should fall monotonically toward the baseline as χ grows, reaching it
once χ is large enough that the MPO is lossless.

## Healing: recover accuracy by retraining the compressed layer

`--heal` (per-layer mode) adds the paper's healing step. After a layer is
truncated to its MPO, that layer's **two MPO tensors are briefly retrained**
(Adam) to minimise the language-model loss, while **every other weight in the
model stays frozen**. The layer therefore stays compressed — its parameter count
is unchanged; only the low-rank factors move — and the run records the healed
perplexity next to the raw one.

```bash
python compactify.py --mode per-layer --preset standard --heal \
    --heal-steps 40 --heal-lr 1e-3 --heal-tokens 16384 --heal-split train \
    --device cuda --ppl-batch-size 16
```

- Healing trains on a **disjoint split** (`--heal-split`, default `train`) so the
  healed perplexity is never measured on the healing tokens.
- Requires the default 2-site MPO (`--mpo-sites 2`) so the truncation factors
  cleanly into the two trainable tensors `A @ B`.
- Extra CSV columns: `perplexity_healed`, `ppl_ratio_healed`,
  `heal_recovered_frac` (fraction of the truncation-induced perplexity increase
  recovered, in [0, 1]), `heal_steps`, `heal_final_loss`, `heal_seconds`.
- The per-layer plot overlays the healed curve as a dashed line (triangles) on
  top of the raw (solid) curve, so the recovery is visible per layer.
- Checkpointing is heal-aware: re-running with `--heal` fills in the healed
  columns for points a prior no-heal run left blank (it does not re-evaluate
  points already healed).

**Cost:** healing runs a short training loop (forward + backward) per (layer, χ)
point, so it is much heavier than the raw probe — use a GPU, keep `--heal-steps`
modest, and profile a subset (`--profile-depths` / `--layer-types`) first.

## What predicts compressibility? (metrics vs. compression)

`analyze_compressibility.py` (module `qllm.compressibility_metrics`) joins the
per-layer MPO sweep to the `layer_analysis` metrics on `param_name` and asks:
**which layer property predicts how far a layer can be compressed before
perplexity degrades?**

It reduces each layer's perplexity-vs-χ curve to a compressibility scalar
(default: negated log perplexity-ratio at the median χ; higher = more
compressible; alternatives `--target chi_at_budget` / `max_compression` with a
`--budget`), then Spearman-correlates each metric against it. **Only the
shape-normalized / scale-free metrics are considered** — the entropies ÷ log k,
the rank ratios ÷ k, the relative spectral gap, the log / RMT-normalized
condition number, and the already-relative perplexity sensitivity — so the
comparison is not confounded by raw scale or matrix size.

```bash
# All eligible layers
python analyze_compressibility.py     results/llms/SmolLM2-135M/compactifai/compactifai_per_layer.csv     results/llms/SmolLM2-135M/layer_analysis/layer_analysis.csv

# Self-attention layers only (q/k/v/o)
python analyze_compressibility.py PER_LAYER.csv LAYER_ANALYSIS.csv --attention-only

# MLP layers only, or a custom subset
python analyze_compressibility.py PER_LAYER.csv LAYER_ANALYSIS.csv --mlp-only
python analyze_compressibility.py PER_LAYER.csv LAYER_ANALYSIS.csv --layer-types o_proj v_proj
```

It prints the metrics ranked by |Spearman ρ| (positive ρ = more of that metric →
more compressible, regardless of `--target`) and writes
`compressibility_vs_metrics.png` (scatter of compressibility vs. each top metric,
coloured by layer type) next to the per-layer CSV. Restricting to a single
family (e.g. `--attention-only`) removes the MLP-vs-attention split that
otherwise dominates the correlations, so what remains reflects variation *within*
that family.

## Quantum disentanglers: is the PQC worth its parameters?

`hybridize.py` (modules `qllm.disentangler`, `qllm.hybrid_sweep`) implements the
hybrid layer of *Quantum Large Language Models via Tensor Network Disentanglers*
(Aizpurua et al.) and measures it **against** the CompactifAI layer above, on the
same probe tokens. `analyze_budget.py` (module `qllm.budget_frontier`) then
answers the question the two methods are really competing over:

> At a perplexity budget **B**, which layer is smaller -- the tensor network, or
> the tensor network with quantum circuits bolted on?

Formally, per layer:

```
N* = min_chi        C(chi)                    s.t.  PPL_classical(chi)   <= B
M* = min_{chi',D,k} C(chi') + w . Q(D, k)     s.t.  PPL_hybrid(chi',D,k) <= B
```

`C` is the classical parameter count of the stored MPO tensors, `Q` that of the
two disentangling circuits, and `w` the price of a quantum parameter in units of
a classical one. Where `M*/N*` is small, the circuits buy real compression.

### The hybrid layer

Each weight matrix is rewritten as `W ~= U MPO_new V^T`, with `U` and `V`
variational quantum circuits acting on the layer's output and input indices and
`MPO_new` the residual tensor network that stays classical. Disentangling moves
correlations out of the network and into the circuits, so `MPO_new` tolerates a
far smaller bond dimension than the MPO of `W` itself -- in the paper, all the
way to `chi' = 1`.

| ingredient | how it is implemented here |
| --- | --- |
| **qubit embedding** | each index is zero-padded to `2^ceil(log2 d)` (the paper's `(576, 192) -> (10, 8)` qubits). The padding is exact but enlarges the residual MPO, so it is a real cost -- see `D = 0` below |
| **circuit ansatz** | brickwall of real orthogonal `k`-qubit gates as **PennyLane `qml.QubitUnitary` operations**, depth `D`. Circuits are realized by `qml.matrix` (`qllm.disentangler.circuit_unitary`) and can be handed to hardware/transpilation via `circuit_ops`. Gates **default to `k = 2`** (the hardware-realistic regime -- the paper runs its two-qubit-gate disentanglers on a real QPU, and transpiling wider gates blows up the physical depth), but there is **no upper cap**: `--gate-sizes 0` is a single register-wide gate (the paper's Table I `10qU, 8qV`), and any `k` is allowed. Depth `D` is the other lever |
| **training** (`--disentangle-optimizer`) | three schemes for the same objective. **`explicit`** (default) is the paper's disentangling algorithm (Appendix A): each gate is set to the polar factor of its environment tensor, `E = A S B -> g <- A B` (Eq. A4), a closed-form optimal step, sweeping U/V until the overlap converges. **`gradient`** optimizes the gate *angles* directly by Adam on a differentiable truncation loss (`--disentangle-gd-steps`, `--disentangle-gd-lr`): each 2-qubit gate is `expm(skew(theta))` with `theta` its `dim so(4) = 6` Lie-algebra angles, so unconstrained descent stays on the orthogonal-gate manifold. The circuit is run **through PennyLane** each step (`circuit_unitary`'s `qml.matrix` keeps the torch autograd graph), so the angles are trained on the PennyLane circuit itself -- no separate torch model. Same gates (`qml.QubitUnitary`), same objective; a closed-form vs. an iterative solver. `gradient` is much slower per optimization (it rebuilds the register unitary every step) and the explicit env-SVD sweep both converges faster and reaches a better optimum for this objective, so it stays the default. **`--fast-gradient`** keeps the same optimizer but forms `U^T W V` with the pure-torch `apply_circuit` local contractions (`O(#gates)`, no `2^n x 2^n` register unitary) instead of PennyLane's `qml.matrix` -- identical result (to numerical noise), ~1.4-1.7x faster end-to-end and far lighter on memory for wide registers (the isolated circuit build alone is 4-9x faster; the per-step SVD and backward dilute that end-to-end). It applies to `gradient` and the gradient half of `explicit+gradient`, and is a no-op for `explicit`. **`explicit+gradient`** runs the sweep first, then warm-starts the gradient optimizer *from the sweep's gates* (each gate `base @ expm(skew(theta))`, `theta = 0` at start, so the polish begins exactly at the explicit result) for `--disentangle-gd-steps` more Adam steps. The polish monotonically descends the gradient loss, which is the *mean per-bond* rank-`chi'` discarded weight -- this coincides with the reported `retained`/`accuracy` only for a single-bond **`balanced`** plan (there the polish reliably helps), whereas on the multi-site **`qubit`** MPO it is a proxy and can move those headline metrics either way even as its own loss falls; compare against `explicit` and keep the better when `retained` is what you optimize |
| **objective** | `min ||U^T W V - T_chi(U^T W V)||`. The target is re-truncated each iteration, so the alternation is **monotone** in the error the compressed layer actually incurs. `--disentangle-target fixed` freezes it at the original's truncation instead, which is the literal objective behind the paper's Eq. (4) accuracy |
| **`Q(D, k)`** | independent gate angles, `dim O(2^k) = 2^(k-1)(2^k - 1)` per gate (`--quantum-param-counting entries` counts `4^k` matrix entries instead, the right figure if a gate is stored as a dense classical tensor) |
| **`D = 0`** | always measured: the circuits are the identity, so that row isolates the qubit-padding overhead from the benefit of the circuits |

Sweeping costs no extra memory. The gates are orthogonal, so the partial product
to the right of the current gate is recovered from the running one by applying
the *old* gate transposed, and the left one accumulates from the freshly updated
gates -- one sweep is `O(#gates)` applications and stores `O(1)` of them. One
disentangling optimization is reused across the whole `chi'` grid (its SVD is
cached), exactly as the classical sweep reuses one SVD per layer, so the heavy
term is the perplexity probe: one evaluation per measured point.

### Tensorization: `balanced` (2-site) vs. `qubit` (the paper's geometry)

`--tensorization` (on both `compactify.py` and `hybridize.py`) chooses how the
MPO factorizes a weight matrix. It applies to **both** curves -- the classical
`C(chi)` and the hybrid `C(chi')` -- so the comparison stays apples-to-apples.

| | `balanced` (default) | `qubit` |
| --- | --- | --- |
| **split** | each index into two coarse factors (`576 -> [24, 24]`) | each index into `log2` dim-2 legs, one MPO site per qubit |
| **`C(chi=1)` for `(576,192)`** | 672 | **36** (matches the paper's Table I: 36, 132, 696, …) |
| **SVDs** | one, cached and re-truncated across the whole sweep | `max(n_out, n_in)` sequential per bond dimension (no cache) |
| **use it for** | fast sweeps, cross-layer scans | matching the paper's parameter counts; the finest compression |

The `qubit` tensorization is exactly the paper's: for the `(192, 576)` layer it
gives 10 sites (8 paired `(2,2)` sites plus 2 input-only `(1,2)` sites at the
tail), storing `36` parameters at bond 1 and `132` at bond 2 -- reproducing
Table I to the integer. Where the two indices need a different qubit count,
`--qubit-align` decides whether the leading (`msb`, default) or trailing (`lsb`)
qubits pair up. It is slower (no single-SVD cache) and the qubit padding is a
real cost the `D=0` row still exposes, but the parameter counts are now the
paper's.

```bash
# Classical CompactifAI sweep with the paper's per-qubit MPO
python compactify.py --mode per-layer --tensorization qubit --layer-types v_proj

# The full M*/N* comparison, both curves on the qubit geometry
python hybridize.py MODEL_ID --profile-depths 10 --layer-types v_proj \
    --tensorization qubit --gate-sizes 2 --circuit-depths 0 1 2 4 8 \
    --solve --budgets 1.003 --q-weights 0 1
```

### Prune the grid first (no model, no tokens)

Whether the hybrid *can* win is settled by arithmetic before any token is
scored: `C_pad(chi') + w Q(D, k) < C(chi)`. `--shapes` answers that offline.

```bash
# SmolLM2-135M's actual layer shapes, against the dense layer
python hybridize.py --shapes v_proj:576x192 q_proj:576x576 up_proj:1536x576 \
    --gate-sizes 2 --circuit-depths 1 2 4 8

# Against a realistic classical operating point instead
python hybridize.py --shapes v_proj:576x192 --gate-sizes 2 \
    --circuit-depths 1 4 --reference-chi 32 --q-weight 1
```

It prints, per `(layer, k, D)`, the circuits' surcharge expressed in bond
dimensions, the largest `chi'` still affordable, and a **VETO** when not even
`chi' = 1` fits. `--dry-run` does the same from a real model's shapes. (The planner and the
measured sweep both accept `k = 0` and wider gates; the default is `k = 2`.)

This matters because the two levers pull against each other: gate size is what
disentangles (the paper's smallest gates barely reduce the entanglement, its
widest ones disentangle at `L = 1`), but `Q` grows as `4^k` per gate -- for
SmolLM2-135M a register-wide gate carries 0.5-2.6 M parameters against layers of
0.1-0.9 M, the paper's own observation that as circuits the disentanglers "carry
at least as many trainable parameters as `W`". Two-qubit gates keep `Q` small
(36-66 parameters for these registers), so the useful lever there is the
brickwall **depth** `D`; register-wide gates (`--gate-sizes 0`) instead
disentangle almost completely at `D = 1` -- the paper's Table I regime -- at the
cost of a large (0.5-2.6 M) quantum-parameter count carried by the circuits.

### Reproduce the paper's Table I (register-wide gates, word-level PPL)

Table I truncates the *disentangled* operator of the `(10qU, 8qV, L=1)` v_proj
layer to bond dimension `chi'` and reports the full-model **word-level** PPL. Use
a register-wide single-layer disentangler (`--gate-sizes 0 --circuit-depths 1`),
the `chi'` grid from the table, and `--word-level`:

```bash
python hybridize.py <PAPER_MODEL> \
    --profile-depths 10 --layer-types v_proj \
    --tensorization qubit --gate-sizes 0 --circuit-depths 1 \
    --disentangle-target-chi 1 --disentangle-target fixed \
    --disentangle-sweeps 40 --restarts 3 \
    --chi 1 2 5 10 50 256 --no-classical --word-level \
    --csv-name table1.csv --ppl-batch-size 32
```

`--gate-sizes 0` builds one register-wide gate per index (a 10-qubit `U` and an
8-qubit `V` for this layer, `Q = 556,416`); `--chi 256` clamps to full rank (the
table's `exact` row). The `MPO_new params` column (`classical_params`: 36, 132,
696, 2,356, 36,948, 401,956) reproduces Table I exactly; `--word-level` matches
its PPL normalisation (a `ppl_unit` column records `word` vs `token`). Omit
`--per-layer-eval-tokens` to score the whole test set as the paper does. Match
the paper's **model** for the absolute baseline (35.292 word-level looks like
SmolLM-135M v1, not SmolLM2-135M); the `∆PPL%` shape reproduces regardless.

### Reconstruction accuracy only (`--no-perplexity`)

The perplexity evaluation is the slow part of the sweep -- every grid point runs
the model over the eval corpus. When you only need the *reconstruction* quality
of the compression (how well `W'` approximates `W`, independent of the LM), pass
`--no-perplexity`: the sweep still truncates / disentangles every point and
records `relative_error` (`‖W-W'‖/‖W‖`), the disentangling `accuracy`, `entropy`,
`retained`, and the parameter counts (`C(chi')`, `Q(D)`, `M*`), but runs **no
forward passes** at all. The `perplexity` columns are left blank, healing is
auto-disabled (it needs the LM loss), and the run is orders of magnitude faster.

This makes it cheap to map accuracy over a large `(D, chi')` grid. To sweep `D`
logarithmically up to depth 256, `chi'` (hybrid bond) logarithmically up to 128,
and `chi` (classical bond) over the same grid:

```bash
python hybridize.py <MODEL> \
    --profile-depths 10 --layer-types v_proj \
    --tensorization qubit --gate-sizes 2 \
    --circuit-depths 0 1 2 4 8 16 32 64 128 256 \
    --chi 1 2 4 8 16 32 64 128 \
    --no-perplexity \
    --disentangle-target-chi 1 --disentangle-target fixed \
    --disentangle-sweeps 40 --restarts 3 \
    --csv-name accuracy_grid.csv
```

Each `D` disentangles once (at target `chi'=1`), then every `chi'` in `--chi`
truncates that same disentangled operator, so `accuracy`/`entropy` are per-`D`
while `relative_error`/`retained` vary per `(D, chi')`. `D=0` is the classical
CompactifAI point (no circuits); `--chi` doubles as both the classical `chi` grid
and the hybrid `chi'` grid. `--solve` is unavailable in this mode (it needs the
perplexity curve) and is skipped with a note.

#### One optimization per `(D, chi')` (`--disentangle-target-per-chi`)

By default a single disentangling optimization per `D` (squeezing to
`--disentangle-target-chi`, i.e. `chi'=1`, the paper's product-operator target)
is reused across the whole `chi'` grid: the circuits are optimal for `chi'=1` and
every larger `chi'` is a post-hoc truncation of that one result, so
`accuracy`/`entropy` are constant down each `D` column. Pass
`--disentangle-target-per-chi` to instead run a **fresh** optimization for every
`(D, chi')` point, each squeezing to `target_chi = chi'` -- the best circuits for
*that* bond dimension. `accuracy`/`entropy`/`retained` then vary per `(D, chi')`,
and `--disentangle-target-chi` is ignored:

```bash
python hybridize.py <MODEL> \
    --profile-depths 10 --layer-types v_proj \
    --tensorization qubit --gate-sizes 2 \
    --circuit-depths 0 1 2 4 8 16 32 64 128 256 \
    --chi 1 2 4 8 16 32 64 128 \
    --no-perplexity --disentangle-target-per-chi \
    --disentangle-target fixed --disentangle-sweeps 40 --restarts 3 \
    --csv-name accuracy_grid_perchi.csv
```

This costs one optimization per grid point instead of one per `D` (here roughly
`8x` more disentangling work), so it is the expensive-but-faithful way to ask
"what is the best each `chi'` can do at each depth". It usually lowers
`relative_error` at larger `chi'`, though not always -- the env-SVD sweep is
non-convex, so a `target_chi = chi'` optimum can occasionally land worse than the
`chi'=1` circuits truncated to `chi'`; `--restarts` mitigates this.

#### Parameter budget (`--max-total-params`)

When you sweep several gate sizes `k` alongside `(chi', D)`, the depth a small
`k` needs to reach a given accuracy is far larger than an all-to-all `k=0` gate
needs, and `Q(D) ~ D` blows up: a deep `k=2` point can cost thousands of quantum
parameters to match what a shallow `k=0` point does in a handful. Those points
are off any sensible Pareto front, so computing them is wasted work. Pass
`--max-total-params M` to cap the budget `M* = C(chi') + Q(D)`: every grid point
over the cap is written as a **NaN row** (its `C`, `Q`, `total_params` are still
recorded, but `relative_error`/`perplexity`/`accuracy` are blank) with **no MPO
truncation, disentangling, healing or model eval**. Crucially, when *every* `chi'`
at a `(D, k)` is over budget, the disentangling optimization itself -- the
expensive part -- is skipped, not just the per-`chi'` evaluations:

```bash
python hybridize.py <MODEL> \
    --profile-depths 10 --layer-types v_proj \
    --tensorization qubit \
    --gate-sizes 0 2 4 \
    --circuit-depths 0 1 2 4 8 16 32 64 128 256 \
    --chi 1 2 4 8 16 32 64 128 \
    --no-perplexity --max-total-params 100000 \
    --csv-name kq_grid.csv
```

The cap applies to the classical surface too (there `Q = 0`, so `M* = C(chi)`).
NaN rows keep the grid rectangular for plotting and count as done for
checkpoint/resume, so the budget only ever removes compute, never grid points.

### Measure both surfaces

```bash
# Both curves for a representative subset of layers, depth-swept at k=2
python hybridize.py --preset standard --gate-sizes 2 --circuit-depths 0 1 2 4 8

# One layer type only
python hybridize.py --gate-sizes 2 --circuit-depths 0 1 2 4 --layer-types v_proj

# Best of several circuit initializations (the optimization is not convex)
python hybridize.py --gate-sizes 2 --circuit-depths 1 2 4 --restarts 3

# Train the circuits by gradient descent instead of the env-SVD sweep
# (--fast-gradient uses the pure-torch circuit path -- same result, faster per step)
python hybridize.py --gate-sizes 2 --circuit-depths 0 1 2 4 \
    --disentangle-optimizer gradient --fast-gradient \
    --disentangle-gd-steps 300 --disentangle-gd-lr 0.05

# Explicit env-SVD sweep, then implicit (Adam) refinement warm-started from it
python hybridize.py --gate-sizes 2 --circuit-depths 0 1 2 4 \
    --disentangle-optimizer explicit+gradient \
    --disentangle-sweeps 40 --disentangle-gd-steps 300 --disentangle-gd-lr 0.03
```

Each `--tensorization` and each `--disentangle-optimizer` writes its **own** CSV
(`hybrid_per_layer.csv`, `hybrid_per_layer_qubit.csv`,
`hybrid_per_layer_gradient.csv`, …) so runs whose `C(chi)` or circuits are not
comparable never share a checkpoint or get mixed in one budget report.

Output is `results/llms/<model>/compactifai/hybrid_per_layer.csv`, one row per
`(layer, method, chi, D, k)` with `tensorization`, `optimizer`,
`classical_params`, `quantum_params`,
`perplexity`, `ppl_ratio` (and, with `--heal`, `perplexity_healed` / `ppl_ratio_healed`), and the disentangling diagnostics
(`disentangle_accuracy` -- the paper's Eq. (4) -- `disentangle_entropy`, and
`disentangle_retained`, the fraction of the layer's weight the target bond
dimension keeps). `method=classical` rows are the pure-TN curve, measured in the
same run so the two budgets are directly comparable. Re-running resumes from the
checkpoint.

### Heal both surfaces to minimize PPL (`--heal`)

`--heal` retrains each swapped-in point against the LM loss (every other layer
dense, the parameter count unchanged) and records its **healed** perplexity, so
the sweep reports the best PPL each method reaches at a given cost. The classical
`C(chi)` rows heal the MPO bond; the hybrid `C(chi'), D` rows heal per
`--heal-mode`: `core` (the `chi'` bond only) or `full` (the bond **and** the
`U`/`V` circuits, all `M*` parameters — the default, since that is where a
starved small-`chi'` bond gains the most). At `D=0` the hybrid `full` point
coincides with the classical bond healing.

```bash
# One layer: sweep chi (classical) and grid chi' x D (hybrid), healing every
# point to minimize PPL, on 32768 eval tokens, checkpointed and resumable.
python hybridize.py HuggingFaceTB/SmolLM2-135M \
    --profile-depths 10 --layer-types v_proj \
    --tensorization qubit --gate-sizes 2 \
    --chi 1 2 4 8 16 --circuit-depths 0 1 2 4 8 16 \
    --heal --heal-mode full \
    --heal-steps 100 --heal-lr 1e-3 --heal-tokens 16384 --heal-split train \
    --per-layer-eval-tokens 32768 --ppl-batch-size 32 --threads 8
```

This adds `perplexity_healed`, `ppl_ratio_healed`, `heal_mode`,
`heal_recovered_frac` (share of the truncation damage healing recovers, in
`[0,1]`), `heal_final_loss` and `heal_seconds` to every row. With `--heal` the
probe defaults to **32768 tokens** if `--per-layer-eval-tokens` is not given, so
the healed perplexities are trustworthy. Healing is checkpoint-aware: a point
counts as done only once it carries a healed perplexity, so adding `--heal` to an
existing cold CSV fills in just the missing healed columns, and the dense
baseline is cached in the CSV and reused on resume.

**Speed.** The perplexity probe dominates, so throughput is set by how well one
evaluation saturates the hardware: the windows of each 32768-token probe are
already batched into one forward pass (raise `--ppl-batch-size` until they fit in
one or two batches), `--threads N` pins the CPU intra-op pool, and one
disentangling optimization is reused across the whole `chi'` column while the
classical SVD is cached across `chi`. On a GPU add `--device cuda` (and
`--dtype bfloat16` for a further ~2x on the probe). Every point is checkpointed,
so an interrupted or extended run never repeats finished work. Healing multiplies
the per-point cost by one short fine-tune plus one extra evaluation, so keep
`--heal-steps` modest and let the checkpoint accumulate results across runs.

### Solve the budget

Either in the same command as the sweep (`hybridize.py --solve`) or afterwards
from the CSV (`analyze_budget.py`) -- both call the same reporting core and both
work for any layers/model:

```bash
# Measure and solve one targeted layer in a single command (any model)
python hybridize.py MODEL_ID --profile-depths 10 --layer-types v_proj \
    --gate-sizes 2 --circuit-depths 0 1 2 4 8 --solve --budgets 1.003 \
    --q-weights 0 1

# Or analyse an existing sweep CSV
python analyze_budget.py results/llms/SmolLM2-135M/compactifai/hybrid_per_layer.csv

# The paper's framing: the circuits run on a QPU and cost no classical memory
python analyze_budget.py HYBRID.csv --q-weight 0

# How does the verdict depend on how a quantum parameter is priced?
python analyze_budget.py HYBRID.csv --q-weight-scan 0 0.01 0.1 1 10

# Tight budget, attention only
python analyze_budget.py HYBRID.csv --budgets 1.002 --attention-only
```

`--solve` reports on the layers the sweep just measured (narrowed by
`--layer-types` / `--profile-depths` when given); `analyze_budget.py` reports on
whatever is in the CSV, filterable the same way. Neither is tied to a particular
model or layer -- the specifics are all arguments.

Budgets are perplexity **ratios** to the dense baseline (`1.01` = at most 1%
worse), which is what the per-layer probe resolves: every point is scored on the
same tokens, so corpus-sampling error largely cancels in `ppl_ratio`. Both
minimizations run over the *measured grid* rather than a fitted curve, so a
non-monotone probe cannot produce a spuriously small `N*`.

It prints:

- the **per-layer table** -- `N*`, `M*`, `M*/N*`, and the winning `(chi', D, k)`;
- **by layer type and by depth** -- median and best `M*/N*`, and how many layers
  the PQC wins, which is the map of *where* the circuits pay;
- the **Pareto front** over both methods together, with the owner of each step;
- the **break-even price** `w*` per layer: the largest `w` at which the hybrid
  still wins. `w*` is the scale-free version of the whole question -- how cheap a
  quantum parameter has to be for that layer to be worth hybridizing -- and it is
  defined even when the hybrid loses at `w = 1`;
- a **model-level rollup**, `N*` and `M*` summed over the comparable layers.

Layers that only one method can bring inside the budget are reported as
`hybrid_only` / `classical_only` rather than scored, so the ratio column never
mixes "smaller" with "the only one that works".

Figures land next to the CSV: `hybrid_vs_classical_ratio.png` (the `M*/N*` map
over layer type x depth, annotated with the winning configuration),
`breakeven_weight.png`, `pareto_front.png`, and `ratio_vs_budget.png` (how the
verdict moves as the budget is loosened).

### Interpreting a result

Three quantities decide every layer, and the tooling separates them:

1. **the padding overhead** -- read it off the `D = 0` rows, where the circuits
   are the identity. A layer whose dimensions sit just above a power of two pays
   for it here before the circuits do anything;
2. **how much the circuits disentangle** -- `disentangle_retained` at `D = 0`
   versus at `D > 0`, and the bond entropy alongside it;
3. **what they cost** -- `Q(D, k)`, and hence `w*`.

A low `M*/N*` needs all three to line up: little padding waste, a circuit wide
enough to actually disentangle, and a price at which its angles are cheaper than
the bond dimensions they save.

### Reproducing a specific layer (e.g. the paper's)

There is no per-paper script: any single layer is a targeted run of the general
tool. The layer of arXiv:2410.17397v2 -- the `(576, 192)` self-attention
projection of block 10 of a compressed SmolLM2 -- is simply

```bash
python hybridize.py HuggingFaceTB/SmolLM2-135M \
    --profile-depths 10 --layer-types v_proj \
    --gate-sizes 2 --circuit-depths 0 1 2 4 8 \
    --solve --budgets 1.003 --q-weights 0 1
```

`--profile-depths 10` picks block 10, `--layer-types v_proj` its `(576, 192)`
projection (`k_proj` is the same shape), and `--budgets 1.003` is the paper's
headline tolerance for that layer (`+0.3%` perplexity). Add `--tensorization
qubit` to match the paper's parameter counts exactly (36 at `chi'=1`, 132 at 2). Point the same command
at any `MODEL_ID`, block, or layer type to compare a different layer. With the
default `k = 2` the disentangling is weaker than the paper's wide-gate circuits
(which is what `M*/N*` at `w > 0` quantifies); pass `--gate-sizes 0` for the
paper's register-wide disentangler.

### Disentangling accuracy vs. number of layers (the paper's Fig. 3)

`disentangle_scaling.py` (module `qllm.disentangle_scaling_cli`) reproduces the
paper's Fig. 3: it sweeps the brickwall depth `L` at a fixed disentangling target
(`chi' = 1`) and records, per gate size, the **disentangling accuracy** `A` (its
Eq. 4, `Tr[T1(W)^T U^T W V] / (||W|| ||T1(W)||)`) and the **mean bond entropy** of
the disentangled operator. For two-qubit gates the accuracy rises -- roughly
logarithmically, saturating -- as `L` grows, the paper's headline observation.

```bash
# The paper's layer (SmolLM2 block 10 v_proj), k=2, L = 1..35 (needs HF)
python disentangle_scaling.py HuggingFaceTB/SmolLM2-135M \
    --block 10 --layer-type v_proj --gate-sizes 2 --max-layers 35

# Offline, on a synthetic (192,576)-shaped matrix
python disentangle_scaling.py --shape 192x576 --gate-sizes 1 2 --max-layers 35
```

It uses the paper's **fixed** bond-1 target (`--disentangle-target fixed`) and the
**qubit** MPO by default, writes `disentangle_scaling.csv`, and plots the Fig. 3
panels (accuracy and entropy vs. `L`, log-`x`). It also records and plots the
**total parameter count `M* = C(chi') + Q(D)`** against `L`: the stored MPO
parameters `C(chi')` (constant, since `chi'` and the tensorization are fixed --
`36` for the paper's qubit MPO at `chi'=1`) plus the circuits' `Q(D)`, which
grows with the brickwall depth. The `classical_params` / `quantum_params` /
`total_params` columns hold the numbers.

The sweep includes **`L = 0`** as an ordinary point -- the left end of every
curve. Zero circuit depth means identity circuits, which collapse the hybrid to
the plain `chi'`-truncated MPO on the *padded* qubit geometry, at `M* = C(chi')`
and `Q(D) = 0` (`36` for the paper's qubit MPO at `chi'=1`). It is the floor the
circuits lift off from; pass `--no-d0` to drop it. Because `L=0` cannot sit on a
log axis, the plot's x-axis switches to symlog (linear through `0`, log beyond)
when the point is present.

Every panel also draws one **tensor-network-only reference line** (`Q(D)=0`, so
it does not depend on `L`), written to the CSV as a marked row (`kind` column):

* **`TN only, no padding`** -- plain CompactifAI: the balanced 2-site MPO on the
  layer's *raw* dimensions, with no power-of-two padding (dash-dot line). It
  usually retains more of the layer than the padded `L=0` truncation but costs
  more parameters, so it is the classical baseline the padded hybrid undercuts.

On the **model path** it adds a further **perplexity-vs-`L`** curve: at
each `L` the target layer is swapped for its
disentangled `chi'`-truncated reconstruction (`U T_{chi'}(MPO_new) V^T`), the full
model is re-scored, and `perplexity / baseline` is plotted against `L` -- so you
see the accuracy gain turn into a shrinking perplexity cost as layers are added.
By default this perplexity is the **cold** cost of the compression: the layer is
swapped in and the model is *not* fine-tuned, so it reflects the raw truncation
(the only optimization is the per-layer disentangling that fits the circuits +
MPO to the original weight). The perplexity probe is controlled by `--dataset` /
`--eval-tokens` / `--max-length` / `--stride` / `--ppl-batch-size`, and skipped
with `--no-perplexity` (unavailable with `--shape`, which has no model).

**Healing** (`--heal core full`, model path) adds recovered-perplexity curves: at
each `L` the swapped-in layer is briefly retrained against the LM loss on a
calibration split (every other layer dense, parameter count unchanged), then
re-scored. Two granularities, compared side by side:

* **`core`** retrains only the `chi'` MPO bond -- `C(chi')` trainable parameters,
  the direct analogue of CompactifAI healing;
* **`full`** retrains the bond **and** the `U`/`V` gates (the gates stay
  orthogonal, optimized on the Lie algebra `g = g0 · expm(skew(θ))`), so it
  trains all `M* = C(chi') + Q(D)` parameters. This gives a starved small-`chi'`
  bond the circuits' extra task-trainable degrees of freedom, which is where
  perplexity actually starts to fall with depth.

Both start exactly at the cold reconstruction (step 0 reproduces the swapped-in
weight) and are recorded as `ppl_core` / `ppl_ratio_core` and `ppl_full` /
`ppl_ratio_full`, drawn on the perplexity panel (cold solid, core dashed, full
dotted). Healing is controlled by `--heal-steps` / `--heal-lr` / `--heal-tokens`
/ `--heal-batch-size` / `--heal-window` / `--heal-split` (default `train`, disjoint
from the eval split). It reuses the same `heal_hybrid` machinery as
`compactifai_heal.py`; at `L=0` (no circuits) `full` coincides with `core`. Note
the healed perplexity depends on the reachable `chi'`: at a `chi'` where the cold
swap already destroys the layer, even full healing has limited room -- raise
`--target-chi` in tandem.

The CSV **checkpoints**: a re-run reuses every `(gate size, L)` point already in
it (and, on the model path, the cached baseline perplexity), so an interrupted
sweep resumes and adding more `L` values only pays for the new points -- the same
resume behaviour as the CompactifAI and hybrid sweeps. A run whose configuration
(layer, MPO geometry, optimizer, target `chi'`, disentangle target) differs from
the CSV is refused; pass `--recompute` to recompute from scratch or `--csv-name`
to write a separate file.

The layer is either a real model weight (`--block` / `--layer-type`, needs the
Hugging Face files) or a synthetic matrix (`--shape`, offline and reproducible).
On the synthetic matrix the **accuracy-vs-`L` trend reproduces faithfully**; the
*entropy-decreases* half of Fig. 3, the perplexity curve, and the exact
magnitudes need the real trained layer (a random matrix has little to
disentangle, and perplexity needs a model). `--optimizer gradient` runs the same
sweep with the gradient-trained circuits.

## Use as a library

```python
from qllm import BenchmarkConfig, run_benchmark, write_result_csv

result = run_benchmark(BenchmarkConfig(model_id="gpt2", max_eval_tokens=20000))
print(result.perplexity, result.total_memory_mb, result.generation_tokens_per_second)
write_result_csv(result, "results/llms/gpt2/benchmark.csv")

# Per-layer analysis
from qllm import LayerAnalysisConfig, run_layer_analysis
run_layer_analysis(LayerAnalysisConfig(model_id="gpt2", sensitivity_eval_tokens=2048))

# Disentangle one weight matrix into circuits + a residual MPO
from qllm import disentangle, hybrid_weight
res = disentangle(weight, gate_size=4, depth=2, target_chi=1)
print(res.quantum_params, res.retained, res.entropy)
approx, classical_params = hybrid_weight(res, chi=2)

# Solve the budget from a hybrid sweep CSV
from qllm import load_curves, solve_all
from qllm.budget_frontier import read_rows
curves = load_curves(read_rows("hybrid_per_layer.csv"))
for s in solve_all(curves, budget=1.01, q_weight=1.0):
    print(s.layer_type, s.depth, s.ratio, s.m_star_chi, s.m_star_circuit_depth)
```

### Project layout

```
qllm/
  benchmark.py         # core: config, loading, perplexity, timing, memory, CSV
  cli.py               # benchmark CLI (single or multi-model sweeps, --resume)
  layer_analysis.py    # per-layer entropy/spectrum/sensitivity + depth reporting
  analyze_cli.py       # layer-analysis CLI
  compactifai.py       # MPO decomposition: index factorization, truncated SVDs
  compactifai_sweep.py # bond-dimension sweep + checkpointed CSV output
  compactifai_heal.py  # healing: brief retraining of a compressed layer
  compactifai_cli.py   # CompactifAI CLI
  compressibility_metrics.py  # which layer metrics predict compressibility
  disentangler.py      # PQC disentanglers: brickwall circuits + environment sweeps
  hybrid_sweep.py      # PPL(chi) and PPL(chi', D, k) on the same probe tokens
  hybrid_planner.py    # offline verdict on (k, D) from parameter counts alone
  qubit_mpo.py         # per-qubit MPO tensorization (paper geometry), both methods
  hybrid_cli.py        # hybrid-sweep CLI (+ --solve: sweep and report M*/N*)
  disentangle_scaling_cli.py  # accuracy/entropy vs. #layers (paper Fig. 3)
  budget_frontier.py   # N*, M*, M*/N*, break-even price, Pareto front
  budget_plots.py      # figures for the budget comparison
  budget_cli.py        # budget-analysis CLI
  __main__.py          # enables `python -m qllm`
benchmark_llm.py       # thin entry point for the benchmark CLI
analyze_layers.py      # thin entry point for the layer-analysis CLI
compactify.py          # thin entry point for the CompactifAI sweep
hybridize.py           # thin entry point for the hybrid PQC+TN sweep
disentangle_scaling.py # thin entry point for the Fig. 3 accuracy-vs-layers sweep
analyze_budget.py      # thin entry point for the budget comparison
analyze_compressibility.py  # thin entry point for the metrics correlation
tests/                 # offline smoke tests (real torch, stubbed network I/O)
models.txt             # example model list for --models-file
```
