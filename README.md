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

> The paper's subsequent **"healing"** (brief retraining) stage is *not*
> performed here, so these are pre-healing perplexities — the raw cost of
> truncation. The paper notes healing recovers most of the gap.

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

## Use as a library

```python
from qllm import BenchmarkConfig, run_benchmark, write_result_csv

result = run_benchmark(BenchmarkConfig(model_id="gpt2", max_eval_tokens=20000))
print(result.perplexity, result.total_memory_mb, result.generation_tokens_per_second)
write_result_csv(result, "results/llms/gpt2/benchmark.csv")

# Per-layer analysis
from qllm import LayerAnalysisConfig, run_layer_analysis
run_layer_analysis(LayerAnalysisConfig(model_id="gpt2", sensitivity_eval_tokens=2048))
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
  compactifai_cli.py   # CompactifAI CLI
  __main__.py          # enables `python -m qllm`
benchmark_llm.py       # thin entry point for the benchmark CLI
analyze_layers.py      # thin entry point for the layer-analysis CLI
compactify.py          # thin entry point for the CompactifAI sweep
models.txt             # example model list for --models-file
```
