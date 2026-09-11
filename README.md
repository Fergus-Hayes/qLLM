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

**Printouts & checkpointing.** Progress is printed as each layer is processed
(spectral line, sensitivity line, running count + ETA). Results are written to
`results/llms/<model-name>/layer_analysis.csv`, which **is** the checkpoint:
each row is flushed as it is computed, and re-running skips any `param_name`
already present — so an interrupted analysis resumes where it stopped. A console
summary of "metric over depth, by layer type" is printed at the end, and (if
`matplotlib` is installed) one `depth_<metric>.png` per metric is saved next to
the CSV. The perplexity-sensitivity probe deliberately uses a small token budget
(it runs once per layer); tune it with `--sensitivity-eval-tokens`,
`--sensitivity-window`, `--epsilon`, and `--sensitivity-samples`.

The benchmark sweep (`benchmark_llm.py`) similarly prints perplexity progress
(running PPL + ETA) and supports `--resume` to skip models already recorded in
their per-model CSV.

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
  benchmark.py       # core: config, loading, perplexity, timing, memory, CSV
  cli.py             # benchmark CLI (single or multi-model sweeps, --resume)
  layer_analysis.py  # per-layer entropy/spectrum/sensitivity + depth reporting
  analyze_cli.py     # layer-analysis CLI
  __main__.py        # enables `python -m qllm`
benchmark_llm.py     # thin entry point for the benchmark CLI
analyze_layers.py    # thin entry point for the layer-analysis CLI
models.txt           # example model list for --models-file
```
