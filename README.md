# qLLM

Utilities for benchmarking (quantized) language models.

## SmolLM2-135M perplexity & inference-time benchmark

`benchmark_smollm2.py` fetches [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
from the Hugging Face Hub and reports:

- **Perplexity** on a text corpus (WikiText-2 by default), computed with a
  sliding-window so every target token is scored with maximal left-context and
  counted exactly once.
- **Inference time** — prefill latency (single forward pass over the prompt)
  and greedy-decoding throughput in tokens/second.
- **Memory footprint** — parameter, buffer, and total memory needed to hold the
  model, measured from the actual tensors so it reflects the loaded dtype
  (e.g. `float16` halves the footprint versus `float32`).

Every run is appended (with a header on first creation) to a CSV at
`results/llms/<model-name>/benchmark.csv` — e.g.
`results/llms/SmolLM2-135M/benchmark.csv` — so baseline and quantized runs
accumulate in one comparable table. Override with `--csv-path`, change the base
directory with `--results-dir`, or disable with `--no-csv`.

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# CPU, full WikiText-2 test set
python benchmark_smollm2.py

# Quick run on a subset of tokens
python benchmark_smollm2.py --max-eval-tokens 20000 --gen-tokens 128

# GPU with half precision
python benchmark_smollm2.py --device cuda --dtype float16
```

Run `python benchmark_smollm2.py --help` for the full list of options
(dataset, window size, stride, prompt, warm-up runs, and flags to skip either
the perplexity or the timing stage).

### Example output

```
============================================================
Model:               HuggingFaceTB/SmolLM2-135M
Device / dtype:      cpu / float32
Parameters:          134,515,008 (134.5M)
Parameter memory:    513.13 MB
Buffer memory:       0.00 MB
Total model memory:  513.13 MB (0.501 GB)
Perplexity:          14.8021 (304985 tokens on Salesforce/wikitext/wikitext-2-raw-v1:test)
Perplexity eval:     2514.85 s
Prefill latency:     57.90 ms (6 prompt tokens)
Generation speed:    16.18 tokens/s (64 new tokens)
============================================================

Saved results to results/llms/SmolLM2-135M/benchmark.csv
```

> Absolute numbers depend on hardware, dtype, and evaluation settings; the
> perplexity figure is deterministic for a given window/stride/dataset.
