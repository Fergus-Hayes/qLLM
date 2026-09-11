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
Perplexity:          16.1234 (270000 tokens on Salesforce/wikitext/wikitext-2-raw-v1:test)
Perplexity eval:     42.10 s
Prefill latency:     35.42 ms (7 prompt tokens)
Generation speed:    28.71 tokens/s (64 new tokens)
============================================================
```

> Absolute numbers depend on hardware, dtype, and evaluation settings; the
> perplexity figure is deterministic for a given window/stride/dataset.
