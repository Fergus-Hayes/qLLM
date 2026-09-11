#!/usr/bin/env python
"""Thin entry point for the qLLM per-layer analysis CLI.

Computes, for every 2-D weight matrix in a model, the Shannon and Renyi-2
entropy of its singular-value spectrum, spectral gap, condition number, and
perplexity sensitivity, then summarises how they vary over layer depth.

    python analyze_layers.py                       # default: SmolLM2-135M
    python analyze_layers.py gpt2 --skip-sensitivity

Results are written to results/llms/<model-name>/layer_analysis.csv, which also
serves as a checkpoint (re-running resumes where it left off). See
``python analyze_layers.py --help`` for all options.
"""

from qllm.analyze_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
