#!/usr/bin/env python
"""Thin entry point for the CompactifAI MPO compression sweep.

Compresses the Self-Attention / MLP weight matrices of each decoder block into
Matrix Product Operators (CompactifAI, Tomut et al.) and measures the model's
perplexity as a function of the MPO bond dimension.

    python compactify.py                          # default: SmolLM2-135M
    python compactify.py --chi 2 4 8 16 32 64
    python compactify.py --svd-workers 4 --ppl-batch-size 16

Results are checkpointed to results/llms/<model-name>/compactifai_sweep.csv.
See ``python compactify.py --help`` for all options.
"""

from qllm.compactifai_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
