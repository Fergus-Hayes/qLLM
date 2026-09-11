#!/usr/bin/env python
"""Thin entry point for the qLLM benchmark CLI.

Equivalent to ``python -m qllm``. Benchmark any Hugging Face causal LM:

    python benchmark_llm.py HuggingFaceTB/SmolLM2-135M
    python benchmark_llm.py --models-file models.txt

See ``python benchmark_llm.py --help`` for all options.
"""

from qllm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
