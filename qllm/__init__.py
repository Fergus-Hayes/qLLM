"""qLLM: utilities for benchmarking (quantized) language models.

The public API is intentionally small: build a :class:`BenchmarkConfig`, then
call :func:`run_benchmark` to fetch a model and measure its perplexity,
inference time, and memory footprint. Results are returned as a
:class:`BenchmarkResult` and can be appended to CSV with :func:`write_result_csv`.
"""

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    write_result_csv,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "write_result_csv",
]

__version__ = "0.1.0"
