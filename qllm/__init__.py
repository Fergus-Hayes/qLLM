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
    window_nlls,
    write_result_csv,
)
from .layer_analysis import (
    LayerAnalysisConfig,
    LayerResult,
    run_layer_analysis,
    spectral_metrics,
)
from .compactifai import (
    build_plan,
    compress_weight,
    mpo_param_count,
)
from .qubit_mpo import (
    build_qubit_plan,
    compress_weight_qubit,
    make_plan,
    plan_compress,
    qubit_mpo_dims,
)
from .compactifai_sweep import CompactifaiConfig, run_sweep
from .compressibility_metrics import correlate as correlate_compressibility
from .disentangler import (
    DisentangleResult,
    circuit_ops,
    circuit_unitary,
    disentangle,
    disentangle_loss,
    hybrid_weight,
    quantum_param_count,
)
from .hybrid_sweep import (
    DEFAULT_GATE_SIZE,
    HybridConfig,
    MAX_GATE_SIZE,
    run_hybrid_sweep,
    validate_gate_sizes,
)
from .hybrid_heal import HybridAdapter, heal_hybrid
from .budget_frontier import (
    LayerSolution,
    load_curves,
    pareto_front,
    solve_all,
    solve_layer,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "window_nlls",
    "write_result_csv",
    "LayerAnalysisConfig",
    "LayerResult",
    "run_layer_analysis",
    "spectral_metrics",
    "CompactifaiConfig",
    "run_sweep",
    "build_plan",
    "compress_weight",
    "mpo_param_count",
    "build_qubit_plan",
    "compress_weight_qubit",
    "make_plan",
    "plan_compress",
    "qubit_mpo_dims",
    "correlate_compressibility",
    "DisentangleResult",
    "circuit_ops",
    "circuit_unitary",
    "disentangle",
    "disentangle_loss",
    "hybrid_weight",
    "quantum_param_count",
    "HybridConfig",
    "DEFAULT_GATE_SIZE",
    "MAX_GATE_SIZE",
    "run_hybrid_sweep",
    "validate_gate_sizes",
    "HybridAdapter",
    "heal_hybrid",
    "LayerSolution",
    "load_curves",
    "pareto_front",
    "solve_all",
    "solve_layer",
]

__version__ = "0.1.0"
