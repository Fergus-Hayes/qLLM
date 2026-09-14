#!/usr/bin/env python3
"""Entry point for the budget-constrained PQC-vs-TN memory comparison.

Given a perplexity budget B, solves per layer

    N* = min_chi      C(chi)               s.t. PPL_classical(chi) <= B
    M* = min_{chi',D} C(chi') + w Q(D)     s.t. PPL_hybrid(chi', D) <= B

and reports where M*/N* is low -- i.e. where the quantum disentangling circuits
are worth their parameters -- with the winning (chi', D), the layer types and
depths that favour them, and the Pareto front.

    python analyze_budget.py results/llms/<model>/compactifai/hybrid_per_layer.csv
"""
from qllm.budget_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
