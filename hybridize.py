#!/usr/bin/env python3
"""Entry point for the hybrid PQC+TN vs. pure TN per-layer sweep.

Measures, on the same probe tokens and for each profiled layer:

* ``PPL_classical(chi)``  -- the CompactifAI tensor-network layer, and
* ``PPL_hybrid(chi', D)`` -- the same layer flanked by disentangling quantum
  circuits of brickwall depth ``D``.

Feed the resulting CSV to ``analyze_budget.py`` to solve the constrained memory
problem at a perplexity budget.

    python hybridize.py --help
"""
from qllm.hybrid_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
