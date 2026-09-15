#!/usr/bin/env python3
"""Entry point: reproduce arXiv:2410.17397v2 Fig. 3 (accuracy/entropy vs. layers).

Sweeps the brickwall depth L at a fixed disentangling target (chi'=1) and records
the disentangling accuracy (Eq. 4) and mean bond entropy per (gate size, L), for a
model layer or a synthetic matrix. See --help.
"""
from qllm.disentangle_scaling_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
