#!/usr/bin/env python3
"""Entry point: M*/N* for the single layer of arXiv:2410.17397v2, with k<=2 gates.

Targets the paper's layer -- the (576, 192) self-attention projection of block 10
of a compressed SmolLM2 -- restricts the disentangling circuits to two-qubit
gates, measures both the pure tensor-network and the PQC+tensor-network
perplexity surfaces on the same probe tokens, and reports

    N* = min_chi     C(chi)               s.t.  PPL_classical(chi)  <= B
    M* = min_{chi',D} C(chi') + w Q(D)    s.t.  PPL_hybrid(chi', D) <= B

with M*/N* and the break-even quantum-parameter price w*.

    python paper_layer.py --help
"""
from qllm.paper_layer_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
