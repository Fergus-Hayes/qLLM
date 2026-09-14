"""M*/N* for the single layer of arXiv:2410.17397v2, with two-qubit gates.

The paper *Quantum Large Language Models via Tensor Network Disentanglers*
replaces one weight matrix -- the self-attention projection of dimension
``(576, 192)`` in the tenth block of a compressed SmolLM2 -- and reports the
perplexity as its disentangled MPO is truncated (its Table I). This CLI targets
exactly that layer and, restricted to the paper's hardware-realistic two-qubit
gates (``ku = kv = 2``), answers the memory question the two methods compete
over:

    N* = min_chi     C(chi)               s.t.  PPL_classical(chi)  <= B
    M* = min_{chi',D} C(chi') + w Q(D)    s.t.  PPL_hybrid(chi', D) <= B

Both surfaces are measured on the same probe tokens, so their budgets are
directly comparable; the report prints N*, M*, M*/N* and the break-even price
w* of a quantum parameter, alongside the paper's own reported figures for the
same layer (which use its full 10-qubit / 8-qubit disentangler, not k=2).

    python paper_layer.py                          # defaults to the paper's layer
    python paper_layer.py --device cuda --ppl-batch-size 16
    python paper_layer.py --budgets 1.001 1.003 --q-weights 0 0.01 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .budget_frontier import load_curves, read_rows, solve_layer
from .hybrid_sweep import (
    HybridConfig,
    MAX_GATE_SIZE,
    run_hybrid_sweep,
    validate_gate_sizes,
)

# The paper's target: SmolLM2, block 10, the self-attention (576, 192)
# projection. In a Llama-style SmolLM2 the (576, 192) matrices are k_proj and
# v_proj (grouped-query attention: 3 KV heads x 64 = 192); either is the paper's
# layer, and v_proj is the default here.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_BLOCK = 10
DEFAULT_LAYER_TYPE = "v_proj"
PAPER_SHAPE = (576, 192)

# arXiv:2410.17397v2, Table I -- the disentangled MPO_new truncated to chi, with
# the FULL disentangler (U = one 10-qubit gate, V = one 8-qubit gate, L = 1).
# dPPL is relative to the original model (W: 110,592 params, PPL 35.292).
PAPER_HYBRID_TABLE = [
    # chi', MPO_new params, dPPL (%)
    (1, 36, 0.26),
    (2, 132, 0.15),
    (5, 696, 0.04),
    (10, 2356, 0.06),
    (50, 36948, 0.02),
]
PAPER_DENSE_PARAMS = 110_592
PAPER_EXACT_PARAMS = 401_956
# The paper's classical reference: the *undisentangled* MPO_old at chi=1 costs
# +9.7% perplexity -- more than 30x the disentangled +0.26%.
PAPER_CLASSICAL_CHI1_DPPL = 9.7


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute M*/N* for the single layer of arXiv:2410.17397v2 "
                    "using two-qubit-gate disentanglers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                        help=f"Model id/path (default: {DEFAULT_MODEL}).")
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                        help=f"Decoder block index (default: {DEFAULT_BLOCK}, "
                             f"the paper's tenth layer).")
    parser.add_argument("--layer-type", default=DEFAULT_LAYER_TYPE,
                        help=f"Self-attention projection to target (default: "
                             f"{DEFAULT_LAYER_TYPE}; the (576, 192) matrix).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--threads", type=int, default=None)

    # Tensor network / circuits.
    parser.add_argument("--gate-sizes", type=int, nargs="+", default=[MAX_GATE_SIZE],
                        help=f"Qubits per gate k (default: {MAX_GATE_SIZE}). "
                             f"Restricted to 1..{MAX_GATE_SIZE}.")
    parser.add_argument("--circuit-depths", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8],
                        help="Brickwall depths D to try (default: 0 1 2 4 8; "
                             "D=0 measures the qubit-padding overhead alone).")
    parser.add_argument("--chi", type=int, nargs="+", default=None,
                        help="Explicit bond dimensions (overrides log spacing).")
    parser.add_argument("--chi-min", type=int, default=1)
    parser.add_argument("--chi-max", type=int, default=None)
    parser.add_argument("--num-chi", type=int, default=10)
    parser.add_argument("--mpo-sites", type=int, default=2)
    parser.add_argument("--disentangle-sweeps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=1,
                        help="Keep the best of this many circuit initializations.")
    parser.add_argument("--quantum-param-counting", default="manifold",
                        choices=["manifold", "entries"])

    # Budget / pricing.
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[1.001, 1.003, 1.01],
                        help="Perplexity budgets as a ratio to the dense "
                             "baseline (1.003 = at most 0.3%%, the paper's "
                             "headline figure for this layer).")
    parser.add_argument("--q-weights", type=float, nargs="+", default=[0.0, 1.0],
                        help="Prices of a quantum parameter (0 = the circuits "
                             "are free QPU memory; 1 = an angle costs a weight).")

    # Probe.
    parser.add_argument("--per-layer-eval-tokens", type=int, default=8192)
    parser.add_argument("--per-layer-stride", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")

    parser.add_argument("--results-dir", default="results/llms")
    parser.add_argument("--csv-name", default="paper_layer.csv")
    parser.add_argument("--no-paper-reference", action="store_true",
                        help="Do not print the paper's reported figures.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def _print_paper_reference() -> None:
    print("\n" + "-" * 78)
    print("Reference: arXiv:2410.17397v2, Table I  (same layer, but the paper's")
    print("full disentangler U,V = 10q,8q; here we use only k<=2 gates)")
    print("-" * 78)
    print(f"    dense W: {PAPER_DENSE_PARAMS:,} params   "
          f"exact MPO_new: {PAPER_EXACT_PARAMS:,} params")
    print(f"    {'chi':>5} {'MPO_new params':>15} {'dPPL vs original':>18}")
    for chi, params, dppl in PAPER_HYBRID_TABLE:
        print(f"    {chi:>5} {params:>15,} {'+' + format(dppl, '.2f') + '%':>18}")
    print(f"    classical (undisentangled MPO) at chi=1: "
          f"+{PAPER_CLASSICAL_CHI1_DPPL:.1f}% "
          f"-- {PAPER_CLASSICAL_CHI1_DPPL / 0.26:.0f}x the disentangled cost.")
    print("    In the paper the disentangler concentrates the layer into ~36")
    print("    classical params at <0.3% PPL; with k<=2 gates the disentangling")
    print("    is weaker, which is exactly what the M*/N* below quantifies.")


def _print_curves(curve) -> None:
    print(f"\nLayer: {curve.param_name}")
    print(f"  type {curve.layer_type}, block {curve.depth}, "
          f"{curve.dense_params:,} dense params")
    print("\n  Classical tensor network  --  PPL_classical(chi):")
    print(f"    {'chi':>5} {'C(chi)':>10} {'C/dense':>9} {'PPL/base':>10}")
    for p in curve.classical:
        print(f"    {p.chi:>5} {p.c_params:>10,} "
              f"{p.c_params / max(1, curve.dense_params):>9.4f} {p.ppl_ratio:>10.5f}")
    print("\n  Hybrid (PQC + tensor network)  --  PPL_hybrid(chi', D), k=2:")
    print("    {:>5} {:>3} {:>10} {:>8} {:>10}".format(
        "chi'", "D", "C(chi')", "Q(D)", "PPL/base"))
    for p in curve.hybrid:
        print(f"    {p.chi:>5} {p.circuit_depth:>3} {p.c_params:>10,} "
              f"{p.q_params:>8,} {p.ppl_ratio:>10.5f}")


def _print_solution(curve, budget: float, q_weight: float) -> None:
    s = solve_layer(curve, budget, q_weight)
    print(f"\n  B = {budget:.4f} x baseline,  w = {q_weight:g}:")
    if s.status == "neither":
        print("    neither method meets this budget on the measured grid.")
        return
    n = (f"N* = {s.n_star:,.0f}  (chi = {s.n_star_chi})"
         if s.n_star_chi >= 0 else "N* = -- (classical never meets B)")
    if s.m_star_chi < 0:
        m = "M* = -- (hybrid never meets B)"
    elif s.m_star_circuit_depth <= 0:
        m = (f"M* = {s.m_star:,.0f}  (chi' = {s.m_star_chi}, no circuit; "
             f"C = {s.m_star_c:,}) -- the padded MPO, circuits switched off")
    else:
        m = (f"M* = {s.m_star:,.0f}  (chi' = {s.m_star_chi}, D = "
             f"{s.m_star_circuit_depth}, k = {s.m_star_gate_size}; "
             f"C = {s.m_star_c:,} + w.Q = {s.m_star_q:,})")
    print(f"    {n}")
    print(f"    {m}")
    if s.status == "both":
        verdict = ("the PQC+TN layer is SMALLER" if s.ratio < 1
                   else "the pure TN layer is smaller" if s.ratio > 1
                   else "the two tie")
        if s.breakeven_weight == float("inf"):
            be = "the PQC wins at any price"
        elif s.breakeven_weight <= 0:
            be = ("the circuits never pay here -- the best hybrid is the "
                  "circuit-free (D=0) MPO")
        else:
            be = f"the PQC wins while a quantum parameter costs < {s.breakeven_weight:.4g}"
        print(f"    M*/N* = {s.ratio:.4f}  ->  {verdict};  break-even: {be}")
    else:
        print(f"    ({s.status}: only one method reaches this budget)")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        gate_sizes = validate_gate_sizes(args.gate_sizes)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    config = HybridConfig(
        model_id=args.model, device=args.device, dtype=args.dtype,
        trust_remote_code=args.trust_remote_code, revision=args.revision,
        mpo_sites=args.mpo_sites,
        chi_min=args.chi_min, chi_max=args.chi_max, num_chi=args.num_chi,
        chi_values=args.chi,
        dataset=args.dataset, dataset_config=args.dataset_config,
        split=args.split, text_column=args.text_column,
        max_length=args.max_length,
        ppl_batch_size=args.ppl_batch_size,
        profile_depths=[args.block], layer_types=[args.layer_type],
        per_layer_eval_tokens=args.per_layer_eval_tokens or None,
        per_layer_stride=args.per_layer_stride,
        circuit_depths=sorted(set(args.circuit_depths)),
        gate_sizes=gate_sizes,
        disentangle_sweeps=args.disentangle_sweeps,
        disentangle_restarts=args.restarts,
        quantum_param_counting=args.quantum_param_counting,
        run_classical=True,
        results_dir=args.results_dir, hybrid_csv_name=args.csv_name,
        make_plots=False, force_recompute=args.recompute,
    )

    print(f"Targeting block {args.block}, self-attention '{args.layer_type}' "
          f"of {args.model}")
    print(f"(the paper's layer is the {PAPER_SHAPE[0]}x{PAPER_SHAPE[1]} "
          f"self-attention projection of block {DEFAULT_BLOCK}).")

    path = run_hybrid_sweep(config)

    curves = load_curves(read_rows(path))
    # Keep only the targeted layer (block + type), in case the CSV carries more.
    curves = {n: c for n, c in curves.items()
              if c.depth == args.block and args.layer_type in c.layer_type}
    if not curves:
        print(f"\nNo rows for block {args.block} / '{args.layer_type}' in {path}.")
        return 1
    if len(curves) > 1:
        print(f"\n(Note: {len(curves)} layers matched '{args.layer_type}' at "
              f"block {args.block}; reporting each.)")

    print("\n" + "=" * 78)
    print("M*/N*  FOR THE PAPER'S LAYER  (two-qubit-gate disentanglers)")
    print("=" * 78)
    for curve in curves.values():
        if curve.dense_params != PAPER_DENSE_PARAMS:
            # Not fatal -- the user may target a different model/layer -- but
            # worth flagging when it is not the paper's 110,592-parameter matrix.
            print(f"\n(Note: this layer has {curve.dense_params:,} params, not "
                  f"the paper's {PAPER_DENSE_PARAMS:,}; it is a different matrix.)")
        _print_curves(curve)
        for budget in args.budgets:
            for q_weight in args.q_weights:
                _print_solution(curve, budget, q_weight)

    if not args.no_paper_reference:
        _print_paper_reference()

    if not args.no_plots:
        try:
            from .budget_plots import plot_pareto
            out_dir = Path(path).parent
            plot_pareto(curves, out_dir, args.q_weights[-1],
                        budget=args.budgets[0], max_panels=len(curves))
        except Exception as exc:                    # noqa: BLE001
            print(f"(Skipping plot: {exc})")

    print(f"\nMeasured surfaces cached in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
