"""CLI for the budget-constrained PQC-vs-TN memory comparison.

Reads the CSV written by ``hybridize.py`` and, for each perplexity budget,
solves per layer

    N* = min_chi      C(chi)                 s.t. PPL_classical(chi) <= B
    M* = min_{chi',D} C(chi') + w Q(D)       s.t. PPL_hybrid(chi', D) <= B

printing where ``M*/N*`` is small, which ``(chi', D)`` wins there, and the
Pareto front, and writing the figures next to the input CSV.

Examples
--------
    # Default: three budgets, quantum parameters priced like classical ones
    python analyze_budget.py results/llms/SmolLM2-135M/compactifai/hybrid_per_layer.csv

    # The paper's framing: the circuits run on a QPU and cost no RAM
    python analyze_budget.py HYBRID.csv --q-weight 0

    # Tight budget, attention only, wider Pareto listing
    python analyze_budget.py HYBRID.csv --budgets 1.002 --attention-only --pareto-limit 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .budget_frontier import (
    _median,
    filter_rows_by_tensorization,
    group_summary,
    tensorizations_in,
    load_curves,
    print_pareto,
    print_report,
    read_rows,
    solve_all,
)
from .budget_plots import make_all

DEFAULT_BUDGETS = [1.001, 1.01, 1.05]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimize layer memory at a perplexity budget, for the pure "
                    "tensor network and for the disentangler+tensor network, "
                    "and compare them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("hybrid_csv",
                        help="hybrid_per_layer.csv from hybridize.py.")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS,
                        help="Perplexity budgets as a ratio to the dense "
                             "baseline (1.01 = at most 1%% worse).")
    parser.add_argument("--q-weight", type=float, default=1.0,
                        help="Price of one quantum parameter in units of one "
                             "classical parameter. 0 = the circuits are free "
                             "classical memory (they live on the QPU); 1 = an "
                             "angle costs exactly what a weight costs.")
    parser.add_argument("--q-weight-scan", type=float, nargs="+", default=None,
                        help="Also report the verdict at these prices.")
    parser.add_argument("--layer-types", nargs="+", default=None,
                        help="Restrict to layer types containing these strings.")
    parser.add_argument("--tensorization", default=None,
                        choices=["balanced", "qubit"],
                        help="Report only rows measured under this MPO geometry "
                             "(required if the CSV mixes geometries).")
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--pareto-limit", type=int, default=6,
                        help="Layers listed in the Pareto-front printout.")
    parser.add_argument("--pareto-panels", type=int, default=9,
                        help="Layers drawn in the Pareto figure.")
    parser.add_argument("--no-pareto", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write figures (default: next to the CSV).")
    return parser.parse_args(argv)


def _filters(args) -> list[str] | None:
    if args.attention_only:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "attn", "attention"]
    if args.mlp_only:
        return ["gate_proj", "up_proj", "down_proj", "mlp", "fc", "ffn"]
    return args.layer_types


def filter_curves(curves: dict, layer_types=None, depths=None) -> dict:
    """Restrict a curve set to layer types (substring) and/or block depths.

    Both filters are general: ``layer_types`` matches any substring against
    ``"<layer_type> <param_name>"`` (so ``"v_proj"`` or ``"attn"`` both work),
    and ``depths`` keeps only those decoder blocks. Passing neither returns the
    curves unchanged.
    """
    out = curves
    if layer_types:
        out = {n: c for n, c in out.items()
               if any(k in f"{c.layer_type} {n}" for k in layer_types)}
    if depths is not None:
        keep = set(depths)
        out = {n: c for n, c in out.items() if c.depth in keep}
    return out


def report_curves(curves: dict, budgets, q_weights, *, pareto: bool = True,
                  pareto_limit: int = 6) -> list:
    """Print the N*/M*/M*-over-N* report for every (budget, quantum-price) pair.

    This is the shared reporting core used both by ``analyze_budget.py`` (from a
    CSV) and by ``hybridize.py --solve`` (right after a sweep). It is agnostic to
    which layers or model produced ``curves``. Returns the solutions from the
    last (q_weight, budget) pair, for downstream plotting.
    """
    solutions = None
    for q_weight in q_weights:
        for budget in budgets:
            solutions = print_report(curves, budget, q_weight)
        if pareto:
            print_pareto(curves, q_weight, pareto_limit)
    return solutions


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.hybrid_csv)
    rows = read_rows(csv_path)

    # C(chi) is not comparable across MPO geometries, so a mixed CSV must be
    # disambiguated rather than reported as one table.
    present = tensorizations_in(rows)
    if args.tensorization:
        rows = filter_rows_by_tensorization(rows, args.tensorization)
        if not rows:
            print(f"No '{args.tensorization}' rows in {csv_path}; "
                  f"it has: {', '.join(present)}.")
            return 1
    elif len(present) > 1:
        print(f"error: {csv_path} mixes tensorizations ({', '.join(present)}); "
              f"pass --tensorization to pick one -- C(chi) is not comparable "
              f"across geometries.")
        return 1

    curves = filter_curves(load_curves(rows), layer_types=_filters(args))
    if not curves:
        print("No layers left after filtering.")
        return 1

    n_hybrid = sum(len(c.hybrid) for c in curves.values())
    n_classical = sum(len(c.classical) for c in curves.values())
    print(f"Loaded {len(curves)} layers from {csv_path}: "
          f"{n_classical} classical points, {n_hybrid} hybrid points.")
    if not n_hybrid or not n_classical:
        print("Both surfaces are needed for the comparison; re-run hybridize.py.")
        return 1

    solutions = report_curves(curves, args.budgets, [args.q_weight],
                              pareto=False)

    if args.q_weight_scan:
        print("\n" + "=" * 78)
        print("SENSITIVITY TO THE PRICE OF A QUANTUM PARAMETER")
        print("=" * 78)
        budget = args.budgets[-1]
        print(f"(at B = {budget:.4f} x baseline)")
        print(f"{'w':>10} {'layers won':>12} {'median M*/N*':>14} {'best M*/N*':>12}")
        for w in sorted(args.q_weight_scan):
            sols = solve_all(curves, budget, w)
            rows = group_summary(sols, "layer_type")
            wins = sum(r["n_wins"] for r in rows)
            comparable = sum(r["n_comparable"] for r in rows)
            ratios = [s.ratio for s in sols if s.ratio == s.ratio]
            print(f"{w:>10g} {f'{wins}/{comparable}':>12} "
                  f"{_median(ratios):>14.4f} "
                  f"{min(ratios, default=float('nan')):>12.4f}")

    if not args.no_pareto:
        print_pareto(curves, args.q_weight, args.pareto_limit)

    if not args.no_plots and solutions is not None:
        out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
        print(f"\nWriting figures to {out_dir} ...")
        scan = list(args.budgets)
        if len(scan) < 2:
            scan = sorted(set(scan) | set(DEFAULT_BUDGETS))
        make_all(curves, solutions, out_dir, args.budgets[-1], args.q_weight,
                 budgets=scan, max_panels=args.pareto_panels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
