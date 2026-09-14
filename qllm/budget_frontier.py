"""Constrained memory optimization: is the PQC worth its parameters?

Given the two perplexity surfaces measured by :mod:`qllm.hybrid_sweep` -- the
classical tensor network ``PPL_classical(chi)`` and the hybrid circuit+network
``PPL_hybrid(chi', D)`` -- and a perplexity budget ``B``, this module solves, per
layer,

    N* = min_chi      C(chi)                 s.t.  PPL_classical(chi) <= B
    M* = min_{chi',D} C(chi') + w . Q(D)     s.t.  PPL_hybrid(chi', D) <= B

and reports where ``M*/N*`` is small -- i.e. where the quantum circuit buys real
compression -- together with the winning ``(chi', D)``, the layer types and
depths that favour it, and the Pareto front of both methods.

Both minimizations run over the *measured grid*, not over a fitted curve: the
feasible set is exactly the points whose measured ``ppl_ratio`` meets the budget.
That keeps the answer robust to a non-monotone probe (a layer whose perplexity
wobbles between neighbouring chi does not get a spuriously small N*).

Pricing the quantum parameters
------------------------------
``w`` (``q_weight``) is the price of one quantum parameter in units of one
classical parameter. ``w = 1`` treats an angle stored for a QPU exactly like a
weight stored in RAM -- the conservative reading, under which the paper's own
wide-gate circuits carry *more* parameters than the layer they replace. ``w = 0``
is the other extreme, the paper's framing that the circuits are the quantum
resource itself and cost no classical memory. Rather than pick one, every layer
also gets a **break-even price** ``w*``: the largest ``w`` at which the hybrid
still beats the classical layer. ``w*`` is the scale-free answer to "how cheap
must a quantum parameter be for this layer to be worth hybridizing".
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

INF = float("inf")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _i(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return -1


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
@dataclass
class ClassicalPoint:
    chi: int
    c_params: int
    ppl_ratio: float


@dataclass
class HybridPoint:
    chi: int
    circuit_depth: int
    gate_size: int
    c_params: int
    q_params: int
    ppl_ratio: float

    def cost(self, q_weight: float) -> float:
        return self.c_params + q_weight * self.q_params


@dataclass
class LayerCurves:
    param_name: str
    layer_type: str
    depth: int
    dense_params: int
    classical: list[ClassicalPoint] = field(default_factory=list)
    hybrid: list[HybridPoint] = field(default_factory=list)


def load_curves(rows: list[dict]) -> dict[str, LayerCurves]:
    """Group the hybrid-sweep CSV into one classical and one hybrid curve per layer."""
    curves: dict[str, LayerCurves] = {}
    for r in rows:
        name = r.get("param_name", "")
        if not name:
            continue
        cur = curves.get(name)
        if cur is None:
            cur = curves[name] = LayerCurves(
                param_name=name, layer_type=r.get("layer_type", ""),
                depth=_i(r.get("depth")), dense_params=_i(r.get("params_original")),
            )
        ratio = _f(r.get("ppl_ratio"))
        if not math.isfinite(ratio):
            continue
        if r.get("method") == "classical":
            cur.classical.append(ClassicalPoint(
                chi=_i(r.get("chi")), c_params=_i(r.get("classical_params")),
                ppl_ratio=ratio))
        elif r.get("method") == "hybrid":
            cur.hybrid.append(HybridPoint(
                chi=_i(r.get("chi")), circuit_depth=_i(r.get("circuit_depth")),
                gate_size=_i(r.get("gate_size")),
                c_params=_i(r.get("classical_params")),
                q_params=_i(r.get("quantum_params")), ppl_ratio=ratio))
    for cur in curves.values():
        cur.classical.sort(key=lambda p: p.chi)
        cur.hybrid.sort(key=lambda p: (p.gate_size, p.circuit_depth, p.chi))
    return curves


# --------------------------------------------------------------------------- #
# The constrained minimizations
# --------------------------------------------------------------------------- #
@dataclass
class LayerSolution:
    param_name: str
    layer_type: str
    depth: int
    budget: float
    q_weight: float
    dense_params: int
    # N* -- classical
    n_star: float
    n_star_chi: int
    # M* -- hybrid
    m_star: float
    m_star_chi: int
    m_star_circuit_depth: int
    m_star_gate_size: int
    m_star_c: int
    m_star_q: int
    ratio: float                 # M*/N*
    status: str                  # both | hybrid_only | classical_only | neither
    breakeven_weight: float      # largest w at which the hybrid still wins
    n_star_fraction: float       # N*/dense  -- classical compression achieved
    m_star_fraction: float       # M*/dense  -- hybrid compression achieved


def solve_layer(curve: LayerCurves, budget: float, q_weight: float = 1.0) -> LayerSolution:
    """Minimize classical / hybrid memory at fixed perplexity budget for one layer.

    ``budget`` is a perplexity *ratio* to the dense baseline (1.01 = "at most 1%
    worse"), which is how the per-layer probe is calibrated: every point is
    scored on the same tokens, so the ratio cancels corpus-sampling error.
    """
    feas_c = [p for p in curve.classical if p.ppl_ratio <= budget]
    feas_h = [p for p in curve.hybrid if p.ppl_ratio <= budget]

    best_c = min(feas_c, key=lambda p: p.c_params) if feas_c else None
    # Ties go to the cheaper circuit: same memory, fewer quantum parameters and
    # then a shallower brickwall, which is the one that is easier to run.
    best_h = min(feas_h, key=lambda p: (p.cost(q_weight), p.q_params,
                                        p.circuit_depth, p.gate_size)) if feas_h else None

    n_star = float(best_c.c_params) if best_c else INF
    m_star = best_h.cost(q_weight) if best_h else INF

    if best_c and best_h:
        status = "both"
    elif best_h:
        status = "hybrid_only"
    elif best_c:
        status = "classical_only"
    else:
        status = "neither"

    ratio = (m_star / n_star) if (best_c and best_h) else float("nan")

    # Break-even price of a quantum parameter: the hybrid point (chi', D) beats
    # N* while  C + w Q < N*, i.e. while  w < (N* - C)/Q. Q = 0 (depth 0) wins at
    # any price when it is already cheaper, and never otherwise.
    breakeven = 0.0
    if best_c:
        for p in feas_h:
            if p.c_params >= n_star:
                continue
            breakeven = INF if p.q_params <= 0 else max(
                breakeven, (n_star - p.c_params) / p.q_params)
            if breakeven == INF:
                break

    dense = curve.dense_params or 1
    return LayerSolution(
        param_name=curve.param_name, layer_type=curve.layer_type, depth=curve.depth,
        budget=budget, q_weight=q_weight, dense_params=curve.dense_params,
        n_star=n_star, n_star_chi=best_c.chi if best_c else -1,
        m_star=m_star, m_star_chi=best_h.chi if best_h else -1,
        m_star_circuit_depth=best_h.circuit_depth if best_h else -1,
        m_star_gate_size=best_h.gate_size if best_h else -1,
        m_star_c=best_h.c_params if best_h else -1,
        m_star_q=best_h.q_params if best_h else -1,
        ratio=ratio, status=status, breakeven_weight=breakeven,
        n_star_fraction=n_star / dense if math.isfinite(n_star) else INF,
        m_star_fraction=m_star / dense if math.isfinite(m_star) else INF,
    )


def solve_all(curves: dict[str, LayerCurves], budget: float,
              q_weight: float = 1.0) -> list[LayerSolution]:
    return [solve_layer(c, budget, q_weight)
            for c in sorted(curves.values(), key=lambda c: (c.layer_type, c.depth))]


# --------------------------------------------------------------------------- #
# Pareto front
# --------------------------------------------------------------------------- #
@dataclass
class ParetoPoint:
    cost: float
    ppl_ratio: float
    method: str
    chi: int
    circuit_depth: int
    gate_size: int


def pareto_front(curve: LayerCurves, q_weight: float = 1.0) -> list[ParetoPoint]:
    """Non-dominated (memory, perplexity) points across *both* methods.

    A point is dominated when another one is no more expensive and no worse in
    perplexity. Sorting by cost and keeping every strict improvement in
    perplexity yields the staircase the budget optimization walks along.
    """
    points = [ParetoPoint(float(p.c_params), p.ppl_ratio, "classical", p.chi, -1, 0)
              for p in curve.classical]
    points += [ParetoPoint(p.cost(q_weight), p.ppl_ratio, "hybrid", p.chi,
                           p.circuit_depth, p.gate_size) for p in curve.hybrid]
    points.sort(key=lambda p: (p.cost, p.ppl_ratio))
    front, best = [], INF
    for p in points:
        if p.ppl_ratio < best - 1e-12:
            front.append(p)
            best = p.ppl_ratio
    return front


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _median(values: list[float]) -> float:
    """Median over the defined values. Infinities are kept: ``w* = inf`` ("the
    hybrid wins at any price") is a real answer, not a missing one."""
    vals = sorted(v for v in values if v == v)
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def group_summary(solutions: list[LayerSolution], key: str) -> list[dict]:
    """Median ratio / break-even price and win count, grouped by ``key``."""
    groups: dict[object, list[LayerSolution]] = defaultdict(list)
    for s in solutions:
        groups[getattr(s, key)].append(s)
    out = []
    for gk, items in groups.items():
        ratios = [s.ratio for s in items]
        comparable = [s for s in items if s.status == "both"]
        wins = [s for s in comparable if s.ratio < 1.0]
        out.append(dict(
            group=gk, n=len(items), n_comparable=len(comparable), n_wins=len(wins),
            median_ratio=_median(ratios),
            min_ratio=min((r for r in ratios if r == r and math.isfinite(r)),
                          default=float("nan")),
            median_breakeven=_median([s.breakeven_weight for s in items]),
            median_n_fraction=_median([s.n_star_fraction for s in items]),
            median_m_fraction=_median([s.m_star_fraction for s in items]),
        ))
    out.sort(key=lambda d: (d["median_ratio"] if d["median_ratio"] == d["median_ratio"]
                            else INF))
    return out


def model_rollup(solutions: list[LayerSolution]) -> dict:
    """Sum N* and M* over the layers where both methods meet the budget."""
    both = [s for s in solutions if s.status == "both"]
    n_total = sum(s.n_star for s in both)
    m_total = sum(s.m_star for s in both)
    dense = sum(s.dense_params for s in both)
    return dict(
        n_layers=len(both), dense_params=dense, n_star_total=n_total,
        m_star_total=m_total,
        ratio=(m_total / n_total) if n_total else float("nan"),
        classical_compression=(n_total / dense) if dense else float("nan"),
        hybrid_compression=(m_total / dense) if dense else float("nan"),
    )


def budget_scan(curves: dict[str, LayerCurves], budgets: list[float],
                q_weight: float = 1.0) -> dict[float, list[LayerSolution]]:
    return {b: solve_all(curves, b, q_weight) for b in budgets}


def weight_scan(curves: dict[str, LayerCurves], budget: float,
                weights: list[float]) -> dict[float, list[LayerSolution]]:
    return {w: solve_all(curves, budget, w) for w in weights}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float, width: int = 10) -> str:
    """Ratios and prices: 3 decimals, with 'n/a' and 'inf' spelled out."""
    if value != value:
        return f"{'n/a':>{width}}"
    if math.isinf(value):
        return f"{'inf':>{width}}"
    if abs(value) >= 1000:
        return f"{value:>{width},.0f}"
    return f"{value:>{width}.3f}"


def _fmt_count(value: float, width: int = 10) -> str:
    """Parameter counts: thousands separators, no false precision."""
    if value != value:
        return f"{'n/a':>{width}}"
    if math.isinf(value):
        return f"{'none':>{width}}"
    return f"{value:>{width},.0f}"


def print_layer_table(solutions: list[LayerSolution], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print("=" * 118)
    head = ("layer", "blk", "N* (C)", "chi", "M* (C+wQ)", "chi'", "D", "k",
            "C(chi')", "Q(D)", "M*/N*", "w*")
    print("{:22} {:>4} {:>11} {:>5} {:>12} {:>5} {:>3} {:>3} {:>10} {:>10} "
          "{:>8} {:>9}".format(*head))
    print("-" * 118)
    for s in sorted(solutions, key=lambda x: (x.ratio if x.ratio == x.ratio else INF)):
        print(f"{s.layer_type[:22]:22} {s.depth:>4} {_fmt_count(s.n_star, 11)} "
              f"{s.n_star_chi:>5} {_fmt_count(s.m_star, 12)} {s.m_star_chi:>5} "
              f"{s.m_star_circuit_depth:>3} {s.m_star_gate_size:>3} "
              f"{s.m_star_c if s.m_star_c >= 0 else 0:>10,} "
              f"{s.m_star_q if s.m_star_q >= 0 else 0:>10,} "
              f"{_fmt(s.ratio, 8)} {_fmt(s.breakeven_weight, 9)}"
              + ("" if s.status == "both" else f"  [{s.status}]"))


def print_group_table(rows: list[dict], label: str) -> None:
    print(f"\nBy {label}  (sorted by median M*/N*, lower = more value in the PQC)")
    print("-" * 100)
    print(f"{label:22} {'n':>4} {'wins':>6} {'median M*/N*':>13} "
          f"{'best M*/N*':>11} {'median w*':>11} {'C/dense':>9} {'M/dense':>9}")
    for r in rows:
        print(f"{str(r['group'])[:22]:22} {r['n']:>4} "
              f"{r['n_wins']:>3}/{r['n_comparable']:<2} "
              f"{_fmt(r['median_ratio'], 13)} {_fmt(r['min_ratio'], 11)} "
              f"{_fmt(r['median_breakeven'], 11)} "
              f"{_fmt(r['median_n_fraction'], 9)} {_fmt(r['median_m_fraction'], 9)}")


def print_report(curves: dict[str, LayerCurves], budget: float,
                 q_weight: float, top: int = 10) -> list[LayerSolution]:
    solutions = solve_all(curves, budget, q_weight)
    print("\n" + "=" * 118)
    print(f"MEMORY AT A PERPLEXITY BUDGET   B = {budget:.4f} x dense baseline   "
          f"|   quantum parameter price w = {q_weight:g}")
    print("=" * 118)
    print("N* = min C(chi) s.t. PPL_classical(chi) <= B        "
          "(pure tensor network)")
    print("M* = min C(chi') + w Q(D) s.t. PPL_hybrid(chi',D) <= B  "
          "(disentangling circuits + tensor network)")
    print_layer_table(solutions)

    infeasible = [s for s in solutions if s.status != "both"]
    if infeasible:
        print(f"\n{len(infeasible)} layer(s) not comparable at this budget:")
        for s in infeasible:
            print(f"    {s.layer_type} d{s.depth}: {s.status}")

    print_group_table(group_summary(solutions, "layer_type"), "layer_type")
    print_group_table(group_summary(solutions, "depth"), "depth")

    wins = [s for s in solutions if s.status == "both" and s.ratio < 1.0]
    print(f"\nThe PQC pays at w = {q_weight:g} for {len(wins)} of "
          f"{len([s for s in solutions if s.status == 'both'])} comparable layers.")
    if wins:
        print(f"Lowest M*/N* (most value in the circuits), top {min(top, len(wins))}:")
        for s in sorted(wins, key=lambda x: x.ratio)[:top]:
            print(f"    {s.layer_type:16} block {s.depth:<3} M*/N* = {s.ratio:.3f} "
                  f"with chi' = {s.m_star_chi}, D = {s.m_star_circuit_depth}, "
                  f"k = {s.m_star_gate_size} "
                  f"(C = {s.m_star_c:,} + w.Q = {s.m_star_q:,})")
    else:
        best = [s for s in solutions if s.status == "both" and
                math.isfinite(s.breakeven_weight)]
        if best:
            top_be = sorted(best, key=lambda x: -x.breakeven_weight)[:top]
            print("No layer wins at this price. Closest, by break-even price w* "
                  "(the PQC pays below it):")
            for s in top_be:
                print(f"    {s.layer_type:16} block {s.depth:<3} "
                      f"w* = {s.breakeven_weight:.4f} "
                      f"(needs a quantum parameter to cost < "
                      f"{s.breakeven_weight:.4f} classical ones)")

    roll = model_rollup(solutions)
    if roll["n_layers"]:
        print(f"\nSummed over the {roll['n_layers']} comparable layers: "
              f"N* = {roll['n_star_total']:,.0f}, M* = {roll['m_star_total']:,.0f} "
              f"-> M*/N* = {roll['ratio']:.3f}")
        print(f"    as a fraction of the dense weights "
              f"({roll['dense_params']:,} params): "
              f"classical {roll['classical_compression']:.4f}, "
              f"hybrid {roll['hybrid_compression']:.4f}")
    return solutions


def print_pareto(curves: dict[str, LayerCurves], q_weight: float = 1.0,
                 limit: int = 6) -> None:
    """The non-dominated staircase, per layer, with the owner of each step."""
    print("\n" + "=" * 90)
    print(f"PARETO FRONT (memory vs. perplexity), quantum parameter price w = {q_weight:g}")
    print("=" * 90)
    for name in sorted(curves)[:limit]:
        curve = curves[name]
        front = pareto_front(curve, q_weight)
        owned = sum(1 for p in front if p.method == "hybrid")
        print(f"\n{curve.layer_type} block {curve.depth}  "
              f"({curve.dense_params:,} dense params) -- {len(front)} steps, "
              f"{owned} owned by the hybrid")
        print(f"    {'cost':>12} {'ppl/base':>9}  method     config")
        for p in front:
            cfg = (f"chi={p.chi}" if p.method == "classical"
                   else f"chi'={p.chi}, D={p.circuit_depth}, k={p.gate_size}")
            print(f"    {p.cost:>12,.0f} {p.ppl_ratio:>9.4f}  {p.method:10} {cfg}")
    if len(curves) > limit:
        print(f"\n({len(curves) - limit} more layers not shown; "
              f"use --pareto-limit to widen.)")
