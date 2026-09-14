"""Figures for the PQC-vs-TN memory comparison at a perplexity budget.

Four views of the same optimization:

* ``hybrid_vs_classical_ratio.png`` -- ``M*/N*`` per (layer type, block depth),
  the map of *where* the circuits buy compression, annotated with the winning
  ``(chi', D)``.
* ``ratio_vs_budget.png`` -- how that verdict moves as the perplexity budget is
  loosened, one curve per layer type.
* ``pareto_front.png`` -- memory vs. perplexity for every measured point, both
  methods, with the non-dominated staircase drawn on top.
* ``breakeven_weight.png`` -- the price a quantum parameter would have to fall
  below for the hybrid to win, per layer.
"""

from __future__ import annotations

import math
from pathlib import Path

from .budget_frontier import (
    LayerCurves,
    LayerSolution,
    pareto_front,
    solve_all,
)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                       # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return None


def _grid(solutions: list[LayerSolution], value):
    """(layer types, depths, matrix[type][depth]) for the heatmap views."""
    types = sorted({s.layer_type for s in solutions})
    depths = sorted({s.depth for s in solutions})
    cell = {(s.layer_type, s.depth): s for s in solutions}
    matrix = [[value(cell[(t, d)]) if (t, d) in cell else float("nan")
               for d in depths] for t in types]
    return types, depths, matrix, cell


def plot_ratio_heatmap(solutions: list[LayerSolution], out_dir: Path,
                       budget: float, q_weight: float) -> None:
    plt = _plt()
    if plt is None or not solutions:
        return
    types, depths, matrix, cell = _grid(solutions, lambda s: s.ratio)
    finite = [v for row in matrix for v in row if v == v and math.isfinite(v)]
    if not finite:
        print("(No comparable layers at this budget: no ratio heatmap.)")
        return

    fig, ax = plt.subplots(figsize=(1.4 * len(depths) + 4, 0.75 * len(types) + 3))
    # Centre the colour scale on 1: below = the PQC pays, above = it does not.
    span = max(abs(math.log10(min(finite))), abs(math.log10(max(finite))), 0.3)
    logm = [[math.log10(v) if v == v and math.isfinite(v) and v > 0 else float("nan")
             for v in row] for row in matrix]
    im = ax.imshow(logm, cmap="RdBu_r", vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(len(depths)), [str(d) for d in depths])
    ax.set_yticks(range(len(types)), types)
    ax.set_xlabel("decoder block (depth)")
    ax.set_title(f"M*/N*  at  B = {budget:.4f} x baseline,  w = {q_weight:g}\n"
                 f"blue = the PQC+TN layer is smaller, red = the pure TN layer is")
    for i, t in enumerate(types):
        for j, d in enumerate(depths):
            s = cell.get((t, d))
            if s is None:
                continue
            if s.ratio != s.ratio or not math.isfinite(s.ratio):
                ax.text(j, i, s.status.split("_")[0][:4], ha="center",
                        va="center", fontsize=6, color="0.4")
                continue
            ax.text(j, i, f"{s.ratio:.2f}\n$\\chi'$={s.m_star_chi} D={s.m_star_circuit_depth} "
                          f"k={s.m_star_gate_size}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(math.log10(s.ratio)) > 0.6 * span else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10 (M*/N*)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hybrid_vs_classical_ratio.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def plot_ratio_vs_budget(curves: dict[str, LayerCurves], budgets: list[float],
                         out_dir: Path, q_weight: float) -> None:
    """Median M*/N* per layer type as the perplexity budget is loosened."""
    plt = _plt()
    if plt is None or not curves:
        return
    from .budget_frontier import _median, group_summary

    types = sorted({c.layer_type for c in curves.values()})
    series = {t: [] for t in types}
    feasible = {t: [] for t in types}
    for b in budgets:
        sols = solve_all(curves, b, q_weight)
        rows = {r["group"]: r for r in group_summary(sols, "layer_type")}
        for t in types:
            r = rows.get(t)
            series[t].append(r["median_ratio"] if r else float("nan"))
            feasible[t].append((r["n_comparable"] / r["n"]) if r and r["n"] else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("tab10")
    xs = [100 * (b - 1) for b in budgets]          # budget as % PPL headroom
    for i, t in enumerate(types):
        axes[0].plot(xs, series[t], marker="o", ms=4, color=cmap(i % 10), label=t)
        axes[1].plot(xs, feasible[t], marker="s", ms=4, color=cmap(i % 10), label=t)
    axes[0].axhline(1.0, ls="--", color="crimson", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].set_xlabel("perplexity budget B  (% above the dense baseline)")
    axes[0].set_ylabel("median M*/N*")
    axes[0].set_title(f"Does the PQC pay?  (w = {q_weight:g}; below the red line it does)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize="small", ncol=2)
    axes[1].set_xscale("symlog", linthresh=0.1)
    axes[1].set_xlabel("perplexity budget B  (% above the dense baseline)")
    axes[1].set_ylabel("fraction of layers with both methods feasible")
    axes[1].set_title("Comparable layers")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ratio_vs_budget.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def plot_pareto(curves: dict[str, LayerCurves], out_dir: Path, q_weight: float,
                budget: float | None = None, max_panels: int = 9) -> None:
    """Memory vs. perplexity for both methods, with the non-dominated staircase."""
    plt = _plt()
    if plt is None or not curves:
        return
    names = sorted(curves, key=lambda n: (curves[n].layer_type, curves[n].depth))
    names = names[:max_panels]
    ncols = min(3, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows),
                             squeeze=False)
    for idx, name in enumerate(names):
        curve = curves[name]
        ax = axes[idx // ncols][idx % ncols]
        cls = [(p.c_params, p.ppl_ratio) for p in curve.classical if p.c_params > 0]
        ax.scatter([c for c, _ in cls], [r for _, r in cls], s=26, marker="o",
                   color="#2a6f97", label="classical TN: C(chi)", zorder=3)
        configs = sorted({(p.gate_size, p.circuit_depth) for p in curve.hybrid})
        cmap = plt.get_cmap("autumn")
        for i, (k, d) in enumerate(configs):
            pts = [(p.cost(q_weight), p.ppl_ratio) for p in curve.hybrid
                   if (p.gate_size, p.circuit_depth) == (k, d)
                   and p.cost(q_weight) > 0]
            frac = i / max(1, len(configs) - 1)
            ax.scatter([c for c, _ in pts], [r for _, r in pts], s=26, marker="^",
                       color=cmap(0.75 * frac), label=f"hybrid D={d}, k={k}", zorder=3)
        front = pareto_front(curve, q_weight)
        if front:
            ax.step([p.cost for p in front], [p.ppl_ratio for p in front],
                    where="post", color="black", lw=1.3, alpha=0.75,
                    label="Pareto front", zorder=2)
        if budget:
            ax.axhline(budget, ls="--", color="crimson", lw=1.1,
                       label=f"budget B={budget:.3f}")
        ax.axvline(curve.dense_params, ls=":", color="0.4", lw=1.1,
                   label="dense layer")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("memory  C + w Q  (parameters)")
        ax.set_ylabel("perplexity / baseline")
        ax.set_title(f"{curve.layer_type}  block {curve.depth}", fontsize=10)
        ax.grid(True, which="both", alpha=0.22)
        if idx == 0:
            ax.legend(fontsize="x-small", ncol=2)
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Pareto front: memory vs. perplexity  (quantum parameter price "
                 f"w = {q_weight:g})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pareto_front.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def plot_breakeven(solutions: list[LayerSolution], out_dir: Path,
                   budget: float) -> None:
    """How cheap a quantum parameter must be, per layer, for the PQC to win."""
    plt = _plt()
    if plt is None or not solutions:
        return
    usable = [s for s in solutions if s.breakeven_weight == s.breakeven_weight]
    if not usable:
        return
    types, depths, matrix, cell = _grid(
        usable, lambda s: s.breakeven_weight if math.isfinite(s.breakeven_weight) else 1e9)
    fig, ax = plt.subplots(figsize=(1.4 * len(depths) + 4, 0.75 * len(types) + 3))
    logm = [[math.log10(v) if v == v and v > 0 else float("nan") for v in row]
            for row in matrix]
    im = ax.imshow(logm, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(depths)), [str(d) for d in depths])
    ax.set_yticks(range(len(types)), types)
    ax.set_xlabel("decoder block (depth)")
    ax.set_title(f"Break-even price w* of a quantum parameter  "
                 f"(B = {budget:.4f} x baseline)\n"
                 f"the PQC+TN layer is smaller whenever w < w*")
    for i, t in enumerate(types):
        for j, d in enumerate(depths):
            s = cell.get((t, d))
            if s is None or s.breakeven_weight != s.breakeven_weight:
                continue
            txt = "inf" if math.isinf(s.breakeven_weight) else f"{s.breakeven_weight:.3g}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10 w*")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "breakeven_weight.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def make_all(curves: dict[str, LayerCurves], solutions: list[LayerSolution],
             out_dir: Path, budget: float, q_weight: float,
             budgets: list[float] | None = None, max_panels: int = 9) -> None:
    plot_ratio_heatmap(solutions, out_dir, budget, q_weight)
    plot_breakeven(solutions, out_dir, budget)
    plot_pareto(curves, out_dir, q_weight, budget, max_panels)
    if budgets and len(budgets) > 1:
        plot_ratio_vs_budget(curves, budgets, out_dir, q_weight)
