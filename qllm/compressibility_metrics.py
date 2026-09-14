"""Relate per-layer compressibility (CompactifAI sweep) to layer_analysis metrics.

Joins two CSVs on ``param_name``:

* ``compactifai_per_layer.csv`` -- one row per (layer, chi) from the per-layer
  MPO sweep. Reduced here to a few *compressibility scalars* per layer.
* ``layer_analysis.csv`` -- one row per layer with its spectral / sensitivity
  metrics.

and reports how strongly each metric ranks with compressibility (Spearman), plus
scatter plots. This answers: *does a layer's spectral shape (entropy, rank,
condition number, gap) or its perplexity-sensitivity predict how far it can be
MPO-compressed before perplexity degrades?*
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

# Metrics from layer_analysis.csv worth correlating (shape-normalized first).
METRIC_COLUMNS = [
    "perplexity_sensitivity",
    "shannon_entropy_normalized", "renyi2_entropy_normalized",
    "effective_rank_ratio", "stable_rank_ratio",
    "relative_spectral_gap", "log_condition_number",
    "condition_number_rmt_ratio",
    "effective_rank", "stable_rank", "condition_number", "spectral_gap",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def compressibility_scalars(per_layer_rows: list[dict], ref_chi: int | None,
                            budget: float) -> dict[str, dict]:
    """Reduce the (layer, chi) sweep to one record of scalars per layer.

    * ``log_ppl_ratio_ref`` -- log perplexity ratio at the reference chi
      (default: the median chi in the sweep). Higher = more damage. Its negation
      is a censoring-free "compressibility" target.
    * ``chi_at_budget`` -- smallest chi whose ppl_ratio <= ``budget`` (lower is
      more compressible); ``inf`` if never within budget.
    * ``max_compression`` -- ``1 - min(compression_ratio)`` over the chi that meet
      the budget (higher = more of the layer removed while staying within budget).
    """
    by_layer = defaultdict(dict)
    meta = {}
    for r in per_layer_rows:
        name = r["param_name"]
        chi = int(r["chi"])
        by_layer[name][chi] = r
        meta.setdefault(name, (r.get("layer_type", ""), int(r["depth"])))

    all_chis = sorted({int(r["chi"]) for r in per_layer_rows})
    if ref_chi is None and all_chis:
        ref_chi = all_chis[len(all_chis) // 2]

    out = {}
    for name, chis in by_layer.items():
        ref = chis.get(ref_chi)
        if ref is None:                       # nearest available chi
            ref = chis[min(chis, key=lambda c: abs(c - (ref_chi or 0)))]
        within = [(int(c), r) for c, r in chis.items() if _f(r["ppl_ratio"]) <= budget]
        chi_at_budget = min((c for c, _ in within), default=float("inf"))
        max_comp = (1 - min(_f(r["compression_ratio"]) for _, r in within)) if within else 0.0
        lt, depth = meta[name]
        out[name] = dict(
            param_name=name, layer_type=lt, depth=depth, ref_chi=ref_chi,
            log_ppl_ratio_ref=math.log(_f(ref["ppl_ratio"])),
            ppl_ratio_ref=_f(ref["ppl_ratio"]),
            relative_error_ref=_f(ref["relative_error"]),
            chi_at_budget=chi_at_budget,
            max_compression=max_comp,
        )
    return out


def spearman(a: list[float], b: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y and math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan")
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        rk = [0.0] * n
        i = 0
        while i < n:                          # average ties
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    xa = [p[0] for p in pairs]; xb = [p[1] for p in pairs]
    ra, rb = ranks(xa), ranks(xb)
    mean = (n - 1) / 2.0
    cov = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    va = math.sqrt(sum((ra[i] - mean) ** 2 for i in range(n)))
    vb = math.sqrt(sum((rb[i] - mean) ** 2 for i in range(n)))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def correlate(per_layer_csv: str | Path, layer_analysis_csv: str | Path,
              target: str = "log_ppl_ratio_ref", ref_chi: int | None = None,
              budget: float = 1.10) -> dict:
    """Join the two CSVs and Spearman-correlate each metric with compressibility.

    ``target`` is the compressibility scalar to correlate against. The sign
    convention below is set so a POSITIVE rho means "more of this metric -> more
    compressible" regardless of which target is chosen.
    """
    scal = compressibility_scalars(_read(per_layer_csv), ref_chi, budget)
    metrics = {r["param_name"]: r for r in _read(layer_analysis_csv)}

    joined = []
    for name, s in scal.items():
        if name in metrics:
            row = dict(s)
            for m in METRIC_COLUMNS:
                row[m] = _f(metrics[name].get(m, "nan"))
            joined.append(row)

    # "compressibility" increases when damage decreases -> flip damage targets.
    flip = target in ("log_ppl_ratio_ref", "ppl_ratio_ref", "relative_error_ref",
                       "chi_at_budget")
    y = [(-row[target] if flip else row[target]) for row in joined]

    results = []
    for m in METRIC_COLUMNS:
        x = [row[m] for row in joined]
        results.append((m, spearman(x, y)))
    results.sort(key=lambda kv: (-(abs(kv[1]) if kv[1] == kv[1] else -1)))
    return {"n_layers": len(joined), "target": target, "budget": budget,
            "joined": joined, "correlations": results}


def print_report(res: dict) -> None:
    print(f"\nCompressibility vs. layer_analysis metrics "
          f"({res['n_layers']} layers joined)")
    print(f"target = compressibility (higher = more compressible), "
          f"derived from '{res['target']}'")
    print("=" * 64)
    print(f"{'metric':30} {'Spearman rho':>12}  interpretation")
    for m, rho in res["correlations"]:
        if rho != rho:
            tag = "n/a"
        elif abs(rho) >= 0.6:
            tag = "STRONG"
        elif abs(rho) >= 0.4:
            tag = "moderate"
        elif abs(rho) >= 0.2:
            tag = "weak"
        else:
            tag = "~none"
        sign = "+more compressible" if rho > 0 else "-less compressible"
        print(f"{m:30} {rho:>+12.3f}  {tag} ({sign})")


def make_plots(res: dict, out_dir: Path, top: int = 6) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                  # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return
    joined = res["joined"]
    if not joined:
        return
    metrics = [m for m, rho in res["correlations"] if rho == rho][:top]
    types = sorted({r["layer_type"] for r in joined})
    cmap = plt.get_cmap("tab10")
    tcol = {t: cmap(i % 10) for i, t in enumerate(types)}

    ncol = 3
    nrow = math.ceil(len(metrics) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    flip = res["target"] in ("log_ppl_ratio_ref", "ppl_ratio_ref",
                             "relative_error_ref", "chi_at_budget")
    for idx, m in enumerate(metrics):
        ax = axes[idx // ncol][idx % ncol]
        for t in types:
            xs = [r[m] for r in joined if r["layer_type"] == t]
            ys = [(-r[res["target"]] if flip else r[res["target"]])
                  for r in joined if r["layer_type"] == t]
            ax.scatter(xs, ys, s=28, color=tcol[t], label=t, alpha=0.8)
        rho = dict(res["correlations"])[m]
        ax.set_xlabel(m)
        ax.set_ylabel(f"compressibility (-{res['target']})")
        ax.set_title(f"{m}  (rho={rho:+.2f})", fontsize=10)
        ax.grid(True, alpha=0.25)
    for j in range(len(metrics), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(types), fontsize="small")
    fig.suptitle("Layer compressibility vs. layer_analysis metrics", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compressibility_vs_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved plot {path}")
