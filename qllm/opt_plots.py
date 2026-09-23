"""Figures for the pre-stage optimisation checks and the Phase A proxy study.

``convergence_doubling.png``  did every configuration converge at the stage budget
``convergence_lr.png``        is the default learning rate on a plateau or a cliff
``ppl_scatter.png``           does per-layer error predict perplexity
``ppl_rho.png``               does the activation-weighted error predict it better
``proxy_scatter.png``         every candidate metric against the damage it cost
``proxy_rho.png``             which candidate ranks perplexity best, per layer

Colour is assigned by the job it does, not by rank: the regime (or the family) is
an *identity*, so it gets categorical slots in a fixed order, and a filtered-out
series never repaints the survivors. The palette is the validated default, used
from slot 1 upward -- five slots clear the adjacent pairlist, and the two-series
scatters clear the stricter all-pairs one. The five-slot set carries a contrast
warning against a light surface, so every row is labelled on the axis as well:
identity is never carried by colour alone, and the CSV beside each figure is the
table view.
"""

from __future__ import annotations

import math
from pathlib import Path

# Validated default categorical palette (light surface #fcfcfb), in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b5b4ae"


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                   # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return None


def _style(ax):
    """Recessive chrome: the data is the ink, the frame is not."""
    ax.grid(True, color=MUTED, linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  saved plot {path}")
    return path


# --------------------------------------------------------------------------- #
def plot_convergence(rows, tol, out_dir):
    """Doubling gap per configuration, one row per regime.

    A strip plot rather than a bar chart: the question is not "what is the mean
    gap" but "is any configuration past the tolerance", and a mean would hide
    exactly the outlier that matters. The tolerance is drawn, so the verdict is
    readable off the figure without consulting the text.
    """
    plt = _plt()
    if plt is None:
        return
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    fig, ax = plt.subplots(figsize=(8.2, 0.7 * len(regimes) + 2.2))
    for i, regime in enumerate(regimes):
        sel = [r for r in rows if r["regime"] == regime]
        colour = SERIES[i % len(SERIES)]
        for r in sel:
            still = r.get("still_improving")
            # Open marker = the best iterate was the last one, so the run was
            # still improving when it was cut off. A second channel beside
            # colour, which the contrast warning obliges.
            ax.plot([r["gap"] * 100], [i + (hash(str(r)) % 7 - 3) * 0.035],
                    marker="o", markersize=6, linestyle="none",
                    markerfacecolor=("none" if still else colour),
                    markeredgecolor=colour, markeredgewidth=1.4, alpha=0.85)
        n_bad = sum(1 for r in sel if not r["converged"])
        ax.annotate(f"{len(sel) - n_bad}/{len(sel)} converged",
                    xy=(1.005, i), xycoords=("axes fraction", "data"),
                    va="center", fontsize=8.5,
                    color=(INK if n_bad == 0 else SERIES[1]))
    ax.axvline(tol * 100, color=INK_2, linewidth=1.2, linestyle="--")
    ax.annotate(f"tolerance {tol:.0%}", xy=(tol * 100, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -11),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes, fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(regimes) - 0.4)
    ax.set_xlabel("error removed by doubling the budget  (%)")
    ax.set_title("Does the stage budget converge?  one dot per configuration\n"
                 "open dot = last tenth of the budget still delivering gain",
                 fontsize=11, color=INK, loc="left")
    _style(ax)
    _save(fig, out_dir, "convergence_doubling.png")
    plt.close(fig)


def plot_lr(rows, default_lr, out_dir):
    """Error against learning rate, one panel per regime, normalised per config.

    Each configuration is divided by its own best error, so the panels answer the
    only question that matters -- how much worse is the default than the best this
    configuration could have done -- without the layer-to-layer spread in absolute
    error swamping it. Individual configurations are recessive; the median is the
    series.
    """
    plt = _plt()
    if plt is None or not rows:
        return
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    lrs = sorted({r["lr"] for r in rows})
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.1 * len(regimes), 3.4),
                             sharey=True, squeeze=False)
    for i, regime in enumerate(regimes):
        ax = axes[0][i]
        keys = dict.fromkeys(
            (r["layer_type"], r["depth"], r["chi"], r["ansatz"])
            for r in rows if r["regime"] == regime)
        curves = []
        for k in keys:
            pts = {r["lr"]: r["err"] for r in rows
                   if r["regime"] == regime
                   and (r["layer_type"], r["depth"], r["chi"], r["ansatz"]) == k}
            ys = [pts.get(l) for l in lrs]
            if any(y is None for y in ys) or min(ys) <= 0:
                continue
            best = min(ys)
            curves.append([y / best for y in ys])
            ax.plot(lrs, curves[-1], color=MUTED, linewidth=1.0, alpha=0.7,
                    label=("configurations" if len(curves) == 1 else None))
        if curves:
            med = [sorted(c[j] for c in curves)[len(curves) // 2]
                   for j in range(len(lrs))]
            ax.plot(lrs, med, color=SERIES[i % len(SERIES)], linewidth=2.0,
                    marker="o", markersize=5, label="median")
        ax.axvline(default_lr, color=INK_2, linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks(lrs)
        ax.set_xticklabels([str(l) for l in lrs], fontsize=8.5)
        ax.set_title(regime, fontsize=9.5, color=INK)
        ax.set_xlabel("learning rate")
        if i == 0:
            ax.set_ylabel("error / this config's best\n(1.0 = best available)")
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2,
                      loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
        _style(ax)
    fig.suptitle(f"Learning-rate sensitivity  (dashed = the default, {default_lr:g})",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, out_dir, "convergence_lr.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_ppl_scatter(rows, out_dir, spearman):
    """Per-layer error against the perplexity it caused, both metrics.

    Two panels on one shared y axis rather than one panel with two x scales: the
    two metrics are different measures, and overlaying them on invented-aligned
    axes would manufacture agreement. Log-log because damage compounds.
    """
    plt = _plt()
    if plt is None or not rows:
        return
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for j, metric in enumerate(("act", "frob")):
        ax = axes[j]
        for i, fam in enumerate(fams):
            pts = [(r[metric], r["dppl"]) for r in rows
                   if r["family"] == fam and r["dppl"] > 0 and r[metric] > 0]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none",
                    marker="o", markersize=6, alpha=0.8,
                    markerfacecolor=SERIES[i], markeredgecolor="#fcfcfb",
                    markeredgewidth=0.8, label=fam)
        rho = spearman.get(metric)
        ax.set_title(("activation-weighted error" if metric == "act"
                      else "Frobenius error")
                     + (f"   pooled rho = {rho:+.2f}" if rho is not None else ""),
                     fontsize=10, color=INK, loc="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("per-layer relative error")
        if j == 0:
            ax.set_ylabel("perplexity increase over baseline")
            ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        _style(ax)
    fig.suptitle("Does the per-layer error predict perplexity?", fontsize=11,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "ppl_scatter.png")
    plt.close(fig)


def plot_ppl_rho(per_layer, out_dir):
    """Rank correlation per layer, activation against Frobenius, grouped by family.

    Signed, with zero drawn: a proxy anti-correlated with perplexity ranks every
    configuration backwards, which an absolute-value axis would hide.
    """
    plt = _plt()
    if plt is None or not per_layer:
        return
    labels = [f"{lt} d{dep}  [{fam}]" for lt, dep, fam, _c in per_layer]
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(labels) + 2.2))
    y = range(len(labels))
    h = 0.36
    for i, (key, colour) in enumerate((("rho_act", SERIES[0]),
                                       ("rho_frob", SERIES[1]))):
        vals = [c[key] for _l, _d, _f, c in per_layer]
        ax.barh([v + (i - 0.5) * h for v in y], vals, height=h * 0.92,
                color=colour, edgecolor="#fcfcfb", linewidth=1.0,
                label=("activation" if key == "rho_act" else "Frobenius"))
    ax.axvline(0.0, color=INK_2, linewidth=1.2)
    ax.axvline(0.5, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("weak-proxy floor", xy=(0.5, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -11),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(min(-1.05, min(c[k] for _l, _d, _f, c in per_layer
                               for k in ("rho_act", "rho_frob")) - 0.05), 1.05)
    ax.set_xlabel("Spearman rho with perplexity   (negative = ranks backwards)")
    ax.set_title("Which per-layer error is the better proxy?", fontsize=11,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.5 / len(labels)))
    _style(ax)
    _save(fig, out_dir, "ppl_rho.png")
    plt.close(fig)


def plot_traces(traces, out_dir, name="convergence_traces.png", max_curves=400):
    """Every optimisation curve, one panel per regime, loss against iteration.

    This is the direct evidence for "all runs converge": a plateaued run is a
    curve that goes flat, and one that is still descending at the right edge is
    visible as such without any summary statistic standing in between. Each curve
    is normalised to its own first value, because the point is the *shape* of the
    descent and absolute losses differ by layer and by objective.

    The gradient phase only -- an environment sweep is a handful of iterations
    and would compress to a couple of pixels beside a 150-step Adam run.
    """
    plt = _plt()
    if plt is None or not traces:
        return
    grad = [t for t in traces if t["phase"] == "gradient"]
    if not grad:
        print("  (no gradient-phase rows to plot)")
        return
    regimes = list(dict.fromkeys(t.get("regime", t["objective"]) for t in grad))
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.1 * len(regimes) + 0.6, 3.5),
                             sharey=True, squeeze=False)
    for i, regime in enumerate(regimes):
        ax = axes[0][i]
        runs = {}
        for t in grad:
            if t.get("regime", t["objective"]) != regime:
                continue
            runs.setdefault(t["run_id"], []).append((t["iter"], t["loss"]))
        drawn = 0
        for pts in runs.values():
            if len(pts) < 2:
                continue
            pts.sort()
            first = pts[0][1]
            if not first:
                continue
            ax.plot([p[0] for p in pts], [p[1] / first for p in pts],
                    color=SERIES[i % len(SERIES)], linewidth=0.9, alpha=0.45)
            drawn += 1
            if drawn >= max_curves:
                break
        ax.set_title(f"{regime}   ({drawn} runs)", fontsize=9.5, color=INK)
        ax.set_xlabel("Adam step")
        if i == 0:
            ax.set_ylabel("loss / its own first value")
        _style(ax)
    fig.suptitle("Every optimisation curve  (flat at the right edge = converged)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase A: which per-layer metric is the proxy?                                #
# --------------------------------------------------------------------------- #

# Seven-odd perturbation families outrun the five validated colour slots, so
# family identity is carried by marker shape as well as hue, and spelled out in
# the legend. The control family is drawn hollow so it reads as the control at a
# glance rather than as one more series.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _mean(vals):
    """Mean over the finite entries; NaN when a correlation was undefined."""
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _rank_corr(x, y):
    """Spearman rho, local so this module imports nothing that imports it back."""
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _family_style(fams):
    """Stable (colour, marker, hollow) per family, control last and hollow."""
    ordered = [f for f in fams if f != "random"] + \
              (["random"] if "random" in fams else [])
    style = {}
    for i, fam in enumerate(ordered):
        if fam == "random":
            style[fam] = (INK_2, "X", True)
        else:
            style[fam] = (SERIES[i % len(SERIES)],
                          MARKERS[i % len(MARKERS)], False)
    return ordered, style


def plot_proxy_scatter(rows, candidates, out_dir):
    """Every candidate metric against the perplexity its perturbation cost.

    One panel per metric on a shared y axis, log-log, because a proxy is only
    useful over orders of magnitude of damage. The panels share the y axis and
    nothing else: each metric keeps its own x scale, since forcing them onto a
    common axis would invent an agreement none of them has earned.

    The spread *within* a family at fixed x is the honest failure mode to look
    for -- a vertical smear means the metric is reading a number that perplexity
    does not follow. The hollow control markers are the sharper test: if they sit
    on the same line as the structured families, the metric is blind to structure.
    """
    plt = _plt()
    if plt is None or not rows:
        return
    fams, style = _family_style(list(dict.fromkeys(r["family"] for r in rows)))

    # A perturbation that makes perplexity go DOWN is the most informative row in
    # the study -- it is the proxy pointing the wrong way -- so the y axis is
    # symlog rather than log. A log axis would drop exactly those rows, and drop
    # them silently, leaving a cleaner-looking figure than the data supports.
    # The linear band is placed at the bottom decile of the observed damage and
    # never more than three decades below the largest, so most points land in the
    # log region. Put it at the median instead and the bulk of the data is crushed
    # into an unlabelled linear smear.
    pos = sorted(abs(r["dppl"]) for r in rows if r["dppl"] != 0)
    lin = max(pos[len(pos) // 10], pos[-1] / 1e3) if pos else 1e-6
    fig, axes = plt.subplots(1, len(candidates),
                             figsize=(3.1 * len(candidates) + 0.8, 4.0),
                             sharey=True, squeeze=False)
    for j, metric in enumerate(candidates):
        ax = axes[0][j]
        for fam in fams:
            colour, marker, hollow = style[fam]
            pts = [(r[metric], r["dppl"]) for r in rows
                   if r["family"] == fam and r[metric] > 0]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none",
                    marker=marker, markersize=6, alpha=0.85,
                    markerfacecolor=("none" if hollow else colour),
                    markeredgecolor=(colour if hollow else "#fcfcfb"),
                    markeredgewidth=(1.3 if hollow else 0.8))
        good = [(r[metric], r["ppl"]) for r in rows if r[metric] > 0]
        rho = _rank_corr([g[0] for g in good], [g[1] for g in good]) \
            if len(good) >= 3 else float("nan")
        ax.set_title(f"{metric}\npooled rho = {rho:+.2f}", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=max(lin, 1e-12), linscale=0.4)
        ax.axhline(0.0, color=INK_2, linewidth=1.0)
        ax.set_xlabel("metric value")
        if j == 0:
            ax.set_ylabel("perplexity change from baseline\n(symlog; zero drawn)")
        _style(ax)
    handles = [plt.Line2D([], [], linestyle="none", marker=style[f][1],
                          markersize=6,
                          markerfacecolor=("none" if style[f][2] else style[f][0]),
                          markeredgecolor=style[f][0],
                          markeredgewidth=(1.3 if style[f][2] else 0.8),
                          label=f + (" (control)" if f == "random" else ""))
               for f in fams]
    fig.suptitle("Does the per-layer metric predict the perplexity it costs?",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
               ncol=min(len(fams), 4), loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(fig, out_dir, "proxy_scatter.png")
    plt.close(fig)


def plot_proxy_rho(rows, candidates, keys, out_dir):
    """Rank correlation with perplexity, per layer, one bar per candidate metric.

    Signed with zero drawn, for the same reason as ``plot_ppl_rho``: a metric
    that ranks backwards is worse than no metric, and an absolute-value axis
    would draw it as a success. The 0.9 line is the threshold the verdict uses,
    so the figure and the printed verdict cannot disagree.
    """
    plt = _plt()
    if plt is None or not rows or not keys:
        return
    groups, labels = [], []
    for lt, dep in keys:
        sel = [r for r in rows if r["layer_type"] == lt and r["depth"] == dep]
        if len(sel) < 5:
            continue
        groups.append([_rank_corr([r[c] for r in sel], [r["ppl"] for r in sel])
                       for c in candidates])
        labels.append(f"{lt} d{dep}  (n={len(sel)})")
    if not groups:
        print("  (no layer has >= 5 perturbations; skipping proxy_rho)")
        return
    means = [_mean([g[i] for g in groups]) for i in range(len(candidates))]
    groups.append(means)
    labels.append("MEAN over layers")

    fig, ax = plt.subplots(
        figsize=(8.4, 0.26 * len(labels) * len(candidates) + 1.5))
    y = range(len(labels))
    h = 0.8 / len(candidates)
    for i, cand in enumerate(candidates):
        off = (i - (len(candidates) - 1) / 2.0) * h
        ax.barh([v + off for v in y], [g[i] for g in groups], height=h * 0.9,
                color=SERIES[i % len(SERIES)], edgecolor="#fcfcfb",
                linewidth=0.9, label=cand)
    ax.axvline(0.0, color=INK_2, linewidth=1.2)
    ax.axvline(0.9, color=MUTED, linewidth=1.0, linestyle="--")
    # Inside the axes, not above them: at two groups the axes box is short and an
    # annotation hung off the top edge lands in the title.
    ax.annotate("verdict threshold 0.9", xy=(0.9, 0.0),
                xycoords=("data", "axes fraction"), xytext=(-5, 5),
                textcoords="offset points", fontsize=8.5, color=INK_2,
                ha="right", va="bottom")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    lo = min(v for g in groups for v in g if v == v)
    ax.set_xlim(min(-1.05, lo - 0.05), 1.05)
    ax.set_xlabel("Spearman rho with perplexity   (negative = ranks backwards)")
    ax.set_title("Which per-layer metric is the better proxy?", fontsize=11,
                 color=INK, loc="left")
    _style(ax)
    fig.legend(frameon=False, fontsize=9, labelcolor=INK_2,
               ncol=min(len(candidates), 4), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out_dir, "proxy_rho.png")
    plt.close(fig)
