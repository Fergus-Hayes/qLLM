"""Reproduce Fig. 3 of arXiv:2410.17397v2: disentangling accuracy vs. layers.

The paper's Fig. 3 plots, for several gate sizes, the **disentangling accuracy**
A (its Eq. 4) and the **mean bond entropy** of the disentangled operator against
the number of brickwall layers L, with the circuits optimized to fully
disentangle the layer (target = the bond-dimension-1 truncation of its MPO). For
the two-qubit-gate case the accuracy rises -- roughly logarithmically, saturating
-- as L grows, while the entropy falls.

This CLI sweeps L (= the brickwall depth D) at a fixed target bond dimension
(chi' = 1) and records, per (gate size, L):

* ``accuracy``  -- Eq. 4: ``Tr[T1(W)^T U^T W V] / (||W|| ||T1(W)||)``, exactly the
  ``DisentangleResult.accuracy`` field, with the target *fixed* to the bond-1
  truncation (``--disentangle-target fixed``, the paper's main-text choice);
* ``entropy``   -- the mean von Neumann entropy over the MPO bonds;
* ``retained``  -- ``||T1(U^T W V)|| / ||W||``, the weight the bond-1 truncation keeps.
* ``classical_params`` / ``quantum_params`` / ``total_params`` -- the hybrid cost
  ``M* = C(chi') + Q(D)``, plotted against L.

The sweep includes ``L = 0`` as an ordinary point (the left end of every curve):
zero circuit depth means identity circuits, i.e. the plain chi'-MPO truncation on
the *padded* qubit geometry, at ``M* = C(chi')`` and ``Q(D)=0``. Pass ``--no-d0``
to drop it.

It also records one tensor-network-only baseline as a marked row (``kind``
column, ``Q(D)=0``), drawn as a horizontal reference line:

* ``tn_nopad`` -- plain CompactifAI: the balanced 2-site MPO on the layer's *raw*
  dimensions, with no power-of-two padding.

It writes a CSV and, if matplotlib is present, the Fig. 3 panels (accuracy,
entropy, M*, and -- on the model path -- perplexity vs. L).

On the model path ``--heal {core,full}`` adds healed perplexity curves: after
swapping in the compressed layer, briefly retrain it against the LM loss (every
other layer dense) and record the recovered perplexity. ``core`` retrains only
the chi' MPO bond (``C(chi')`` params); ``full`` retrains the bond **and** the
U/V circuits (``M*`` params) -- give both to see how much the circuits' extra
task-trainable degrees of freedom recover beyond the starved small-chi' bond.

The CSV is a checkpoint: a re-run reuses every (gate size, L) point already in
it for the same configuration -- and, on the model path, the cached baseline
perplexity -- so an interrupted sweep resumes where it stopped and extra L
values only cost the new points. A run whose configuration (layer, MPO
geometry, optimizer, target chi', disentangle target) differs from the CSV is
refused; pass ``--recompute`` to overwrite it or ``--csv-name`` to keep both.

The layer is either a real model weight (``MODEL --block B --layer-type T``,
needs the Hugging Face files) or a synthetic matrix of a given shape
(``--shape 192x576``), which runs offline and is reproducible.

    # The paper's layer (SmolLM2 block 10 v_proj), two-qubit gates, L = 1..35
    python disentangle_scaling.py HuggingFaceTB/SmolLM2-135M \
        --block 10 --layer-type v_proj --gate-sizes 2 --max-layers 35

    # Offline, on a synthetic (576,192)-shaped matrix
    python disentangle_scaling.py --shape 192x576 --gate-sizes 1 2 --max-layers 35
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from .compactifai import mpo_param_count
from .disentangler import (
    disentangle,
    hybrid_weight,
    n_qubits_for,
    quantum_param_count,
)
from .hybrid_heal import heal_hybrid, make_heal_batches
from .hybrid_sweep import DEFAULT_GATE_SIZE, validate_gate_sizes
from .layer_analysis import parse_layer_info
from .qubit_mpo import make_plan, plan_bond_entropy, plan_compress

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def log_spaced_layers(max_layers: int, num_points: int) -> list[int]:
    """De-duplicated, log-spaced integer layer counts in ``[1, max_layers]``."""
    if max_layers <= 1 or num_points <= 1:
        return [max(1, max_layers)]
    vals = {int(round(max_layers ** (i / (num_points - 1)))) for i in range(num_points)}
    return sorted(v for v in vals if 1 <= v <= max_layers)


def parse_shape(text: str) -> tuple[int, int]:
    rows, _, cols = text.lower().partition("x")
    return int(rows), int(cols)


def synthetic_weight(d_out: int, d_in: int, rank: int, noise: float,
                     seed: int) -> torch.Tensor:
    """A structured ``(d_out, d_in)`` matrix: low-rank core + noise (reproducible).

    A pure Gaussian matrix is near-maximally entangled (flat spectrum); a trained
    layer is not. This gives it non-trivial, disentangle-able structure so the
    accuracy-vs-L curve is meaningful, the way the paper's real layer is.
    """
    g = torch.Generator().manual_seed(seed)
    r = max(1, min(rank, d_out, d_in))
    core = torch.randn(d_out, r, generator=g) @ torch.randn(r, d_in, generator=g)
    return core / math.sqrt(r) + noise * torch.randn(d_out, d_in, generator=g)


def load_model_layer(args):
    """Load the model + tokenizer and return the live target-layer parameter.

    Returns ``(model, tokenizer, device, name, param)`` -- ``param`` is the actual
    ``nn.Parameter``, so its weight can be swapped for the disentangled
    reconstruction and restored while measuring perplexity.
    """
    from .benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device
    dev = resolve_device(args.device)
    cfg = BenchmarkConfig(model_id=args.model, device=args.device, dtype=args.dtype,
                          trust_remote_code=args.trust_remote_code,
                          revision=args.revision)
    model, tok = load_model_and_tokenizer(cfg, dev)
    matches = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        lt, depth = parse_layer_info(name)
        if depth == args.block and args.layer_type in lt:
            matches.append((name, param))
    if not matches:
        raise RuntimeError(
            f"No weight matched block {args.block} / '{args.layer_type}'. Check "
            f"--block/--layer-type against the model's parameter names.")
    if len(matches) > 1:
        names = ", ".join(n for n, _ in matches)
        print(f"(Note: {len(matches)} weights matched; using the first: {names})")
    name, param = matches[0]
    return model, tok, dev, name, param


ROW_FIELDS = ["kind", "gate_size", "n_layers", "d_out", "d_in", "n_out_qubits",
              "n_in_qubits", "target_chi", "disentangle_target",
              "tensorization", "optimizer",
              "accuracy", "entropy", "retained", "retained_classical",
              "classical_params", "quantum_params", "total_params",
              "perplexity", "ppl_baseline", "ppl_ratio",
              "ppl_core", "ppl_ratio_core", "ppl_full", "ppl_ratio_full",
              "sweeps_run", "seconds", "source"]

# Reference rows carried alongside the swept (D>=1) circuits: the depth-0 hybrid
# (identity circuits -> the padded tensor-network truncation) and the plain
# CompactifAI tensor network on the *unpadded* matrix. Both have Q(D)=0.
KIND_SWEEP, KIND_TN_D0, KIND_TN_NOPAD = "sweep", "tn_d0", "tn_nopad"


def _read_rows(path: Path) -> list[dict]:
    """Existing checkpoint rows (empty if the CSV does not exist yet)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce arXiv:2410.17397v2 Fig. 3: disentangling accuracy "
                    "and bond entropy vs. the number of brickwall layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                   help=f"Model id (default: {DEFAULT_MODEL}); ignored with --shape.")
    p.add_argument("--shape", default=None,
                   help="Use a synthetic matrix of this shape (e.g. 192x576) "
                        "instead of a model layer -- runs offline.")
    p.add_argument("--block", type=int, default=10,
                   help="Decoder block index of the layer (default: 10).")
    p.add_argument("--layer-type", default="v_proj",
                   help="Layer-type substring to select (default: v_proj).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="float32",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--revision", default=None)

    p.add_argument("--gate-sizes", type=int, nargs="+", default=[2],
                   help=f"Qubits per gate k (default: {DEFAULT_GATE_SIZE}; no "
                        f"upper cap, k=0 = register-wide gate).")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Explicit brickwall depths L to sweep (overrides "
                        "--max-layers/--num-points). L=0 is allowed: it is the "
                        "no-circuit (tensor-network-only) point.")
    p.add_argument("--max-layers", type=int, default=35,
                   help="Largest L (default: 35, the paper's k=2 range).")
    p.add_argument("--num-points", type=int, default=10,
                   help="Number of log-spaced L values up to --max-layers.")
    p.add_argument("--no-d0", action="store_true",
                   help="Drop the L=0 (no-circuit / TN-only) point that is "
                        "otherwise included as the left end of each curve.")
    p.add_argument("--target-chi", type=int, default=1,
                   help="Target bond dimension the circuits disentangle to "
                        "(default: 1, the paper's fully-disentangled target).")
    p.add_argument("--disentangle-target", default="fixed",
                   choices=["fixed", "adaptive"],
                   help="'fixed' (default) is the paper's Eq. 4 target -- the "
                        "bond-1 truncation of the original operator.")
    p.add_argument("--tensorization", default="qubit", choices=["balanced", "qubit"],
                   help="MPO geometry (default: qubit, the paper's).")
    p.add_argument("--optimizer", default="explicit",
                   choices=["explicit", "gradient", "explicit+gradient"],
                   help="Training scheme (default: explicit, the paper's env-SVD). "
                        "'explicit+gradient' refines the sweep with Adam afterwards.")
    p.add_argument("--fast-gradient", action="store_true",
                   help="gradient/explicit+gradient: evaluate the loss with pure-torch "
                        "apply_circuit contractions instead of PennyLane qml.matrix "
                        "(same result, faster per step; no effect on explicit).")
    p.add_argument("--gradient-objective", default="disentangle-loss",
                   choices=["disentangle-loss", "relative-error"],
                   help="gradient/explicit+gradient: minimize the per-bond discarded "
                        "weight ('disentangle-loss', default) or the joint "
                        "reconstruction error at target_chi ('relative-error').")
    p.add_argument("--disentangle-sweeps", type=int, default=40,
                   help="explicit: max environment sweeps per L.")
    p.add_argument("--disentangle-gd-steps", type=int, default=200)
    p.add_argument("--disentangle-gd-lr", type=float, default=0.05)
    p.add_argument("--restarts", type=int, default=1,
                   help="Keep the best of this many circuit initializations.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic-rank", type=int, default=24,
                   help="Rank of the synthetic matrix's low-rank core (--shape).")
    p.add_argument("--synthetic-noise", type=float, default=0.3)

    # Perplexity vs. L (model path only): swap the target layer for its
    # disentangled chi'-truncated reconstruction at each L and re-score the model.
    p.add_argument("--no-perplexity", action="store_true",
                   help="Skip the perplexity-vs-L curve (accuracy/entropy only). "
                        "Perplexity is unavailable with --shape (no model).")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024,
                   help="Sliding-window stride (default: non-overlapping).")
    p.add_argument("--eval-tokens", type=int, default=8192,
                   help="Tokens per perplexity evaluation (one per L point).")
    p.add_argument("--ppl-batch-size", type=int, default=8)

    # Healing (model path only): after swapping in the compressed layer, briefly
    # retrain it against the LM loss and record the healed perplexity per L.
    p.add_argument("--heal", nargs="+", default=["none"],
                   choices=["none", "core", "full"],
                   help="Healed perplexity curves to add (model path). 'core' "
                        "retrains only the chi' MPO bond; 'full' retrains the bond "
                        "AND the U/V circuits (M* params). Give both to compare.")
    p.add_argument("--heal-steps", type=int, default=100,
                   help="Adam steps per healed point (per mode).")
    p.add_argument("--heal-lr", type=float, default=0.02)
    p.add_argument("--heal-tokens", type=int, default=4096,
                   help="Calibration tokens used for healing.")
    p.add_argument("--heal-batch-size", type=int, default=2)
    p.add_argument("--heal-window", type=int, default=None,
                   help="Healing sequence length (default: --max-length).")
    p.add_argument("--heal-split", default="train",
                   help="Dataset split for the healing calibration set "
                        "(default: train -- disjoint from the eval split).")

    p.add_argument("--recompute", action="store_true",
                   help="Ignore any existing CSV and recompute every point from "
                        "scratch (default: resume, skipping points already done "
                        "and reusing the cached baseline perplexity).")
    p.add_argument("--results-dir", default="results/llms")
    p.add_argument("--csv-name", default="disentangle_scaling.csv")
    p.add_argument("--out-dir", default=None,
                   help="Where to write CSV + plots (default: derived from model).")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def _out_dir(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    name = ("synthetic" if args.shape
            else args.model.rstrip("/").split("/")[-1])
    return Path(args.results_dir) / name / "disentangle_scaling"


def _finite(rows: list[dict], key: str) -> bool:
    for r in rows:
        try:
            if math.isfinite(float(r[key])):
                return True
        except (KeyError, TypeError, ValueError):
            pass
    return False


def plot_scaling(rows: list[dict], out_dir: Path, target_chi: int,
                 refs: list[dict] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                          # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return
    by_k: dict[int, list[dict]] = {}
    for r in rows:
        by_k.setdefault(int(r["gate_size"]), []).append(r)
    ks = sorted(by_k)
    cmap = plt.get_cmap("viridis")

    # Beyond accuracy + entropy, an M* panel (total parameters vs. L) is added
    # whenever it was recorded, and a perplexity-ratio panel when perplexity was
    # measured (model path only).
    has_m = _finite(rows, "total_params")
    has_ppl = _finite(rows, "ppl_ratio")
    baseline = None
    if has_ppl:
        for r in rows:
            try:
                baseline = float(r["ppl_baseline"]); break
            except (KeyError, TypeError, ValueError):
                pass

    n_panels = 2 + int(has_m) + int(has_ppl)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 4 * n_panels), sharex=True)
    axes = list(axes)
    ax_a, ax_s = axes[0], axes[1]
    idx = 2
    ax_m = axes[idx] if has_m else None
    idx += int(has_m)
    ax_p = axes[idx] if has_ppl else None

    def _pts(k, key):
        pts = sorted(by_k[k], key=lambda r: int(r["n_layers"]))
        xy = [(int(r["n_layers"]), float(r[key])) for r in pts
              if _is_finite(r.get(key))]
        return [x for x, _ in xy], [y for _, y in xy]

    # Healed perplexity curves, when present (cold = solid; core = dashed;
    # full = dotted), so a single per-k colour carries all three.
    heal_curves = [(m, key, style) for m, key, style in
                   (("core", "ppl_ratio_core", (0, (5, 2))),
                    ("full", "ppl_ratio_full", (0, (1, 1))))
                   if _finite(rows, key)]

    for i, k in enumerate(ks):
        col = cmap(i / max(1, len(ks) - 1))
        lab = f"{k}q gates"
        ax_a.plot(*_pts(k, "accuracy"), marker="o", color=col, label=lab)
        ax_s.plot(*_pts(k, "entropy"), marker="s", color=col, label=lab)
        if ax_m is not None:
            ax_m.plot(*_pts(k, "total_params"), marker="D", color=col, label=lab)
        if ax_p is not None:
            plab = f"{lab} (cold)" if heal_curves else lab
            ax_p.plot(*_pts(k, "ppl_ratio"), marker="^", color=col, label=plab)
            for mode, key, style in heal_curves:
                ax_p.plot(*_pts(k, key), marker="v", color=col, ls=style,
                          label=f"{lab} (+heal {mode})")

    # Horizontal reference line: the plain TN on the *unpadded* matrix (does not
    # depend on L). The D=0 point is drawn as an ordinary point on each curve.
    ref_styles = {
        KIND_TN_NOPAD: dict(ls="-.", color="black", lw=1.4,
                            label="TN only (no padding)"),
    }

    def _ref_line(ax, key):
        for r in (refs or []):
            st = ref_styles.get(r.get("kind"))
            if st and _is_finite(r.get(key)):
                ax.axhline(float(r[key]), **st)

    _ref_line(ax_a, "accuracy")
    _ref_line(ax_s, "entropy")

    ax_a.set_ylabel("disentangling accuracy $A$  (Eq. 4)")
    ax_a.set_title(f"Fig. 3 reproduction: disentangling vs. layers "
                   f"(target $\\chi'$ = {target_chi})")
    ax_s.set_ylabel("mean bond entropy (nats)")
    panels = [ax_a, ax_s]
    if ax_m is not None:
        _ref_line(ax_m, "total_params")
        ax_m.set_ylabel(f"total parameters $M^*$\n$C(\\chi'={target_chi}) + Q(D)$")
        panels.append(ax_m)
    if ax_p is not None:
        if baseline is not None:
            ax_p.axhline(1.0, ls="--", color="crimson", lw=1.2, label="dense baseline")
        _ref_line(ax_p, "ppl_ratio")
        ax_p.set_ylabel(f"perplexity / baseline\n($\\chi'$={target_chi} layer swapped in)")
        panels.append(ax_p)
    panels[-1].set_xlabel("number of layers $L$")
    # L=0 (the no-circuit point) cannot sit on a log axis, so use symlog -- linear
    # through 0 up to 1, log beyond -- whenever a curve includes it.
    has_zero = any(int(r["n_layers"]) == 0 for r in rows)
    for ax in panels:
        if has_zero:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=-0.15)
        else:
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "disentangling_accuracy_vs_layers.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        gate_sizes = validate_gate_sizes(args.gate_sizes)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    # Requested healing modes, in a stable order ("core" before "full").
    heal_modes = [m for m in ("core", "full") if m in set(args.heal)]

    # --- weight source, and (model path only) the live layer + a perplexity probe
    model = tokenizer = device = param = orig = None
    input_ids = None
    measure_ppl = False
    baseline = float("nan")
    if args.shape:
        d_out, d_in = parse_shape(args.shape)
        weight = synthetic_weight(d_out, d_in, args.synthetic_rank,
                                  args.synthetic_noise, args.seed)
        source = f"synthetic:{d_out}x{d_in}:rank{args.synthetic_rank}"
        print(f"Synthetic layer {d_out}x{d_in} "
              f"(rank {args.synthetic_rank} + noise {args.synthetic_noise}).")
        if not args.no_perplexity:
            print("(Perplexity vs. L needs a model; unavailable with --shape.)")
    else:
        model, tokenizer, device, name, param = load_model_layer(args)
        orig = param.detach().to("cpu", torch.float32, copy=True)
        weight = orig
        d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
        source = f"{args.model}:{name}"
        print(f"Layer {name}  ({d_out}x{d_in}).")
        measure_ppl = not args.no_perplexity

    n_out, n_in = n_qubits_for(d_out), n_qubits_for(d_in)
    # L=0 is a regular point (no circuits -> the plain padded-TN truncation); it
    # is the left end of every curve unless --no-d0 drops it.
    min_L = 1 if args.no_d0 else 0
    if args.layers:
        layers = sorted({int(x) for x in args.layers if x >= min_L})
    else:
        layers = log_spaced_layers(args.max_layers, args.num_points)
        if not args.no_d0:
            layers = [0] + layers
    print(f"Register {n_out}q x {n_in}q | gate sizes {gate_sizes} | "
          f"L = {layers} | target chi' = {args.target_chi} | "
          f"{args.disentangle_target} target | {args.optimizer} | {args.tensorization} MPO")

    # M* = C(chi') + Q(D): the stored MPO parameters (constant across L, since
    # chi' and the tensorization geometry are fixed) plus the circuits' Q(D),
    # which grows with the brickwall depth. Both are pure functions of the layer
    # geometry, so they are the same numbers the disentangler records -- computed
    # here directly so every point (including resumed ones) gets M* without a
    # re-optimization. The plan is built from the original weight (build_qubit_plan
    # accounts for the power-of-two padding internally), so its leg dims -- hence
    # C(chi') -- match the disentangler's.
    _ref_plan = make_plan(weight, args.tensorization, 2, svd_cache=False, align="msb")
    classical_params = mpo_param_count(_ref_plan.out_dims, _ref_plan.in_dims,
                                       args.target_chi)

    def _params_for(k: int, L: int) -> tuple[int, int, int]:
        q = quantum_param_count(n_out, n_in, k, L, "manifold")
        return classical_params, q, classical_params + q

    def evaluate():
        from .benchmark import perplexity_over_ids
        ppl, _tokens, _secs = perplexity_over_ids(
            model, input_ids, device, args.max_length, args.stride,
            batch_size=args.ppl_batch_size, max_eval_tokens=args.eval_tokens,
            progress=False)
        return ppl

    def _ppl_of(recon: torch.Tensor) -> float:
        """Score the model with the target layer swapped for ``recon``, then restore."""
        with torch.no_grad():
            param.copy_(recon.to(dtype=param.dtype, device=param.device))
        ppl = evaluate()
        with torch.no_grad():
            param.copy_(orig.to(dtype=param.dtype, device=param.device))
        return ppl

    out_dir = _out_dir(args)
    path = out_dir / args.csv_name

    def _write(rows: list[dict]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # --- resume: reuse points already checkpointed for this same configuration.
    cur_cfg = (source, args.tensorization, args.optimizer,
               str(args.target_chi), args.disentangle_target)

    def _row_cfg(r: dict) -> tuple:
        # disentangle_target is a newer column; when absent assume the current one.
        return (str(r.get("source", "")), str(r.get("tensorization", "")),
                str(r.get("optimizer", "")), str(r.get("target_chi", "")),
                str(r.get("disentangle_target") or args.disentangle_target))

    existing = [] if args.recompute else _read_rows(path)
    # Reference rows (kind != sweep) are always recomputed and carry a different
    # geometry (the unpadded TN), so they are excluded from the swept-grid's
    # configuration check and resume bookkeeping.
    sweep_existing = [r for r in existing if (r.get("kind") or KIND_SWEEP) == KIND_SWEEP]
    conflict = next((r for r in sweep_existing if _row_cfg(r) != cur_cfg), None)
    if conflict is not None:
        print(f"error: {path} holds rows from a different configuration "
              f"({_row_cfg(conflict)} vs {cur_cfg}). Re-run with --recompute to "
              f"overwrite it, or --csv-name to write to a separate file.")
        return 2

    # A checkpointed point is "done" only if it also carries the perplexity we
    # are asking for now (cold, plus each requested heal mode), so a run can be
    # extended later with perplexity or a new heal mode.
    def _row_done(r: dict) -> bool:
        if not measure_ppl:
            return True
        if not _is_finite(r.get("ppl_ratio")):
            return False
        return all(_is_finite(r.get(f"ppl_ratio_{m}")) for m in heal_modes)

    seeded: list[dict] = []
    done_points: set[tuple[int, int]] = set()
    for r in sweep_existing:
        try:
            rk, rl = int(r["gate_size"]), int(r["n_layers"])
        except (KeyError, ValueError, TypeError):
            continue
        if _row_done(r):
            r["kind"] = KIND_SWEEP
            r["disentangle_target"] = r.get("disentangle_target") or args.disentangle_target
            if not _is_finite(r.get("total_params")):    # backfill pre-M* rows
                cp, qp, tp = _params_for(rk, rl)
                r["classical_params"], r["quantum_params"], r["total_params"] = cp, qp, tp
            seeded.append(r)
            done_points.add((rk, rl))

    cached_baseline = float("nan")
    if measure_ppl and not args.recompute:
        for r in existing:
            if _is_finite(r.get("ppl_baseline")):
                cached_baseline = float(r["ppl_baseline"])
                break

    # Reference row (unpadded TN): reuse a checkpointed one only if it matches
    # this run and, on the model path, already carries a perplexity ratio. The
    # D=0 (no-circuit) point is now an ordinary sweep point, not a reference.
    def _find_ref(kind: str):
        if args.recompute:
            return None
        for r in existing:
            if (r.get("kind") == kind
                    and str(r.get("source", "")) == source
                    and str(r.get("target_chi", "")) == str(args.target_chi)
                    and (str(r.get("disentangle_target") or args.disentangle_target)
                         == args.disentangle_target)
                    and str(r.get("optimizer", "")) == args.optimizer
                    and (not measure_ppl or _is_finite(r.get("ppl_ratio")))):
                return r
        return None
    ref_cache = {KIND_TN_NOPAD: _find_ref(KIND_TN_NOPAD)}
    need_ref_eval = measure_ppl and any(v is None for v in ref_cache.values())

    grid = [(k, L) for k in gate_sizes for L in layers]
    total = len(grid)
    reused = sum(1 for kl in grid if kl in done_points)
    to_run = total - reused
    if reused:
        print(f"Resuming: {reused}/{total} grid points already in {path.name}; "
              f"{to_run} to run.")

    # --- baseline perplexity: reuse the checkpoint's value, else measure once.
    #     Tokenize whenever any sweep point or reference still needs evaluating.
    if measure_ppl:
        if math.isfinite(cached_baseline):
            baseline = cached_baseline
            print(f"\nBaseline (dense) perplexity from checkpoint = {baseline:.4f}")
        if to_run or need_ref_eval:
            from .benchmark import BenchmarkConfig, tokenize_corpus
            load_cfg = BenchmarkConfig(
                model_id=args.model, device=args.device, dtype=args.dtype,
                trust_remote_code=args.trust_remote_code, revision=args.revision,
                dataset=args.dataset, dataset_config=args.dataset_config,
                split=args.split, text_column=args.text_column)
            input_ids = tokenize_corpus(tokenizer, load_cfg)
            if not math.isfinite(baseline):
                print(f"\nBaseline (dense) perplexity on {args.eval_tokens} tokens ...")
                baseline = evaluate()
                print(f"Baseline perplexity = {baseline:.4f}")

    # --- healing calibration set (model path, only when a heal mode is requested
    #     and there is work to do). Uses a disjoint split from the eval corpus.
    heal_batches: list = []
    if measure_ppl and heal_modes and to_run:
        from .benchmark import BenchmarkConfig, tokenize_corpus
        heal_cfg = BenchmarkConfig(
            model_id=args.model, device=args.device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code, revision=args.revision,
            dataset=args.dataset, dataset_config=args.dataset_config,
            split=args.heal_split, text_column=args.text_column)
        heal_ids = tokenize_corpus(tokenizer, heal_cfg)
        heal_window = args.heal_window or args.max_length
        heal_batches = make_heal_batches(heal_ids, heal_window,
                                         args.heal_batch_size, args.heal_tokens)
        print(f"Healing: modes {heal_modes} | {len(heal_batches)} calib batches "
              f"(<= {args.heal_tokens} tok, window {heal_window}) | "
              f"{args.heal_steps} Adam steps @ lr {args.heal_lr} "
              f"(split '{args.heal_split}')")

    def _heal_ppls(res) -> dict:
        """Healed perplexity + ratio columns for a disentangled point (per mode)."""
        cols: dict = {}
        for mode in heal_modes:
            healed, _n, _loss = heal_hybrid(
                model, name, res, args.target_chi, mode, heal_batches, device,
                steps=args.heal_steps, lr=args.heal_lr)
            hppl = _ppl_of(healed)
            hratio = hppl / baseline if baseline else float("nan")
            cols[f"ppl_{mode}"] = round(hppl, 4)
            cols[f"ppl_ratio_{mode}"] = round(hratio, 6)
        return cols

    # --- reference row: the plain TN on the *unpadded* matrix (Q(D)=0). Computed
    #     once, up front, so every checkpoint carries it. (The D=0 point is an
    #     ordinary sweep point on the curve, handled by the main loop.)
    def _ref_ppl_fields(recon):
        ppl = _ppl_of(recon) if measure_ppl else float("nan")
        ratio = ppl / baseline if measure_ppl and baseline else float("nan")
        return dict(perplexity=round(ppl, 4) if measure_ppl else "",
                    ppl_baseline=round(baseline, 4) if measure_ppl else "",
                    ppl_ratio=round(ratio, 6) if measure_ppl else "")

    def _reference_rows() -> list[dict]:
        wnorm = float(torch.linalg.norm(weight)) or 1.0
        out: list[dict] = []

        cached = ref_cache[KIND_TN_NOPAD]
        if cached is not None:
            cached["kind"] = KIND_TN_NOPAD
            out.append(cached)
        else:
            plan_np = make_plan(weight, "balanced", 2, svd_cache=False)
            recon_np, p_np = plan_compress(weight, plan_np, args.target_chi)
            ret_np = float(torch.linalg.norm(recon_np)) / wnorm
            out.append(dict(
                kind=KIND_TN_NOPAD, gate_size=-1, n_layers=0, d_out=d_out, d_in=d_in,
                n_out_qubits=n_out, n_in_qubits=n_in, target_chi=args.target_chi,
                disentangle_target=args.disentangle_target,
                tensorization="balanced(no pad)", optimizer=args.optimizer,
                accuracy=round(ret_np, 6),
                entropy=round(plan_bond_entropy(weight, plan_np), 6),
                retained=round(ret_np, 6), retained_classical=round(ret_np, 6),
                classical_params=p_np, quantum_params=0, total_params=p_np,
                sweeps_run=0, seconds=0.0, source=source,
                **_ref_ppl_fields(recon_np)))
        return out

    ref_rows = _reference_rows()

    rows: list[dict] = list(seeded)
    start = time.perf_counter()
    done = 0
    ppl_head = f"{'ppl':>9} {'x base':>7} " if measure_ppl else ""
    print(f"\n{'k':>3} {'L':>4} {'accuracy':>9} {'entropy':>9} {'retained':>9} "
          f"{'M*':>9} {ppl_head}{'sec':>6}")
    for r in ref_rows:                                  # unpadded-TN baseline
        pstr = ""
        if measure_ppl and _is_finite(r.get("ppl_ratio")):
            pstr = f"{float(r['perplexity']):>9.4f} {float(r['ppl_ratio']):>7.4f} "
        print(f"{'TN0':>3} {'-':>4} {float(r['accuracy']):>9.4f} "
              f"{float(r['entropy']):>9.4f} {float(r['retained']):>9.4f} "
              f"{int(float(r['total_params'])):>9} {pstr}{'':>6}")
    for k in gate_sizes:
        for L in layers:
            if (k, L) in done_points:
                continue
            res = disentangle(
                weight, gate_size=k, depth=L, target_chi=args.target_chi,
                target_mode=args.disentangle_target, tensorization=args.tensorization,
                optimizer=args.optimizer, sweeps=args.disentangle_sweeps,
                gd_steps=args.disentangle_gd_steps, gd_lr=args.disentangle_gd_lr,
                restarts=args.restarts, seed=args.seed,
                fast_gradient=args.fast_gradient,
                gradient_objective=args.gradient_objective)
            ppl = float("nan")
            heal_cols: dict = {}
            if measure_ppl:
                # Swap in the disentangled layer at the target bond dimension,
                # score the full model, then restore the dense weight.
                approx, _params = hybrid_weight(res, args.target_chi)
                ppl = _ppl_of(approx)
                if heal_modes:
                    heal_cols = _heal_ppls(res)
            ratio = ppl / baseline if measure_ppl and baseline else float("nan")
            cparams, qparams, tparams = _params_for(k, L)
            rows.append(dict(
                kind=KIND_SWEEP,
                gate_size=k, n_layers=L, d_out=d_out, d_in=d_in,
                n_out_qubits=n_out, n_in_qubits=n_in, target_chi=args.target_chi,
                disentangle_target=args.disentangle_target,
                tensorization=args.tensorization, optimizer=args.optimizer,
                accuracy=round(res.accuracy, 6), entropy=round(res.entropy, 6),
                retained=round(res.retained, 6),
                retained_classical=round(res.retained_classical, 6),
                classical_params=cparams, quantum_params=qparams,
                total_params=tparams,
                perplexity=round(ppl, 4) if measure_ppl else "",
                ppl_baseline=round(baseline, 4) if measure_ppl else "",
                ppl_ratio=round(ratio, 6) if measure_ppl else "",
                sweeps_run=res.sweeps_run, seconds=round(res.seconds, 2),
                source=source, **heal_cols))
            done += 1
            ppl_str = f"{ppl:>9.4f} {ratio:>7.4f} " if measure_ppl else ""
            heal_str = "".join(
                f" {m[:4]}:{float(heal_cols[f'ppl_ratio_{m}']):.4f}" for m in heal_modes
            ) if heal_cols else ""
            print(f"{k:>3} {L:>4} {res.accuracy:>9.4f} {res.entropy:>9.4f} "
                  f"{res.retained:>9.4f} {tparams:>9} {ppl_str}{res.seconds:>6.1f}{heal_str}")
            _write(rows + ref_rows)                     # checkpoint after each point

    _write(rows + ref_rows)             # persist references (and any resumed rows)

    if measure_ppl and orig is not None:               # ensure the model is pristine
        with torch.no_grad():
            param.copy_(orig.to(dtype=param.dtype, device=param.device))

    elapsed = time.perf_counter() - start
    ran_msg = f"{done} run" + (f" + {reused} reused" if reused else "")
    print(f"\n{ran_msg} = {reused + done}/{total} points in {elapsed:.0f}s. CSV: {path}")

    # Unpadded-TN reference line alongside the per-k swept trends.
    def _ref_note(kind: str, label: str) -> None:
        r = next((x for x in ref_rows if x.get("kind") == kind), None)
        if r is None:
            return
        line = (f"  {label}: accuracy {float(r['accuracy']):.4f}, "
                f"M* {int(float(r['total_params']))} params")
        if measure_ppl and _is_finite(r.get("ppl_ratio")):
            line += f", perplexity/base {float(r['ppl_ratio']):.4f}"
        print(line)
    _ref_note(KIND_TN_NOPAD, "TN only, no padding")

    for k in gate_sizes:
        ka = sorted([r for r in rows if int(r["gate_size"]) == k],
                    key=lambda r: int(r["n_layers"]))
        if len(ka) >= 2:
            lo, hi = ka[0], ka[-1]
            lo_a, hi_a = float(lo["accuracy"]), float(hi["accuracy"])
            trend = "increases" if hi_a > lo_a else "flat/decreases"
            line = (f"  k={k}: accuracy {lo_a:.4f} (L={lo['n_layers']}) -> "
                    f"{hi_a:.4f} (L={hi['n_layers']})  [{trend} with L]")
            if _is_finite(lo.get("total_params")) and _is_finite(hi.get("total_params")):
                line += (f";  M* {int(float(lo['total_params']))} -> "
                         f"{int(float(hi['total_params']))} params")
            if measure_ppl and _is_finite(lo.get("ppl_ratio")) and _is_finite(hi.get("ppl_ratio")):
                line += (f";  perplexity/base {float(lo['ppl_ratio']):.4f} -> "
                         f"{float(hi['ppl_ratio']):.4f}")
            print(line)
            for mode in heal_modes:
                key = f"ppl_ratio_{mode}"
                if _is_finite(lo.get(key)) and _is_finite(hi.get(key)):
                    print(f"       +heal({mode}): perplexity/base "
                          f"{float(lo[key]):.4f} -> {float(hi[key]):.4f}")
    if not args.no_plots:
        plot_scaling(rows, out_dir, args.target_chi, ref_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
