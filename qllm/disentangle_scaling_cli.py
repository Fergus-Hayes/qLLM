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

It writes a CSV and, if matplotlib is present, the two Fig. 3 panels.

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
    pad_to_qubits,
    quantum_param_count,
)
from .hybrid_sweep import MAX_GATE_SIZE, validate_gate_sizes
from .layer_analysis import parse_layer_info
from .qubit_mpo import make_plan

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


ROW_FIELDS = ["gate_size", "n_layers", "d_out", "d_in", "n_out_qubits",
              "n_in_qubits", "target_chi", "disentangle_target",
              "tensorization", "optimizer",
              "accuracy", "entropy", "retained", "retained_classical",
              "classical_params", "quantum_params", "total_params",
              "perplexity", "ppl_baseline", "ppl_ratio",
              "sweeps_run", "seconds", "source"]


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
                   help=f"Qubits per gate k (default: 2; 1..{MAX_GATE_SIZE}).")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Explicit brickwall depths L to sweep (overrides "
                        "--max-layers/--num-points).")
    p.add_argument("--max-layers", type=int, default=35,
                   help="Largest L (default: 35, the paper's k=2 range).")
    p.add_argument("--num-points", type=int, default=10,
                   help="Number of log-spaced L values up to --max-layers.")
    p.add_argument("--target-chi", type=int, default=1,
                   help="Target bond dimension the circuits disentangle to "
                        "(default: 1, the paper's fully-disentangled target).")
    p.add_argument("--disentangle-target", default="fixed",
                   choices=["fixed", "adaptive"],
                   help="'fixed' (default) is the paper's Eq. 4 target -- the "
                        "bond-1 truncation of the original operator.")
    p.add_argument("--tensorization", default="qubit", choices=["balanced", "qubit"],
                   help="MPO geometry (default: qubit, the paper's).")
    p.add_argument("--optimizer", default="explicit", choices=["explicit", "gradient"],
                   help="Training scheme (default: explicit, the paper's env-SVD).")
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


def plot_scaling(rows: list[dict], out_dir: Path, target_chi: int) -> None:
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

    for i, k in enumerate(ks):
        col = cmap(i / max(1, len(ks) - 1))
        lab = f"{k}q gates"
        ax_a.plot(*_pts(k, "accuracy"), marker="o", color=col, label=lab)
        ax_s.plot(*_pts(k, "entropy"), marker="s", color=col, label=lab)
        if ax_m is not None:
            ax_m.plot(*_pts(k, "total_params"), marker="D", color=col, label=lab)
        if ax_p is not None:
            ax_p.plot(*_pts(k, "ppl_ratio"), marker="^", color=col, label=lab)

    ax_a.set_ylabel("disentangling accuracy $A$  (Eq. 4)")
    ax_a.set_title(f"Fig. 3 reproduction: disentangling vs. layers "
                   f"(target $\\chi'$ = {target_chi})")
    ax_s.set_ylabel("mean bond entropy (nats)")
    panels = [ax_a, ax_s]
    if ax_m is not None:
        cvals = [float(r["classical_params"]) for r in rows
                 if _is_finite(r.get("classical_params"))]
        if cvals:
            ax_m.axhline(cvals[0], ls=":", color="gray", lw=1.2,
                         label=f"classical $C(\\chi'={target_chi})$ only")
        ax_m.set_ylabel(f"total parameters $M^*$\n$C(\\chi'={target_chi}) + Q(D)$")
        panels.append(ax_m)
    if ax_p is not None:
        if baseline is not None:
            ax_p.axhline(1.0, ls="--", color="crimson", lw=1.2, label="dense baseline")
        ax_p.set_ylabel(f"perplexity / baseline\n($\\chi'$={target_chi} layer swapped in)")
        panels.append(ax_p)
    panels[-1].set_xlabel("number of layers $L$")
    for ax in panels:
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
    layers = (sorted({int(x) for x in args.layers if x >= 1}) if args.layers
              else log_spaced_layers(args.max_layers, args.num_points))
    print(f"Register {n_out}q x {n_in}q | gate sizes {gate_sizes} | "
          f"L = {layers} | target chi' = {args.target_chi} | "
          f"{args.disentangle_target} target | {args.optimizer} | {args.tensorization} MPO")

    # M* = C(chi') + Q(D): the stored MPO parameters (constant across L, since
    # chi' and the tensorization geometry are fixed) plus the circuits' Q(D),
    # which grows with the brickwall depth. Both are pure functions of the layer
    # geometry, so they are the same numbers the disentangler records -- computed
    # here directly so every point (including resumed ones) gets M* without a
    # re-optimization.
    _ref_plan = make_plan(pad_to_qubits(weight)[0], args.tensorization, 2,
                          svd_cache=False, align="msb")  # disentangle()'s default
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
    conflict = next((r for r in existing if _row_cfg(r) != cur_cfg), None)
    if conflict is not None:
        print(f"error: {path} holds rows from a different configuration "
              f"({_row_cfg(conflict)} vs {cur_cfg}). Re-run with --recompute to "
              f"overwrite it, or --csv-name to write to a separate file.")
        return 2

    # A checkpointed point is "done" only if it also carries the perplexity we
    # are asking for now, so a --no-perplexity run can be extended with it later.
    seeded: list[dict] = []
    done_points: set[tuple[int, int]] = set()
    for r in existing:
        try:
            rk, rl = int(r["gate_size"]), int(r["n_layers"])
        except (KeyError, ValueError, TypeError):
            continue
        if not measure_ppl or _is_finite(r.get("ppl_ratio")):
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

    grid = [(k, L) for k in gate_sizes for L in layers]
    total = len(grid)
    reused = sum(1 for kl in grid if kl in done_points)
    to_run = total - reused
    if reused:
        print(f"Resuming: {reused}/{total} grid points already in {path.name}; "
              f"{to_run} to run.")

    # --- baseline perplexity: reuse the checkpoint's value, else measure once.
    if measure_ppl:
        if math.isfinite(cached_baseline):
            baseline = cached_baseline
            print(f"\nBaseline (dense) perplexity from checkpoint = {baseline:.4f}")
        if to_run:
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

    rows: list[dict] = list(seeded)
    start = time.perf_counter()
    done = 0
    ppl_head = f"{'ppl':>9} {'x base':>7} " if measure_ppl else ""
    print(f"\n{'k':>3} {'L':>4} {'accuracy':>9} {'entropy':>9} {'retained':>9} "
          f"{'M*':>9} {ppl_head}{'sec':>6}")
    for k in gate_sizes:
        for L in layers:
            if (k, L) in done_points:
                continue
            res = disentangle(
                weight, gate_size=k, depth=L, target_chi=args.target_chi,
                target_mode=args.disentangle_target, tensorization=args.tensorization,
                optimizer=args.optimizer, sweeps=args.disentangle_sweeps,
                gd_steps=args.disentangle_gd_steps, gd_lr=args.disentangle_gd_lr,
                restarts=args.restarts, seed=args.seed)
            ppl = float("nan")
            if measure_ppl:
                # Swap in the disentangled layer at the target bond dimension,
                # score the full model, then restore the dense weight.
                approx, _params = hybrid_weight(res, args.target_chi)
                with torch.no_grad():
                    param.copy_(approx.to(dtype=param.dtype, device=param.device))
                ppl = evaluate()
                with torch.no_grad():
                    param.copy_(orig.to(dtype=param.dtype, device=param.device))
            ratio = ppl / baseline if measure_ppl and baseline else float("nan")
            cparams, qparams, tparams = _params_for(k, L)
            rows.append(dict(
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
                source=source))
            done += 1
            ppl_str = f"{ppl:>9.4f} {ratio:>7.4f} " if measure_ppl else ""
            print(f"{k:>3} {L:>4} {res.accuracy:>9.4f} {res.entropy:>9.4f} "
                  f"{res.retained:>9.4f} {tparams:>9} {ppl_str}{res.seconds:>6.1f}")
            _write(rows)                                # checkpoint after each point

    if not existing and not rows:                      # nothing done, nothing cached
        _write(rows)
    elif reused and not done:                          # fully resumed: rewrite as-is
        _write(rows)

    if measure_ppl and orig is not None:               # ensure the model is pristine
        with torch.no_grad():
            param.copy_(orig.to(dtype=param.dtype, device=param.device))

    elapsed = time.perf_counter() - start
    ran_msg = f"{done} run" + (f" + {reused} reused" if reused else "")
    print(f"\n{ran_msg} = {reused + done}/{total} points in {elapsed:.0f}s. CSV: {path}")
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
    if not args.no_plots:
        plot_scaling(rows, out_dir, args.target_chi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
