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

from .disentangler import disentangle, n_qubits_for
from .hybrid_sweep import MAX_GATE_SIZE, validate_gate_sizes
from .layer_analysis import parse_layer_info

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


def load_model_weight(model_id: str, block: int, layer_type: str, device: str,
                      dtype: str, trust_remote_code: bool, revision):
    """Fetch one decoder-block weight matrix (block index + layer-type substring)."""
    from .benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device
    dev = resolve_device(device)
    cfg = BenchmarkConfig(model_id=model_id, device=device, dtype=dtype,
                          trust_remote_code=trust_remote_code, revision=revision)
    model, _tok = load_model_and_tokenizer(cfg, dev)
    matches = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        lt, depth = parse_layer_info(name)
        if depth == block and layer_type in lt:
            matches.append((name, param.detach().to("cpu", torch.float32)))
    if not matches:
        raise RuntimeError(
            f"No weight matched block {block} / '{layer_type}'. Check --block/"
            f"--layer-type against the model's parameter names.")
    if len(matches) > 1:
        names = ", ".join(n for n, _ in matches)
        print(f"(Note: {len(matches)} weights matched; using the first: {names})")
    return matches[0]


ROW_FIELDS = ["gate_size", "n_layers", "d_out", "d_in", "n_out_qubits",
              "n_in_qubits", "target_chi", "tensorization", "optimizer",
              "accuracy", "entropy", "retained", "retained_classical",
              "sweeps_run", "seconds", "source"]


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
    fig, (ax_a, ax_s) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    cmap = plt.get_cmap("viridis")
    ks = sorted(by_k)
    for i, k in enumerate(ks):
        pts = sorted(by_k[k], key=lambda r: int(r["n_layers"]))
        xs = [int(r["n_layers"]) for r in pts]
        col = cmap(i / max(1, len(ks) - 1))
        ax_a.plot(xs, [float(r["accuracy"]) for r in pts], marker="o", color=col,
                  label=f"{k}q gates")
        ax_s.plot(xs, [float(r["entropy"]) for r in pts], marker="s", color=col,
                  label=f"{k}q gates")
    ax_a.set_ylabel("disentangling accuracy $A$  (Eq. 4)")
    ax_a.set_title(f"Fig. 3 reproduction: disentangling vs. layers "
                   f"(target $\\chi'$ = {target_chi})")
    ax_s.set_ylabel("mean bond entropy (nats)")
    ax_s.set_xlabel("number of layers $L$")
    for ax in (ax_a, ax_s):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "disentangling_accuracy_vs_layers.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        gate_sizes = validate_gate_sizes(args.gate_sizes)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if args.shape:
        d_out, d_in = parse_shape(args.shape)
        weight = synthetic_weight(d_out, d_in, args.synthetic_rank,
                                  args.synthetic_noise, args.seed)
        source = f"synthetic:{d_out}x{d_in}:rank{args.synthetic_rank}"
        print(f"Synthetic layer {d_out}x{d_in} "
              f"(rank {args.synthetic_rank} + noise {args.synthetic_noise}).")
    else:
        name, weight = load_model_weight(args.model, args.block, args.layer_type,
                                         args.device, args.dtype,
                                         args.trust_remote_code, args.revision)
        d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
        source = f"{args.model}:{name}"
        print(f"Layer {name}  ({d_out}x{d_in}).")

    n_out, n_in = n_qubits_for(d_out), n_qubits_for(d_in)
    layers = (sorted({int(x) for x in args.layers if x >= 1}) if args.layers
              else log_spaced_layers(args.max_layers, args.num_points))
    print(f"Register {n_out}q x {n_in}q | gate sizes {gate_sizes} | "
          f"L = {layers} | target chi' = {args.target_chi} | "
          f"{args.disentangle_target} target | {args.optimizer} | {args.tensorization} MPO")

    out_dir = _out_dir(args)
    path = out_dir / args.csv_name
    rows: list[dict] = []
    total = len(gate_sizes) * len(layers)
    start = time.perf_counter()
    done = 0
    print(f"\n{'k':>3} {'L':>4} {'accuracy':>9} {'entropy':>9} {'retained':>9} {'sec':>6}")
    for k in gate_sizes:
        for L in layers:
            res = disentangle(
                weight, gate_size=k, depth=L, target_chi=args.target_chi,
                target_mode=args.disentangle_target, tensorization=args.tensorization,
                optimizer=args.optimizer, sweeps=args.disentangle_sweeps,
                gd_steps=args.disentangle_gd_steps, gd_lr=args.disentangle_gd_lr,
                restarts=args.restarts, seed=args.seed)
            rows.append(dict(
                gate_size=k, n_layers=L, d_out=d_out, d_in=d_in,
                n_out_qubits=n_out, n_in_qubits=n_in, target_chi=args.target_chi,
                tensorization=args.tensorization, optimizer=args.optimizer,
                accuracy=round(res.accuracy, 6), entropy=round(res.entropy, 6),
                retained=round(res.retained, 6),
                retained_classical=round(res.retained_classical, 6),
                sweeps_run=res.sweeps_run, seconds=round(res.seconds, 2),
                source=source))
            done += 1
            print(f"{k:>3} {L:>4} {res.accuracy:>9.4f} {res.entropy:>9.4f} "
                  f"{res.retained:>9.4f} {res.seconds:>6.1f}")
            out_dir.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:      # checkpoint after each point
                w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
                w.writeheader()
                w.writerows(rows)

    elapsed = time.perf_counter() - start
    print(f"\n{done}/{total} points in {elapsed:.0f}s. CSV: {path}")
    for k in gate_sizes:
        ka = [r for r in rows if r["gate_size"] == k]
        if len(ka) >= 2:
            lo, hi = ka[0], ka[-1]
            trend = "increases" if hi["accuracy"] > lo["accuracy"] else "flat/decreases"
            print(f"  k={k}: accuracy {lo['accuracy']:.4f} (L={lo['n_layers']}) -> "
                  f"{hi['accuracy']:.4f} (L={hi['n_layers']})  [{trend} with L]")
    if not args.no_plots:
        plot_scaling(rows, out_dir, args.target_chi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
