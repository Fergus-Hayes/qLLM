#!/usr/bin/env python3
"""Phase 0: does a weight matrix have exploitable qubit-level structure?

For every selected layer this computes, with NO training:

* the **two-qubit mutual information** I(a;b) of W seen as an operator state on
  ``n_out + n_in`` qubits, split into the three blocks that have different
  remedies in this pipeline:
    - ``row-row``  -> sets the topology of the disentangler U
    - ``col-col``  -> sets the topology of the disentangler V
    - ``row-col``  -> the MPO site pairing / index ordering (U and V cannot fix it)
* a **matched random null**: Gaussian matrices of the same shape, padded the same
  way, so the finite-size floor (and the zero-padding artefact) is subtracted
  rather than mistaken for signal. ``mi_*_ratio`` > 1 is the only evidence of
  structure that counts.
* the **MPO bond entropy** (the quantity disentangling actually tries to reduce).
* **head alignment**: with head_dim a power of two the low ``log2(head_dim)``
  qubits are the within-head dimension and the high qubits are the head index,
  so the MI can be attributed to head-index vs within-head couplings.

The decision this feeds: if ``mi_rr_ratio`` and ``mi_cc_ratio`` sit at ~1, the
weights are pairwise structureless at the qubit level and no ansatz derived from
pairwise statistics can help -- which would itself explain why only the
(over-complete) all-to-all ansatz wins.

    python analyze_structure.py layers/ --out structure.csv --save-mi mi/
    python analyze_structure.py models/SmolLM2-135M --local-files-only \
        --types q_proj v_proj --depths 10 15 20 --crop-pow2
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import torch

from qllm.disentangler import n_qubits_for, pad_to_qubits
from qllm.layer_analysis import parse_layer_info
from qllm.qubit_mpo import make_plan, plan_bond_entropy


# --------------------------------------------------------------------------- #
# Mutual information of the operator state
# --------------------------------------------------------------------------- #
def operator_state(W: torch.Tensor) -> torch.Tensor:
    """|W> : W flattened and normalised, as a state on n_out + n_in qubits.

    Row-major flattening puts the row qubits first, most-significant bit first --
    the same convention ``apply_gate`` uses (qubit 0 is the MSB).
    """
    psi = W.reshape(-1).to(torch.float64)
    nrm = psi.norm()
    if nrm == 0:
        raise ValueError("all-zero weight matrix")
    return psi / nrm


def _entropy(rho: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(rho).clamp_min(1e-15)
    return float(-(ev * ev.log2()).sum())


def pair_mi(psi_flat: torch.Tensor, q: int):
    """(single-qubit entropies, MI matrix) for a q-qubit state."""
    psi = psi_flat.reshape((2,) * q)
    s1 = []
    for a in range(q):
        M = psi.movedim(a, 0).reshape(2, -1)
        s1.append(_entropy(M @ M.T))
    mi = torch.zeros(q, q, dtype=torch.float64)
    for a, b in itertools.combinations(range(q), 2):
        M = psi.movedim((a, b), (0, 1)).reshape(4, -1)
        val = max(0.0, s1[a] + s1[b] - _entropy(M @ M.T))
        mi[a, b] = mi[b, a] = val
    return torch.tensor(s1, dtype=torch.float64), mi


def _mean_max(vals):
    if not vals:
        return 0.0, 0.0
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.max())


def block_stats(mi: torch.Tensor, n: int, m: int) -> dict:
    """MI split into row-row / col-col / row-col blocks."""
    q = n + m
    rr = [float(mi[a, b]) for a, b in itertools.combinations(range(n), 2)]
    cc = [float(mi[a, b]) for a, b in itertools.combinations(range(n, q), 2)]
    rc = [float(mi[a, b]) for a in range(n) for b in range(n, q)]
    out = {}
    for tag, vals in (("rr", rr), ("cc", cc), ("rc", rc)):
        mean, mx = _mean_max(vals)
        out[f"mi_{tag}_mean"], out[f"mi_{tag}_max"] = mean, mx
    return out


def head_stats(mi: torch.Tensor, n: int, m: int, hb_out: int, hb_in: int) -> dict:
    """Attribute MI to head-index vs within-head qubits (MSB-first: head bits lead)."""
    Hr, Wr = list(range(hb_out)), list(range(hb_out, n))
    Hc = list(range(n, n + hb_in))
    Wc = list(range(n + hb_in, n + m))
    def cross(A, B):
        return [float(mi[a, b]) for a in A for b in B if a != b]
    def within(A):
        return [float(mi[a, b]) for a, b in itertools.combinations(A, 2)]
    groups = {
        "head_x_head_cross_reg": cross(Hr, Hc),          # block-diagonal signature
        "head_intra_reg": within(Hr) + within(Hc),       # U/V work among head bits
        "within_intra_reg": within(Wr) + within(Wc),     # U/V work among within-head bits
        "head_x_within": cross(Hr, Wr) + cross(Hc, Wc),
    }
    out = {}
    for tag, vals in groups.items():
        mean, mx = _mean_max(vals)
        out[f"mi_{tag}_mean"], out[f"mi_{tag}_max"] = mean, mx
    return out


# --------------------------------------------------------------------------- #
# Per-layer analysis
# --------------------------------------------------------------------------- #
def prepare(W: torch.Tensor, mode: str):
    """Return (matrix, n_out, n_in) either zero-padded or cropped to a power of two.

    ``pad`` matches what the sweep actually compresses, but the zero block is itself
    structure and shows up in the MI -- which is why the null is padded too.
    ``crop`` takes the largest power-of-two sub-block: artefact-free, and for a
    head_dim that is a power of two it keeps a whole number of heads.
    """
    if mode == "crop":
        r = 1 << int(math.floor(math.log2(W.shape[0])))
        c = 1 << int(math.floor(math.log2(W.shape[1])))
        return W[:r, :c].contiguous().to(torch.float32), r.bit_length() - 1, c.bit_length() - 1
    return pad_to_qubits(W)


# A null draw depends only on the matrix shape and the prepare() mode, never on the
# layer's contents, so every layer of the same shape shares one set of draws. With a
# sweep of 12 layers spanning 2 shapes that is a 6x saving, which is what makes a
# large --null-reps (and therefore a usable error bar) affordable.
_NULL_CACHE: dict = {}


def null_distribution(shape, mode: str, reps: int, seed: int) -> dict:
    """Per-draw MI block means for Gaussian matrices of ``shape``, cached by shape."""
    key = (int(shape[0]), int(shape[1]), mode, int(reps), int(seed))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    gen = torch.Generator().manual_seed(seed)
    draws = {t: [] for t in ("rr", "cc", "rc", "offdiag")}
    for _ in range(reps):
        G = torch.randn(int(shape[0]), int(shape[1]), generator=gen)
        Mg, ng, mg = prepare(G, mode)
        _s, Ig = pair_mi(operator_state(Mg), ng + mg)
        b = block_stats(Ig, ng, mg)
        offg = Ig[~torch.eye(ng + mg, dtype=bool)]
        for t in ("rr", "cc", "rc"):
            draws[t].append(b[f"mi_{t}_mean"])
        draws["offdiag"].append(float(offg.mean()))
    _NULL_CACHE[key] = draws
    return draws


def analyse(W: torch.Tensor, mode: str, head_dim: int, null_reps: int,
            seed: int) -> tuple[dict, torch.Tensor]:
    M, n, m = prepare(W, mode)
    q = n + m
    s1, mi = pair_mi(operator_state(M), q)
    row = {"n_out": n, "n_in": m, "q": q,
           "padded_rows": 1 << n, "padded_cols": 1 << m,
           "s1_mean": float(s1.mean()), "s1_min": float(s1.min()),
           "s1_max": float(s1.max())}
    row.update(block_stats(mi, n, m))

    off = mi[~torch.eye(q, dtype=bool)]
    row["mi_offdiag_mean"] = float(off.mean())
    row["mi_offdiag_max"] = float(off.max())
    row["mi_concentration"] = float(off.max() / off.mean().clamp_min(1e-15))

    # Matched null: same original shape, same prepare() -> same padding artefact.
    # The observation is a single matrix, so the uncertainty lives entirely in the
    # null; the error bar is the spread of the null draws, and the question each
    # block answers is "how many null sigmas above the null mean is this layer?".
    draws = null_distribution(W.shape, mode, null_reps, seed) if null_reps > 0 else None
    nan = float("nan")
    for tag in ("rr", "cc", "rc", "offdiag"):
        obs = row[f"mi_{tag}_mean"]
        vals = (draws or {}).get(tag) or []
        if not vals:
            for k in ("null_mi_%s_mean", "null_mi_%s_std", "mi_%s_ratio", "mi_%s_z"):
                row[k % tag] = nan
            row[f"mi_{tag}_exceed"], row[f"mi_{tag}_nreps"] = 0, 0
            continue
        v = torch.tensor(vals, dtype=torch.float64)
        mu = float(v.mean())
        sd = float(v.std(unbiased=True)) if len(vals) > 1 else nan
        row[f"null_mi_{tag}_mean"] = mu
        row[f"null_mi_{tag}_std"] = sd
        row[f"mi_{tag}_ratio"] = (obs / mu) if mu > 0 else nan
        row[f"mi_{tag}_z"] = ((obs - mu) / sd) if (sd == sd and sd > 0) else nan
        # Distribution-free companion to z: null draws that reached the observation.
        row[f"mi_{tag}_exceed"] = int(sum(1 for x in vals if x >= obs))
        row[f"mi_{tag}_nreps"] = len(vals)

    # Head attribution (only when head_dim is a power of two and fits the register).
    hb_out = n - int(round(math.log2(head_dim))) if head_dim > 1 else 0
    hb_in = m - int(round(math.log2(head_dim))) if head_dim > 1 else 0
    if head_dim > 1 and (head_dim & (head_dim - 1)) == 0 and hb_out >= 0 and hb_in >= 0:
        row["head_dim"] = head_dim
        row["head_bits_out"], row["head_bits_in"] = hb_out, hb_in
        row.update(head_stats(mi, n, m, hb_out, hb_in))
    else:
        row["head_dim"] = 0

    # What disentangling is actually trying to reduce.
    plan = make_plan(M, "qubit", 0)
    row["bond_entropy"] = float(plan_bond_entropy(M, plan))
    return row, mi


# --------------------------------------------------------------------------- #
def load_layers(source: str, args):
    """Layers from an extract_layers.py directory, or straight from a model."""
    from qllm.hybrid_sweep import HybridConfig, _layers_from_dir
    cfg = HybridConfig(model_id=source, dtype=args.dtype,
                       profile_depths=args.depths, layer_types=args.types,
                       local_files_only=args.local_files_only)
    p = Path(source)
    is_layer_dir = p.is_dir() and (
        (p / "manifest.csv").exists() or any(p.glob("*.safetensors")) or any(p.glob("*.pt")))
    if is_layer_dir:
        layers, _keep, _avail = _layers_from_dir(source, cfg)
        return layers
    from qllm.benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device
    from qllm.hybrid_sweep import _select_profile_layers
    model, _tok = load_model_and_tokenizer(
        BenchmarkConfig(model_id=source, device="cpu", dtype=args.dtype,
                        local_files_only=args.local_files_only), resolve_device("cpu"))
    layers, _keep, _avail = _select_profile_layers(model, cfg)
    return [(n, p.detach().cpu()) for n, p in layers]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="extract_layers.py output dir, OR a model id / local dir")
    ap.add_argument("--types", nargs="+", default=None, help="layer-type substrings")
    ap.add_argument("--depths", type=int, nargs="+", default=None, help="block indices")
    ap.add_argument("--out", default="structure.csv", help="output CSV")
    ap.add_argument("--save-mi", default=None,
                    help="directory to save each layer's full MI matrix as .npy")
    ap.add_argument("--crop-pow2", action="store_true",
                    help="crop to the largest power-of-two block instead of zero-padding "
                         "(artefact-free cross-check; keeps whole heads when head_dim is 2^k)")
    ap.add_argument("--head-dim", type=int, default=64,
                    help="attention head dimension for head attribution (0 to disable)")
    ap.add_argument("--null-reps", type=int, default=2,
                    help="random matrices averaged for the matched null (0 to skip)")
    ap.add_argument("--max-qubits", type=int, default=24,
                    help="skip layers needing more than this many total qubits")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", "--offline", action="store_true",
                    dest="local_files_only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layers = load_layers(args.source, args)
    if not layers:
        raise SystemExit("No layers matched.")
    mode = "crop" if args.crop_pow2 else "pad"
    mi_dir = Path(args.save_mi) if args.save_mi else None
    if mi_dir:
        mi_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(layers)} layer(s), mode={mode}, "
          f"null_reps={args.null_reps}, head_dim={args.head_dim}\n")
    rows = []
    for name, W in layers:
        W = W.detach().cpu().to(torch.float32)
        lt, depth = parse_layer_info(name)
        q = n_qubits_for(W.shape[0]) + n_qubits_for(W.shape[1])
        if q > args.max_qubits:
            print(f"  skip {name}: {q} qubits > --max-qubits {args.max_qubits}")
            continue
        row, mi = analyse(W, mode, args.head_dim, args.null_reps, args.seed)
        row.update({"param_name": name, "layer_type": lt, "depth": depth,
                    "rows": int(W.shape[0]), "cols": int(W.shape[1]), "mode": mode})
        rows.append(row)
        if mi_dir is not None:
            import numpy as np
            np.save(mi_dir / f"block{depth:03d}.{lt}.mi.npy", mi.numpy())
        def _rz(tag):
            return f"{row[f'mi_{tag}_ratio']:6.2f} (z={row[f'mi_{tag}_z']:+7.1f})"
        print(f"  {lt:<16} d{depth:<3} {row['n_out']}qx{row['n_in']}q "
              f"S={row['bond_entropy']:.3f}  rr {_rz('rr')}  cc {_rz('cc')}  "
              f"rc {_rz('rc')}")

    order = ["param_name", "layer_type", "depth", "rows", "cols", "mode",
             "n_out", "n_in", "q", "padded_rows", "padded_cols", "bond_entropy"]
    keys = order + [k for k in rows[0] if k not in order]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nWrote {len(rows)} row(s) to {Path(args.out).resolve()}")
    if mi_dir:
        print(f"MI matrices in {mi_dir.resolve()}")

    # Verdict: the Phase-0 gate, judged on significance rather than a bare ratio.
    # One-sided, Bonferroni-corrected for len(rows) layers x 3 blocks: at 36 tests
    # alpha = 0.05/36 puts the bar at z ~ 3.
    import statistics as st
    ntest = max(1, len(rows) * 3)
    zcrit = 3.0
    labels = {"rr": "U (row-row)", "cc": "V (col-col)", "rc": "MPO ordering (row-col)"}
    print(f"\n{'block':<26}{'mean ratio':>12}{'max z':>10}{'layers z>3':>12}")
    sig = {}
    for tag, lab in labels.items():
        zs = [r[f"mi_{tag}_z"] for r in rows if r[f"mi_{tag}_z"] == r[f"mi_{tag}_z"]]
        nsig = sum(1 for z in zs if z > zcrit)
        sig[tag] = nsig
        mr = st.mean(r[f"mi_{tag}_ratio"] for r in rows)
        mz = max(zs) if zs else float("nan")
        print(f"{lab:<26}{mr:>12.2f}{mz:>10.1f}{nsig:>8}/{len(rows)}")
    reps = rows[0].get("mi_rr_nreps", 0)
    print(f"(null: {reps} draws per shape; bar is z>{zcrit:.0f}, one-sided Bonferroni "
          f"over {ntest} tests)")
    hit = [labels[t] for t in ("rr", "cc", "rc") if sig[t] > 0]
    if not hit:
        print("VERDICT: no block is significantly above the matched-null floor in any "
              "layer. Ansatze derived from pairwise statistics are unlikely to help -- "
              "consider the all-to-all pruning route instead.")
    else:
        print("VERDICT: significant structure in: " + ", ".join(hit))


if __name__ == "__main__":
    raise SystemExit(main())
