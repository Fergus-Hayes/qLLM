#!/usr/bin/env python3
"""Phase A: is the compression target wrong?

The sweep minimizes ``||W - W'||_F``, but capability depends on the layer's output
``Wx`` over the inputs that actually occur. That error is ``tr[(W-W') H (W-W')^T]``
with ``H = E[x x^T]``, so the whole input distribution enters as one matrix.

    # 1. capture H over calibration text (needs the model; ~1 min for SmolLM2-135M)
    python activation_aware.py capture models/SmolLM2-135M --local-files-only \
        --types k_proj o_proj q_proj v_proj --depths 10 15 20 --out cov/

    # 2. re-score compressions under both metrics (needs only weights + H)
    python activation_aware.py score --layers-dir models/SmolLM2-135M/layers \
        --cov cov/ --out activation_aware.csv

``score`` reports, per layer and bond dimension, the Frobenius error the sweep has
been minimizing next to the output error that actually matters, their ratio (the
headroom), and -- at matched parameter count -- the best plain low-rank
approximation against the best *whitened* one, which is the exact optimum for the
output error and the reference any activation-aware scheme must beat.
"""
import argparse
import csv
import math
from pathlib import Path

import torch

from qllm.activation_stats import (
    _sqrt_and_inv,
    covariance_spectrum,
    input_covariance,
    low_rank_frobenius,
    low_rank_whitened,
    magic_matched_null,
    output_relative_error,
    sparse_lowrank_params,
    sparse_lowrank_whitened,
    stabilizer_renyi_entropy,
)
from qllm.compactifai import relative_error
from qllm.layer_analysis import parse_layer_info
from qllm.qubit_mpo import make_plan, plan_compress


def _load_layers(args):
    from qllm.hybrid_sweep import HybridConfig, _layers_from_dir
    cfg = HybridConfig(model_id=args.layers_dir or "", dtype="float32",
                       profile_depths=args.depths, layer_types=args.types)
    layers, _k, _a = _layers_from_dir(args.layers_dir, cfg)
    return layers


# --------------------------------------------------------------------------- #
def _calibration_ids(args):
    """Token ids for calibration, from the cheapest source that works.

    ``H = E[x x^T]`` needs activations, so it needs token *ids* -- never a tokenizer
    and never the Hub. ``--token-ids`` therefore skips both entirely; the tokenizer
    is imported only if text still has to be turned into ids.
    """
    if args.token_ids:
        obj = torch.load(args.token_ids, map_location="cpu")
        if isinstance(obj, dict):
            obj = next(iter(obj.values()))
        ids = obj.long()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        print(f"Calibration ids from {args.token_ids}: {tuple(ids.shape)}")
        return ids

    from transformers import AutoTokenizer
    src = args.tokenizer or args.model
    try:
        tok = AutoTokenizer.from_pretrained(src, local_files_only=args.local_files_only)
    except Exception as exc:                                     # noqa: BLE001
        raise SystemExit(
            f"Could not load a tokenizer from '{src}':\n  {type(exc).__name__}: {exc}\n\n"
            f"Capturing H needs token ids, not a tokenizer. Any of these unblocks it:\n"
            f"  --tokenizer HuggingFaceTB/SmolLM2-135M --local-files-only   "
            f"(use the copy in your HF cache)\n"
            f"  --token-ids ids.pt                                         "
            f"(pre-tokenized ids; no tokenizer, no dataset)\n"
            f"  --text-file corpus.txt                                     "
            f"(local text, still needs a tokenizer)\n"
            f"  pip install sentencepiece                                  "
            f"(some tokenizers need it to convert slow -> fast)"
        ) from exc
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if args.text_file:
        text = Path(args.text_file).read_text()
        ids = tok(text, return_tensors="pt").input_ids
        print(f"Calibration ids from {args.text_file}: {tuple(ids.shape)}")
        return ids

    from qllm.benchmark import BenchmarkConfig, tokenize_corpus
    bcfg = BenchmarkConfig(model_id=args.model, dataset=args.dataset,
                           dataset_config=args.dataset_config, split=args.split,
                           local_files_only=args.local_files_only)
    return tokenize_corpus(tok, bcfg)


def cmd_capture(args):
    from transformers import AutoModelForCausalLM

    from qllm.benchmark import DTYPE_MAP, resolve_device
    from qllm.compactifai_heal import make_heal_batches
    from qllm.hybrid_sweep import HybridConfig, _select_profile_layers

    device = resolve_device(args.device)
    # Load the model directly rather than through load_model_and_tokenizer: the
    # tokenizer is not needed to capture activations, and a local save with missing
    # or unconvertible tokenizer files must not block the capture.
    print(f"Loading '{args.model}'"
          f"{' (local files only)' if args.local_files_only else ''} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE_MAP.get(args.dtype, torch.float32),
        local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    hcfg = HybridConfig(model_id=args.model, profile_depths=args.depths,
                        layer_types=args.types)
    layers, _keep, _avail = _select_profile_layers(model, hcfg)
    if not layers:
        raise SystemExit("No layers matched --types/--depths.")
    names = [n for n, _p in layers]
    print(f"Capturing H = E[x x^T] for {len(names)} layer(s).")

    ids = _calibration_ids(args)
    batches = make_heal_batches(ids, args.window, args.batch_size, args.calib_tokens)
    if not batches:
        raise SystemExit("No calibration batches; raise --calib-tokens or lower --window.")
    tok_n = sum(b.numel() for b in batches)
    print(f"Calibration: {len(batches)} batch(es), {tok_n:,} tokens "
          f"(window {args.window}, batch {args.batch_size})")

    def prog(i, n):
        if i % max(1, n // 10) == 0 or i == n:
            print(f"  {i}/{n} batches", flush=True)

    cov, counts = input_covariance(model, names, batches, device=device, progress=prog)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    man = []
    for name, h in cov.items():
        lt, dep = parse_layer_info(name)
        fn = f"block{dep:03d}.{lt}.cov.safetensors"
        save_file({name: h.to(torch.float32).contiguous()}, str(out / fn))
        spec = covariance_spectrum(h)
        man.append({"file": fn, "param_name": name, "layer_type": lt, "depth": dep,
                    "tokens": counts[name], **{k: spec[k] for k in
                    ("dim", "stable_rank", "rank90", "rank99", "participation")}})
        print(f"  {lt:<18} d{dep:<3} dim={spec['dim']:<5} stable_rank={spec['stable_rank']:8.1f}"
              f"  rank90={spec['rank90']:<5} rank99={spec['rank99']:<5}")
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(man)
    print(f"\nSaved {len(man)} covariance matrices to {out.resolve()}")


# --------------------------------------------------------------------------- #
def cmd_score(args):
    from safetensors.torch import load_file
    covdir = Path(args.cov)
    cov = {}
    for r in csv.DictReader(open(covdir / "manifest.csv")):
        cov[r["param_name"]] = next(iter(load_file(covdir / r["file"]).values()))
    layers = _load_layers(args)
    layers = [(n, w) for n, w in layers if n in cov]
    if not layers:
        raise SystemExit("No layer had a matching covariance; check --cov / --layers-dir.")

    rows = []
    print(f"{'layer':<17}{'d':>3}{'chi':>5}{'frob err':>11}{'output err':>12}"
          f"{'headroom':>10}{'LR frob':>10}{'LR whit':>10}{'gain':>8}")
    for name, W in layers:
        lt, dep = parse_layer_info(name)
        H = cov[name]
        spec = covariance_spectrum(H)
        plan = make_plan(W, "qubit", 0)
        for chi in args.chi:
            approx, cpar = plan_compress(W, plan, chi)
            fe = float(relative_error(W, approx))
            oe = output_relative_error(W, approx, H)
            # Matched-parameter plain low-rank, scored by the SAME output metric:
            # Frobenius-optimal vs the exact H-weighted optimum (whitened).
            # Rank whose cost r*(m+n) matches the MPO's classical parameters. The
            # floor at 1 means that at small chi' the rank-1 map is far MORE
            # expensive than the MPO, so the MPO-vs-low-rank comparison is not
            # matched there and is flagged; the whitening gain is unaffected, being
            # frobenius-optimal vs whitened-optimal at the SAME rank.
            per_rank = W.shape[0] + W.shape[1]
            r = max(1, int(cpar // per_rank))
            lr_params = r * per_rank
            lr_f = output_relative_error(W, low_rank_frobenius(W, r), H)
            lr_w = output_relative_error(W, low_rank_whitened(W, H, r, args.damp), H)
            rows.append(dict(layer_type=lt, depth=dep, chi=chi, mpo_params=int(cpar),
                             lr_rank=r, lr_params=int(lr_params),
                             lr_param_ratio=round(lr_params / cpar, 2) if cpar else float("nan"),
                             matched=bool(lr_params <= 1.25 * cpar), frob_err=round(fe, 6), output_err=round(oe, 6),
                             headroom=round(fe / oe, 4) if oe > 0 else float("nan"),
                             lowrank_frob_outerr=round(lr_f, 6),
                             lowrank_whitened_outerr=round(lr_w, 6),
                             whitening_gain=round(lr_f / lr_w, 4) if lr_w > 0 else float("nan"),
                             stable_rank=round(spec["stable_rank"], 2),
                             rank90=spec["rank90"], rank99=spec["rank99"]))
            flag = "" if lr_params <= 1.25 * cpar else f"  [LR uses {lr_params/cpar:.0f}x params]"
            print(f"{lt:<17}{dep:>3}{chi:>5}{fe:>11.4f}{oe:>12.4f}"
                  f"{fe/oe if oe>0 else float('nan'):>10.2f}{lr_f:>10.4f}{lr_w:>10.4f}"
                  f"{lr_f/lr_w if lr_w>0 else float('nan'):>8.2f}{flag}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    import statistics as st
    sr = st.mean(r["stable_rank"] for r in rows)
    dim = int(st.mean(covariance_spectrum(cov[n])["dim"] for n, _ in layers))
    print(f"\nH: mean stable rank {sr:.1f} of {dim} input dimensions")

    # Aggregate PER chi, never across it: how much data-awareness buys depends on
    # the budget -- there is nothing to exploit at chi'=1 and a lot at large chi'.
    print(f"\n{'chi':>5}{'headroom':>12}{'whitening gain':>18}")
    best_gain, best_chi = 0.0, None
    for chi in sorted({r["chi"] for r in rows}):
        sel = [r for r in rows if r["chi"] == chi]
        hr = st.mean(r["headroom"] for r in sel if r["headroom"] == r["headroom"])
        wg = st.mean(r["whitening_gain"] for r in sel
                     if r["whitening_gain"] == r["whitening_gain"])
        nm = sum(1 for r in sel if not r["matched"])
        note = f"   (MPO vs LR not parameter-matched in {nm}/{len(sel)})" if nm else ""
        print(f"{chi:>5}{hr:>12.2f}x{wg:>17.2f}x{note}")
        if wg > best_gain:
            best_gain, best_chi = wg, chi

    # The whitening gain is the load-bearing number. Headroom compares the SAME
    # approximation under two metrics, so it only departs from 1 when the
    # compression error happens to sit in low-energy input directions; a truncation
    # error that is roughly isotropic leaves it at 1 even when H is very
    # anisotropic. The gain instead compares the best data-blind approximation with
    # the best data-aware one at equal parameters, which is the question being asked.
    if best_gain < 1.2:
        print(f"\nVERDICT: whitening buys at most {best_gain:.2f}x at any tested chi' -- "
              f"H is too close to isotropic for activation-aware compression to pay "
              f"here. Phase B/C not worth building.")
    else:
        print(f"\nVERDICT: data-aware compression wins {best_gain:.2f}x at equal "
              f"parameters (best at chi'={best_chi}) -- an output-error objective is "
              f"worth building (Phase B/C).")


# --------------------------------------------------------------------------- #
def cmd_curve(args):
    """Output error against parameter count, for each method on its own grid.

    No budget matching is needed: both families are sampled densely and laid on a
    common parameter axis, so "what does budget X buy" is read off directly.

    A whitened rank-r map costs exactly ``r*(m+n)``, the same as any rank-r
    factorization -- ``W' = trunc(W H^(1/2)) H^(-1/2)`` is still rank r, and the
    ``H^(-1/2)`` is absorbed into the right factor at build time. The whitening is a
    compile-time change of objective, not a runtime cost, and because it is the
    exact optimum of the H-weighted error it is a hard lower bound on what *any*
    rank-r approximation can achieve on the output metric.

    ``sparse`` and ``sparse+low-rank-w`` are the classes that can hold the one
    structure the diagnostics did find. Their nonzeros are charged for their indices
    as well as their values (see :func:`sparse_lowrank_params`), so sparsity gets no
    more of a free ride than a quantum gate does; the budget axis stays one number
    that every method is spending from.
    """
    from safetensors.torch import load_file

    from qllm.compactifai import log_spaced_ints

    covdir = Path(args.cov)
    cov = {}
    for r in csv.DictReader(open(covdir / "manifest.csv")):
        cov[r["param_name"]] = next(iter(load_file(covdir / r["file"]).values()))
    layers = [(n, w) for n, w in _load_layers(args) if n in cov]
    if not layers:
        raise SystemExit("No layer had a matching covariance.")

    rows = []
    for name, W in layers:
        lt, dep = parse_layer_info(name)
        H = cov[name]
        m, n = int(W.shape[0]), int(W.shape[1])
        dense = m * n
        plan = make_plan(W, "qubit", 0)

        ranks = sorted(set(log_spaced_ints(1, min(m, n), args.points)))
        for r in ranks:
            for tag, fn in (("low-rank", low_rank_frobenius),
                            ("low-rank-whitened", None)):
                approx = (low_rank_whitened(W, H, r, args.damp) if fn is None
                          else fn(W, r))
                rows.append(dict(layer_type=lt, depth=dep, method=tag, knob=r,
                                 params=r * (m + n), dense=dense,
                                 param_frac=round(r * (m + n) / dense, 5),
                                 output_err=round(output_relative_error(W, approx, H), 6),
                                 frob_err=round(float(relative_error(W, approx)), 6)))
        for chi in sorted(set(log_spaced_ints(1, plan.max_chi, args.points))):
            approx, cpar = plan_compress(W, plan, chi)
            rows.append(dict(layer_type=lt, depth=dep, method="mpo", knob=chi,
                             params=int(cpar), dense=dense,
                             param_frac=round(cpar / dense, 5),
                             output_err=round(output_relative_error(W, approx, H), 6),
                             frob_err=round(float(relative_error(W, approx)), 6)))

        # Sparse, and sparse on top of the whitened low-rank. The split between the
        # two budgets is a free knob, so it is swept rather than guessed: every
        # (rank, nnz) pair lands on the same parameter axis and the budget table
        # below picks the best split for each budget by itself.
        # The grids stop at the largest budget the table reads: a point costing more
        # than that can never be selected, and the per-row weighted refit gets
        # expensive exactly there (its solves are O(nnz_per_row^3) per row).
        cap = max(args.budgets) * dense
        k_max = max(1, min(n, int(cap / sparse_lowrank_params(
            m, n, 0, 1, args.value_bits, args.index_bits))))
        nnzs = sorted(set(log_spaced_ints(1, k_max, args.sparse_points)))
        sp_ranks = [0] + sorted(set(log_spaced_ints(1, min(m, n), args.sparse_points)))
        n_sp = 0
        for r in sp_ranks:
            for k in nnzs:
                cost = sparse_lowrank_params(m, n, r, k, args.value_bits,
                                             args.index_bits)
                if cost > cap:
                    continue
                approx, _inf = sparse_lowrank_whitened(
                    W, H, r, k, args.damp, args.sparse_iters)
                n_sp += 1
                rows.append(dict(
                    layer_type=lt, depth=dep,
                    method="sparse" if r == 0 else "sparse+low-rank-w",
                    knob=(k if r == 0 else r * 10000 + k), params=round(cost, 1),
                    dense=dense, param_frac=round(cost / dense, 5),
                    output_err=round(output_relative_error(W, approx, H), 6),
                    frob_err=round(float(relative_error(W, approx)), 6)))
        print(f"  {lt:<18} d{dep:<3} {m}x{n}: "
              f"{len(ranks)} ranks + {args.points} chi + "
              f"{n_sp} sparse", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    # What each budget actually buys: the best point a method reaches without
    # exceeding it. Budgets are fractions of the dense layer.
    import statistics as st
    methods = ["mpo", "low-rank", "low-rank-whitened", "sparse", "sparse+low-rank-w"]
    print(f"\nBest output error at or under a parameter budget "
          f"(mean over {len(layers)} layers)")
    print(f"{'budget':>10}" + "".join(f"{mth:>21}" for mth in methods))
    best_at = {}
    for frac in args.budgets:
        cells = []
        for mth in methods:
            vals = []
            for name, W in layers:
                lt, dep = parse_layer_info(name)
                cand = [r["output_err"] for r in rows
                        if r["layer_type"] == lt and r["depth"] == dep
                        and r["method"] == mth and r["param_frac"] <= frac]
                if cand:
                    vals.append(min(cand))
            cells.append(st.mean(vals) if vals else float("nan"))
        best_at[frac] = dict(zip(methods, cells))
        line = f"{frac:>9.1%}" + "".join(f"{c:>21.4f}" for c in cells)
        print(line)
    print("\n(low-rank-whitened is the exact optimum of the output error over all "
          "rank-r maps,\n so it lower-bounds every rank-r method; it costs the same "
          "r*(m+n) as plain low-rank.)")
    idx_bits = (args.index_bits if args.index_bits is not None
                else max(1, int(math.ceil(math.log2(max(2, max(int(W.shape[1])
                                                               for _n, W in layers)))))))
    print(f"(a sparse nonzero is charged "
          f"{1 + idx_bits / args.value_bits:.3f} parameter-equivalents: one "
          f"{args.value_bits}-bit value\n plus a {idx_bits}-bit column index. "
          f"Values only would be 1.000.)")

    # Does adding sparsity to the whitened low-rank actually buy anything, at a
    # budget it has to pay for? This is the question the build was made to answer.
    print("\nSparse gain over whitened low-rank, at equal total budget")
    print(f"{'budget':>10}{'LR-whitened':>16}{'best sparse mix':>18}{'gain':>10}")
    fired = []
    for frac in args.budgets:
        lr = best_at[frac]["low-rank-whitened"]
        cand = [v for k, v in best_at[frac].items()
                if k in ("sparse", "sparse+low-rank-w") and v == v]
        mix = min(cand) if cand else float("nan")
        # A budget no method can afford says nothing; skip it rather than score it.
        gain = lr / mix if (lr == lr and mix == mix and mix > 0) else float("nan")
        if gain == gain and gain > 1.05:
            fired.append((frac, gain))
        print(f"{frac:>9.1%}{lr:>16.4f}{mix:>18.4f}{gain:>9.2f}x")
    # The question is per layer, not on average: the hypothesis under test is that
    # *some* layers want a different class at *some* error threshold, and a mean
    # over layers is exactly the statistic that would hide it.
    print("\nPer layer: best sparse mix vs whitened low-rank (gain at each budget)")
    print(f"{'layer':<20}{'depth':>6}" + "".join(f"{f:>9.1%}" for f in args.budgets))
    per_layer = {}
    for name, W in layers:
        lt, dep = parse_layer_info(name)
        cells = []
        for frac in args.budgets:
            def _best(mths):
                c = [r["output_err"] for r in rows
                     if r["layer_type"] == lt and r["depth"] == dep
                     and r["method"] in mths and r["param_frac"] <= frac]
                return min(c) if c else float("nan")
            lr = _best(("low-rank-whitened",))
            mix = _best(("sparse", "sparse+low-rank-w"))
            cells.append(lr / mix if (lr == lr and mix == mix and mix > 0)
                         else float("nan"))
        per_layer[(lt, dep)] = cells
        print(f"{lt:<20}{dep:>6}" + "".join(
            ("     n/a" if c != c else f"{c:>8.2f}x") for c in cells))

    if fired:
        f, g = max(fired, key=lambda t: t[1])
        won = sorted((k for k, v in per_layer.items()
                      if any(c == c and c > 1.05 for c in v)),
                     key=lambda k: -max(c for c in per_layer[k] if c == c))
        print(f"\nVERDICT: sparsity pays -- up to {g:.2f}x lower output error than "
              f"whitened low-rank\n  at the same total parameter count (best at the "
              f"{f:.1%} budget), indices charged.\n  It wins on {len(won)}/"
              f"{len(per_layer)} layers"
              + (f", led by " + ", ".join(f"{lt} d{d}" for lt, d in won[:3])
                 if won else "") + ".")
    else:
        won = [k for k, v in per_layer.items()
               if any(c == c and c > 1.05 for c in v)]
        print("\nVERDICT: sparsity does not pay on average. Whitened low-rank is "
              "within 5% of the\n  best sparse mix at every budget, once nonzeros "
              f"are charged for their indices.\n  Per layer it still wins on "
              f"{len(won)}/{len(per_layer)}"
              + (": " + ", ".join(f"{lt} d{d}" for lt, d in won[:5]) if won
                 else " -- nowhere.") + "")


# --------------------------------------------------------------------------- #
def _crop_pow2(v: torch.Tensor):
    """Largest power-of-two prefix of a vector, and its qubit count."""
    q = int(math.floor(math.log2(v.numel())))
    return v[: 1 << q].contiguous(), q


def _participation(v: torch.Tensor) -> float:
    """Effective number of entries carrying the vector's energy (inverse Simpson).

    Equals ``len(v)`` for a perfectly spread vector and 1 for a spike, so
    ``participation / len`` is a scale-free sparsity score.
    """
    p = (v.double() ** 2)
    tot = float(p.sum())
    if tot <= 0:
        return float("nan")
    p = p / tot
    return float(1.0 / (p ** 2).sum())


def _bond_entropy(v: torch.Tensor) -> float:
    """Entanglement entropy of a vector across a balanced qubit cut (bits).

    A vector that factorizes over the qubit register has entropy 0 and can be
    stored as a small MPS / produced by a shallow circuit; a generic vector is
    near-maximal. This is what decides whether the important subspace can be
    encoded for less than one parameter per entry.
    """
    w, q = _crop_pow2(v)
    nrm = float(torch.linalg.norm(w.double()))
    if nrm <= 0 or q < 2:
        return float("nan")
    a = q // 2
    m = (w.double() / nrm).reshape(1 << a, -1)
    sv = torch.linalg.svdvals(m).clamp_min(0)
    p = (sv ** 2)
    p = p[p > 1e-15]
    return float(-(p * p.log2()).sum())


def _head_share(v: torch.Tensor, head_dim: int) -> float:
    """Energy fraction in the single strongest head block.

    Only meaningful where the register really is split into attention heads. MLP
    projections have no head structure, so pass ``head_dim=0`` there and this
    returns NaN rather than an arbitrary blocking of the coordinates.
    """
    if head_dim <= 1:
        return float("nan")
    n = v.numel() // head_dim
    if n < 2:
        return float("nan")
    e = (v.double()[: n * head_dim] ** 2).reshape(n, head_dim).sum(1)
    tot = float(e.sum())
    return float(e.max() / tot) if tot > 0 else float("nan")


def _null_stats(dim: int, head_dim: int, reps: int, seed: int, cache={}):
    """Same measures on Haar-random unit vectors of the same length."""
    key = (dim, head_dim, reps, seed)
    if key in cache:
        return cache[key]
    g = torch.Generator().manual_seed(seed)
    pr, be, hs = [], [], []
    for _ in range(reps):
        v = torch.randn(dim, generator=g, dtype=torch.float64)
        v = v / torch.linalg.norm(v)
        pr.append(_participation(v))
        be.append(_bond_entropy(v))
        hs.append(_head_share(v, head_dim))
    import statistics as st
    hs = [x for x in hs if x == x]
    out = {"participation": st.mean(pr), "bond_entropy": st.mean(be),
           "head_share": st.mean(hs) if hs else float("nan")}
    cache[key] = out
    return out


def cmd_vectors(args):
    """Are the top singular vectors of W H^(1/2) structured?

    Whitened low-rank is the exact optimum over rank-r maps, so beating it means
    encoding the same dominant subspace for less than the r*(m+n) parameters a
    factorization costs. That is possible only if the leading singular vectors are
    themselves structured -- concentrated on few entries, confined to a head block,
    or low-entanglement across the qubit register. Each is measured against a
    Haar-random null of the same length: a ratio near 1 means the vector looks
    generic and nothing structured can encode it more cheaply.
    """
    from safetensors.torch import load_file

    covdir = Path(args.cov)
    cov = {}
    for r in csv.DictReader(open(covdir / "manifest.csv")):
        cov[r["param_name"]] = next(iter(load_file(covdir / r["file"]).values()))
    layers = [(n, w) for n, w in _load_layers(args) if n in cov]
    if not layers:
        raise SystemExit("No layer had a matching covariance.")

    rows = []
    print(f"{'layer':<17}{'d':>3}{'side':>7}{'rank':>6}{'sparsity':>11}"
          f"{'bond ent':>11}{'head':>8}   (x null)")
    for name, W in layers:
        lt, dep = parse_layer_info(name)
        S, _inv = _sqrt_and_inv(cov[name], args.damp)
        u, sv, vh = torch.linalg.svd(W.double() @ S, full_matrices=False)
        for side, mat in (("left(U)", u.T), ("right(V)", vh)):
            dim = mat.shape[1]
            null = _null_stats(dim, args.head_dim, args.null_reps, args.seed)
            for r in args.ranks:
                r = min(r, mat.shape[0])
                vecs = [mat[i] for i in range(r)]
                pr = sum(_participation(v) for v in vecs) / r
                be = sum(_bond_entropy(v) for v in vecs) / r
                hs = sum(_head_share(v, args.head_dim) for v in vecs) / r
                kept = (1 << int(math.floor(math.log2(dim)))) / dim
                rows.append(dict(layer_type=lt, depth=dep, side=side, rank=r, dim=dim,
                                 crop_kept=round(kept, 3),
                                 participation=round(pr, 2),
                                 participation_ratio=round(pr / null["participation"], 4),
                                 bond_entropy=round(be, 4),
                                 bond_entropy_ratio=round(be / null["bond_entropy"], 4),
                                 head_share=round(hs, 4),
                                 head_share_ratio=round(hs / null["head_share"], 4)))
                print(f"{lt:<17}{dep:>3}{side:>7}{r:>6}"
                      f"{pr / null['participation']:>10.2f}x"
                      f"{be / null['bond_entropy']:>10.2f}x"
                      f"{hs / null['head_share']:>7.2f}x")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    import statistics as st

    def _m(sel, key):
        v = [r[key] for r in sel if r[key] == r[key]]
        return st.mean(v) if v else float("nan")

    # Broken down by layer type: the question is whether SOME layers admit a
    # cheaper-than-r*(m+n) encoding, not whether the average does.
    print(f"\n{'layer type':<20}{'sparsity':>11}{'bond entropy':>15}{'head conc.':>13}"
          f"{'best bond ent @ rank>=4':>26}")
    verdicts = {}
    for lt in sorted({r["layer_type"] for r in rows}):
        sel = [r for r in rows if r["layer_type"] == lt]
        deep = [r["bond_entropy_ratio"] for r in sel if r["rank"] >= 4]
        best = min(deep) if deep else float("nan")
        print(f"{lt:<20}{_m(sel, 'participation_ratio'):>10.2f}x"
              f"{_m(sel, 'bond_entropy_ratio'):>14.2f}x"
              f"{_m(sel, 'head_share_ratio'):>12.2f}x{best:>25.2f}x")
        verdicts[lt] = best
    print("(sparsity and bond entropy BELOW 1 mean structure; head concentration "
          "ABOVE 1 means structure.\n head concentration is NaN where --head-dim 0 "
          "disabled it, e.g. MLP layers with no heads.)")

    # Only bond entropy at a USEFUL rank decides whether a circuit/ancilla ansatz
    # can pay: sparsity is a classical encoding, and rank 1 is too inaccurate to
    # matter. A ratio here of ~0.4 or below is the threshold worth chasing.
    good = {lt: b for lt, b in verdicts.items() if b == b and b <= 0.4}
    print("\nDeciding quantity -- lowest bond entropy at rank >= 4, per layer type:")
    if good:
        print("  candidates for a circuit / ancilla ansatz: "
              + ", ".join(f"{lt} ({b:.2f}x)" for lt, b in sorted(good.items())))
        print("  these vectors are low-entanglement at a rank that matters, so they "
              "could plausibly\n  be produced by a shallow circuit for less than one "
              "parameter per entry.")
    else:
        b = min((v for v in verdicts.values() if v == v), default=float("nan"))
        print(f"  none below 0.4x (best {b:.2f}x). At the ranks that deliver useful "
              f"MSE the dominant\n  vectors are near-generic, so a shallow circuit "
              f"cannot encode them cheaply and the\n  saving an ancilla ansatz could "
              f"offer is small.")


# --------------------------------------------------------------------------- #
def _magic_null(dim: int, reps: int, seed: int, cache={}):
    """Mean M2 of Haar-random real vectors of this length."""
    key = (dim, reps, seed)
    if key in cache:
        return cache[key]
    g = torch.Generator().manual_seed(seed)
    import statistics as st
    vals = [stabilizer_renyi_entropy(torch.randn(dim, generator=g, dtype=torch.float64))
            for _ in range(reps)]
    cache[key] = st.mean(vals)
    return cache[key]


def cmd_magic(args):
    """Do the whitened singular vectors have Clifford (stabilizer) structure?

    Entanglement entropy provably cannot answer this: a random stabilizer state has
    near-maximal entanglement across any cut yet costs ZERO continuous parameters,
    so the `vectors` diagnostic reads it as generic. The stabilizer Renyi entropy
    M2 closes that gap -- it is exactly 0 for a stabilizer state and large for a
    generic one.

    A ratio near 0 would mean the dominant subspace is Clifford-like and could be
    prepared essentially for free, which is precisely the class a measurement-based
    or ancilla-assisted ansatz reaches and the earlier diagnostics were blind to.
    """
    from safetensors.torch import load_file

    covdir = Path(args.cov)
    cov = {}
    for r in csv.DictReader(open(covdir / "manifest.csv")):
        cov[r["param_name"]] = next(iter(load_file(covdir / r["file"]).values()))
    layers = [(n, w) for n, w in _load_layers(args) if n in cov]
    if not layers:
        raise SystemExit("No layer had a matching covariance.")

    rows = []
    print(f"{'layer':<17}{'d':>3}{'side':>9}{'rank':>6}{'M2':>8}{'haar':>8}"
          f"{'matched':>9}{'vs haar':>9}{'vs matched':>12}")
    for name, W in layers:
        lt, dep = parse_layer_info(name)
        S, _inv = _sqrt_and_inv(cov[name], args.damp)
        u, sv, vh = torch.linalg.svd(W.double() @ S, full_matrices=False)
        for side, mat in (("left(U)", u.T), ("right(V)", vh)):
            dim = mat.shape[1]
            null = _magic_null(dim, args.null_reps, args.seed)
            for r in args.ranks:
                r = min(r, mat.shape[0])
                vals = [stabilizer_renyi_entropy(mat[i]) for i in range(r)]
                mn = [magic_matched_null(mat[i], args.matched_reps, args.seed)
                      for i in range(r)]
                m2 = sum(vals) / r
                mnull = sum(mn) / r
                rows.append(dict(layer_type=lt, depth=dep, side=side, rank=r, dim=dim,
                                 m2_bits=round(m2, 4), haar_null_bits=round(null, 4),
                                 matched_null_bits=round(mnull, 4),
                                 magic_ratio_haar=round(m2 / null, 4) if null > 0 else float("nan"),
                                 magic_ratio=round(m2 / mnull, 4) if mnull > 0 else float("nan")))
                print(f"{lt:<17}{dep:>3}{side:>9}{r:>6}{m2:>8.2f}{null:>8.2f}"
                      f"{mnull:>9.2f}{m2 / null:>8.2f}x"
                      f"{(m2 / mnull if mnull > 0 else float('nan')):>11.2f}x")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {Path(args.out).resolve()}")

    import statistics as st
    print(f"\n{'layer type':<20}{'mean vs matched':>18}{'lowest @ rank>=4':>20}")
    lowest = {}
    for lt in sorted({r["layer_type"] for r in rows}):
        sel = [r for r in rows if r["layer_type"] == lt]
        deep = [r["magic_ratio"] for r in sel if r["rank"] >= 4]
        lo = min(deep) if deep else float("nan")
        lowest[lt] = lo
        print(f"{lt:<20}{st.mean(r['magic_ratio'] for r in sel):>19.2f}x{lo:>19.2f}x")
    print("(ratios are against a SPARSITY-MATCHED null -- the observed magnitudes\n"
          " randomly rearranged. A Haar null is wrong here: a spike is a basis state,\n"
          " so any sparse vector scores low for reasons unrelated to Clifford\n"
          " structure -- a random 4-sparse vector reads 0.10x of Haar but 1.00x of\n"
          " matched. 0 = genuine stabilizer structure; 1 = magic fully explained by\n"
          " the magnitude profile.)")
    cand = {lt: v for lt, v in lowest.items() if v == v and v < 0.5}
    if cand:
        print("\nVERDICT: Clifford-like structure in " + ", ".join(
            f"{lt} ({v:.2f}x)" for lt, v in sorted(cand.items()))
            + ".\n  The earlier diagnostics were blind to this, and it is exactly what "
              "an ancilla /\n  measurement-based ansatz can exploit cheaply. Worth "
              "pursuing.")
    else:
        b = min((v for v in lowest.values() if v == v), default=float("nan"))
        print(f"\nVERDICT: no stabilizer structure (lowest {b:.2f}x of the matched null). "
              f"The dominant\n  subspace is generic in magic as well as in "
              f"entanglement and sparsity, closing the\n  one identified blind spot "
              f"in the earlier analysis.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run the model and save H = E[x x^T] per layer")
    c.add_argument("model", help="Hub id OR a local save_pretrained directory")
    c.add_argument("--out", default="cov", help="output directory for the H matrices")
    c.add_argument("--types", nargs="+", default=None)
    c.add_argument("--depths", type=int, nargs="+", default=None)
    c.add_argument("--device", default="cpu")
    c.add_argument("--dtype", default="float32")
    c.add_argument("--local-files-only", "--offline", action="store_true",
                   dest="local_files_only")
    c.add_argument("--dataset", default="Salesforce/wikitext")
    c.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    c.add_argument("--split", default="train")
    c.add_argument("--tokenizer", default=None,
                   help="load the tokenizer from here instead of the model directory "
                        "(e.g. the Hub id, which resolves from your local HF cache)")
    c.add_argument("--token-ids", default=None,
                   help="pre-tokenized ids (.pt, [T] or [1,T]) -- needs no tokenizer "
                        "and no dataset, so it works fully offline")
    c.add_argument("--text-file", default=None,
                   help="tokenize this local text file instead of the Hub dataset")
    c.add_argument("--calib-tokens", type=int, default=65536)
    c.add_argument("--window", type=int, default=512)
    c.add_argument("--batch-size", type=int, default=2)
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("score", help="score compressions under both metrics")
    s.add_argument("--layers-dir", required=True, help="extract_layers.py output")
    s.add_argument("--cov", required=True, help="capture output directory")
    s.add_argument("--chi", type=int, nargs="+", default=[1, 4, 16, 64])
    s.add_argument("--types", nargs="+", default=None)
    s.add_argument("--depths", type=int, nargs="+", default=None)
    s.add_argument("--damp", type=float, default=1e-6,
                   help="relative damping for H^(-1/2) in the whitened low-rank build")
    s.add_argument("--out", default="activation_aware.csv")
    s.set_defaults(func=cmd_score)

    cu = sub.add_parser("curve", help="output error vs parameter count per method")
    cu.add_argument("--layers-dir", required=True)
    cu.add_argument("--cov", required=True)
    cu.add_argument("--types", nargs="+", default=None)
    cu.add_argument("--depths", type=int, nargs="+", default=None)
    cu.add_argument("--points", type=int, default=12,
                    help="grid points per method (log spaced)")
    cu.add_argument("--budgets", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.25, 0.50],
                    help="parameter budgets as a fraction of the dense layer")
    cu.add_argument("--damp", type=float, default=1e-6)
    cu.add_argument("--sparse-points", type=int, default=7,
                    help="grid points for rank and for nnz/row in the sparse mix "
                         "(the (rank, nnz) split is swept, not guessed)")
    cu.add_argument("--sparse-iters", type=int, default=3,
                    help="alternating low-rank/sparse refinement steps")
    cu.add_argument("--value-bits", type=int, default=16,
                    help="width of a stored weight, for charging sparse indices")
    cu.add_argument("--index-bits", type=int, default=None,
                    help="bits per sparse index (default: ceil(log2(cols)))")
    cu.add_argument("--out", default="activation_curve.csv")
    cu.set_defaults(func=cmd_curve)

    ve = sub.add_parser("vectors", help="is the dominant whitened subspace structured?")
    ve.add_argument("--layers-dir", required=True)
    ve.add_argument("--cov", required=True)
    ve.add_argument("--types", nargs="+", default=None)
    ve.add_argument("--depths", type=int, nargs="+", default=None)
    ve.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 16])
    ve.add_argument("--head-dim", type=int, default=64)
    ve.add_argument("--null-reps", type=int, default=64)
    ve.add_argument("--damp", type=float, default=1e-6)
    ve.add_argument("--seed", type=int, default=0)
    ve.add_argument("--out", default="activation_vectors.csv")
    ve.set_defaults(func=cmd_vectors)

    mg = sub.add_parser("magic", help="do the whitened vectors have Clifford structure?")
    mg.add_argument("--layers-dir", required=True)
    mg.add_argument("--cov", required=True)
    mg.add_argument("--types", nargs="+", default=None)
    mg.add_argument("--depths", type=int, nargs="+", default=None)
    mg.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8])
    mg.add_argument("--null-reps", type=int, default=8)
    mg.add_argument("--matched-reps", type=int, default=3,
                    help="draws for the sparsity-matched null (the deciding control)")
    mg.add_argument("--damp", type=float, default=1e-6)
    mg.add_argument("--seed", type=int, default=0)
    mg.add_argument("--out", default="activation_magic.csv")
    mg.set_defaults(func=cmd_magic)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
