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
from pathlib import Path

import torch

from qllm.activation_stats import (
    covariance_spectrum,
    input_covariance,
    low_rank_frobenius,
    low_rank_whitened,
    output_relative_error,
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

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
