#!/usr/bin/env python3
"""Extract selected weight matrices (by layer type and block depth) and save each
to its own file.

Type and depth mean exactly what they do in the sweep's ``--layer-types`` and
``--profile-depths``: ``--types`` matches the role substring (``q_proj`` hits
``self_attn.q_proj``), ``--depths`` matches the decoder-block index. Selection
reuses the repo's include/exclude filters, so the files you get are precisely the
weights ``hybridize.py`` would compress.

    # every attention projection in blocks 10, 15, 20 of the local model
    python extract_layers.py models/SmolLM2-135M --local-files-only \
        --types k_proj o_proj q_proj v_proj --depths 10 15 20 --out layers/

    # one layer, as a raw torch tensor file
    python extract_layers.py HuggingFaceTB/SmolLM2-135M \
        --types q_proj --depths 5 --format pt

Each file holds one 2-D tensor keyed by its full parameter name; a manifest.csv
in the output directory records name, type, depth, shape and file.
"""
import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from qllm.compactifai_sweep import DEFAULT_EXCLUDE, DEFAULT_INCLUDE
from qllm.layer_analysis import parse_layer_info

DTYPES = {"auto": None, "float32": torch.float32,
          "float16": torch.float16, "bfloat16": torch.bfloat16}


def select(model, include, exclude, types, depths):
    """(name, type, depth, param) for 2-D weights passing the type/depth filters.

    Mirrors qllm.compactifai_sweep.select_layers, then narrows to the requested
    types (role substring) and depths (exact block index), as the sweep does.
    """
    import re
    inc, exc = re.compile(include), re.compile(exclude)
    want_types = tuple(types) if types else None
    want_depths = set(depths) if depths else None
    out = []
    for name, param in model.named_parameters():
        if param.ndim != 2 or exc.search(name) or not inc.search(name):
            continue
        layer_type, depth = parse_layer_info(name)
        if depth < 0:                                   # not in a numbered block
            continue
        if want_types is not None and not any(w in layer_type for w in want_types):
            continue
        if want_depths is not None and depth not in want_depths:
            continue
        out.append((name, layer_type, depth, param))
    return out


def save_tensor(tensor, name, path, fmt):
    """Write one tensor to `path` in the chosen format, keyed by its param name."""
    if fmt == "safetensors":
        from safetensors.torch import save_file
        save_file({name: tensor.contiguous()}, str(path))
    elif fmt == "pt":
        torch.save({name: tensor}, str(path))          # dict keeps the name
    elif fmt == "npy":
        import numpy as np
        np.save(str(path), tensor.numpy())
    else:                                               # pragma: no cover
        raise ValueError(f"unknown format {fmt!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hub id OR a local dir saved by save_pretrained")
    ap.add_argument("--types", nargs="+", default=None,
                    help="Layer-type substrings to keep (e.g. q_proj v_proj). "
                         "Default: all eligible types.")
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="Decoder-block indices to keep (e.g. 10 15 20). "
                         "Default: all depths.")
    ap.add_argument("--out", default="layers", help="Output directory (default: ./layers)")
    ap.add_argument("--format", default="safetensors", choices=["safetensors", "pt", "npy"])
    ap.add_argument("--dtype", default="auto", choices=DTYPES,
                    help="Cast weights before saving (default: keep model dtype).")
    ap.add_argument("--local-files-only", "--offline", action="store_true",
                    dest="local_files_only", help="Never contact the Hub.")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--include", default=DEFAULT_INCLUDE, help="Eligibility include regex.")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="Eligibility exclude regex.")
    args = ap.parse_args()

    print(f"Loading '{args.model}'"
          f"{' (local files only)' if args.local_files_only else ''} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPES[args.dtype],
        trust_remote_code=args.trust_remote_code, revision=args.revision,
        local_files_only=args.local_files_only)
    model.eval()

    layers = select(model, args.include, args.exclude, args.types, args.depths)
    if not layers:
        raise SystemExit("No layers matched the given --types/--depths and "
                         "--include/--exclude filters.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = {"safetensors": "safetensors", "pt": "pt", "npy": "npy"}[args.format]

    manifest = []
    with torch.no_grad():
        for name, layer_type, depth, param in layers:
            fname = f"block{depth:03d}.{layer_type}.{ext}"
            path = out_dir / fname
            save_tensor(param.detach().cpu(), name, path, args.format)
            rows, cols = param.shape
            print(f"  {name:<45} [{rows}x{cols}] -> {path}")
            manifest.append({"file": fname, "param_name": name,
                             "layer_type": layer_type, "depth": depth,
                             "rows": rows, "cols": cols, "dtype": str(param.dtype)})

    man_path = out_dir / "manifest.csv"
    with open(man_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nSaved {len(manifest)} layer file(s) + manifest to {out_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
