#!/usr/bin/env python3
"""Run both pre-stage checks in order, then say whether the stages should run.

    python preflight.py models/SmolLM2-135M \
        --layers-dir models/SmolLM2-135M/layers --token-ids ids.pt \
        --types q_proj k_proj v_proj o_proj --depths 10 20 --offline

Convergence first (it needs no model and fails fast), then the perplexity
correlation. Either can veto the stages, for different reasons:

* a regime that has not converged at the stage budget produces numbers that
  understate what its ansatz can do, so stage comparisons between regimes would
  be measuring the budget rather than the method;
* a per-layer error that does not track perplexity means the stages are ranking
  configurations by something that is not model quality -- and an *anti*-correlated
  one means they would be ranked backwards.

Exit status is 0 only when both pass, so this can gate a longer run:

    python preflight.py ... && python stages.py stage1 ...
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([sys.executable, *cmd]).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model dir or hub id (the correlation needs it)")
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--cov", default=None, help="H directory for the "
                                                "activation-mse regime")
    ap.add_argument("--types", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--token-ids", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--offline", "--local-files-only", action="store_true",
                    dest="offline")
    ap.add_argument("--out-dir", default="preflight")
    ap.add_argument("--ansatze", nargs="+",
                    default=["brickwall-k4D2", "brickwall-k2D4", "rope-pair"])
    ap.add_argument("--chis", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--gd-steps", type=int, default=150)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--skip", nargs="+", default=[], choices=["convergence", "ppl"])
    # Explicit per-script passthrough. A single shared "extra" would forward
    # ppl-only flags to convergence.py and make it exit 2 on an unknown option,
    # which looks exactly like a failed check.
    ap.add_argument("--conv-args", default="",
                    help='extra flags for convergence.py, as ONE quoted string '
                         '(e.g. --conv-args "--sweeps 8 --tol 0.02")')
    ap.add_argument("--ppl-args", default="",
                    help='extra flags for ppl_correlation.py, as ONE quoted '
                         'string (e.g. --ppl-args "--window 512 --batch-size 4")')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figs"
    codes = {}

    if "convergence" not in args.skip:
        cmd = ["convergence.py", "--layers-dir", args.layers_dir,
               "--types", *args.types, "--ansatze", *args.ansatze,
               "--chis", *[str(c) for c in args.chis],
               "--gd-steps", str(args.gd_steps),
               "--out", str(out / "convergence.csv"), "--figs", str(figs)]
        if args.depths:
            cmd += ["--depths", *[str(d) for d in args.depths]]
        if args.cov:
            cmd += ["--cov", args.cov]
        codes["convergence"] = run(cmd + shlex.split(args.conv_args))

    if "ppl" not in args.skip:
        cmd = ["ppl_correlation.py", args.model, "--layers-dir", args.layers_dir,
               "--types", *args.types, "--points", str(args.points),
               "--out", str(out / "ppl_correlation.csv"), "--figs", str(figs)]
        if args.depths:
            cmd += ["--depths", *[str(d) for d in args.depths]]
        if args.token_ids:
            cmd += ["--token-ids", args.token_ids]
        if args.text_file:
            cmd += ["--text-file", args.text_file]
        if args.offline:
            cmd += ["--local-files-only"]
        codes["ppl"] = run(cmd + shlex.split(args.ppl_args))

    print("\n" + "=" * 72)
    for name, code in codes.items():
        print(f"  {name:<12} {'ok' if code == 0 else f'FAILED (exit {code})'}")
    print(f"  artefacts    {out.resolve()}")
    bad = [n for n, c in codes.items() if c != 0]
    if bad:
        print("\nPREFLIGHT FAILED: " + ", ".join(bad) + ". Read the verdicts above "
              "and the\nfigures before running the stages.")
        return 1
    print("\nPREFLIGHT COMPLETE. Both checks ran -- read their VERDICT lines: a "
          "non-converged\nregime or a weak/negative correlation is a reason not to "
          "trust stage rankings,\neven though both scripts exited 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
