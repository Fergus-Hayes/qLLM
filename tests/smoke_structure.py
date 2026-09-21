"""Smoke test for analyze_structure.py (Phase 0 diagnostic).

The Phase-0 decision rests on this tool telling structure from noise, so the test
plants three known structures and asserts each produces its own signature:

* Gaussian noise -> every MI block sits at the matched-null floor (ratio ~ 1).
* Head-block     -> mass in row-col (the MPO site pairing), and the head
                    attribution isolates the head-index cross-register coupling.
* Low rank       -> mass in row-row / col-col (what U and V can actually fix).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import csv  # noqa: E402


def _write(d, depth, W):
    name = f"model.layers.{depth}.self_attn.q_proj.weight"
    save_file({name: W.contiguous()},
              os.path.join(d, f"block{depth:03d}.self_attn.q_proj.safetensors"))


def main():
    torch.manual_seed(0)
    d = tempfile.mkdtemp()
    N, heads, hd = 64, 8, 8          # 6 qubits per side, 8 heads x 8 dims -> 3 head bits
    _write(d, 0, torch.randn(N, N))                                   # null
    blk = torch.zeros(N, N)
    for i in range(heads):
        blk[i*hd:(i+1)*hd, i*hd:(i+1)*hd] = torch.randn(hd, hd)
    _write(d, 1, blk)                                                 # head-block
    _write(d, 2, torch.randn(N, 2) @ torch.randn(2, N))               # low rank

    out = os.path.join(d, "s.csv")
    subprocess.run([sys.executable, os.path.join(ROOT, "analyze_structure.py"), d,
                    "--head-dim", str(hd), "--null-reps", "2", "--out", out],
                   check=True, env={**os.environ, "PYTHONPATH": ROOT},
                   stdout=subprocess.DEVNULL)
    rows = {int(r["depth"]): r for r in csv.DictReader(open(out))}
    g = lambda i, k: float(rows[i][k])  # noqa: E731

    # Null layer: nothing anywhere above the matched-null floor.
    for tag in ("rr", "cc", "rc"):
        assert g(0, f"mi_{tag}_ratio") < 2.0, \
            f"null layer flagged structure in {tag}: {g(0, f'mi_{tag}_ratio')}"

    # Head-block: row-col dominates, and it is the head-index cross-register pairs.
    assert g(1, "mi_rc_ratio") > 10 * max(g(1, "mi_rr_ratio"), g(1, "mi_cc_ratio")), \
        "head-block should concentrate in row-col"
    assert g(1, "mi_head_x_head_cross_reg_mean") > 10 * g(1, "mi_within_intra_reg_mean"), \
        "head attribution failed to isolate the head-index coupling"
    assert int(rows[1]["head_bits_out"]) == 3, rows[1]["head_bits_out"]

    # Low rank: within-register mass, and NOT an ordering problem.
    assert min(g(2, "mi_rr_ratio"), g(2, "mi_cc_ratio")) > 5, "low-rank should flag U/V"
    assert g(2, "mi_rc_ratio") < 2.0, "low-rank is not a row-col/ordering problem"

    # Bond entropy: the head-block tensor is genuinely easier than noise.
    assert g(1, "bond_entropy") < g(0, "bond_entropy"), "bond entropy ordering wrong"

    print(f"  null   ratios rr/cc/rc = {g(0,'mi_rr_ratio'):.2f}/{g(0,'mi_cc_ratio'):.2f}"
          f"/{g(0,'mi_rc_ratio'):.2f}  (floor)")
    print(f"  head   ratios rr/cc/rc = {g(1,'mi_rr_ratio'):.2f}/{g(1,'mi_cc_ratio'):.2f}"
          f"/{g(1,'mi_rc_ratio'):.2f}  (row-col)")
    print(f"  lowrank ratios rr/cc/rc = {g(2,'mi_rr_ratio'):.2f}/{g(2,'mi_cc_ratio'):.2f}"
          f"/{g(2,'mi_rc_ratio'):.2f}  (row-row/col-col)")
    shutil.rmtree(d, ignore_errors=True)
    print("\nSTRUCTURE SMOKE PASSED")


if __name__ == "__main__":
    main()
