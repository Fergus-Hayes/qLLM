"""End-to-end smoke test for the PQC disentangler and the budget optimization.

Real torch throughout -- only the network I/O (HF model + dataset) is stubbed to
a tiny in-memory GPT-2. Stages:

  1. Circuit algebra: gates are orthogonal, circuits invert exactly, and the
     index-juggling helpers agree with the explicit matrix form.
  2. The environment sweep is monotone in its objective.
  3. Disentangling is monotone, exact at full bond dimension, and a wider gate
     disentangles more (the paper's gate-size trade-off).
  4. The hybrid per-layer sweep runs and writes both surfaces to CSV.
  5. The budget optimization picks the right points on hand-made curves, and
     runs on the CSV from stage 4.
"""
import os, sys, random, shutil, string, time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.tokenization_utils_base import BatchEncoding

torch.manual_seed(0)
random.seed(0)

SCRATCH = os.environ.get("QLLM_SMOKE_DIR", "/tmp/qllm-smoke-hybrid")


def rel(a, b):
    return float((a - b).norm() / a.norm())


# --------------------------------------------------------------------------- #
# 1. Circuit algebra
# --------------------------------------------------------------------------- #
from qllm.disentangler import (
    apply_circuit, apply_right, build_circuit, disentangle, gate_positions,
    hybrid_weight, quantum_param_count, sweep_circuit,
)
from qllm.compactifai import build_plan, compress_weight, full_rank_chi

def _isfinite(x):
    import math as _m
    try:
        return _m.isfinite(float(x))
    except (TypeError, ValueError):
        return False


print("=== 1. Circuit algebra ===")
# The gates are genuine PennyLane circuits (qml.QubitUnitary on brickwall wires).
import pennylane as qml
from qllm.disentangler import circuit_ops, circuit_unitary
_ops = circuit_ops(build_circuit(6, 2, 2, init="random",
                                 generator=torch.Generator().manual_seed(1)))
assert _ops and all(isinstance(o, qml.QubitUnitary) for o in _ops), \
    "disentangler gates must be PennyLane QubitUnitary ops"
print(f"  gates are PennyLane ops: {type(_ops[0]).__name__} on wires "
      f"{_ops[0].wires.tolist()} (x{len(_ops)})")
for n, k, D in [(6, 2, 3), (5, 2, 4), (8, 3, 2), (4, 4, 1)]:
    gates = build_circuit(n, k, D, init="random",
                          generator=torch.Generator().manual_seed(1))
    X = torch.randn(1 << n, 7)
    Y = apply_circuit(gates, X, n)
    assert torch.allclose(apply_circuit(gates, Y, n, transpose=True), X, atol=1e-5)
    U = apply_circuit(gates, torch.eye(1 << n), n)          # explicit matrix
    # The PennyLane-composed register unitary equals the applied circuit.
    assert torch.allclose(circuit_unitary(gates, n), U, atol=1e-5)
    assert torch.allclose(U @ U.T, torch.eye(1 << n), atol=1e-5), "not orthogonal"
    assert torch.allclose(U @ X, Y, atol=1e-5), "circuit != matrix product"
    R = torch.randn(9, 1 << n)
    assert torch.allclose(apply_right(gates, R, n), R @ U.T, atol=1e-5)
    assert torch.allclose(apply_right(gates, R, n, transpose=True), R @ U, atol=1e-5)
print("  gates orthogonal, circuits exactly invertible, left/right forms agree")

# Gradient checkpointing (deep circuits): segmenting the chain must not change the
# forward OR the gradient -- it only trades recompute for memory, so a deep D=256
# circuit trains in a few GB instead of OOM-ing.
from qllm.disentangler import _ckpt_segments, _gates_from_angles
_pos = gate_positions(8, 2, 12)                          # ~42 gates > 32 -> checkpointed
assert _ckpt_segments(len(_pos)) is not None, "a deep chain should checkpoint"
assert _ckpt_segments(4) is None, "a shallow chain should not checkpoint"
_pad = torch.randn(1 << 8, 1 << 8)


def _fwd_grad(segs):
    th = [torch.zeros(6, requires_grad=True) for _ in _pos]   # so(4) = 6 angles per k=2 gate
    gg = _gates_from_angles(th, _pos, None)
    y = apply_circuit(gg, _pad, 8, transpose=True, checkpoint_segments=segs)
    (y * y).sum().backward()
    return y.detach(), torch.cat([t.grad.flatten() for t in th])


_y0, _g0 = _fwd_grad(None)
_y1, _g1 = _fwd_grad(_ckpt_segments(len(_pos)))
assert torch.allclose(_y0, _y1, atol=1e-6) and torch.allclose(_g0, _g1, atol=1e-6), \
    "gradient checkpointing changed the forward/gradient"
print(f"  gradient checkpointing exact over {len(_pos)} gates "
      f"({_ckpt_segments(len(_pos))} segments): forward & gradient match")
assert len(gate_positions(10, 10, 5)) == 1, "a register-wide gate must not stack"
assert quantum_param_count(10, 8, 10, 1) == 1024 * 1023 // 2 + 256 * 255 // 2
print(f"  Q for the paper's (10qU, 8qV, L=1): "
      f"{quantum_param_count(10, 8, 10, 1):,} parameters")

# The two sweeps see the SAME objective <W, U M V^T>, the V side through a
# transposition. Getting that convention wrong would still reconstruct exactly
# (U, V orthogonal), so it is checked against the dense matrices directly.
n_out, n_in = 6, 5
Wt = torch.randn(1 << n_out, 1 << n_in)
Mt = torch.randn(1 << n_out, 1 << n_in)
ug = build_circuit(n_out, 2, 3, init="random", generator=torch.Generator().manual_seed(1))
vg = build_circuit(n_in, 2, 3, init="random", generator=torch.Generator().manual_seed(2))
Um = apply_circuit(ug, torch.eye(1 << n_out), n_out)
Vm = apply_circuit(vg, torch.eye(1 << n_in), n_in)
f_ref = float((Wt * (Um @ Mt @ Vm.T)).sum())
f_u = float((Wt * apply_circuit(ug, apply_right(vg, Mt, n_in), n_out)).sum())
f_v = float((Wt.T * apply_circuit(vg, apply_circuit(ug, Mt, n_out).T.contiguous(),
                                  n_in)).sum())
assert abs(f_u - f_ref) < 1e-3 and abs(f_v - f_ref) < 1e-3, (f_ref, f_u, f_v)
cur = apply_right(vg, apply_circuit(ug, Wt, n_out, transpose=True), n_in, transpose=True)
assert torch.allclose(cur, Um.T @ Wt @ Vm, atol=1e-4), "MPO_new != U^T W V"
# A V-only sweep must raise the objective on its own.
vg2 = build_circuit(n_in, 2, 3, init="identity")
src = apply_circuit(ug, Mt, n_out).T.contiguous()
v_before = float((Wt.T * apply_circuit(vg2, src, n_in)).sum())
for _ in range(4):
    v_after = sweep_circuit(vg2, Wt.T.contiguous(), src, n_in)
assert v_after > v_before, (v_before, v_after)
print(f"  both sweeps optimize the same objective ({f_ref:.4f}); the V side "
      f"alone lifts it {v_before:.1f} -> {v_after:.1f}")

# --------------------------------------------------------------------------- #
# 2. Environment sweep is monotone
# --------------------------------------------------------------------------- #
print("\n=== 2. Environment sweep ===")
n = 6
gates = build_circuit(n, 2, 4, init="identity")
A, B = torch.randn(1 << n, 12), torch.randn(1 << n, 12)
vals = [float((A * apply_circuit(gates, B, n)).sum())]
for _ in range(6):
    vals.append(sweep_circuit(gates, A, B, n))
assert all(vals[i + 1] >= vals[i] - 1e-5 for i in range(len(vals) - 1)), vals
print(f"  overlap increases monotonically: {vals[0]:.3f} -> {vals[-1]:.3f}")

# --------------------------------------------------------------------------- #
# 3. Disentangling a structured matrix
# --------------------------------------------------------------------------- #
print("\n=== 3. Disentangling ===")
g = torch.Generator().manual_seed(3)
d_out, d_in = 96, 48
W = (torch.randn(d_out, 8, generator=g) @ torch.randn(8, d_in, generator=g) / 8
     + 0.3 * torch.randn(d_out, d_in, generator=g))
plan_c = build_plan(W, 2)
err_classical = rel(W, compress_weight(W, plan_c, 1)[0])
retained = {}
for k, D in [(2, 1), (2, 4), (4, 2), (7, 1)]:
    res = disentangle(W, gate_size=k, depth=D, target_chi=1, sweeps=20)
    hist = res.history
    assert all(hist[i + 1] >= hist[i] - 1e-5 for i in range(len(hist) - 1)), hist
    exact = full_rank_chi(res.plan.out_dims, res.plan.in_dims)
    assert rel(W, hybrid_weight(res, exact)[0]) < 1e-4, "full chi must be exact"
    retained[(k, D)] = res.retained
    print(f"  k={k} D={D}: Q={res.quantum_params:>6,}  retained "
          f"{res.retained_classical:.4f} -> {res.retained:.4f}  "
          f"entropy {res.entropy:.3f}  chi'=1 err "
          f"{rel(W, hybrid_weight(res, 1)[0]):.4f} (classical {err_classical:.4f})")
assert retained[(7, 1)] > retained[(2, 1)], "a wider gate must disentangle more"
# Depth 0 leaves the circuits at identity, so the hybrid layer must then BE the
# padded classical MPO -- that row is the padding overhead, nothing else.
from qllm.disentangler import pad_to_qubits
res0 = disentangle(W, gate_size=2, depth=0, target_chi=2)
pad, _, _ = pad_to_qubits(W)
ref0, _ = compress_weight(pad, build_plan(pad, 2), 2)
assert torch.allclose(hybrid_weight(res0, 2)[0], ref0[:d_out, :d_in], atol=1e-5), \
    "D=0 must reduce to the padded classical MPO"
assert all(v >= retained[(2, 1)] - 1e-6 or k == 2 for (k, _D), v in retained.items())
print("  monotone, exact at full chi, wider gates disentangle more")

# --------------------------------------------------------------------------- #
# 4. Hybrid per-layer sweep on a tiny real model
# --------------------------------------------------------------------------- #
print("\n=== 4. Hybrid per-layer sweep ===")
cfg = GPT2Config(vocab_size=512, n_positions=128, n_embd=64, n_layer=2, n_head=4,
                 n_inner=128, activation_function="gelu_new")
model = GPT2LMHeadModel(cfg)
model.eval()


class ByteTok:
    pad_token_id = eos_token_id = 0
    pad_token = "\0"
    model_max_length = 10 ** 8

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[b % cfg.vocab_size for b in text.encode()[:5000]]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids):
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return bytes(int(i) % 256 for i in seq).decode("latin-1", "ignore")


class FakeDS:
    def __init__(self):
        self._t = " ".join("".join(random.choices(string.ascii_lowercase,
                                                  k=random.randint(2, 8)))
                           for _ in range(400))

    def __getitem__(self, key):
        return [self._t]


import qllm.benchmark as B
import qllm.hybrid_sweep as H

tok = ByteTok()
B.load_model_and_tokenizer = lambda c, d: (model.to(d), tok)
H.load_model_and_tokenizer = lambda c, d: (model.to(d), tok)
B.load_dataset = H.load_dataset = lambda *a, **k: FakeDS()

shutil.rmtree(SCRATCH, ignore_errors=True)
t0 = time.perf_counter()
hcfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.\d+\.(attn|mlp)\.", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    chi_min=2, chi_max=16, num_chi=3, num_depths=1,
    circuit_depths=[0, 2], gate_sizes=[1, 2], disentangle_sweeps=6,
    max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=256, ppl_batch_size=4,
    results_dir=SCRATCH, make_plots=False,
)
path = H.run_hybrid_sweep(hcfg)
rows = H._read_rows(path)
methods = {r["method"] for r in rows}
assert methods == {"classical", "hybrid"}, methods
assert all(r["ppl_ratio"] not in ("", "nan") for r in rows)
print(f"  {len(rows)} rows ({sum(r['method'] == 'classical' for r in rows)} classical, "
      f"{sum(r['method'] == 'hybrid' for r in rows)} hybrid) in "
      f"{time.perf_counter() - t0:.1f}s")

# Re-running must be a no-op: every point is already checkpointed.
before = open(path).read()
H.run_hybrid_sweep(hcfg)
assert open(path).read() == before, "checkpoint did not suppress recomputation"
print("  re-run resumed from the checkpoint without recomputing")

# --------------------------------------------------------------------------- #
# 5. Budget optimization
# --------------------------------------------------------------------------- #
print("\n=== 5. Budget optimization ===")
from qllm.budget_frontier import (
    ClassicalPoint, HybridPoint, LayerCurves, load_curves, pareto_front,
    print_report, read_rows, solve_layer,
)

# Hand-made curves with a known answer: classical needs chi=8 (C=800) to meet a
# 1% budget; the hybrid meets it with chi'=1 (C=100) plus Q=200 at depth 2.
curve = LayerCurves("w", "q_proj", 0, dense_params=10_000)
curve.classical = [ClassicalPoint(2, 200, 1.30), ClassicalPoint(4, 400, 1.05),
                   ClassicalPoint(8, 800, 1.004), ClassicalPoint(16, 1600, 1.000)]
curve.hybrid = [HybridPoint(1, 0, 0, 100, 0, 1.40),
                HybridPoint(1, 2, 2, 100, 200, 1.005),
                HybridPoint(2, 2, 2, 200, 200, 1.002),
                HybridPoint(4, 2, 2, 400, 200, 1.000)]

s = solve_layer(curve, budget=1.01, q_weight=1.0)
assert (s.n_star, s.n_star_chi) == (800.0, 8), s
assert (s.m_star, s.m_star_chi, s.m_star_circuit_depth,
        s.m_star_gate_size) == (300.0, 1, 2, 2), s
assert abs(s.ratio - 300 / 800) < 1e-9 and s.status == "both"
# w* : the cheapest feasible hybrid point costs C=100, so it wins while
# w < (800 - 100)/200 = 3.5.
assert abs(s.breakeven_weight - 3.5) < 1e-9, s.breakeven_weight
print(f"  N*={s.n_star:.0f} (chi={s.n_star_chi})  M*={s.m_star:.0f} "
      f"(chi'={s.m_star_chi}, D={s.m_star_circuit_depth})  M*/N*={s.ratio:.3f}  "
      f"w*={s.breakeven_weight:.2f}")

# Pricing the circuits differently must flip the verdict, not crash.
assert solve_layer(curve, 1.01, q_weight=0.0).m_star == 100.0
assert solve_layer(curve, 1.01, q_weight=10.0).ratio > 1.0
# A budget nothing can meet leaves the layer non-comparable rather than wrong.
assert solve_layer(curve, 0.99, 1.0).status == "neither"
# A layer only the hybrid can bring inside the budget is flagged, not scored.
only = LayerCurves("w2", "v_proj", 1, dense_params=10_000)
only.classical = [ClassicalPoint(2, 200, 1.20), ClassicalPoint(4, 400, 1.08)]
only.hybrid = [HybridPoint(1, 4, 2, 100, 300, 1.004)]
oh = solve_layer(only, 1.01, 1.0)
assert oh.status == "hybrid_only" and oh.ratio != oh.ratio, oh
assert oh.breakeven_weight == 0.0, oh   # nothing classical to beat
print("  q_weight flips the verdict; an unreachable budget is reported, not faked")

front = pareto_front(curve, 1.0)
assert [p.cost for p in front] == sorted(p.cost for p in front)
assert all(front[i + 1].ppl_ratio < front[i].ppl_ratio for i in range(len(front) - 1))
print(f"  Pareto front is a strict staircase ({len(front)} steps, "
      f"{sum(p.method == 'hybrid' for p in front)} owned by the hybrid)")

curves = load_curves(read_rows(path))
assert curves, "no curves loaded from the sweep CSV"
print_report(curves, budget=1.5, q_weight=0.0)

try:
    from qllm.budget_plots import make_all
    from pathlib import Path
    from qllm.budget_frontier import solve_all
    out = Path(SCRATCH) / "figs"
    make_all(curves, solve_all(curves, 1.5, 0.0), out, 1.5, 0.0,
             budgets=[1.05, 1.5, 2.0])
    print(f"  figures: {sorted(p.name for p in out.glob('*.png'))}")
except Exception as exc:                            # noqa: BLE001
    print(f"  (plots skipped: {exc})")

# --------------------------------------------------------------------------- #
# 6. Two-qubit-gate restriction and the general "sweep + solve" CLI
# --------------------------------------------------------------------------- #
print("\n=== 6. gate-size range (no cap) and hybridize.py --solve ===")
from qllm.hybrid_sweep import DEFAULT_GATE_SIZE, validate_gate_sizes

assert validate_gate_sizes([2]) == [2] and validate_gate_sizes([1, 2]) == [1, 2]
assert validate_gate_sizes(None) == [DEFAULT_GATE_SIZE], "default gate size is 2"
assert validate_gate_sizes([0]) == [0], "k=0 (register-wide) must be allowed"
assert validate_gate_sizes([2, 8, 10]) == [2, 8, 10], "no upper cap on gate size"
try:
    validate_gate_sizes([-1])
except ValueError:
    pass
else:
    raise AssertionError("negative gate size should have been rejected")
print(f"  default k={DEFAULT_GATE_SIZE}, no upper cap (k=0 = register-wide); negatives rejected")

from qllm.hybrid_cli import main as hybridize_main

# A general, argument-driven targeted run: one block, one layer type, solved in
# the same command. Nothing here is specialized to a particular model or layer.
shutil.rmtree(SCRATCH + "-solve", ignore_errors=True)
rc = hybridize_main([
    "tiny-gpt2", "--profile-depths", "0", "--layer-types", "c_proj",
    "--gate-sizes", "2", "--circuit-depths", "0", "1", "2",
    "--chi", "1", "2", "4",
    "--per-layer-eval-tokens", "256", "--max-length", "64",
    "--ppl-batch-size", "4", "--solve", "--budgets", "1.01", "1.05",
    "--q-weights", "0", "1", "--no-pareto",
    "--results-dir", SCRATCH + "-solve",
])
assert rc == 0, f"hybridize --solve returned {rc}"
# It must refuse a gate size the build does not allow.
rc_bad = hybridize_main(["tiny-gpt2", "--gate-sizes", "-1"])
assert rc_bad == 2, "hybridize should reject negative gate sizes"
print("  hybridize --solve ran a targeted run end-to-end; negative k rejected")

# --------------------------------------------------------------------------- #
# 7. Qubit-level MPO tensorization (the paper's geometry), both methods
# --------------------------------------------------------------------------- #
print("\n=== 7. Qubit MPO tensorization ===")
from qllm.qubit_mpo import build_qubit_plan, compress_weight_qubit
from qllm.compactifai import mpo_param_count, full_rank_chi as _frc

# Parameter counts must match arXiv:2410.17397 Table I for the (192,576) layer.
Wp = torch.randn(192, 576)
qplan = build_qubit_plan(Wp)
paper = {1: 36, 2: 132, 5: 696, 10: 2356, 50: 36948}
for chi, want in paper.items():
    got = mpo_param_count(qplan.out_dims, qplan.in_dims, chi)
    assert got == want, f"qubit MPO chi={chi}: {got} != paper {want}"
print("  param counts match paper Table I: "
      + ", ".join(f"chi={c}->{v}" for c, v in paper.items()))
# Exact at full rank, monotone in chi.
errs = [float((Wp - compress_weight_qubit(Wp, qplan, c)[0]).norm() / Wp.norm())
        for c in (1, 2, 8, _frc(qplan.out_dims, qplan.in_dims))]
assert errs[0] >= errs[1] >= errs[2] >= errs[3] and errs[-1] < 1e-4, errs
print(f"  monotone in chi, exact at full rank (err {errs[-1]:.1e})")

# The disentangler and both sweep curves accept tensorization="qubit".
r0 = disentangle(Wp, gate_size=2, depth=0, target_chi=1, tensorization="qubit")
assert type(r0.plan).__name__ == "QubitMPOPlan"
c1, cp = compress_weight_qubit(Wp, qplan, 1)
h1, hp = hybrid_weight(r0, 1)
assert (cp, hp) == (36, 36) and torch.allclose(c1, h1, atol=1e-5), \
    "D=0 hybrid must equal the classical qubit MPO"
r2 = disentangle(Wp, gate_size=2, depth=2, target_chi=1, tensorization="qubit", sweeps=6)
assert float((Wp - hybrid_weight(r2, _frc(r2.plan.out_dims, r2.plan.in_dims))[0]).norm()
             / Wp.norm()) < 1e-4
print("  disentangler + hybrid_weight run on the qubit residual; D=0 == classical")

# End-to-end sweep in qubit mode (both curves), on the tiny model.
shutil.rmtree(SCRATCH + "-qubit", ignore_errors=True)
qcfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.\d+\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    tensorization="qubit", chi_min=1, chi_max=8, num_chi=3, num_depths=1,
    circuit_depths=[0, 1], gate_sizes=[2], disentangle_sweeps=4,
    max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=256, ppl_batch_size=4,
    results_dir=SCRATCH + "-qubit", make_plots=False)
qpath = H.run_hybrid_sweep(qcfg)
qrows = H._read_rows(qpath)
assert qrows and {r["method"] for r in qrows} == {"classical", "hybrid"}
assert {r["tensorization"] for r in qrows} == {"qubit"}, "rows must record the geometry"
print(f"  hybrid sweep ran in qubit mode: {len(qrows)} rows, both surfaces")

# Healing in the hybrid sweep: classical rows heal the MPO bond, hybrid rows heal
# bond + circuits (full). Both record a healed PPL; checkpoint counts a point done
# only when its healed column is present, and D=0 hybrid healing == classical.
shutil.rmtree(SCRATCH + "-heal-sweep", ignore_errors=True)
hheal = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    tensorization="qubit", chi_values=[1, 2], circuit_depths=[0, 2], gate_sizes=[2],
    num_depths=1, disentangle_sweeps=4, max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=256, ppl_batch_size=4,
    heal=True, heal_mode="full", heal_steps=4, heal_lr=0.02, heal_tokens=256,
    heal_batch=2, heal_split="train", results_dir=SCRATCH + "-heal-sweep", make_plots=False)
hpath = H.run_hybrid_sweep(hheal)
hrows = H._read_rows(hpath)
assert hrows and all(_isfinite(r["ppl_ratio_healed"]) for r in hrows), \
    "every point must carry a healed perplexity"
assert {r["heal_mode"] for r in hrows if r["method"] == "classical"} == {"core"}
assert {r["heal_mode"] for r in hrows if r["method"] == "hybrid"} == {"full"}
# D=0 hybrid full healing coincides with the classical MPO-bond healing at each chi.
_cl = {r["chi"]: r["ppl_ratio_healed"] for r in hrows if r["method"] == "classical"}
_d0 = {r["chi"]: r["ppl_ratio_healed"] for r in hrows
       if r["method"] == "hybrid" and r["circuit_depth"] == "0"}
assert _cl == _d0, "D=0 hybrid healing must equal classical bond healing"
# Resume: everything is done AND the baseline is reused from the checkpoint.
import io as _io2, contextlib as _ctx2
_b = _io2.StringIO()
with _ctx2.redirect_stdout(_b):
    H.run_hybrid_sweep(hheal)
_out = _b.getvalue()
assert "from checkpoint" in _out and "points to run: 0" in _out, \
    "heal resume must skip done points and reuse the baseline"
print(f"  healing in the sweep: {len(hrows)} healed rows (classical->core, hybrid->full); "
      f"D=0 == classical; resume skips all")

# --- gradient ("implicit") training scheme, alongside the explicit env-SVD one
Wg = (torch.randn(64, 32) @ torch.randn(32, 48) / 32 + 0.2 * torch.randn(64, 48))
gexp = disentangle(Wg, gate_size=2, depth=2, target_chi=1, optimizer="explicit", sweeps=20)
ggd = disentangle(Wg, gate_size=2, depth=2, target_chi=1, optimizer="gradient",
                  gd_steps=200, gd_lr=0.05)
assert gexp.optimizer == "explicit" and ggd.optimizer == "gradient"
# gradient stays on the gate manifold (orthogonal gates) and is exact at full rank
frg = full_rank_chi(ggd.plan.out_dims, ggd.plan.in_dims)
assert rel(Wg, hybrid_weight(ggd, frg)[0]) < 1e-4, "gradient not exact at full rank"
# both beat the circuit-free baseline (the loss went down)
assert ggd.retained >= ggd.retained_classical - 1e-6
assert gexp.retained >= gexp.retained_classical - 1e-6
# the trained gates are still PennyLane ops
assert all(isinstance(o, qml.QubitUnitary) for o in circuit_ops(ggd.u_gates))
print(f"  gradient scheme: retained {ggd.retained_classical:.4f} -> {ggd.retained:.4f} "
      f"(explicit -> {gexp.retained:.4f}); gates are QubitUnitary, exact at full rank")

# a gradient sweep writes its own CSV (no mixing with the explicit run)
shutil.rmtree(SCRATCH + "-gd", ignore_errors=True)
gdcfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.\d+\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    disentangle_optimizer="gradient", disentangle_gd_steps=40, disentangle_gd_lr=0.1,
    chi_min=1, chi_max=8, num_chi=2, num_depths=1,
    circuit_depths=[0, 1], gate_sizes=[2],
    max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=256, ppl_batch_size=4,
    results_dir=SCRATCH + "-gd", make_plots=False)
gdpath = H.run_hybrid_sweep(gdcfg)
exppath = H.hybrid_csv_path(H.HybridConfig(model_id="tiny-gpt2", results_dir=SCRATCH + "-gd"))
assert "gradient" in gdpath.name and gdpath != exppath, "optimizer must isolate the CSV"
gdrows = H._read_rows(gdpath)
assert {r["optimizer"] for r in gdrows if r["method"] == "hybrid"} == {"gradient"}
print(f"  gradient sweep isolated in {gdpath.name}; hybrid rows tagged 'gradient'")

# The two geometries must never share a checkpoint file (else one masks the
# other's points and the solver mixes non-comparable C(chi)).
from qllm.budget_frontier import filter_rows_by_tensorization, tensorizations_in
bpath = H.hybrid_csv_path(H.HybridConfig(model_id="tiny-gpt2", tensorization="balanced",
                                         results_dir=SCRATCH + "-qubit"))
assert bpath != qpath and "qubit" in qpath.name and "qubit" not in bpath.name
mixed = [{"tensorization": "balanced"}, {"tensorization": "qubit"}, {}]
assert tensorizations_in(mixed) == ["balanced", "qubit"]        # {} -> balanced
assert len(filter_rows_by_tensorization(mixed, "qubit")) == 1
print("  geometries land in separate files; mixed-CSV rows are detectable")

# --------------------------------------------------------------------------- #
# 8. Fig. 3 reproduction CLI: disentangling accuracy vs. number of layers
# --------------------------------------------------------------------------- #
print("\n=== 8. Disentangling accuracy vs. layers (Fig. 3 CLI) ===")
from qllm.disentangle_scaling_cli import main as scaling_main, log_spaced_layers

assert log_spaced_layers(35, 5)[0] == 1 and log_spaced_layers(35, 5)[-1] == 35
shutil.rmtree(SCRATCH + "-fig3", ignore_errors=True)
rc = scaling_main([
    "--shape", "16x64", "--gate-sizes", "2", "--layers", "0", "1", "4", "12",
    "--target-chi", "1", "--disentangle-sweeps", "20",
    "--results-dir", SCRATCH + "-fig3", "--no-plots",
])
assert rc == 0
import csv as _csv
from pathlib import Path as _Path
fig3_csv = _Path(SCRATCH + "-fig3") / "synthetic" / "disentangle_scaling" / "disentangle_scaling.csv"
with fig3_csv.open() as f:
    all_rows = list(_csv.DictReader(f))
scan = sorted((r for r in all_rows if r["kind"] == "sweep"), key=lambda r: int(r["n_layers"]))
accs = [float(r["accuracy"]) for r in scan]
assert accs[-1] > accs[0], f"accuracy should rise with L: {accs}"
assert all(r["target_chi"] == "1" and r["tensorization"] == "qubit" for r in scan)
assert all(r["ppl_ratio"] in ("", "nan") for r in scan), "no perplexity without a model"
print(f"  accuracy(Eq.4) rises with L for k=2: {accs[0]:.4f} (L={scan[0]['n_layers']}) -> "
      f"{accs[-1]:.4f} (L={scan[-1]['n_layers']})")
# D=0 is an ordinary sweep point (no circuits): Q=0, M* = C(chi'), the left end
# of the curve, and its accuracy == the classical retained fraction.
d0 = next(r for r in scan if int(r["n_layers"]) == 0)
assert int(d0["gate_size"]) == 2 and int(d0["quantum_params"]) == 0, "D=0 has no circuits"
assert int(d0["total_params"]) == int(d0["classical_params"]), "D=0 M* = C(chi')"
assert float(d0["accuracy"]) == float(d0["retained_classical"]), "D=0 accuracy == retained_classical"
# M* = C(chi') + Q(D): classical part constant, total grows with L (more layers)
assert all(int(r["total_params"]) == int(r["classical_params"]) + int(r["quantum_params"])
           for r in scan), "total_params must equal classical + quantum"
for gk in {int(r["gate_size"]) for r in scan}:
    ks = sorted((r for r in scan if int(r["gate_size"]) == gk), key=lambda r: int(r["n_layers"]))
    assert len({int(r["classical_params"]) for r in ks}) == 1, "C(chi') is fixed across L"
    assert int(ks[-1]["total_params"]) > int(ks[0]["total_params"]), "M* grows with L"
m_lo, m_hi = int(scan[0]["total_params"]), int(sorted(scan, key=lambda r: int(r["total_params"]))[-1]["total_params"])
print(f"  D=0 is a regular point: acc={float(d0['accuracy']):.4f}, M*=C(chi')={d0['total_params']}; "
      f"M* grows {m_lo} -> {m_hi}")
# Reference row: the plain TN on the *unpadded* matrix (D=0 is no longer one).
refs = {r["kind"]: r for r in all_rows if r["kind"].startswith("tn_")}
assert set(refs) == {"tn_nopad"}, f"only the unpadded-TN reference remains: {set(refs)}"
nopad = refs["tn_nopad"]
assert int(nopad["quantum_params"]) == 0, "reference has Q=0"
assert nopad["tensorization"] == "balanced(no pad)", "unpadded TN uses the raw-dims MPO"
print(f"  TN-no-pad reference: acc={float(nopad['accuracy']):.4f} M*={nopad['total_params']}")
# no upper cap now; negative k refused; runs offline from a synthetic shape
assert scaling_main(["--shape", "8x8", "--gate-sizes", "-1"]) == 2
print("  runs offline on a synthetic shape; negative k rejected")

# On the model path it additionally measures the perplexity-vs-L curve (the
# target layer swapped for its disentangled chi'=1 reconstruction at each L).
shutil.rmtree(SCRATCH + "-fig3ppl", ignore_errors=True)
rc = scaling_main([
    "tiny-gpt2", "--block", "0", "--layer-type", "c_proj", "--gate-sizes", "2",
    "--layers", "1", "8", "--target-chi", "1", "--disentangle-sweeps", "12",
    "--max-length", "64", "--stride", "64", "--eval-tokens", "256",
    "--ppl-batch-size", "4", "--results-dir", SCRATCH + "-fig3ppl", "--no-plots",
])
assert rc == 0
ppl_csv = _Path(SCRATCH + "-fig3ppl") / "tiny-gpt2" / "disentangle_scaling" / "disentangle_scaling.csv"
with ppl_csv.open() as f:
    prows = [r for r in _csv.DictReader(f) if r["kind"] == "sweep"]
assert prows and all(float(r["ppl_ratio"]) > 0 and r["perplexity"] for r in prows), \
    "model path must record a finite perplexity per L"
# The unpadded-TN reference gets its perplexity swapped in on the model path too.
with ppl_csv.open() as f:
    prefs = {r["kind"]: r for r in _csv.DictReader(f) if r["kind"].startswith("tn_")}
assert set(prefs) == {"tn_nopad"} and all(float(r["ppl_ratio"]) > 0 for r in prefs.values()), \
    "the unpadded-TN reference must carry a perplexity ratio on the model path"
print(f"  model path adds perplexity vs L: x{float(prows[0]['ppl_ratio']):.4f} -> "
      f"x{float(prows[-1]['ppl_ratio']):.4f} baseline; "
      f"TN-no-pad x{float(prefs['tn_nopad']['ppl_ratio']):.4f}")

# Resume: re-running with an extra L reuses the finished points and the cached
# baseline (no re-evaluation) instead of recomputing from scratch.
import io as _io
import contextlib as _ctx
prev_base = prows[0]["ppl_baseline"]
buf = _io.StringIO()
with _ctx.redirect_stdout(buf):
    rc = scaling_main([
        "tiny-gpt2", "--block", "0", "--layer-type", "c_proj", "--gate-sizes", "2",
        "--layers", "1", "4", "8", "--target-chi", "1", "--disentangle-sweeps", "12",
        "--max-length", "64", "--stride", "64", "--eval-tokens", "256",
        "--ppl-batch-size", "4", "--results-dir", SCRATCH + "-fig3ppl", "--no-plots",
    ])
out = buf.getvalue()
assert rc == 0
assert "from checkpoint" in out, "baseline should be reused from the checkpoint CSV"
assert "Resuming: 2/3" in out, f"two of three points should be reused:\n{out}"
with ppl_csv.open() as f:
    prows2 = {int(r["n_layers"]): r for r in _csv.DictReader(f) if r["kind"] == "sweep"}
assert set(prows2) == {1, 4, 8}, "added L=4 without dropping the finished points"
assert all(prows2[L]["ppl_baseline"] == prev_base for L in prows2), \
    "the reused baseline must match the checkpointed one"
assert prows2[1]["accuracy"] == prows[0]["accuracy"], "finished rows kept verbatim"
print(f"  resume: reused 2/3 points + cached baseline {prev_base}, ran only L=4")

# A conflicting configuration is refused; --recompute overwrites it.
assert scaling_main([
    "tiny-gpt2", "--block", "0", "--layer-type", "c_proj", "--gate-sizes", "2",
    "--layers", "1", "--optimizer", "gradient", "--disentangle-gd-steps", "2",
    "--max-length", "64", "--stride", "64", "--eval-tokens", "256",
    "--ppl-batch-size", "4", "--results-dir", SCRATCH + "-fig3ppl", "--no-plots",
]) == 2
rc = scaling_main([
    "tiny-gpt2", "--block", "0", "--layer-type", "c_proj", "--gate-sizes", "2",
    "--layers", "1", "--recompute", "--disentangle-sweeps", "12",
    "--max-length", "64", "--stride", "64", "--eval-tokens", "256",
    "--ppl-batch-size", "4", "--results-dir", SCRATCH + "-fig3ppl", "--no-plots",
])
assert rc == 0
with ppl_csv.open() as f:
    fresh = list(_csv.DictReader(f))
assert sum(1 for r in fresh if r["kind"] == "sweep") == 1, "--recompute rewrites the CSV fresh"
assert {r["kind"] for r in fresh if r["kind"].startswith("tn_")} == {"tn_nopad"}, \
    "--recompute recomputes the unpadded-TN reference too"
print("  refuses a conflicting config; --recompute overwrites")

# Healing: both core (chi' bond only) and full (bond + U/V circuits) modes add a
# healed perplexity column; the un-healed adapter reproduces the cold weight.
from qllm.disentangler import disentangle as _dis, hybrid_weight as _hw
from qllm.hybrid_heal import HybridAdapter as _HA
_res = _dis(torch.randn(32, 48, generator=torch.Generator().manual_seed(0)),
            gate_size=2, depth=3, target_chi=1, tensorization="qubit", sweeps=6, seed=0)
_cold, _C = _hw(_res, 1)
import torch.nn as _nn
_lin = _nn.Linear(48, 32, bias=False)
with torch.no_grad():
    _lin.weight.copy_(torch.randn(32, 48, generator=torch.Generator().manual_seed(0)))
_a_core, _a_full = _HA(_lin, _res, 1, "core"), _HA(_lin, _res, 1, "full")
for _a in (_a_core, _a_full):
    assert float(torch.linalg.norm(_a.effective_weight() - _cold)) < 1e-4, \
        "healing adapter must start at the cold reconstruction"
assert _a_core.n_params == _C, "core adapter trains exactly C(chi') params"
assert _a_full.n_params > _a_core.n_params, "full adapter adds the Q(D) circuit params"
print(f"  heal adapter: core trains {_a_core.n_params} params (=C), "
      f"full trains {_a_full.n_params} (=M*)")

shutil.rmtree(SCRATCH + "-heal", ignore_errors=True)
rc = scaling_main([
    "tiny-gpt2", "--block", "0", "--layer-type", "c_proj", "--gate-sizes", "2",
    "--layers", "0", "2", "--target-chi", "1", "--disentangle-sweeps", "8",
    "--max-length", "64", "--stride", "64", "--eval-tokens", "256", "--ppl-batch-size", "4",
    "--heal", "core", "full", "--heal-steps", "6", "--heal-lr", "0.02",
    "--heal-tokens", "256", "--heal-batch-size", "2", "--heal-window", "32",
    "--results-dir", SCRATCH + "-heal", "--no-plots",
])
assert rc == 0
heal_csv = _Path(SCRATCH + "-heal") / "tiny-gpt2" / "disentangle_scaling" / "disentangle_scaling.csv"
with heal_csv.open() as f:
    hsweep = {int(r["n_layers"]): r for r in _csv.DictReader(f) if r["kind"] == "sweep"}
assert hsweep and all(float(r["ppl_ratio_core"]) > 0 and float(r["ppl_ratio_full"]) > 0
                      for r in hsweep.values()), "core and full healed perplexity must be recorded"
assert hsweep[0]["ppl_ratio_core"] == hsweep[0]["ppl_ratio_full"], \
    "at L=0 (no circuits) full healing == core healing"
print(f"  --heal core/full records healed perplexity; L=0 core==full "
      f"(x{float(hsweep[0]['ppl_ratio_core']):.4f})")


# --------------------------------------------------------------------------- #
# 9. Perplexity error vs. number of tokens evaluated
# --------------------------------------------------------------------------- #
print("\n=== 9. Perplexity error vs. tokens ===")
import numpy as _np
from qllm.benchmark import window_nlls as _wnll, perplexity_over_ids as _pov
from qllm.ppl_error_cli import analyze as _analyze, _fit_power, main as _pe_main

_ids = torch.randint(0, 512, (1, 8192))
_ppl_ref, _ntok_ref, _ = _pov(model, _ids, "cpu", 64, 64, batch_size=8, progress=False)
_nll, _nt = _wnll(model, _ids, "cpu", 64, 64, batch_size=8, progress=False)
assert abs(_np.exp(_nll.sum() / _nt.sum()) - _ppl_ref) < 1e-2, "window_nlls must reproduce the PPL"
assert int(_nt.sum()) == _ntok_ref, "window token counts must match"
_budgets = [512, 1024, 2048, 4096, int(_nt.sum())]
_rows, _full, _sigma = _analyze(_nll, _nt, _budgets, 300, _np.random.default_rng(0))
_re = [r["ppl_rel_err"] for r in _rows]
assert _re[0] > _re[-1] > 0, f"relative error must fall with tokens: {_re}"
_C, _p = _fit_power([r["tokens"] for r in _rows], _re)
assert -0.75 < _p < -0.25, f"error should scale ~N^-0.5, got exponent {_p:.3f}"
# bootstrap and i.i.d.-ideal estimates agree to a small factor on ~i.i.d. tokens
_ratio = _rows[-1]["ppl_rel_err"] / _rows[-1]["iid_rel_err"]
assert 0.5 < _ratio < 2.0, f"bootstrap vs i.i.d. mismatch: {_ratio}"
print(f"  window_nlls reproduces PPL; rel_err ~ {_C:.3g} * N^{_p:.3f} "
      f"(N^-0.5 ideal); falls {_re[0]:.4f} -> {_re[-1]:.4f}")

# the full CLI writes a CSV + solves target-token budgets (paired mode too)
import qllm.benchmark as _B
_orig_tok = _B.tokenize_corpus
_B.tokenize_corpus = lambda t, c: torch.randint(0, 512, (1, 6144))
try:
    _rc = _pe_main(["tiny-gpt2", "--max-length", "64", "--stride", "64",
                    "--max-eval-tokens", "6144", "--ppl-batch-size", "8",
                    "--num-budgets", "6", "--bootstrap", "200",
                    "--compare-chi", "2", "--block", "0", "--layer-type", "c_proj",
                    "--tensorization", "qubit", "--out-dir", SCRATCH + "-ppl-err",
                    "--no-plots"])
finally:
    _B.tokenize_corpus = _orig_tok
assert _rc == 0
_pe_csv = _Path(SCRATCH + "-ppl-err") / "perplexity_error.csv"
_pe_rows = list(_csv.DictReader(_pe_csv.open()))
assert _pe_rows and "ratio_rel_err" in _pe_rows[0], "paired ratio columns must be written"
print(f"  ppl_error CLI: {len(_pe_rows)} budgets + paired ratio error, CSV written")


# --------------------------------------------------------------------------- #
# 10. Register-wide gates + word-level PPL (Table I mechanism)
# --------------------------------------------------------------------------- #
print("\n=== 10. Register-wide gates + word-level PPL (Table I) ===")
import math as _m10
_g10 = torch.Generator().manual_seed(0)
_W10 = ((torch.randn(96, 20, generator=_g10) @ torch.randn(20, 48, generator=_g10))
        / _m10.sqrt(20) + 0.3 * torch.randn(96, 48, generator=_g10))
_r2 = disentangle(_W10, gate_size=2, depth=1, target_chi=1, tensorization="qubit", sweeps=20)
_rw = disentangle(_W10, gate_size=16, depth=1, target_chi=1, tensorization="qubit", sweeps=20)
assert _rw.retained > 3 * _r2.retained, "register-wide must disentangle far more than k=2"
assert _rw.entropy < _r2.entropy - 0.5, "register-wide must collapse the entropy"
print(f"  register-wide D=1 concentrates entanglement: retained {_r2.retained:.3f} (k=2) -> "
      f"{_rw.retained:.3f}; entropy {_r2.entropy:.3f} -> {_rw.entropy:.3f}")

def _wsweep(word, tag):
    shutil.rmtree(SCRATCH + tag, ignore_errors=True)
    cfg = H.HybridConfig(
        model_id="tiny-gpt2", device="cpu", dtype="float32",
        include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
        tensorization="qubit", chi_values=[1, 2], circuit_depths=[1], gate_sizes=[0],
        num_depths=1, disentangle_sweeps=6, max_length=64, stride=64, per_layer_stride=64,
        per_layer_eval_tokens=256, ppl_batch_size=4, run_classical=False,
        word_level=word, results_dir=SCRATCH + tag, make_plots=False)
    return H._read_rows(H.run_hybrid_sweep(cfg))

_wr = _wsweep(True, "-word")
_tr = _wsweep(False, "-word-tok")
assert _wr and all(r["ppl_unit"] == "word" for r in _wr), "rows must be tagged word-level"
assert all(r["ppl_unit"] == "token" for r in _tr), "token run must be tagged token"
# --gate-sizes 0 becomes the register-wide gate width (> 2), not 0
assert all(int(r["gate_size"]) > 2 for r in _wr), "register-wide gate width recorded"
_w = {r["chi"]: float(r["perplexity"]) for r in _wr}
_t = {r["chi"]: float(r["perplexity"]) for r in _tr}
assert any(abs(_w[c] - _t[c]) > 1e-3 for c in _w), "word-level PPL must differ from token-level"
print(f"  hybrid sweep: --gate-sizes 0 -> register-wide (k={_wr[0]['gate_size']}); "
      f"--word-level tags ppl_unit=word and rescales PPL")


# --------------------------------------------------------------------------- #
# 11. Relative-error-only mode (--no-perplexity)
# --------------------------------------------------------------------------- #
print("\n=== 11. Relative-error-only mode (--no-perplexity) ===")
shutil.rmtree(SCRATCH + "-noppl", ignore_errors=True)
# A model-eval that would blow up if it were called -- proves no forward happens.
_bad_ppl = H.perplexity_over_ids
H.perplexity_over_ids = lambda *a, **k: (_ for _ in ()).throw(AssertionError("PPL called!"))
try:
    npcfg = H.HybridConfig(
        model_id="tiny-gpt2", device="cpu", dtype="float32",
        include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
        tensorization="qubit", chi_values=[1, 2, 4], circuit_depths=[0, 2, 8],
        gate_sizes=[2], num_depths=1, disentangle_sweeps=6,
        measure_perplexity=False, heal=True,   # heal must be auto-disabled
        results_dir=SCRATCH + "-noppl", make_plots=False)
    nppath = H.run_hybrid_sweep(npcfg)
finally:
    H.perplexity_over_ids = _bad_ppl
nprows = H._read_rows(nppath)
assert nprows, "no-perplexity sweep wrote no rows"
assert all(r["perplexity"] in ("", "nan") for r in nprows), "perplexity must be blank/nan"
assert all(_isfinite(r["relative_error"]) for r in nprows), "relative_error must be recorded"
assert all(r["ppl_ratio_healed"] in ("", "nan") for r in nprows), "healing must be off without PPL"
# relative_error varies with chi' (truncation), accuracy is per-D (fixed target)
_h = [r for r in nprows if r["method"] == "hybrid" and int(r["circuit_depth"]) == 8]
_by_chi = sorted(_h, key=lambda r: int(r["chi"]))
assert float(_by_chi[0]["relative_error"]) > float(_by_chi[-1]["relative_error"]), \
    "relative_error should fall as chi' grows"
assert len({r["disentangle_accuracy"] for r in _h}) == 1, "accuracy is per-D, constant across chi'"
print(f"  --no-perplexity: {len(nprows)} rows, relative_error + accuracy only, no forward passes, heal auto-off")

# --------------------------------------------------------------------------- #
# 12. Per-(D, chi') disentangling (--disentangle-target-per-chi)
# --------------------------------------------------------------------------- #
print("\n=== 12. Per-(D, chi') disentangling (target_chi' = chi') ===")
shutil.rmtree(SCRATCH + "-perchi", ignore_errors=True)
pccfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    tensorization="qubit", chi_values=[1, 2, 4], circuit_depths=[0, 8],
    gate_sizes=[2], num_depths=1, disentangle_sweeps=6,
    measure_perplexity=False, disentangle_target_per_chi=True,
    results_dir=SCRATCH + "-perchi", make_plots=False)
pcpath = H.run_hybrid_sweep(pccfg)
pcrows = H._read_rows(pcpath)
assert pcrows, "per-chi sweep wrote no rows"
assert all(_isfinite(r["relative_error"]) for r in pcrows), "relative_error must be recorded"
# With a fresh optimization per (D, chi') the disentangling stats are no longer a
# per-D constant: at fixed D>0 the accuracy/retained now vary across chi' (each
# row squeezed to its own target_chi = chi'), unlike the shared-optimization mode
# asserted constant in stage 11.
_pc = [r for r in pcrows if r["method"] == "hybrid" and int(r["circuit_depth"]) == 8]
assert len(_pc) >= 2, "need multiple chi' at D=8"
assert len({r["disentangle_accuracy"] for r in _pc}) > 1, \
    "per-(D, chi') accuracy must vary across chi' (target_chi' = chi')"
assert len({r["disentangle_retained"] for r in _pc}) > 1, \
    "per-(D, chi') retained must vary across chi'"
# The default (shared) mode over the same grid keeps them constant -- the contrast.
shutil.rmtree(SCRATCH + "-shared", ignore_errors=True)
shcfg = H.HybridConfig(**{**pccfg.__dict__, "disentangle_target_per_chi": False,
                          "results_dir": SCRATCH + "-shared"})
shrows = H._read_rows(H.run_hybrid_sweep(shcfg))
_sh = [r for r in shrows if r["method"] == "hybrid" and int(r["circuit_depth"]) == 8]
assert len({r["disentangle_accuracy"] for r in _sh}) == 1, \
    "shared mode: accuracy constant across chi' (one optimization at target_chi')"
print(f"  per-(D,chi'): {len(pcrows)} rows; D=8 accuracy over chi' = "
      f"{sorted(float(r['disentangle_accuracy']) for r in _pc)} "
      f"(shared mode: constant {_sh[0]['disentangle_accuracy']})")

# --------------------------------------------------------------------------- #
# 13. explicit+gradient: implicit refinement after the explicit sweep
# --------------------------------------------------------------------------- #
print("\n=== 13. explicit+gradient (implicit refinement after explicit) ===")
torch.manual_seed(0)
Wr = torch.randn(64, 64)
# The chained mode is warm-started from the explicit gates and descends
# disentangle_loss. On a single-bond BALANCED plan that loss aligns with the
# reported retained, so the polish does not lose ground there.
be = disentangle(Wr, gate_size=2, depth=4, target_chi=2, tensorization="balanced",
                 sweeps=12, optimizer="explicit", seed=0)
bg = disentangle(Wr, gate_size=2, depth=4, target_chi=2, tensorization="balanced",
                 sweeps=12, optimizer="explicit+gradient", gd_steps=300, gd_lr=0.02, seed=0)
assert bg.optimizer == "explicit+gradient", bg.optimizer
assert bg.retained >= be.retained - 1e-6, \
    f"balanced polish must not lose retained: {be.retained} -> {bg.retained}"
# Full-rank reconstruction stays exact (U, V remain orthogonal after refinement).
_ex = full_rank_chi(bg.plan.out_dims, bg.plan.in_dims)
_approx, _ = hybrid_weight(bg, _ex)
assert rel(Wr, _approx) < 1e-4, f"refined circuits must stay orthogonal: {rel(Wr, _approx)}"
# Depth 0 has no gates: the chained mode is a no-op that still yields a result.
b0 = disentangle(Wr, gate_size=2, depth=0, target_chi=2, tensorization="qubit",
                 sweeps=6, optimizer="explicit+gradient", gd_steps=50, seed=0)
assert not b0.u_gates and not b0.v_gates, "depth 0 must have no gates"
print(f"  balanced retained: explicit {be.retained:.4f} -> explicit+gradient "
      f"{bg.retained:.4f} (+{bg.retained - be.retained:.4f}); full-rank recon exact; "
      f"depth-0 no-op ok")

# --------------------------------------------------------------------------- #
# 14. Parameter budget (--max-total-params): over-cap points -> NaN rows
# --------------------------------------------------------------------------- #
print("\n=== 14. Parameter budget (--max-total-params) ===")
shutil.rmtree(SCRATCH + "-cap", ignore_errors=True)
CAP = 200
capcfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    tensorization="qubit", chi_values=[1, 2, 4], circuit_depths=[0, 8],
    gate_sizes=[2], num_depths=1, disentangle_sweeps=6,
    measure_perplexity=False, max_total_params=CAP,
    results_dir=SCRATCH + "-cap", make_plots=False)
caprows = H._read_rows(H.run_hybrid_sweep(capcfg))
_hy = [r for r in caprows if r["method"] == "hybrid"]
_over = [r for r in _hy if r["relative_error"] in ("", "nan")]
_kept = [r for r in _hy if r["relative_error"] not in ("", "nan")]
assert _over and _kept, "cap must both skip some points and keep others"
# Every skipped row is over budget with its parameter counts still recorded;
# every kept row is within budget and actually computed.
for r in _over:
    assert int(r["total_params"]) > CAP, f"skipped row not over cap: {r['total_params']}"
    assert int(r["classical_params"]) > 0 and r["perplexity"] in ("", "nan")
    assert r["disentangle_sweeps"] == "0", "skipped row must run no sweeps"
for r in _kept:
    assert int(r["total_params"]) <= CAP, f"kept row over cap: {r['total_params']}"
# The whole disentangling optimization is skipped when every chi' at a (D, k) is
# over budget: here D=8 (Q large) exceeds the cap for all chi', so no D=8 row is
# ever computed (all NaN), while D=0 (Q=0) keeps its small-chi' rows.
_d8 = [r for r in _hy if int(r["circuit_depth"]) == 8]
assert _d8 and all(r["relative_error"] in ("", "nan") for r in _d8), \
    "D=8 exceeds cap for every chi' -> optimization skipped, all rows NaN"
assert any(int(r["circuit_depth"]) == 0 and r["relative_error"] not in ("", "nan")
           for r in _hy), "D=0 small-chi' points stay within budget and compute"
print(f"  cap {CAP}: {len(_kept)} computed, {len(_over)} over-budget NaN rows; "
      f"D=8 optimization skipped entirely")

# --------------------------------------------------------------------------- #
# 15. --fast-gradient: pure-torch apply_circuit matches PennyLane qml.matrix
# --------------------------------------------------------------------------- #
print("\n=== 15. --fast-gradient (torch apply_circuit == PennyLane qml.matrix) ===")
torch.manual_seed(0)
Wg = torch.randn(64, 64)
_kw = dict(gate_size=2, depth=6, target_chi=2, tensorization="qubit",
           optimizer="gradient", gd_steps=60, gd_lr=0.03, seed=0)
slow = disentangle(Wg, fast_gradient=False, **_kw)
fast = disentangle(Wg, fast_gradient=True, **_kw)
# Same optimization, two ways of forming U^T W V -> same optimum (numerical noise).
assert abs(slow.retained - fast.retained) < 1e-3, \
    f"fast-gradient retained diverged: {slow.retained} vs {fast.retained}"
a_slow, _ = hybrid_weight(slow, 2)
a_fast, _ = hybrid_weight(fast, 2)
assert rel(a_slow, a_fast) < 1e-2, f"fast-gradient W' diverged: {rel(a_slow, a_fast)}"
# Circuits stay orthogonal -> exact reconstruction at full rank.
_exg = full_rank_chi(fast.plan.out_dims, fast.plan.in_dims)
assert rel(Wg, hybrid_weight(fast, _exg)[0]) < 1e-4, "fast-gradient broke orthogonality"
# It also composes with the explicit+gradient polish (just must run + stay exact).
polish = disentangle(Wg, gate_size=2, depth=4, target_chi=2, tensorization="balanced",
                     optimizer="explicit+gradient", gd_steps=80, gd_lr=0.02, seed=0,
                     fast_gradient=True)
assert rel(Wg, hybrid_weight(polish, full_rank_chi(polish.plan.out_dims,
                                                    polish.plan.in_dims))[0]) < 1e-4
print(f"  fast==slow: retained {slow.retained:.5f} vs {fast.retained:.5f}, "
      f"W' reldiff {rel(a_slow, a_fast):.1e}; explicit+gradient --fast-gradient exact")

# --------------------------------------------------------------------------- #
# 16. --gradient-objective relative-error: polish lowers relative_error
# --------------------------------------------------------------------------- #
print("\n=== 16. --gradient-objective relative-error ===")
torch.manual_seed(0)
Wr = torch.randn(16, 16)   # 4q x 4q qubit MPO (multi-site), small & fast


def _re_at(res, chi):
    a, _ = hybrid_weight(res, chi)
    return rel(Wr, a)


_kw = dict(gate_size=2, depth=4, target_chi=1, tensorization="qubit", sweeps=20,
           seed=0)
ex = disentangle(Wr, optimizer="explicit", **_kw)
# The relative-error objective, warm-started from the explicit gates, minimizes
# the true reconstruction error at target_chi -- so it can only match or beat
# explicit there (unlike disentangle-loss, which the user saw *worsen* it).
reo = disentangle(Wr, optimizer="explicit+gradient", gd_steps=40, gd_lr=0.03,
                  fast_gradient=True, gradient_objective="relative-error", **_kw)
assert _re_at(reo, 1) <= _re_at(ex, 1) + 1e-4, \
    f"relative-error objective must not worsen relative_error at target: " \
    f"{_re_at(ex, 1)} -> {_re_at(reo, 1)}"
# Minimizing the reconstruction error maximizes retained at the target bond.
assert reo.retained >= ex.retained - 1e-4, \
    f"relative-error objective should not lose retained: {ex.retained} -> {reo.retained}"
# Circuits stay orthogonal -> exact full-rank reconstruction.
assert rel(Wr, hybrid_weight(reo, full_rank_chi(reo.plan.out_dims,
                                                reo.plan.in_dims))[0]) < 1e-4
# The default objective is unchanged (disentangle-loss).
assert "relative-error" in __import__("qllm.disentangler", fromlist=["x"]).GRADIENT_OBJECTIVES
print(f"  relative-error obj: RE@target {_re_at(ex, 1):.5f} -> {_re_at(reo, 1):.5f}, "
      f"retained {ex.retained:.4f} -> {reo.retained:.4f} (exact full-rank recon)")

# --------------------------------------------------------------------------- #
# 17. Per-layer parameter cap (--max-total-params 0 caps at params_original)
# --------------------------------------------------------------------------- #
# (Process-parallel --jobs is validated separately in tests/smoke_parallel.py,
# which needs a __main__-guarded entry for the forkserver start method.)
print("\n=== 17. Per-layer cap (--max-total-params 0) ===")
shutil.rmtree(SCRATCH + "-cap0", ignore_errors=True)
cap0cfg = H.HybridConfig(
    model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.0\.attn\.c_proj", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    tensorization="qubit", chi_values=[1, 2, 4, 16, 64], circuit_depths=[1, 4, 16],
    gate_sizes=[2, 3], num_depths=1, disentangle_sweeps=6,
    disentangle_optimizer="explicit", measure_perplexity=False,
    max_total_params=0, results_dir=SCRATCH + "-cap0", make_plots=False)
cap0rows = H._read_rows(H.run_hybrid_sweep(cap0cfg))
_hy = [r for r in cap0rows if r["method"] == "hybrid"]
_kept = [r for r in _hy if r["relative_error"] not in ("", "nan")]
_skip = [r for r in _hy if r["relative_error"] in ("", "nan")]
assert _kept and _skip, "cap=0 must keep some points and skip others"
# --max-total-params 0 caps each layer at its own params_original: kept rows are
# within it, skipped rows exceed it.
for r in _kept:
    assert int(r["total_params"]) <= int(r["params_original"]), \
        f"cap=0 kept a row over params_original: {r['total_params']} > {r['params_original']}"
for r in _skip:
    assert int(r["total_params"]) > int(r["params_original"]), \
        f"cap=0 skipped a row within budget: {r['total_params']} <= {r['params_original']}"
# The classical surface honours it too (Q=0, so it caps C(chi) at params_original).
_cl_skip = [r for r in cap0rows if r["method"] == "classical" and r["relative_error"] in ("", "nan")]
for r in _cl_skip:
    assert int(r["classical_params"]) > int(r["params_original"])
print(f"  cap=0: kept {len(_kept)} hybrid (<= params_original), skipped {len(_skip)} "
      f"(> params_original), {len(_cl_skip)} classical skipped")

print("\n=== 18. Load layers from files (--layers-dir) ===")
# _layers_from_dir sources weights from extract_layers.py files and applies the same
# depth/type filters as _select_profile_layers -- no full model needed.
import tempfile as _tf
from safetensors.torch import save_file as _save_file
_ld = _tf.mkdtemp()
_specs = {  # full param name -> (rows, cols); two types across two depths
    "model.layers.0.self_attn.q_proj.weight": (16, 16),
    "model.layers.0.self_attn.v_proj.weight": (8, 16),
    "model.layers.1.self_attn.q_proj.weight": (16, 16),
    "model.layers.2.self_attn.q_proj.weight": (16, 16),
}
import csv as _csv
_man = []
for _nm, (_r, _c) in _specs.items():
    _lt, _dep = H.parse_layer_info(_nm)
    _f = f"block{_dep:03d}.{_lt}.safetensors"
    _save_file({_nm: torch.randn(_r, _c)}, os.path.join(_ld, _f))
    _man.append({"file": _f, "param_name": _nm, "layer_type": _lt, "depth": _dep})
with open(os.path.join(_ld, "manifest.csv"), "w", newline="") as _fh:
    _w = _csv.DictWriter(_fh, fieldnames=list(_man[0].keys())); _w.writeheader(); _w.writerows(_man)

_cfg_ld = H.HybridConfig(model_id="unused", tensorization="qubit",
                         profile_depths=[0, 2], layer_types=["q_proj"], layers_dir=_ld)
_layers, _keep, _avail = H._layers_from_dir(_ld, _cfg_ld)
assert _avail == [0, 1, 2], _avail                         # every depth present in the dir
assert _keep == [0, 2], _keep                              # narrowed by profile_depths
_got = sorted(n for n, _ in _layers)
assert _got == ["model.layers.0.self_attn.q_proj.weight",
                "model.layers.2.self_attn.q_proj.weight"], _got   # type+depth filtered
assert all(t.ndim == 2 for _, t in _layers)
# Glob fallback: same result with no manifest.csv present.
os.remove(os.path.join(_ld, "manifest.csv"))
_layers2, _, _ = H._layers_from_dir(_ld, _cfg_ld)
assert sorted(n for n, _ in _layers2) == _got, "glob fallback disagreed with manifest"
shutil.rmtree(_ld, ignore_errors=True)
print(f"  --layers-dir: {len(_got)} layer(s) selected from files (q_proj @ depths 0,2); "
      f"manifest and glob paths agree")

print("\n=== 19. Head-block ansatz (head-aligned gate layout) ===")
# The head-index qubits are the HIGH bits (qubit 0 = MSB), so they form a
# contiguous block at the start of the register and "dense on the head index" is an
# ordinary gate (0, head_bits) -- expressible in the existing Gate machinery.
from qllm.disentangler import head_block_positions, quantum_param_count as _qpc  # noqa: E402
assert head_block_positions(10, 64, 0, 0) == [(0, 4)], head_block_positions(10, 64, 0, 0)
assert head_block_positions(8, 64, 0, 0) == [(0, 2)], head_block_positions(8, 64, 0, 0)
assert head_block_positions(6, 64, 0, 0) == [], "head_dim == register -> no head bits"
# Optional within-head brickwall lands entirely above the head bits.
_hp = head_block_positions(10, 64, 2, 2)
assert _hp[0] == (0, 4) and all(s >= 4 for s, _k in _hp[1:]), _hp
# Q = 2 * dim SO(2^4) = 2 * 120 for a 10q x 10q layer.
assert _qpc(10, 10, 0, 0, "manifold", "head-block", 64) == 240
# The brickwall path is untouched by the new parameters.
assert gate_positions(10, 2, 4) == gate_positions(10, 2, 4, "brickwall", 0)
# And it actually disentangles: identity init means it can only improve on the MPO.
_wq = torch.randn(64, 64)
_rh = disentangle(_wq, gate_size=0, depth=0, target_chi=1, tensorization="qubit",
                    sweeps=5, ansatz="head-block", head_dim=16)
assert _rh.quantum_params == 2 * (4 * 3 // 2), _rh.quantum_params
assert _rh.retained >= _rh.retained_classical - 1e-6, "head-block fell below the MPO"
print(f"  head-block on 10q/head_dim 64 -> [(0,4)], Q=240; 6q/head_dim 16 -> "
      f"Q={_rh.quantum_params}, retained {_rh.retained_classical:.4f} -> {_rh.retained:.4f}")

print("\nALL HYBRID SMOKE STAGES PASSED")
