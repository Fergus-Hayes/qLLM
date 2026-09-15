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
print("\n=== 6. k<=2 restriction and hybridize.py --solve ===")
from qllm.hybrid_sweep import MAX_GATE_SIZE, validate_gate_sizes

assert validate_gate_sizes([2]) == [2] and validate_gate_sizes([1, 2]) == [1, 2]
for bad in ([0], [3], [2, 4]):
    try:
        validate_gate_sizes(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"gate sizes {bad} should have been rejected")
print(f"  gate sizes capped at k = {MAX_GATE_SIZE}; k=0 and k>2 rejected")

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
rc_bad = hybridize_main(["tiny-gpt2", "--gate-sizes", "4"])
assert rc_bad == 2, "hybridize should reject k>2"
print("  hybridize --solve ran a targeted run end-to-end and rejected k>2")

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
    "--shape", "16x64", "--gate-sizes", "2", "--layers", "1", "4", "12",
    "--target-chi", "1", "--disentangle-sweeps", "20",
    "--results-dir", SCRATCH + "-fig3", "--no-plots",
])
assert rc == 0
import csv as _csv
from pathlib import Path as _Path
fig3_csv = _Path(SCRATCH + "-fig3") / "synthetic" / "disentangle_scaling" / "disentangle_scaling.csv"
with fig3_csv.open() as f:
    scan = sorted(_csv.DictReader(f), key=lambda r: int(r["n_layers"]))
accs = [float(r["accuracy"]) for r in scan]
assert accs[-1] > accs[0], f"accuracy should rise with L: {accs}"
assert all(r["target_chi"] == "1" and r["tensorization"] == "qubit" for r in scan)
assert all(r["ppl_ratio"] in ("", "nan") for r in scan), "no perplexity without a model"
print(f"  accuracy(Eq.4) rises with L for k=2: {accs[0]:.4f} (L=1) -> {accs[-1]:.4f} "
      f"(L={scan[-1]['n_layers']})")
# M* = C(chi') + Q(D): classical part constant, total grows with L (more layers)
assert all(int(r["total_params"]) == int(r["classical_params"]) + int(r["quantum_params"])
           for r in scan), "total_params must equal classical + quantum"
for gk in {int(r["gate_size"]) for r in scan}:
    ks = sorted((r for r in scan if int(r["gate_size"]) == gk), key=lambda r: int(r["n_layers"]))
    assert len({int(r["classical_params"]) for r in ks}) == 1, "C(chi') is fixed across L"
    assert int(ks[-1]["total_params"]) > int(ks[0]["total_params"]), "M* grows with L"
m_lo, m_hi = int(scan[0]["total_params"]), int(sorted(scan, key=lambda r: int(r["total_params"]))[-1]["total_params"])
print(f"  M* = C(chi')+Q(D) recorded and grows with L: {m_lo} -> {m_hi} params")
# k>2 refused, and it runs offline from a synthetic shape (no model needed)
assert scaling_main(["--shape", "8x8", "--gate-sizes", "4"]) == 2
print("  runs offline on a synthetic shape; rejects k>2")

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
    prows = list(_csv.DictReader(f))
assert prows and all(float(r["ppl_ratio"]) > 0 and r["perplexity"] for r in prows), \
    "model path must record a finite perplexity per L"
print(f"  model path adds perplexity vs L: x{float(prows[0]['ppl_ratio']):.4f} -> "
      f"x{float(prows[-1]['ppl_ratio']):.4f} baseline")

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
    prows2 = {int(r["n_layers"]): r for r in _csv.DictReader(f)}
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
    assert len(list(_csv.DictReader(f))) == 1, "--recompute rewrites the CSV fresh"
print("  refuses a conflicting config; --recompute overwrites")

print("\nALL HYBRID SMOKE STAGES PASSED")
