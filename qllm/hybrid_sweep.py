"""Per-layer sweep of the hybrid PQC+TN layer against the pure TN layer.

For every profiled layer this measures **two perplexity surfaces on the same
probe tokens**, so their budgets are directly comparable:

* ``method=classical`` -- the CompactifAI layer: ``W`` replaced by its MPO at bond
  dimension ``chi``. Classical cost ``C(chi)``, no quantum cost.
* ``method=hybrid``    -- the layer of *Quantum LLMs via Tensor Network
  Disentanglers*: ``W ~= U MPO_new(chi') V^T`` with brickwall circuits of depth
  ``D``. Classical cost ``C(chi')`` (the residual MPO, over the qubit-padded
  indices), quantum cost ``Q(D)`` (the circuits' variational parameters).

``D = 0`` is always included: the circuits are then the identity, so that row
isolates the cost of the power-of-two qubit padding from the benefit of the
circuits.

One disentangling optimization is reused across the whole ``chi'`` grid -- the
residual operator is disentangled once per ``(layer, D)`` and its cached SVD is
truncated at each bond dimension, mirroring how the classical sweep reuses one
SVD per layer. The heavy term is therefore the perplexity probe, one evaluation
per row, and everything is checkpointed per row so an interrupted sweep resumes.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import (
    BenchmarkConfig,
    load_model_and_tokenizer,
    perplexity_over_ids,
    resolve_device,
    tokenize_corpus,
)
from .compactifai import (
    full_rank_chi,
    log_spaced_ints,
    relative_error,
)
from .compactifai_sweep import (
    CompactifaiConfig,
    _read_rows,
    _write_rows,
    compactifai_dir,
    evenly_spaced,
    select_layers,
    tensorized_name,
)
from .disentangler import disentangle, hybrid_weight, n_qubits_for
from .qubit_mpo import make_plan, plan_compress
from .layer_analysis import parse_layer_info

# The circuits are restricted to at most two-qubit gates. This is the
# hardware-realistic regime -- the paper runs only its two-qubit-gate
# disentanglers on a real QPU (its "ku = kv = 2" configuration), transpiling
# wider gates is what blows up the physical depth -- and it is where the
# quantum parameter count Q ~ 4^k per gate stays affordable against the layer.
MAX_GATE_SIZE = 2


def validate_gate_sizes(gate_sizes) -> list[int]:
    """Keep the requested gate sizes, rejecting anything wider than two qubits.

    ``k = 0`` (one gate spanning the whole register) and ``k > 2`` are refused:
    both leave the two-qubit-gate regime this build is restricted to.
    """
    sizes = sorted({int(k) for k in (gate_sizes or [MAX_GATE_SIZE])})
    bad = [k for k in sizes if k < 1 or k > MAX_GATE_SIZE]
    if bad:
        raise ValueError(
            f"gate size(s) {bad} are outside the allowed 1..{MAX_GATE_SIZE} "
            f"qubits; this build considers only k <= {MAX_GATE_SIZE} "
            f"(two-qubit gates). Drop k=0 (whole register) and k>2.")
    return sizes


@dataclass
class HybridConfig(CompactifaiConfig):
    """CompactifAI settings plus the disentangler (PQC) knobs."""
    # Circuit ansatz.
    circuit_depths: list[int] | None = None    # D values (0 = no circuit)
    gate_sizes: list[int] | None = None        # qubits per gate k (0 = whole register)
    disentangle_target_chi: int = 1            # bond dimension the circuits aim at
    disentangle_sweeps: int = 12
    disentangle_tol: float = 1e-6
    disentangle_init: str = "identity"
    disentangle_target_mode: str = "adaptive"  # 'adaptive' or the paper's 'fixed'
    disentangle_restarts: int = 1
    disentangle_seed: int = 0
    quantum_param_counting: str = "manifold"   # 'manifold' (angles) or 'entries'
    max_qubits: int = 13                       # skip layers needing a bigger register
    run_classical: bool = True                 # also record the pure-TN curve
    hybrid_csv_name: str = "hybrid_per_layer.csv"


@dataclass
class HybridRow:
    """One measured point of either surface."""
    timestamp: str
    model_id: str
    param_name: str
    layer_type: str
    depth: int                 # decoder block index
    method: str                # 'classical' | 'hybrid'
    tensorization: str         # 'balanced' | 'qubit' -- the MPO geometry
    chi: int                   # bond dimension (chi for classical, chi' for hybrid)
    circuit_depth: int         # D (-1 for classical rows)
    gate_size: int
    rows: int
    cols: int
    padded_rows: int
    padded_cols: int
    n_out_qubits: int
    n_in_qubits: int
    params_original: int
    classical_params: int      # C(chi) or C(chi')
    quantum_params: int        # Q(D), zero for classical rows
    total_params: int          # C + Q (unweighted; weighting happens downstream)
    compression_ratio: float   # total_params / params_original
    relative_error: float
    perplexity: float
    ppl_baseline: float
    ppl_ratio: float
    disentangle_accuracy: float
    disentangle_entropy: float
    disentangle_retained: float
    disentangle_sweeps: int
    disentangle_seconds: float
    eval_tokens: int
    eval_seconds: float


def hybrid_csv_path(config: HybridConfig) -> Path:
    return compactifai_dir(config) / tensorized_name(
        config.hybrid_csv_name, config.tensorization)


def default_depths(max_depth: int = 8, count: int = 4) -> list[int]:
    """``[0, 1, 2, 4, 8, ...]``: 0 (padding only) plus a geometric ladder."""
    depths = [0]
    d = 1
    while d <= max_depth and len(depths) <= count:
        depths.append(d)
        d *= 2
    return depths


def _select_profile_layers(model, config: HybridConfig):
    """The same block/type subsetting the CompactifAI per-layer profile uses."""
    layers = select_layers(model, config)
    available = sorted({parse_layer_info(n)[1] for n, _ in layers})
    if config.profile_depths is not None:
        keep = sorted(set(config.profile_depths) & set(available))
    elif config.num_depths:
        keep = evenly_spaced(available, config.num_depths)
    else:
        keep = available
    layers = [(n, p) for n, p in layers if parse_layer_info(n)[1] in set(keep)]
    if config.layer_types:
        wanted = tuple(config.layer_types)
        layers = [(n, p) for n, p in layers
                  if any(w in parse_layer_info(n)[0] for w in wanted)]
    return layers, keep, available


def run_hybrid_sweep(config: HybridConfig) -> Path:
    """Measure PPL(chi) for the TN layer and PPL(chi', D) for the PQC+TN layer."""
    device = resolve_device(config.device)
    load_cfg = BenchmarkConfig(
        model_id=config.model_id, device=config.device, dtype=config.dtype,
        trust_remote_code=config.trust_remote_code, revision=config.revision,
        dataset=config.dataset, dataset_config=config.dataset_config,
        split=config.split, text_column=config.text_column,
    )
    model, tokenizer = load_model_and_tokenizer(load_cfg, device)

    layers, keep, available = _select_profile_layers(model, config)
    if not layers:
        raise RuntimeError("No layers selected for the hybrid sweep.")

    # A layer needs ceil(log2 d) qubits per index; the disentangler works on the
    # padded 2^n x 2^m box, so very wide layers are opt-in via --max-qubits.
    too_big = [(n, p) for n, p in layers
               if max(n_qubits_for(p.shape[0]), n_qubits_for(p.shape[1])) > config.max_qubits]
    if too_big:
        names = ", ".join(sorted({parse_layer_info(n)[0] for n, _ in too_big}))
        print(f"Skipping {len(too_big)} layer(s) needing more than "
              f"{config.max_qubits} qubits ({names}); raise --max-qubits to include them.")
        layers = [(n, p) for n, p in layers if (n, p) not in too_big]
    if not layers:
        raise RuntimeError("Every selected layer exceeds --max-qubits.")

    depths = sorted({int(d) for d in (config.circuit_depths or default_depths())
                     if d >= 0})
    gate_sizes = validate_gate_sizes(config.gate_sizes)
    types_present = sorted({parse_layer_info(n)[0] for n, _ in layers})
    print(f"\nHybrid plan: {len(keep)} of {len(available)} decoder blocks {keep}")
    print(f"             {len(types_present)} layer types {types_present}")
    print(f"             circuit depths D = {depths}, gate sizes k = "
          + ", ".join(str(k) for k in gate_sizes) + " qubit(s)")
    print(f"             disentangling target chi = {config.disentangle_target_chi}, "
          f"Q counted as gate {config.quantum_param_counting}")

    originals = {name: p.detach().to("cpu", copy=True) for name, p in layers}

    # Classical plans (one cached SVD per layer, reused across chi).
    classical_plans = {name: make_plan(originals[name], config.tensorization,
                                       config.mpo_sites, svd_cache=config.svd_cache,
                                       align=config.qubit_align)
                       for name, _ in layers}

    auto_max = max(plan.max_chi for plan in classical_plans.values())
    chi_max = config.chi_max if config.chi_max else auto_max
    if config.chi_values:
        chis = sorted({int(c) for c in config.chi_values if c >= 1})
    else:
        chis = log_spaced_ints(config.chi_min, chi_max, config.num_chi)
    # The hybrid's premise is that the circuits let chi' be tiny, so the grid
    # always reaches down to 1 even when the classical grid starts higher.
    hybrid_chis = sorted(set(chis) | {1})
    print(f"             bond dimensions chi = {chis}")
    print(f"             hybrid bond dimensions chi' = {hybrid_chis}")

    input_ids = tokenize_corpus(tokenizer, load_cfg)
    budget = config.per_layer_eval_tokens
    probe_stride = config.per_layer_stride or config.max_length

    def evaluate():
        return perplexity_over_ids(
            model, input_ids, device, config.max_length, probe_stride,
            batch_size=config.ppl_batch_size, max_eval_tokens=budget, progress=False,
        )

    path = hybrid_csv_path(config)
    existing = [] if config.force_recompute else _read_rows(path)
    rows_by_key = {(r.get("param_name"), r.get("method"), (r.get("tensorization") or "balanced"), str(r.get("chi")),
                    str(r.get("circuit_depth")), str(r.get("gate_size"))): r
                   for r in existing}

    print(f"\nBaseline (all layers dense) perplexity on {budget} tokens ...")
    baseline, eval_tokens, _ = evaluate()
    print(f"Baseline perplexity = {baseline:.4f}")

    # Enumerate the work: classical points first (cheap, no optimization), then
    # the hybrid grid grouped by (layer, D) so one disentangling serves all chi'.
    todo: list[tuple] = []
    for name, param in layers:
        if config.run_classical:
            exact = full_rank_chi(classical_plans[name].out_dims,
                                  classical_plans[name].in_dims)
            for chi in sorted({min(int(c), exact) for c in chis}):
                key = (name, "classical", config.tensorization, str(chi), "-1", "0")
                if key not in rows_by_key:
                    todo.append(("classical", name, param, chi, -1))
    hybrid_jobs = []
    for name, param in layers:
        register = max(n_qubits_for(param.shape[0]), n_qubits_for(param.shape[1]))
        seen: set[tuple[int, int]] = set()
        for k in gate_sizes:
            k_eff = register if k == 0 else min(k, register)
            # Depth 0 has no gates at all, so it is the same point for every k;
            # likewise two k values that clamp to the same width.
            for d in depths:
                k_row = 0 if d == 0 else k_eff
                if (k_row, d) in seen:
                    continue
                seen.add((k_row, d))
                missing = [c for c in hybrid_chis
                           if (name, "hybrid", config.tensorization, str(c),
                               str(d), str(k_row)) not in rows_by_key]
                if missing:
                    hybrid_jobs.append((name, param, d, k_eff, k_row, missing))

    n_hybrid_points = sum(len(j[-1]) for j in hybrid_jobs)
    print(f"\n{len(layers)} layers | classical points to run: {len(todo)} | "
          f"hybrid points to run: {n_hybrid_points} "
          f"({len(hybrid_jobs)} disentangling optimizations)")
    windows = max(1, math.ceil((budget or input_ids.size(1)) / probe_stride))
    print(f"Each point = one perplexity evaluation over {budget} tokens "
          f"({windows} windows of {config.max_length}, stride {probe_stride}).")
    print(f"Checkpoint / results CSV: {path}")

    field_names = [f.name for f in fields(HybridRow)]

    def checkpoint():
        ordered = sorted(rows_by_key.values(),
                         key=lambda r: (str(r["param_name"]), str(r["method"]),
                                        int(r["gate_size"]), int(r["circuit_depth"]),
                                        int(r["chi"])))
        _write_rows(path, ordered, field_names)

    def record(row: HybridRow):
        rows_by_key[(row.param_name, row.method, row.tensorization, str(row.chi),
                     str(row.circuit_depth), str(row.gate_size))] = asdict(row)
        checkpoint()

    start = time.perf_counter()
    total_points = len(todo) + n_hybrid_points
    done = 0

    # ---- Classical surface: PPL(chi) with the layer's MPO, as in CompactifAI.
    for _kind, name, param, chi, _d in todo:
        plan = classical_plans[name]
        orig = originals[name]
        approx, params_mpo = plan_compress(orig, plan, chi)
        err = relative_error(orig, approx)
        with torch.no_grad():
            param.copy_(approx.to(dtype=param.dtype, device=param.device))
        ppl, eval_tokens, secs = evaluate()
        with torch.no_grad():
            param.copy_(orig.to(dtype=param.dtype, device=param.device))
        layer_type, block = parse_layer_info(name)
        done += 1
        record(HybridRow(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id, param_name=name, layer_type=layer_type,
            depth=block, method="classical",
            tensorization=config.tensorization, chi=chi, circuit_depth=-1,
            gate_size=0, rows=int(orig.shape[0]), cols=int(orig.shape[1]),
            padded_rows=int(orig.shape[0]), padded_cols=int(orig.shape[1]),
            n_out_qubits=0, n_in_qubits=0, params_original=plan.dense_params,
            classical_params=params_mpo, quantum_params=0, total_params=params_mpo,
            compression_ratio=round(params_mpo / plan.dense_params, 6),
            relative_error=round(err, 6), perplexity=round(ppl, 4),
            ppl_baseline=round(baseline, 4),
            ppl_ratio=round(ppl / baseline, 6) if baseline else float("nan"),
            disentangle_accuracy=float("nan"), disentangle_entropy=float("nan"),
            disentangle_retained=float("nan"), disentangle_sweeps=0,
            disentangle_seconds=0.0, eval_tokens=eval_tokens,
            eval_seconds=round(secs, 2),
        ))
        elapsed = time.perf_counter() - start
        print(f"  [{done}/{total_points}] classical {layer_type} d{block} "
              f"chi={chi:<4} C={params_mpo:<8,} rel.err={err:.4f} "
              f"ppl={ppl:.4f} (x{ppl / baseline:.4f})  "
              f"elapsed={elapsed:.0f}s eta={elapsed / done * (total_points - done):.0f}s")

    # ---- Hybrid surface: PPL(chi', D) with the disentangling circuits in place.
    for name, param, d, k_eff, k_row, missing in hybrid_jobs:
        orig = originals[name]
        layer_type, block = parse_layer_info(name)
        print(f"  disentangling {layer_type} d{block} D={d} k={k_eff} ...")
        res = disentangle(
            orig, gate_size=k_eff, depth=d,
            tensorization=config.tensorization, qubit_align=config.qubit_align,
            target_chi=config.disentangle_target_chi, n_sites=config.mpo_sites,
            sweeps=config.disentangle_sweeps, tol=config.disentangle_tol,
            init=config.disentangle_init, seed=config.disentangle_seed,
            param_counting=config.quantum_param_counting,
            target_mode=config.disentangle_target_mode,
            restarts=config.disentangle_restarts,
        )
        print(f"    {res.n_out_qubits}q x {res.n_in_qubits}q, Q(D)={res.quantum_params:,} "
              f"| retained {res.retained_classical:.4f} -> {res.retained:.4f} "
              f"| accuracy {res.accuracy:.4f} | entropy {res.entropy:.4f} "
              f"| {res.sweeps_run} sweeps in {res.seconds:.1f}s")
        exact = full_rank_chi(res.plan.out_dims, res.plan.in_dims)
        for chi in sorted({min(int(c), exact) for c in missing}):
            approx, c_params = hybrid_weight(res, chi)
            err = relative_error(orig, approx)
            with torch.no_grad():
                param.copy_(approx.to(dtype=param.dtype, device=param.device))
            ppl, eval_tokens, secs = evaluate()
            with torch.no_grad():
                param.copy_(orig.to(dtype=param.dtype, device=param.device))
            q_params = res.quantum_params
            dense = int(orig.shape[0]) * int(orig.shape[1])
            done += 1
            record(HybridRow(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model_id=config.model_id, param_name=name, layer_type=layer_type,
                depth=block, method="hybrid",
                tensorization=config.tensorization, chi=chi, circuit_depth=d,
                gate_size=k_row,
                rows=int(orig.shape[0]), cols=int(orig.shape[1]),
                padded_rows=res.padded_shape[0], padded_cols=res.padded_shape[1],
                n_out_qubits=res.n_out_qubits, n_in_qubits=res.n_in_qubits,
                params_original=dense, classical_params=c_params,
                quantum_params=q_params, total_params=c_params + q_params,
                compression_ratio=round((c_params + q_params) / dense, 6),
                relative_error=round(err, 6), perplexity=round(ppl, 4),
                ppl_baseline=round(baseline, 4),
                ppl_ratio=round(ppl / baseline, 6) if baseline else float("nan"),
                disentangle_accuracy=round(res.accuracy, 6),
                disentangle_entropy=round(res.entropy, 6),
                disentangle_retained=round(res.retained, 6),
                disentangle_sweeps=res.sweeps_run,
                disentangle_seconds=round(res.seconds, 2),
                eval_tokens=eval_tokens, eval_seconds=round(secs, 2),
            ))
            elapsed = time.perf_counter() - start
            print(f"  [{done}/{total_points}] hybrid    {layer_type} d{block} "
                  f"k={k_row} D={d} chi'={chi:<4} C={c_params:<8,} Q={q_params:<8,} "
                  f"rel.err={err:.4f} ppl={ppl:.4f} (x{ppl / baseline:.4f})  "
                  f"elapsed={elapsed:.0f}s "
                  f"eta={elapsed / done * (total_points - done):.0f}s")

    with torch.no_grad():
        for name, param in layers:
            param.copy_(originals[name].to(dtype=param.dtype, device=param.device))
    print("\nOriginal weights restored.")
    print(f"\nHybrid sweep complete. Results in {path}")
    return path
