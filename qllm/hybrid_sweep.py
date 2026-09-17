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

By default one disentangling optimization is reused across the whole ``chi'``
grid -- the residual operator is disentangled once per ``(layer, D)`` (squeezing
to ``disentangle_target_chi``) and its cached SVD is truncated at each bond
dimension, mirroring how the classical sweep reuses one SVD per layer. With
``disentangle_target_per_chi`` a fresh optimization is run for every
``(layer, D, chi')`` point instead, each squeezing to ``target_chi = chi'`` -- the
best circuits for that bond, at one optimization per grid point. The heavy term
is otherwise the perplexity probe, one evaluation per row, and everything is
checkpointed per row so an interrupted sweep resumes.

With ``config.heal`` each point is additionally healed -- briefly retrained
against the LM loss (all other layers dense) to minimise its perplexity -- and
the healed perplexity recorded. Classical rows heal the MPO bond (via
:func:`~qllm.compactifai_heal.heal_layer` for the balanced geometry, or the
depth-0 hybrid adapter for the qubit geometry); hybrid rows heal per
``config.heal_mode`` (``core`` = the chi' bond, ``full`` = bond + U/V circuits)
through :func:`~qllm.hybrid_heal.heal_hybrid`. The dense baseline is cached in
the checkpoint, and a point counts as done only once it carries a healed value.
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
from .disentangler import disentangle, hybrid_weight, n_qubits_for, pad_to_qubits
from .qubit_mpo import make_plan, plan_compress
from .layer_analysis import parse_layer_info
from .compactifai_heal import heal_layer, make_heal_batches
from .hybrid_heal import heal_hybrid

# Gates default to two qubits -- the hardware-realistic regime the paper runs on
# a QPU, where the quantum parameter count Q ~ 4^k per gate stays affordable.
# There is no upper cap: wider gates, up to a single register-wide gate (k=0),
# are allowed so the paper's Table I "10qU, 8qV, L=1" register-wide configuration
# is reachable. Wider gates disentangle in fewer layers but blow up the physical
# depth once transpiled to hardware-native gates -- a deliberate trade-off.
DEFAULT_GATE_SIZE = 2
MAX_GATE_SIZE = DEFAULT_GATE_SIZE          # backward-compatible alias (the default)


def validate_gate_sizes(gate_sizes) -> list[int]:
    """Normalise the requested gate sizes -- no upper cap.

    ``k >= 1`` is a ``k``-qubit gate; ``k = 0`` is a single gate spanning the
    whole register (the paper's register-wide configuration). When none are
    given the default is a single two-qubit gate. Only negative widths are
    refused.
    """
    sizes = sorted({int(k) for k in (gate_sizes if gate_sizes else [DEFAULT_GATE_SIZE])})
    bad = [k for k in sizes if k < 0]
    if bad:
        raise ValueError(
            f"gate size(s) {bad} are negative; use k>=1 for a k-qubit gate or "
            f"k=0 for a single register-wide gate.")
    return sizes


@dataclass
class HybridConfig(CompactifaiConfig):
    """CompactifAI settings plus the disentangler (PQC) knobs."""
    # Circuit ansatz.
    circuit_depths: list[int] | None = None    # D values (0 = no circuit)
    gate_sizes: list[int] | None = None        # qubits per gate k (0 = whole register)
    disentangle_target_chi: int = 1            # bond dimension the circuits aim at
    disentangle_optimizer: str = "explicit"    # 'explicit' (env-SVD) or 'gradient' (Adam)
    disentangle_sweeps: int = 12               # explicit: max environment sweeps
    disentangle_gd_steps: int = 200            # gradient: Adam steps
    disentangle_gd_lr: float = 0.05            # gradient: Adam learning rate
    disentangle_tol: float = 1e-6
    disentangle_init: str = "identity"
    disentangle_target_mode: str = "adaptive"  # 'adaptive' or the paper's 'fixed'
    disentangle_target_per_chi: bool = False   # re-optimize per (D, chi') with target_chi = chi'
    disentangle_restarts: int = 1
    disentangle_seed: int = 0
    quantum_param_counting: str = "manifold"   # 'manifold' (angles) or 'entries'
    max_qubits: int = 13                       # skip layers needing a bigger register
    run_classical: bool = True                 # also record the pure-TN curve
    hybrid_csv_name: str = "hybrid_per_layer.csv"
    heal_mode: str = "full"                    # hybrid healing: "core" (chi' bond) or "full" (+U/V circuits)
    word_level: bool = False                   # report word-level PPL (Table I) instead of per-token
    measure_perplexity: bool = True            # False -> relative error / accuracy only (no model eval)


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
    optimizer: str             # 'explicit' | 'gradient' (classical rows: '-')
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
    ppl_unit: str = "token"                    # "token" or "word" (--word-level)
    # Healing (optional): brief LM-loss retraining of the swapped-in layer.
    heal_mode: str = "-"
    perplexity_healed: float = float("nan")
    ppl_ratio_healed: float = float("nan")
    heal_recovered_frac: float = float("nan")
    heal_final_loss: float = float("nan")
    heal_seconds: float = 0.0


def _heal_cols(mode, ppl_h, ratio_h, recovered, loss, secs) -> dict:
    """Assemble the six healing CSV columns (nan-safe rounding)."""
    def _r(x, n):
        return round(x, n) if isinstance(x, float) and x == x else x
    return dict(heal_mode=mode, perplexity_healed=_r(ppl_h, 4),
                ppl_ratio_healed=_r(ratio_h, 6), heal_recovered_frac=_r(recovered, 4),
                heal_final_loss=_r(loss, 4), heal_seconds=round(secs, 2))


def hybrid_csv_path(config: HybridConfig) -> Path:
    # Isolate each geometry AND training scheme in its own checkpoint file, so
    # non-comparable rows never mix (C(chi) differs by geometry; the hybrid curve
    # differs by optimizer).
    name = tensorized_name(config.hybrid_csv_name, config.tensorization)
    name = tensorized_name(name, config.disentangle_optimizer, default="explicit")
    return compactifai_dir(config) / name


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
    train = (f"gradient (Adam, {config.disentangle_gd_steps} steps @ "
             f"{config.disentangle_gd_lr})" if config.disentangle_optimizer == "gradient"
             else f"explicit (env-SVD, <= {config.disentangle_sweeps} sweeps)")
    print(f"             disentangling target chi = {config.disentangle_target_chi}, "
          f"Q counted as gate {config.quantum_param_counting}")
    print(f"             training scheme = {train}")

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

    budget = config.per_layer_eval_tokens
    probe_stride = config.per_layer_stride or config.max_length
    measure_ppl = config.measure_perplexity
    heal_on = config.heal and measure_ppl
    if config.heal and not measure_ppl:
        print("(--no-perplexity: healing is disabled -- it needs the LM loss.)")

    input_ids = None
    word_scale = 1.0
    if measure_ppl:
        input_ids = tokenize_corpus(tokenizer, load_cfg)

        def _token_evaluate():
            return perplexity_over_ids(
                model, input_ids, device, config.max_length, probe_stride,
                batch_size=config.ppl_batch_size, max_eval_tokens=budget, progress=False,
            )

        # Word-level PPL (Table I): PPL_word = PPL_token ** (scored_tokens/words).
        if config.word_level:
            span = min(budget or input_ids.size(1), input_ids.size(1))
            try:
                n_words = max(1, len(tokenizer.decode(input_ids[0, :span]).split()))
            except Exception:                          # noqa: BLE001
                n_words = span
            word_scale = span / n_words
            print(f"Word-level PPL: {span} tokens / {n_words} words -> exponent {word_scale:.4f}")

        def evaluate():
            ppl, tok, secs = _token_evaluate()
            return (ppl ** word_scale if config.word_level else ppl), tok, secs
    else:
        print("\nPerplexity disabled (--no-perplexity): recording relative error, "
              "disentangling accuracy, entropy and retained only.")

        def evaluate():
            return float("nan"), 0, 0.0

    path = hybrid_csv_path(config)
    existing = [] if config.force_recompute else _read_rows(path)
    rows_by_key = {(r.get("param_name"), r.get("method"), (r.get("tensorization") or "balanced"), str(r.get("chi")),
                    str(r.get("circuit_depth")), str(r.get("gate_size"))): r
                   for r in existing}

    def _present(key):
        r = rows_by_key.get(key)
        if r is None:
            return False
        if not heal_on:
            return True
        try:
            return math.isfinite(float(r.get("ppl_ratio_healed", "nan")))
        except (TypeError, ValueError):
            return False

    baseline = float("nan")
    eval_tokens = 0
    if measure_ppl:
        if not config.force_recompute:
            for _r in existing:
                try:
                    baseline = float(_r["ppl_baseline"]); break
                except (KeyError, ValueError, TypeError):
                    continue
        if not math.isfinite(baseline):
            print(f"\nBaseline (all layers dense) perplexity on {budget} tokens ...")
            baseline, eval_tokens, _ = evaluate()
            print(f"Baseline perplexity = {baseline:.4f}")
        else:
            eval_tokens = budget or input_ids.size(1)
            print(f"\nBaseline perplexity from checkpoint = {baseline:.4f}")

    # ---- Healing calibration set (disjoint split), tokenized once and reused.
    heal_batches: list = []
    _res0: dict = {}
    if heal_on:
        heal_cfg = BenchmarkConfig(
            model_id=config.model_id, device=config.device, dtype=config.dtype,
            trust_remote_code=config.trust_remote_code, revision=config.revision,
            dataset=config.heal_dataset or config.dataset,
            dataset_config=config.dataset_config, split=config.heal_split,
            text_column=config.text_column)
        heal_ids = tokenize_corpus(tokenizer, heal_cfg)
        heal_batches = make_heal_batches(heal_ids, config.max_length,
                                         config.heal_batch, config.heal_tokens)
        print(f"Healing on: {config.heal_steps} Adam steps @ lr {config.heal_lr} | "
              f"{config.heal_tokens} tok from '{config.heal_dataset or config.dataset}:"
              f"{config.heal_split}' ({len(heal_batches)} batches) | "
              f"classical -> MPO bond, hybrid -> {config.heal_mode}.")

    def _res0_for(name, orig):
        """A depth-0 disentangle (identity circuits) for qubit-geometry MPO-bond healing."""
        r = _res0.get(name)
        if r is None:
            r = disentangle(orig, gate_size=1, depth=0, tensorization=config.tensorization,
                            qubit_align=config.qubit_align, n_sites=config.mpo_sites,
                            target_chi=config.disentangle_target_chi, seed=config.disentangle_seed)
            _res0[name] = r
        return r

    def _recovered(ppl_cold, ppl_healed):
        damage = ppl_cold - baseline
        if damage <= 1e-9:
            return 1.0
        return max(0.0, min(1.0, (ppl_cold - ppl_healed) / damage))

    def _heal_and_eval(name, param, orig, chi, mode, *, plan=None, res=None):
        """Heal the swapped-in layer, score PPL, restore. Returns (ppl, loss, secs)."""
        t0 = time.perf_counter()
        if res is not None:
            healed, _n, loss = heal_hybrid(model, name, res, chi, mode, heal_batches,
                                           device, config.heal_steps, config.heal_lr)
        else:
            healed, _n, loss = heal_layer(model, name, plan, chi, heal_batches, device,
                                          steps=config.heal_steps, lr=config.heal_lr)
        with torch.no_grad():
            param.copy_(healed.to(dtype=param.dtype, device=param.device))
        ppl_h, _tok, _secs = evaluate()
        with torch.no_grad():
            param.copy_(orig.to(dtype=param.dtype, device=param.device))
        return ppl_h, float(loss), time.perf_counter() - t0

    # Enumerate the work: classical points first (cheap, no optimization), then
    # the hybrid grid grouped by (layer, D) so one disentangling serves all chi'.
    todo: list[tuple] = []
    for name, param in layers:
        if config.run_classical:
            exact = full_rank_chi(classical_plans[name].out_dims,
                                  classical_plans[name].in_dims)
            for chi in sorted({min(int(c), exact) for c in chis}):
                key = (name, "classical", config.tensorization, str(chi), "-1", "0")
                if not _present(key):
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
                           if not _present((name, "hybrid", config.tensorization, str(c),
                                            str(d), str(k_row)))]
                if missing:
                    hybrid_jobs.append((name, param, d, k_eff, k_row, missing))

    n_hybrid_points = sum(len(j[-1]) for j in hybrid_jobs)
    _n_opts = n_hybrid_points if config.disentangle_target_per_chi else len(hybrid_jobs)
    _opt_note = ("one per (D, chi'), target_chi'=chi'"
                 if config.disentangle_target_per_chi
                 else f"reused across chi', target_chi'={config.disentangle_target_chi}")
    print(f"\n{len(layers)} layers | classical points to run: {len(todo)} | "
          f"hybrid points to run: {n_hybrid_points} "
          f"({_n_opts} disentangling optimizations, {_opt_note})")
    if measure_ppl:
        windows = max(1, math.ceil((budget or input_ids.size(1)) / probe_stride))
        print(f"Each point = one perplexity evaluation over {budget} tokens "
              f"({windows} windows of {config.max_length}, stride {probe_stride}).")
    else:
        print("Each point = one MPO truncation / disentangle, no model eval.")
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
        ppl, secs = float("nan"), 0.0
        if measure_ppl:
            with torch.no_grad():
                param.copy_(approx.to(dtype=param.dtype, device=param.device))
            ppl, eval_tokens, secs = evaluate()
            with torch.no_grad():
                param.copy_(orig.to(dtype=param.dtype, device=param.device))
        layer_type, block = parse_layer_info(name)
        hmode, ph, pr, frac, hloss, hsec = "-", float("nan"), float("nan"), float("nan"), float("nan"), 0.0
        if heal_on and heal_batches:
            if config.tensorization == "balanced":
                ph, hloss, hsec = _heal_and_eval(name, param, orig, chi, "core", plan=plan)
            else:
                ph, hloss, hsec = _heal_and_eval(name, param, orig, chi, "core",
                                                 res=_res0_for(name, orig))
            pr = ph / baseline if baseline else float("nan")
            frac = _recovered(ppl, ph); hmode = "core"
        done += 1
        record(HybridRow(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id, param_name=name, layer_type=layer_type,
            depth=block, method="classical",
            tensorization=config.tensorization, optimizer="-", chi=chi, circuit_depth=-1,
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
            ppl_unit="word" if config.word_level else "token",
            **_heal_cols(hmode, ph, pr, frac, hloss, hsec),
        ))
        elapsed = time.perf_counter() - start
        print(f"  [{done}/{total_points}] classical {layer_type} d{block} "
              f"chi={chi:<4} C={params_mpo:<8,} rel.err={err:.4f} "
              + (f"ppl={ppl:.4f} (x{ppl / baseline:.4f})" if measure_ppl else "")
              + (f" heal x{pr:.4f}" if heal_on and pr == pr else "")
              + f"  elapsed={elapsed:.0f}s eta={elapsed / done * (total_points - done):.0f}s")

    # ---- Hybrid surface: PPL(chi', D) with the disentangling circuits in place.
    per_chi = config.disentangle_target_per_chi

    def _disentangle_op(op, k_eff, d, target_chi):
        return disentangle(
            op, gate_size=k_eff, depth=d,
            tensorization=config.tensorization, qubit_align=config.qubit_align,
            target_chi=target_chi, n_sites=config.mpo_sites,
            sweeps=config.disentangle_sweeps, tol=config.disentangle_tol,
            init=config.disentangle_init, seed=config.disentangle_seed,
            param_counting=config.quantum_param_counting,
            target_mode=config.disentangle_target_mode,
            optimizer=config.disentangle_optimizer,
            gd_steps=config.disentangle_gd_steps, gd_lr=config.disentangle_gd_lr,
            restarts=config.disentangle_restarts,
        )

    def _print_res(res):
        print(f"    {res.n_out_qubits}q x {res.n_in_qubits}q, Q(D)={res.quantum_params:,} "
              f"| retained {res.retained_classical:.4f} -> {res.retained:.4f} "
              f"| accuracy {res.accuracy:.4f} | entropy {res.entropy:.4f} "
              f"| {res.sweeps_run} sweeps in {res.seconds:.1f}s")

    for name, param, d, k_eff, k_row, missing in hybrid_jobs:
        orig = originals[name]
        layer_type, block = parse_layer_info(name)
        # Full-rank bond of the padded operator: geometry only, so it is the same
        # whatever the circuits or target_chi are -- clamp the chi' grid to it.
        _padded, _no, _ni = pad_to_qubits(orig)
        _geom = make_plan(_padded, config.tensorization, config.mpo_sites,
                          svd_cache=False, align=config.qubit_align)
        exact = full_rank_chi(_geom.out_dims, _geom.in_dims)
        chi_list = sorted({min(int(c), exact) for c in missing})

        # Default: one optimization at ``disentangle_target_chi`` reused across the
        # whole chi' grid (the residual SVD is truncated per chi', paper Table I).
        # ``disentangle_target_per_chi``: a fresh optimization per (D, chi') that
        # squeezes to ``target_chi = chi'`` -- best circuits for each bond, at N x
        # the disentangling cost.
        res_shared = None
        if not per_chi:
            print(f"  disentangling {layer_type} d{block} D={d} k={k_eff} "
                  f"target_chi'={config.disentangle_target_chi} ...")
            res_shared = _disentangle_op(orig, k_eff, d, config.disentangle_target_chi)
            _print_res(res_shared)

        for chi in chi_list:
            if per_chi:
                print(f"  disentangling {layer_type} d{block} D={d} k={k_eff} "
                      f"target_chi'={chi} ...")
                res = _disentangle_op(orig, k_eff, d, chi)
                _print_res(res)
            else:
                res = res_shared
            approx, c_params = hybrid_weight(res, chi)
            err = relative_error(orig, approx)
            ppl, secs = float("nan"), 0.0
            if measure_ppl:
                with torch.no_grad():
                    param.copy_(approx.to(dtype=param.dtype, device=param.device))
                ppl, eval_tokens, secs = evaluate()
                with torch.no_grad():
                    param.copy_(orig.to(dtype=param.dtype, device=param.device))
            q_params = res.quantum_params
            dense = int(orig.shape[0]) * int(orig.shape[1])
            hmode, ph, pr, frac, hloss, hsec = "-", float("nan"), float("nan"), float("nan"), float("nan"), 0.0
            if heal_on and heal_batches:
                ph, hloss, hsec = _heal_and_eval(name, param, orig, chi, config.heal_mode, res=res)
                pr = ph / baseline if baseline else float("nan")
                frac = _recovered(ppl, ph); hmode = config.heal_mode
            done += 1
            record(HybridRow(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model_id=config.model_id, param_name=name, layer_type=layer_type,
                depth=block, method="hybrid",
                tensorization=config.tensorization,
                optimizer=config.disentangle_optimizer, chi=chi, circuit_depth=d,
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
                ppl_unit="word" if config.word_level else "token",
                **_heal_cols(hmode, ph, pr, frac, hloss, hsec),
            ))
            elapsed = time.perf_counter() - start
            print(f"  [{done}/{total_points}] hybrid    {layer_type} d{block} "
                  f"k={k_row} D={d} chi'={chi:<4} C={c_params:<8,} Q={q_params:<8,} "
                  f"rel.err={err:.4f} "
                  + (f"ppl={ppl:.4f} (x{ppl / baseline:.4f})" if measure_ppl else "")
                  + (f" heal x{pr:.4f}" if heal_on and pr == pr else "")
                  + f"  elapsed={elapsed:.0f}s "
                  f"eta={elapsed / done * (total_points - done):.0f}s")

    with torch.no_grad():
        for name, param in layers:
            param.copy_(originals[name].to(dtype=param.dtype, device=param.device))
    print("\nOriginal weights restored.")
    print(f"\nHybrid sweep complete. Results in {path}")
    return path
