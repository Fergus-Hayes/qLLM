"""Sweep the MPO bond dimension of a CompactifAI-compressed model vs. perplexity.

For each bond dimension ``chi`` (logarithmically spaced), every eligible
Self-Attention / MLP weight matrix in the decoder blocks is replaced by its
truncated MPO reconstruction, and the model's perplexity is measured.

Speed strategy
--------------
* The corpus is tokenized **once** and re-scored for every ``chi``.
* Each layer's SVD is computed **once** and cached; a 2-site MPO needs exactly
  one SVD, so every ``chi`` in the sweep is produced by truncation alone. This
  turns ``n_chi x n_layers`` SVDs into ``n_layers``.
* The cache build can run across threads (``--svd-workers``); LAPACK releases
  the GIL, so independent layer SVDs overlap.
* Perplexity scoring batches sliding windows into single forward passes
  (``--ppl-batch-size``) and honours ``--device``/``--dtype``/``--threads``.
* Results are checkpointed per ``chi``, so an interrupted sweep resumes.

Per-layer mode optionally applies the paper's *healing* step (``--heal``): after
each layer is truncated, its MPO tensors are briefly retrained (every other
weight frozen) and the healed perplexity recorded alongside the raw one.
"""

from __future__ import annotations

import csv
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import (
    BYTES_PER_MB,
    BenchmarkConfig,
    load_model_and_tokenizer,
    perplexity_over_ids,
    resolve_device,
    tokenize_corpus,
)
from .compactifai import (
    full_rank_chi,
    log_spaced_ints,
    mpo_bond_dims,
    mpo_param_count,
    relative_error,
)
from .compactifai_heal import heal_layer, make_heal_batches
from .layer_analysis import parse_layer_info
from .qubit_mpo import make_plan, plan_compress

# Default targets: the Self-Attention and MLP sub-modules inside decoder blocks,
# which is exactly the set CompactifAI tensorizes (embeddings / head excluded).
# Combined with the 2-D and in-a-numbered-block filters, this selects the
# projection weight matrices for essentially any transformer naming scheme.
DEFAULT_INCLUDE = r"(self_attn|attention|attn|mlp|feed_forward|ffn)\."
DEFAULT_EXCLUDE = r"(embed|lm_head|wte|wpe|norm|layernorm|\.bias$)"
# The "last MLP layer" of a block (its output projection) -- the paper reports
# this one is the most sensitive to compression.
MLP_OUTPUT_RE = r"(mlp|feed_forward|ffn)\.(down_proj|c_proj|w2|fc2|down)\b"


@dataclass
class CompactifaiConfig:
    model_id: str
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    revision: str | None = None

    # Compression.
    tensorization: str = "balanced"     # "balanced" (2-site) or "qubit" (paper geometry)
    qubit_align: str = "msb"            # qubit MPO: pair leading ("msb") or trailing ("lsb")
    mpo_sites: int = 2
    chi_min: int = 2
    chi_max: int | None = None          # None -> auto (largest useful chi)
    num_chi: int = 12
    chi_values: list[int] | None = None  # explicit override
    include_pattern: str = DEFAULT_INCLUDE
    exclude_pattern: str = DEFAULT_EXCLUDE
    exclude_down_proj: bool = False     # paper: last MLP layer is most sensitive
    min_depth: int | None = None        # only compress blocks with depth >= this
    force_all: bool = False             # compress even when MPO is not smaller
    svd_workers: int = 1
    svd_cache: bool = True

    # Evaluation corpus.
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    text_column: str = "text"
    max_length: int = 1024
    stride: int = 512
    max_eval_tokens: int | None = 20000
    ppl_batch_size: int = 8

    # Per-layer profiling (one layer compressed at a time -> a curve per layer).
    profile_depths: list[int] | None = None   # explicit block indices
    num_depths: int | None = None             # or: auto-pick N evenly spaced depths
    layer_types: list[str] | None = None      # restrict to these layer types
    per_layer_eval_tokens: int | None = 8192  # budget per evaluation
    per_layer_stride: int | None = None       # default: non-overlapping windows

    # Healing: brief retraining of the compressed layer's MPO tensors (all other
    # weights frozen) to recover accuracy lost to truncation. Uses a disjoint
    # train split so healed perplexity is not measured on the healing data.
    heal: bool = False
    heal_steps: int = 40
    heal_lr: float = 1e-3
    heal_tokens: int = 16384
    heal_batch: int = 2
    heal_split: str = "train"
    heal_dataset: str | None = None           # default: same as eval dataset

    # Output.
    results_dir: str = "results/llms"
    csv_name: str = "compactifai_sweep.csv"
    layer_csv_name: str = "compactifai_layers.csv"
    per_layer_csv_name: str = "compactifai_per_layer.csv"
    write_layer_csv: bool = True
    make_plots: bool = True
    force_recompute: bool = False


@dataclass
class SweepRow:
    timestamp: str
    model_id: str
    device: str
    dtype: str
    chi: int
    mpo_sites: int
    layers_compressed: int
    layers_eligible: int
    params_total_original: int
    params_total_compressed: int
    param_reduction_pct: float
    memory_original_mb: float
    memory_compressed_mb: float
    memory_reduction_pct: float
    compressed_layer_params_original: int
    compressed_layer_params_mpo: int
    layer_compression_pct: float
    mean_relative_error: float
    perplexity: float
    ppl_baseline: float
    ppl_ratio: float
    eval_tokens: int
    dataset: str
    compress_seconds: float
    eval_seconds: float


@dataclass
class LayerRow:
    chi: int
    param_name: str
    layer_type: str
    depth: int
    rows: int
    cols: int
    out_dims: str
    in_dims: str
    bond_dims: str
    params_original: int
    params_mpo: int
    compression_ratio: float
    compressed: bool
    relative_error: float


@dataclass
class PerLayerRow:
    """One (layer, chi) point: only that layer is compressed, all others dense."""
    timestamp: str
    model_id: str
    param_name: str
    layer_type: str
    depth: int
    chi: int
    rows: int
    cols: int
    params_original: int
    params_mpo: int
    compression_ratio: float
    relative_error: float
    perplexity: float
    ppl_baseline: float
    ppl_ratio: float
    perplexity_healed: float
    ppl_ratio_healed: float
    heal_recovered_frac: float
    heal_steps: int
    heal_final_loss: float
    heal_seconds: float
    eval_tokens: int
    eval_seconds: float


# --------------------------------------------------------------------------- #
# CSV helpers (atomic rewrite + per-chi checkpoint)
# --------------------------------------------------------------------------- #
def _write_rows(path: Path, rows: list[dict], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
    tmp.replace(path)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def compactifai_dir(config: CompactifaiConfig) -> Path:
    """<results-dir>/<model>/compactifai/ -- all CompactifAI outputs live here."""
    name = config.model_id.rstrip("/").split("/")[-1]
    return Path(config.results_dir) / name / "compactifai"


def _migrate_legacy(old_path: Path, new_path: Path) -> None:
    """Move a file written by the older flat layout, preserving checkpoints."""
    if old_path.exists() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.replace(new_path)
        print(f"Migrated {old_path} -> {new_path}")


def tensorized_name(name: str, tensorization: str) -> str:
    """Insert a non-default tensorization into a CSV name so geometries never
    share a checkpoint file (``compactifai_per_layer.csv`` ->
    ``compactifai_per_layer_qubit.csv``). ``balanced`` keeps the plain name."""
    if tensorization == "balanced":
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_{tensorization}{dot}{ext}" if dot else f"{name}_{tensorization}"


def sweep_csv_path(config: CompactifaiConfig) -> Path:
    name = config.model_id.rstrip("/").split("/")[-1]
    base = Path(config.results_dir) / name
    fname = tensorized_name(config.csv_name, config.tensorization)
    new_path = compactifai_dir(config) / fname
    _migrate_legacy(base / fname, new_path)
    return new_path


def layer_csv_path(config: CompactifaiConfig) -> Path:
    name = config.model_id.rstrip("/").split("/")[-1]
    base = Path(config.results_dir) / name
    fname = tensorized_name(config.layer_csv_name, config.tensorization)
    new_path = compactifai_dir(config) / fname
    _migrate_legacy(base / fname, new_path)
    return new_path


def per_layer_csv_path(config: CompactifaiConfig) -> Path:
    return compactifai_dir(config) / tensorized_name(
        config.per_layer_csv_name, config.tensorization)


def evenly_spaced(values: list[int], count: int) -> list[int]:
    """Pick ``count`` evenly spaced entries from ``values``, keeping both ends."""
    values = sorted(values)
    if count <= 0 or count >= len(values):
        return values
    if count == 1:
        return [values[0]]
    idx = {round(i * (len(values) - 1) / (count - 1)) for i in range(count)}
    return [values[i] for i in sorted(idx)]


# --------------------------------------------------------------------------- #
# Layer selection
# --------------------------------------------------------------------------- #
def select_layers(model, config: CompactifaiConfig) -> list[tuple[str, torch.nn.Parameter]]:
    """Eligible 2-D weight matrices: SA/MLP projections inside decoder blocks."""
    include = re.compile(config.include_pattern)
    exclude = re.compile(config.exclude_pattern)
    selected = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        if exclude.search(name) or not include.search(name):
            continue
        layer_type, depth = parse_layer_info(name)
        if depth < 0:                      # not inside a numbered decoder block
            continue
        if config.min_depth is not None and depth < config.min_depth:
            continue
        if config.exclude_down_proj and re.search(MLP_OUTPUT_RE, name):
            continue          # paper: block-output MLP layer is most sensitive
        selected.append((name, param))
    return selected


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(config: CompactifaiConfig) -> Path:
    device = resolve_device(config.device)
    load_cfg = BenchmarkConfig(
        model_id=config.model_id, device=config.device, dtype=config.dtype,
        trust_remote_code=config.trust_remote_code, revision=config.revision,
        dataset=config.dataset, dataset_config=config.dataset_config,
        split=config.split, text_column=config.text_column,
    )
    model, tokenizer = load_model_and_tokenizer(load_cfg, device)

    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = next(model.parameters()).element_size()

    layers = select_layers(model, config)
    if not layers:
        raise RuntimeError("No eligible layers matched; check --include/--exclude.")
    layer_params = sum(p.numel() for _, p in layers)
    print(f"\nEligible layers for MPO compression: {len(layers)} "
          f"({layer_params:,} params = {100 * layer_params / total_params:.1f}% of the model)")

    # ---- Originals (kept on CPU so restoring is exact and device memory is free)
    originals = {name: p.detach().to("cpu", copy=True) for name, p in layers}

    # ---- Build decomposition plans (cached SVDs), optionally across threads
    print(f"\nBuilding MPO plans ({config.mpo_sites}-site"
          f"{', cached SVD' if config.svd_cache else ''}, "
          f"{config.svd_workers} worker(s)) ...")
    t0 = time.perf_counter()
    plans: dict[str, object] = {}
    done = {"n": 0}

    def _build(item):
        name, param = item
        plan = make_plan(originals[name], config.tensorization, config.mpo_sites,
                         svd_cache=config.svd_cache, align=config.qubit_align)
        plans[name] = plan
        done["n"] += 1
        if done["n"] % max(1, len(layers) // 10) == 0 or done["n"] == len(layers):
            print(f"    plans {done['n']}/{len(layers)}  "
                  f"({time.perf_counter() - t0:.0f}s)")
        return name

    if config.svd_workers > 1:
        with ThreadPoolExecutor(max_workers=config.svd_workers) as pool:
            list(pool.map(_build, layers))
    else:
        for item in layers:
            _build(item)
    plan_seconds = time.perf_counter() - t0
    print(f"Plans ready in {plan_seconds:.1f}s "
          f"(SVDs reused for every bond dimension).")

    # ---- Bond dimensions (logarithmic spacing)
    auto_max = max(plans[n].max_chi for n, _ in layers)
    chi_max = config.chi_max if config.chi_max else auto_max
    if config.chi_values:
        chis = sorted({int(c) for c in config.chi_values if c >= 1})
    else:
        chis = log_spaced_ints(config.chi_min, chi_max, config.num_chi)
    print(f"\nBond dimensions (log-spaced, max useful chi = {auto_max}): {chis}")

    # ---- Corpus tokenized once, re-scored for every chi
    input_ids = tokenize_corpus(tokenizer, load_cfg)
    eval_tokens_avail = input_ids.size(1)
    print(f"Corpus tokenized once: {eval_tokens_avail} tokens "
          f"(using up to {config.max_eval_tokens or eval_tokens_avail}).")

    def evaluate(progress=True):
        return perplexity_over_ids(
            model, input_ids, device, config.max_length, config.stride,
            batch_size=config.ppl_batch_size,
            max_eval_tokens=config.max_eval_tokens, progress=progress,
        )

    # ---- Checkpoint state
    path = sweep_csv_path(config)
    layer_path = layer_csv_path(config)
    existing = _read_rows(path)
    done_chis = set()
    if not config.force_recompute:
        for row in existing:
            try:
                done_chis.add(int(row["chi"]))
            except (KeyError, ValueError):
                pass
    results_by_chi = {}
    for row in existing:
        try:
            results_by_chi[int(row["chi"])] = row
        except (KeyError, ValueError):
            pass
    layer_rows_existing = [] if config.force_recompute else _read_rows(layer_path)

    # ---- Baseline (uncompressed) perplexity
    baseline = None
    for row in existing:
        try:
            baseline = float(row["ppl_baseline"])
            break
        except (KeyError, ValueError, TypeError):
            continue
    if baseline is None or not math.isfinite(baseline) or config.force_recompute:
        print("\n=== Baseline (uncompressed) perplexity ===")
        baseline, _, _ = evaluate()
        print(f"Baseline perplexity = {baseline:.4f}")
    else:
        print(f"\nBaseline perplexity from checkpoint = {baseline:.4f}")

    todo = [c for c in chis if c not in done_chis]
    print(f"\n{len(chis)} bond dimensions requested, {len(chis) - len(todo)} "
          f"already in checkpoint, {len(todo)} to run.")
    print(f"Checkpoint / results CSV: {path}")

    memory_original_mb = total_params * bytes_per_param / BYTES_PER_MB
    sweep_start = time.perf_counter()

    for idx, chi in enumerate(todo, start=1):
        print(f"\n[{idx}/{len(todo)}] === chi = {chi} ===")
        t_c = time.perf_counter()
        n_compressed = 0
        sum_orig, sum_mpo, errors = 0, 0, []
        layer_rows = []

        for name, param in layers:
            plan = plans[name]
            orig = originals[name]
            params_mpo = mpo_param_count(plan.out_dims, plan.in_dims, chi)
            beneficial = params_mpo < plan.dense_params
            layer_type, depth = parse_layer_info(name)

            if beneficial or config.force_all:
                approx, params_mpo = plan_compress(orig, plan, chi)
                err = relative_error(orig, approx)
                with torch.no_grad():
                    param.copy_(approx.to(dtype=param.dtype, device=param.device))
                n_compressed += 1
                sum_orig += plan.dense_params
                sum_mpo += params_mpo
                errors.append(err)
                compressed = True
            else:
                # Not worth compressing at this chi -> keep the dense weights.
                with torch.no_grad():
                    param.copy_(orig.to(dtype=param.dtype, device=param.device))
                sum_orig += plan.dense_params
                sum_mpo += plan.dense_params
                err = 0.0
                compressed = False

            if config.write_layer_csv:
                layer_rows.append(asdict(LayerRow(
                    chi=chi, param_name=name, layer_type=layer_type, depth=depth,
                    rows=int(param.shape[0]), cols=int(param.shape[1]),
                    out_dims="x".join(map(str, plan.out_dims)),
                    in_dims="x".join(map(str, plan.in_dims)),
                    bond_dims="x".join(map(str, mpo_bond_dims(plan.out_dims, plan.in_dims, chi))),
                    params_original=plan.dense_params,
                    params_mpo=params_mpo,
                    compression_ratio=round(params_mpo / plan.dense_params, 6),
                    compressed=compressed,
                    relative_error=round(err, 6),
                )))

        compress_seconds = time.perf_counter() - t_c
        mean_err = sum(errors) / len(errors) if errors else 0.0
        params_compressed_total = total_params - sum_orig + sum_mpo
        mem_compressed = params_compressed_total * bytes_per_param / BYTES_PER_MB
        print(f"    compressed {n_compressed}/{len(layers)} layers in "
              f"{compress_seconds:.1f}s | layer params "
              f"{sum_orig:,} -> {sum_mpo:,} "
              f"({100 * (1 - sum_mpo / sum_orig):.1f}% smaller) | "
              f"mean rel.err {mean_err:.4f}")
        print(f"    model params {total_params:,} -> {params_compressed_total:,} "
              f"({100 * (1 - params_compressed_total / total_params):.1f}% smaller), "
              f"{memory_original_mb:.1f} MB -> {mem_compressed:.1f} MB")

        ppl, eval_tokens, eval_seconds = evaluate()
        print(f"    perplexity = {ppl:.4f}  (baseline {baseline:.4f}, "
              f"x{ppl / baseline:.3f})  [{eval_seconds:.1f}s]")

        row = SweepRow(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id, device=device, dtype=config.dtype,
            chi=chi, mpo_sites=config.mpo_sites,
            layers_compressed=n_compressed, layers_eligible=len(layers),
            params_total_original=total_params,
            params_total_compressed=params_compressed_total,
            param_reduction_pct=round(100 * (1 - params_compressed_total / total_params), 4),
            memory_original_mb=round(memory_original_mb, 3),
            memory_compressed_mb=round(mem_compressed, 3),
            memory_reduction_pct=round(100 * (1 - mem_compressed / memory_original_mb), 4),
            compressed_layer_params_original=sum_orig,
            compressed_layer_params_mpo=sum_mpo,
            layer_compression_pct=round(100 * (1 - sum_mpo / sum_orig), 4),
            mean_relative_error=round(mean_err, 6),
            perplexity=round(ppl, 4),
            ppl_baseline=round(baseline, 4),
            ppl_ratio=round(ppl / baseline, 4) if baseline else float("nan"),
            eval_tokens=eval_tokens, dataset=f"{config.dataset}/{config.dataset_config}:{config.split}",
            compress_seconds=round(compress_seconds, 2),
            eval_seconds=round(eval_seconds, 2),
        )
        results_by_chi[chi] = asdict(row)
        _write_rows(path, [results_by_chi[c] for c in sorted(results_by_chi)],
                    [f.name for f in fields(SweepRow)])

        if config.write_layer_csv:
            keep = [r for r in layer_rows_existing if str(r.get("chi")) != str(chi)]
            layer_rows_existing = keep + layer_rows
            layer_rows_existing.sort(key=lambda r: (int(r["chi"]), str(r["param_name"])))
            _write_rows(layer_path, layer_rows_existing, [f.name for f in fields(LayerRow)])

        elapsed = time.perf_counter() - sweep_start
        eta = elapsed / idx * (len(todo) - idx)
        print(f"    checkpointed -> {path}  "
              f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)")

    # ---- Restore the pristine model
    with torch.no_grad():
        for name, param in layers:
            param.copy_(originals[name].to(dtype=param.dtype, device=param.device))
    print("\nOriginal weights restored.")

    print_summary(_read_rows(path))
    if config.make_plots:
        plot_global_sweep(_read_rows(path), compactifai_dir(config))
    print(f"\nSweep complete. Results in {path}")
    if config.write_layer_csv:
        print(f"Per-layer detail in {layer_csv_path(config)}")
    return path


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 78)
    print("COMPACTIFAI BOND-DIMENSION SWEEP")
    print("=" * 78)
    print(f"{'chi':>6} {'params':>14} {'model -%':>9} {'layer -%':>9} "
          f"{'rel.err':>8} {'perplexity':>11} {'x base':>7}")
    for r in sorted(rows, key=lambda x: int(x["chi"])):
        try:
            print(f"{int(r['chi']):>6} {int(r['params_total_compressed']):>14,} "
                  f"{float(r['param_reduction_pct']):>8.2f}% "
                  f"{float(r['layer_compression_pct']):>8.2f}% "
                  f"{float(r['mean_relative_error']):>8.4f} "
                  f"{float(r['perplexity']):>11.4f} "
                  f"{float(r['ppl_ratio']):>7.3f}")
        except (KeyError, ValueError, TypeError):
            continue
    try:
        print(f"\nBaseline (uncompressed) perplexity: {float(rows[0]['ppl_baseline']):.4f}")
    except (KeyError, ValueError, TypeError):
        pass


# --------------------------------------------------------------------------- #
# Per-layer profiling: perplexity vs. bond dimension, one layer at a time
# --------------------------------------------------------------------------- #
def run_per_layer_sweep(config: CompactifaiConfig) -> Path:
    """Compress ONE layer at a time and measure perplexity vs. bond dimension.

    This is the paper's layer-sensitivity profiling: with every other layer left
    dense, the curve for a layer isolates how much truncation *that* layer
    tolerates. Cost is ``n_layers x n_chi`` evaluations, so it uses its own
    (smaller) token budget, ``--per-layer-eval-tokens``, and is checkpointed per
    (layer, chi) pair.
    """
    device = resolve_device(config.device)
    load_cfg = BenchmarkConfig(
        model_id=config.model_id, device=config.device, dtype=config.dtype,
        trust_remote_code=config.trust_remote_code, revision=config.revision,
        dataset=config.dataset, dataset_config=config.dataset_config,
        split=config.split, text_column=config.text_column,
    )
    model, tokenizer = load_model_and_tokenizer(load_cfg, device)

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
    if not layers:
        raise RuntimeError("No layers selected for per-layer profiling.")

    types_present = sorted({parse_layer_info(n)[0] for n, _ in layers})
    print(f"\nProfiling plan: {len(keep)} of {len(available)} decoder blocks "
          f"{keep}")
    print(f"                {len(types_present)} layer types {types_present}")

    originals = {name: p.detach().to("cpu", copy=True) for name, p in layers}

    print(f"                {len(layers)} layers to profile")
    print("\nBuilding MPO plans (cached SVDs) ...")
    t0 = time.perf_counter()
    plans = {}
    for i, (name, _) in enumerate(layers, start=1):
        plans[name] = make_plan(originals[name], config.tensorization, config.mpo_sites,
                                svd_cache=config.svd_cache, align=config.qubit_align)
        if i % max(1, len(layers) // 10) == 0 or i == len(layers):
            print(f"    plans {i}/{len(layers)} ({time.perf_counter() - t0:.0f}s)")

    auto_max = max(plans[n].max_chi for n, _ in layers)
    chi_max = config.chi_max if config.chi_max else auto_max
    if config.chi_values:
        chis = sorted({int(c) for c in config.chi_values if c >= 1})
    else:
        chis = log_spaced_ints(config.chi_min, chi_max, config.num_chi)
    print(f"Bond dimensions (log-spaced): {chis}")

    input_ids = tokenize_corpus(tokenizer, load_cfg)
    budget = config.per_layer_eval_tokens
    # Non-overlapping windows by default: the probe compares conditions on the
    # SAME tokens, so overlap buys little while costing ~2x the forward passes.
    probe_stride = config.per_layer_stride or config.max_length

    def evaluate():
        return perplexity_over_ids(
            model, input_ids, device, config.max_length, probe_stride,
            batch_size=config.ppl_batch_size, max_eval_tokens=budget, progress=False,
        )

    # Healing corpus: a DISJOINT split (train by default) so healed perplexity is
    # never measured on the healing tokens. Tokenized once and reused per layer.
    heal_batches = []
    if config.heal:
        if config.mpo_sites != 2 or not config.svd_cache:
            raise RuntimeError("Healing requires --mpo-sites 2 and the SVD cache on.")
        heal_cfg = BenchmarkConfig(
            model_id=config.model_id, dataset=config.heal_dataset or config.dataset,
            dataset_config=config.dataset_config, split=config.heal_split,
            text_column=config.text_column,
        )
        heal_ids = tokenize_corpus(tokenizer, heal_cfg)
        heal_batches = make_heal_batches(heal_ids, config.max_length,
                                         config.heal_batch, config.heal_tokens)
        print(f"Healing enabled: {config.heal_steps} Adam steps @ lr {config.heal_lr} "
              f"on {config.heal_tokens} tokens from "
              f"'{config.heal_dataset or config.dataset}:{config.heal_split}' "
              f"({len(heal_batches)} batches of {config.heal_batch}x{config.max_length}); "
              f"only the compressed layer's MPO tensors are trained.")

    path = per_layer_csv_path(config)
    existing = [] if config.force_recompute else _read_rows(path)

    def _row_complete(r):
        # When healing is requested, a checkpointed row counts as done only if it
        # already carries a finite healed perplexity (so a prior no-heal run does
        # not mask a heal run).
        if not config.heal:
            return True
        try:
            return math.isfinite(float(r.get("perplexity_healed", "nan")))
        except (TypeError, ValueError):
            return False

    done = {(r.get("param_name"), str(r.get("chi"))) for r in existing if _row_complete(r)}
    rows_by_key = {(r.get("param_name"), str(r.get("chi"))): r for r in existing}

    print(f"\nBaseline (all layers dense) perplexity on {budget} tokens ...")
    baseline, eval_tokens, _ = evaluate()
    print(f"Baseline perplexity = {baseline:.4f}")

    # Per-layer chi list: a chi above a layer's full rank leaves every bond
    # untruncated, so the reconstruction is identical to the exact one. Clamping
    # per layer drops those duplicate points instead of re-evaluating them.
    layer_chis = {}
    for name, _ in layers:
        exact = full_rank_chi(plans[name].out_dims, plans[name].in_dims)
        layer_chis[name] = sorted({min(int(c), exact) for c in chis})

    total_points = sum(len(layer_chis[n]) for n, _ in layers)
    todo = [(n, p, c) for (n, p) in layers
            for c in layer_chis[n] if (n, str(c)) not in done]
    print(f"\n{len(layers)} layers x up to {len(chis)} chi = {total_points} points "
          f"(chi clamped per layer to its exact-reconstruction rank); "
          f"{len(todo)} to evaluate, {total_points - len(todo)} checkpointed.")
    windows = max(1, math.ceil((budget or input_ids.size(1)) / probe_stride))
    print(f"Each point = one perplexity evaluation over {budget} tokens "
          f"({windows} windows of {config.max_length}, stride {probe_stride}) "
          f"with only that layer compressed.")
    print(f"Checkpoint / results CSV: {path}")

    start = time.perf_counter()
    for i, (name, param, chi) in enumerate(todo, start=1):
        plan = plans[name]
        orig = originals[name]
        approx, params_mpo = plan_compress(orig, plan, chi)
        err = relative_error(orig, approx)
        with torch.no_grad():
            param.copy_(approx.to(dtype=param.dtype, device=param.device))
        ppl, eval_tokens, secs = evaluate()
        with torch.no_grad():                      # restore immediately
            param.copy_(orig.to(dtype=param.dtype, device=param.device))

        # Healing: retrain this layer's MPO tensors (others frozen), then re-score.
        ppl_healed, ratio_healed, recovered = float("nan"), float("nan"), float("nan")
        heal_loss, heal_secs = float("nan"), 0.0
        if config.heal and heal_batches:
            th = time.perf_counter()
            healed_w, _, heal_loss = heal_layer(
                model, name, plan, chi, heal_batches, device,
                steps=config.heal_steps, lr=config.heal_lr, log=False,
            )
            with torch.no_grad():
                param.copy_(healed_w.to(dtype=param.dtype, device=param.device))
            ppl_healed, _, _ = evaluate()
            with torch.no_grad():
                param.copy_(orig.to(dtype=param.dtype, device=param.device))
            heal_secs = time.perf_counter() - th
            ratio_healed = ppl_healed / baseline if baseline else float("nan")
            # Fraction of the truncation damage recovered by healing, in [0, 1].
            damage = ppl - baseline
            recovered = max(0.0, min(1.0, (ppl - ppl_healed) / damage)) if damage > 1e-9 else 1.0

        layer_type, depth = parse_layer_info(name)
        row = asdict(PerLayerRow(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id, param_name=name, layer_type=layer_type,
            depth=depth, chi=chi, rows=int(param.shape[0]), cols=int(param.shape[1]),
            params_original=plan.dense_params, params_mpo=params_mpo,
            compression_ratio=round(params_mpo / plan.dense_params, 6),
            relative_error=round(err, 6), perplexity=round(ppl, 4),
            ppl_baseline=round(baseline, 4),
            ppl_ratio=round(ppl / baseline, 4) if baseline else float("nan"),
            perplexity_healed=round(ppl_healed, 4) if ppl_healed == ppl_healed else float("nan"),
            ppl_ratio_healed=round(ratio_healed, 4) if ratio_healed == ratio_healed else float("nan"),
            heal_recovered_frac=round(recovered, 4) if recovered == recovered else float("nan"),
            heal_steps=config.heal_steps if config.heal else 0,
            heal_final_loss=round(heal_loss, 4) if heal_loss == heal_loss else float("nan"),
            heal_seconds=round(heal_secs, 2),
            eval_tokens=eval_tokens, eval_seconds=round(secs, 2),
        ))
        rows_by_key[(name, str(chi))] = row
        ordered = sorted(rows_by_key.values(),
                         key=lambda r: (str(r["param_name"]), int(r["chi"])))
        _write_rows(path, ordered, [f.name for f in fields(PerLayerRow)])

        elapsed = time.perf_counter() - start
        eta = elapsed / i * (len(todo) - i)
        if i == 1 and len(todo) > 1:
            print(f"    (first point took {elapsed:.1f}s -> estimated total "
                  f"{elapsed * len(todo) / 60:.0f} min for {len(todo)} points; "
                  f"reduce with --profile-depths / --num-chi / --per-layer-eval-tokens)")
        heal_str = (f" -> healed {ppl_healed:.4f} (x{ratio_healed:.3f}, "
                    f"{recovered*100:.0f}% recovered)"
                    if config.heal and heal_batches else "")
        print(f"  [{i}/{len(todo)}] {layer_type} d{depth} chi={chi:<4} "
              f"ratio={params_mpo / plan.dense_params:.3f} rel.err={err:.4f} "
              f"ppl={ppl:.4f} (x{ppl / baseline:.3f}){heal_str}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    with torch.no_grad():
        for name, param in layers:
            param.copy_(originals[name].to(dtype=param.dtype, device=param.device))
    print("\nOriginal weights restored.")

    if config.make_plots:
        plot_per_layer(_read_rows(path), compactifai_dir(config))
    print(f"\nPer-layer profiling complete. Results in {path}")
    return path


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                       # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return None


def plot_per_layer(rows: list[dict], out_dir: Path) -> None:
    """Perplexity vs. bond dimension, one curve per layer.

    Panels are grouped by layer type; within a panel each curve is one decoder
    block (depth), coloured light->dark with increasing depth. The dashed line is
    the uncompressed baseline.
    """
    plt = _import_pyplot()
    if plt is None or not rows:
        return

    data = {}
    baseline = None
    for r in rows:
        try:
            lt, depth, chi = r["layer_type"], int(r["depth"]), int(r["chi"])
            ppl = float(r["perplexity"])
            baseline = float(r["ppl_baseline"])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(ppl):
            continue
        healed = None
        try:
            hv = float(r.get("perplexity_healed", "nan"))
            healed = hv if math.isfinite(hv) else None
        except (TypeError, ValueError):
            healed = None
        data.setdefault(lt, {}).setdefault(depth, []).append((chi, ppl, healed))

    if not data:
        print("(No per-layer rows to plot.)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    types = sorted(data)
    ncols = min(3, len(types))
    nrows = math.ceil(len(types) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows),
                             squeeze=False, sharex=True)
    depths = sorted({d for t in data.values() for d in t})
    dmin, dmax = (min(depths), max(depths)) if depths else (0, 1)
    cmap = plt.get_cmap("viridis")

    for idx, lt in enumerate(types):
        ax = axes[idx // ncols][idx % ncols]
        any_healed = False
        for depth in sorted(data[lt]):
            pts = sorted(data[lt][depth])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            frac = (depth - dmin) / (dmax - dmin) if dmax > dmin else 0.5
            ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=cmap(frac),
                    label=f"block {depth}")
            hpts = [(c, h) for (c, _p, h) in pts if h is not None]
            if hpts:
                any_healed = True
                ax.plot([c for c, _ in hpts], [h for _, h in hpts], marker="^",
                        ms=3, lw=1.2, ls="--", color=cmap(frac))
        if baseline and math.isfinite(baseline):
            ax.axhline(baseline, ls="--", lw=1.2, color="crimson",
                       label="baseline (dense)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lt, fontsize=11)
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_ylabel("perplexity")
        ax.grid(True, which="both", alpha=0.25)
        if len(data[lt]) <= 12:
            ax.legend(fontsize="x-small", ncol=2)

    for j in range(len(types), nrows * ncols):     # hide unused panels
        axes[j // ncols][j % ncols].axis("off")

    if depths and len(depths) > 12:
        norm = plt.Normalize(vmin=dmin, vmax=dmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axes, label="decoder block (depth)", fraction=0.02, pad=0.01)

    healed_any = any(h is not None for lt in data.values() for dd in lt.values()
                     for (_c, _p, h) in dd)
    subtitle = "  (solid = truncated, dashed ^ = healed)" if healed_any else ""
    fig.suptitle("CompactifAI: perplexity vs. MPO bond dimension "
                 "(one layer compressed at a time)" + subtitle, fontsize=13)
    path = out_dir / "perplexity_vs_bond_dimension_per_layer.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")

    # Companion: every layer on a single axes, coloured by layer type.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = plt.get_cmap("tab10")
    for i, lt in enumerate(types):
        first = True
        for depth in sorted(data[lt]):
            pts = sorted(data[lt][depth])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.0, alpha=0.55,
                    color=colors(i % 10), label=lt if first else None)
            hpts = [(c, h) for (c, _p, h) in pts if h is not None]
            if hpts:
                ax.plot([c for c, _ in hpts], [h for _, h in hpts], lw=1.0,
                        ls="--", alpha=0.55, color=colors(i % 10))
            first = False
    if baseline and math.isfinite(baseline):
        ax.axhline(baseline, ls="--", lw=1.3, color="black", label="baseline (dense)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("perplexity")
    ax.set_title("Perplexity vs. bond dimension for every profiled layer")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small", ncol=2)
    path = out_dir / "perplexity_vs_bond_dimension_all_layers.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")


def plot_global_sweep(rows: list[dict], out_dir: Path) -> None:
    """Perplexity and compression vs. bond dimension for the whole-model sweep."""
    plt = _import_pyplot()
    if plt is None or not rows:
        return
    pts = []
    baseline = None
    for r in rows:
        try:
            pts.append((int(r["chi"]), float(r["perplexity"]),
                        float(r["param_reduction_pct"])))
            baseline = float(r["ppl_baseline"])
        except (KeyError, ValueError, TypeError):
            continue
    if not pts:
        return
    pts.sort()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color="#2a6f97",
            label="all layers compressed")
    if baseline and math.isfinite(baseline):
        ax.axhline(baseline, ls="--", color="crimson", label="baseline (dense)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("perplexity")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker="s", ms=4,
             color="#888", alpha=0.7, label="model params removed (%)")
    ax2.set_ylabel("model parameters removed (%)")
    ax.set_title("CompactifAI whole-model sweep")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize="small", loc="best")
    path = out_dir / "perplexity_vs_bond_dimension_global.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot {path}")
