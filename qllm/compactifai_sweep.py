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
    build_plan,
    compress_weight,
    log_spaced_ints,
    mpo_bond_dims,
    mpo_param_count,
    relative_error,
)
from .layer_analysis import parse_layer_info

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

    # Output.
    results_dir: str = "results/llms"
    csv_name: str = "compactifai_sweep.csv"
    layer_csv_name: str = "compactifai_layers.csv"
    write_layer_csv: bool = True
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


def sweep_csv_path(config: CompactifaiConfig) -> Path:
    name = config.model_id.rstrip("/").split("/")[-1]
    return Path(config.results_dir) / name / config.csv_name


def layer_csv_path(config: CompactifaiConfig) -> Path:
    name = config.model_id.rstrip("/").split("/")[-1]
    return Path(config.results_dir) / name / config.layer_csv_name


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
        plan = build_plan(originals[name], config.mpo_sites, cache=config.svd_cache)
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
                approx, params_mpo = compress_weight(orig, plan, chi)
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
