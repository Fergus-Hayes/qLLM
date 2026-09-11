"""Per-layer spectral and sensitivity analysis for causal language models.

For every 2-D weight matrix ("layer") in a model this computes:

* **Shannon entropy** of the normalized singular-value spectrum
  ``p_i = sigma_i / sum(sigma)`` — ``H = -sum p_i log p_i`` (nats). Its
  exponential is the *effective rank* of the matrix.
* **Renyi-2 (collision) entropy** ``H_2 = -log sum(p_i^2)``.
* **Spectral gap** — the difference between the first two singular values
  ``sigma_1 - sigma_2``.
* **Condition number** ``sigma_max / sigma_min``.
* **Perplexity sensitivity** — the relative change in perplexity produced by a
  small relative perturbation of the layer's weights
  (``sensitivity = (PPL' - PPL) / PPL / epsilon`` with ``||dW|| / ||W|| = epsilon``).

Results are printed as they are computed and appended to a CSV. The CSV doubles
as a checkpoint: layers already present are skipped on a re-run, so an
interrupted analysis resumes where it stopped.
"""

from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset

from .benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|decoder\.layers)\.(\d+)\.")


@dataclass
class LayerAnalysisConfig:
    model_id: str
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    revision: str | None = None

    # Corpus used for the perplexity-sensitivity probe.
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    text_column: str = "text"

    # Perplexity-sensitivity probe (kept small: it runs once per layer).
    run_sensitivity: bool = True
    sensitivity_eval_tokens: int = 4096
    sensitivity_window: int = 1024
    epsilon: float = 0.01
    sensitivity_samples: int = 1
    sensitivity_seed: int = 0

    # Output.
    results_dir: str = "results/llms"
    csv_name: str = "layer_analysis.csv"
    make_plots: bool = True
    force_recompute: bool = False   # ignore checkpoint and recompute every layer


@dataclass
class LayerResult:
    timestamp: str
    model_id: str
    param_name: str
    layer_type: str
    depth: int
    rows: int
    cols: int
    num_params: int
    sigma_1: float
    sigma_2: float
    sigma_min: float
    spectral_gap: float
    condition_number: float
    shannon_entropy: float
    shannon_entropy_normalized: float
    renyi2_entropy: float
    effective_rank: float
    perplexity_sensitivity: float
    ppl_baseline: float
    ppl_perturbed: float
    epsilon: float
    sensitivity_samples: int


# --------------------------------------------------------------------------- #
# Layer identification
# --------------------------------------------------------------------------- #
def parse_layer_info(param_name: str) -> tuple[str, int]:
    """Return (layer_type, depth) for a parameter name.

    ``depth`` is the transformer block index, or -1 for parameters that do not
    live inside a numbered block (embeddings, final norm, lm_head). ``layer_type``
    is the module path with the block index stripped, so weights of the same role
    at different depths share a type (e.g. ``self_attn.q_proj``).
    """
    match = _LAYER_RE.search(param_name)
    name = param_name[:-len(".weight")] if param_name.endswith(".weight") else param_name
    if match:
        depth = int(match.group(1))
        layer_type = name[match.end():] if match.end() <= len(name) else name
        layer_type = layer_type or name
    else:
        depth = -1
        layer_type = name
    return layer_type, depth


def iter_weight_matrices(model):
    """Yield (param_name, parameter) for every 2-D weight matrix in the model."""
    for name, param in model.named_parameters():
        if param.ndim == 2:
            yield name, param


# --------------------------------------------------------------------------- #
# Spectral metrics
# --------------------------------------------------------------------------- #
def spectral_metrics(weight: torch.Tensor) -> dict:
    """Compute singular-value-based metrics for a 2-D weight matrix."""
    w = weight.detach().to(torch.float32).cpu()
    s = torch.linalg.svdvals(w)                    # descending singular values
    s = s[s > 0] if (s > 0).any() else s           # guard all-zero matrices
    total = float(s.sum())

    if total <= 0 or s.numel() == 0:
        return dict(sigma_1=0.0, sigma_2=0.0, sigma_min=0.0, spectral_gap=0.0,
                    condition_number=float("inf"), shannon_entropy=0.0,
                    shannon_entropy_normalized=0.0, renyi2_entropy=0.0,
                    effective_rank=0.0)

    p = s / total
    logp = torch.log(p)
    shannon = float(-(p * logp).sum())
    renyi2 = float(-torch.log((p * p).sum()))
    n = s.numel()
    shannon_norm = shannon / math.log(n) if n > 1 else 0.0

    sigma_1 = float(s[0])
    sigma_2 = float(s[1]) if n > 1 else 0.0
    sigma_min = float(s[-1])
    condition = sigma_1 / sigma_min if sigma_min > 0 else float("inf")

    return dict(
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        sigma_min=sigma_min,
        spectral_gap=sigma_1 - sigma_2,
        condition_number=condition,
        shannon_entropy=shannon,
        shannon_entropy_normalized=shannon_norm,
        renyi2_entropy=renyi2,
        effective_rank=math.exp(shannon),
    )


# --------------------------------------------------------------------------- #
# Perplexity sensitivity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quick_perplexity(model, input_ids: torch.Tensor, device: str, window: int) -> float:
    """Perplexity over a fixed token block using non-overlapping windows."""
    seq_len = input_ids.size(1)
    nll_sum, n_tokens = 0.0, 0
    for begin in range(0, seq_len, window):
        ids = input_ids[:, begin:begin + window].to(device)
        if ids.size(1) < 2:
            break
        loss = model(ids, labels=ids).loss
        n = ids.size(1) - 1
        nll_sum += float(loss) * n
        n_tokens += n
    if n_tokens == 0:
        return float("nan")
    return math.exp(nll_sum / n_tokens)


@torch.no_grad()
def perplexity_sensitivity(
    model,
    param: torch.Tensor,
    input_ids: torch.Tensor,
    device: str,
    window: int,
    baseline_ppl: float,
    epsilon: float,
    n_samples: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    """Relative PPL change per unit relative weight perturbation.

    A perturbation ``dW`` with ``||dW|| / ||W|| = epsilon`` is applied, the
    perplexity re-measured, and the weights restored. Returns the averaged
    sensitivity and the mean perturbed perplexity.
    """
    weight = param.data
    original = weight.clone()
    w_norm = weight.norm()
    if float(w_norm) == 0.0:
        return 0.0, baseline_ppl

    sens_values, ppl_values = [], []
    for _ in range(max(1, n_samples)):
        noise = torch.randn(weight.shape, generator=generator,
                            dtype=weight.dtype, device=weight.device)
        scale = epsilon * w_norm / (noise.norm() + 1e-12)
        weight.add_(scale * noise)
        ppl = quick_perplexity(model, input_ids, device, window)
        weight.copy_(original)                     # restore before next sample
        ppl_values.append(ppl)
        if baseline_ppl > 0 and math.isfinite(ppl):
            sens_values.append((ppl - baseline_ppl) / baseline_ppl / epsilon)

    sensitivity = sum(sens_values) / len(sens_values) if sens_values else float("nan")
    mean_ppl = sum(ppl_values) / len(ppl_values) if ppl_values else float("nan")
    return sensitivity, mean_ppl


# --------------------------------------------------------------------------- #
# CSV / checkpoint helpers
# --------------------------------------------------------------------------- #
def layer_csv_path(config: LayerAnalysisConfig) -> Path:
    model_name = config.model_id.rstrip("/").split("/")[-1]
    return Path(config.results_dir) / model_name / config.csv_name


CSV_FIELDS = [f.name for f in fields(LayerResult)]


def read_rows_by_name(csv_path: Path) -> dict[str, dict]:
    """Return existing rows keyed by param_name (checkpoint state)."""
    rows: dict[str, dict] = {}
    if not csv_path.exists():
        return rows
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("param_name"):
                rows[row["param_name"]] = row
    return rows


def _is_finite_field(row: dict, field_name: str) -> bool:
    try:
        return math.isfinite(float(row.get(field_name, "")))
    except (TypeError, ValueError):
        return False


def row_is_complete(row: dict, need_sensitivity: bool) -> bool:
    """A checkpointed row is complete only if the requested metrics are present.

    Spectral metrics are always expected; the perplexity-sensitivity column must
    additionally be a finite number when sensitivity is requested. This prevents
    a spectral-only (``--skip-sensitivity``) run from masking a later run that
    does want sensitivity.
    """
    if not _is_finite_field(row, "shannon_entropy"):
        return False
    if need_sensitivity and not _is_finite_field(row, "perplexity_sensitivity"):
        return False
    return True


def write_all_rows(csv_path: Path, rows_in_order: list[dict]) -> None:
    """Atomically (re)write the whole CSV, so updates never duplicate a row."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in_order:
            writer.writerow(row)
        f.flush()
    tmp.replace(csv_path)


def read_all_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Reporting: depth vs. layer type
# --------------------------------------------------------------------------- #
METRICS_FOR_DEPTH = [
    "shannon_entropy", "renyi2_entropy", "effective_rank",
    "spectral_gap", "condition_number", "perplexity_sensitivity",
]


def print_depth_summary(rows: list[dict]) -> None:
    """Print how each metric varies over depth, grouped by layer type."""
    block_rows = [r for r in rows if int(r["depth"]) >= 0]
    if not block_rows:
        print("\n(No block-level layers found for depth summary.)")
        return

    by_type: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in block_rows:
        by_type[r["layer_type"]][int(r["depth"])] = r

    print("\n" + "=" * 72)
    print("METRICS OVER LAYER DEPTH (by layer type)")
    print("=" * 72)
    for metric in METRICS_FOR_DEPTH:
        print(f"\n### {metric}")
        for layer_type in sorted(by_type):
            depths = sorted(by_type[layer_type])
            vals = []
            for d in depths:
                try:
                    vals.append(float(by_type[layer_type][d][metric]))
                except (ValueError, KeyError):
                    vals.append(float("nan"))
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                trend = f"min={min(finite):.4g} max={max(finite):.4g} mean={sum(finite)/len(finite):.4g}"
            else:
                trend = "n/a"
            preview = ", ".join(f"{v:.3g}" for v in vals[:8])
            more = " ..." if len(vals) > 8 else ""
            print(f"  {layer_type:<28} depth[0..{depths[-1]}]: {preview}{more}   ({trend})")


def make_depth_plots(rows: list[dict], out_dir: Path) -> None:
    """Save one PNG per metric: value vs. depth, one line per layer type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(Skipping plots: matplotlib unavailable: {exc})")
        return

    block_rows = [r for r in rows if int(r["depth"]) >= 0]
    if not block_rows:
        return
    by_type: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in block_rows:
        by_type[r["layer_type"]][int(r["depth"])] = r

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS_FOR_DEPTH:
        plt.figure(figsize=(8, 5))
        plotted = False
        for layer_type in sorted(by_type):
            depths = sorted(by_type[layer_type])
            xs, ys = [], []
            for d in depths:
                try:
                    y = float(by_type[layer_type][d][metric])
                except (ValueError, KeyError):
                    continue
                if math.isfinite(y):
                    xs.append(d)
                    ys.append(y)
            if xs:
                plt.plot(xs, ys, marker="o", label=layer_type)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("layer depth (block index)")
        plt.ylabel(metric)
        plt.title(f"{metric} over layer depth")
        plt.legend(fontsize="small")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"depth_{metric}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  saved plot {path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_eval_ids(tokenizer, config: LayerAnalysisConfig) -> torch.Tensor:
    print(f"Loading sensitivity corpus "
          f"'{config.dataset}/{config.dataset_config}' [{config.split}] ...")
    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split)
    text = "\n\n".join(dataset[config.text_column])
    ids = tokenizer(text, return_tensors="pt").input_ids
    if config.sensitivity_eval_tokens:
        ids = ids[:, :config.sensitivity_eval_tokens]
    print(f"Sensitivity probe uses {ids.size(1)} tokens.")
    return ids


def run_layer_analysis(config: LayerAnalysisConfig) -> Path:
    """Analyze every 2-D layer, printing and checkpointing to CSV as it goes."""
    device = resolve_device(config.device)
    load_cfg = BenchmarkConfig(
        model_id=config.model_id, device=config.device, dtype=config.dtype,
        trust_remote_code=config.trust_remote_code, revision=config.revision,
    )
    model, tokenizer = load_model_and_tokenizer(load_cfg, device)

    csv_path = layer_csv_path(config)
    existing = read_rows_by_name(csv_path)

    matrices = list(iter_weight_matrices(model))
    total = len(matrices)
    need_sensitivity = config.run_sensitivity

    # Per-metric checkpoint: recompute a layer if it is missing OR if sensitivity
    # is now requested but its stored value is not a finite number.
    remaining = [
        (n, p) for n, p in matrices
        if config.force_recompute
        or n not in existing
        or not row_is_complete(existing[n], need_sensitivity)
    ]
    skipped = total - len(remaining)
    augment = sum(
        1 for n, _ in matrices
        if n in existing and not config.force_recompute
        and _is_finite_field(existing[n], "shannon_entropy")
        and not row_is_complete(existing[n], need_sensitivity)
    )
    print(f"\nFound {total} weight matrices; {skipped} complete in checkpoint, "
          f"{len(remaining)} to compute ({augment} need sensitivity added).")
    print(f"Checkpoint / results CSV: {csv_path}")

    # Results keyed by name, seeded from the checkpoint so skipped rows survive
    # the atomic rewrite. Order follows the model's parameter order.
    order = [n for n, _ in matrices]
    order += [n for n in existing if n not in set(order)]
    results: dict[str, dict] = dict(existing)

    # Baseline perplexity for the sensitivity probe (only if any layer needs it).
    eval_ids = None
    baseline_ppl = float("nan")
    generator = torch.Generator(device="cpu")
    if need_sensitivity and remaining:
        eval_ids = _load_eval_ids(tokenizer, config)
        window = min(config.sensitivity_window, eval_ids.size(1))
        print("Computing baseline perplexity for sensitivity probe ...")
        t0 = time.perf_counter()
        baseline_ppl = quick_perplexity(model, eval_ids, device, window)
        print(f"Baseline PPL = {baseline_ppl:.4f} ({time.perf_counter() - t0:.1f}s)")
        if not math.isfinite(baseline_ppl) or baseline_ppl <= 0:
            print("    WARNING: baseline perplexity is not a finite positive "
                  "number; sensitivity will be NaN. This often happens with "
                  "--dtype float16 on CPU — try --dtype float32 or --device cuda.")

    start = time.perf_counter()
    for i, (name, param) in enumerate(remaining, start=1):
        layer_type, depth = parse_layer_info(name)
        rows_, cols_ = param.shape[0], param.shape[1]
        print(f"\n[{i}/{len(remaining)}] {name}  "
              f"(type={layer_type}, depth={depth}, shape={rows_}x{cols_})")

        sm = spectral_metrics(param)
        print(f"    spectral: sigma1={sm['sigma_1']:.4g} gap={sm['spectral_gap']:.4g} "
              f"cond={sm['condition_number']:.4g} "
              f"H={sm['shannon_entropy']:.4f} H2={sm['renyi2_entropy']:.4f} "
              f"eff_rank={sm['effective_rank']:.2f}")

        sensitivity, ppl_pert = float("nan"), float("nan")
        if config.run_sensitivity and eval_ids is not None:
            generator.manual_seed(config.sensitivity_seed + i)
            window = min(config.sensitivity_window, eval_ids.size(1))
            sensitivity, ppl_pert = perplexity_sensitivity(
                model, param, eval_ids, device, window,
                baseline_ppl, config.epsilon, config.sensitivity_samples, generator,
            )
            print(f"    sensitivity: dPPL/PPL per eps={config.epsilon} -> "
                  f"{sensitivity:.4g}  (PPL' = {ppl_pert:.4f})")

        result = LayerResult(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id,
            param_name=name,
            layer_type=layer_type,
            depth=depth,
            rows=rows_,
            cols=cols_,
            num_params=rows_ * cols_,
            sigma_1=round(sm["sigma_1"], 6),
            sigma_2=round(sm["sigma_2"], 6),
            sigma_min=round(sm["sigma_min"], 8),
            spectral_gap=round(sm["spectral_gap"], 6),
            condition_number=round(sm["condition_number"], 4)
                if math.isfinite(sm["condition_number"]) else sm["condition_number"],
            shannon_entropy=round(sm["shannon_entropy"], 6),
            shannon_entropy_normalized=round(sm["shannon_entropy_normalized"], 6),
            renyi2_entropy=round(sm["renyi2_entropy"], 6),
            effective_rank=round(sm["effective_rank"], 4),
            perplexity_sensitivity=round(sensitivity, 6)
                if math.isfinite(sensitivity) else sensitivity,
            ppl_baseline=round(baseline_ppl, 4) if math.isfinite(baseline_ppl) else baseline_ppl,
            ppl_perturbed=round(ppl_pert, 4) if math.isfinite(ppl_pert) else ppl_pert,
            epsilon=config.epsilon,
            sensitivity_samples=config.sensitivity_samples,
        )
        # Update in place and rewrite the whole CSV: this checkpoints after every
        # layer and updates existing rows (e.g. adding sensitivity) without ever
        # writing a duplicate param_name.
        results[name] = asdict(result)
        write_all_rows(csv_path, [results[n] for n in order if n in results])
        done_now = skipped + i
        elapsed = time.perf_counter() - start
        eta = elapsed / i * (len(remaining) - i)
        print(f"    checkpointed ({done_now}/{total})  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    all_rows = read_all_rows(csv_path)
    print_depth_summary(all_rows)
    if config.make_plots:
        print("\nGenerating depth plots ...")
        make_depth_plots(all_rows, csv_path.parent)

    print(f"\nLayer analysis complete. Results in {csv_path}")
    return csv_path
