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

    # Random-matrix (Marchenko-Pastur) condition-number baseline. The empirical
    # baseline is the median condition number of this many iid Gaussian matrices
    # of the same shape (cached per shape); it works for square matrices too,
    # where the analytic MP-edge ratio diverges. Set 0 to disable.
    rmt_samples: int = 5
    rmt_seed: int = 0

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
    aspect_ratio: float
    # raw (scale-/shape-dependent) spectral metrics
    sigma_1: float
    sigma_2: float
    sigma_min: float
    spectral_gap: float
    condition_number: float
    shannon_entropy: float
    renyi2_entropy: float
    effective_rank: float
    stable_rank: float
    # shape-normalized / scale-free variants (comparable across layer shapes)
    shannon_entropy_normalized: float
    renyi2_entropy_normalized: float
    effective_rank_ratio: float
    stable_rank_ratio: float
    relative_spectral_gap: float
    log_condition_number: float
    condition_number_mp_ratio: float
    condition_number_rmt_ratio: float
    # perplexity sensitivity (already relative -> comparable)
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


@torch.no_grad()
def expected_condition_number(m: int, n: int, samples: int, seed: int) -> float:
    """Median condition number of iid Gaussian matrices of shape (m, n).

    This is the random-matrix baseline used to normalize a layer's condition
    number. Unlike the analytic Marchenko-Pastur edge ratio (which diverges for
    square matrices), it is finite and well-defined for every shape, and the
    median is robust to the heavy tail of the condition-number distribution near
    square. The condition number is scale-invariant, so the entry variance is
    irrelevant. The seed is derived from the shape so the baseline is
    reproducible regardless of the order layers are processed in.
    """
    if samples <= 0:
        return float("nan")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + m * 1_000_003 + n)
    conds = []
    for _ in range(samples):
        g = torch.randn(m, n, generator=generator, dtype=torch.float32)
        s = torch.linalg.svdvals(g)
        s_min = float(s[-1])
        conds.append(float(s[0]) / s_min if s_min > 0 else float("inf"))
    conds.sort()
    return conds[len(conds) // 2]        # median


# --------------------------------------------------------------------------- #
# Spectral metrics
# --------------------------------------------------------------------------- #
def spectral_metrics(weight: torch.Tensor) -> dict:
    """Compute singular-value-based metrics for a 2-D weight matrix.

    Alongside the raw metrics this returns *shape-normalized* variants that are
    comparable across matrices of different dimensions (see module/README notes):

    * entropies divided by ``log k`` (k = number of singular values) -> [0, 1];
    * effective / stable rank divided by ``k`` -> (0, 1];
    * relative spectral gap ``(sigma_1 - sigma_2) / sigma_1`` (scale-free);
    * ``log10`` condition number, and the condition number divided by the
      Marchenko-Pastur bulk expectation for a random matrix of the same aspect
      ratio (removes the size/shape bias for rectangular matrices).
    """
    w = weight.detach().to(torch.float32).cpu()
    m, ncol = int(w.shape[0]), int(w.shape[1])
    k = min(m, ncol)                               # number of singular values
    aspect = k / max(m, ncol)                      # gamma in (0, 1]

    s = torch.linalg.svdvals(w)                    # length k, descending, >= 0
    sigma_1 = float(s[0]) if k > 0 else 0.0
    sigma_2 = float(s[1]) if k > 1 else 0.0
    sigma_min = float(s[-1]) if k > 0 else 0.0
    fro_sq = float((s * s).sum())
    total = float(s.sum())

    zero = dict(sigma_1=sigma_1, sigma_2=sigma_2, sigma_min=sigma_min,
                spectral_gap=0.0, relative_spectral_gap=0.0,
                condition_number=float("inf"), log_condition_number=float("inf"),
                condition_number_mp_ratio=float("nan"),
                shannon_entropy=0.0, shannon_entropy_normalized=0.0,
                renyi2_entropy=0.0, renyi2_entropy_normalized=0.0,
                effective_rank=0.0, effective_rank_ratio=0.0,
                stable_rank=0.0, stable_rank_ratio=0.0, aspect_ratio=aspect)
    if total <= 0 or sigma_1 <= 0:
        return zero

    pos = s[s > 0]
    p = pos / total
    shannon = float(-(p * torch.log(p)).sum())
    renyi2 = float(-torch.log((p * p).sum()))
    log_k = math.log(k) if k > 1 else 1.0
    shannon_norm = shannon / log_k if k > 1 else 0.0
    renyi2_norm = renyi2 / log_k if k > 1 else 0.0

    effective_rank = math.exp(shannon)
    stable_rank = fro_sq / (sigma_1 * sigma_1)     # ||W||_F^2 / sigma_1^2 in [1, k]

    condition = sigma_1 / sigma_min if sigma_min > 0 else float("inf")
    log_condition = math.log10(condition) if math.isfinite(condition) else float("inf")

    # Marchenko-Pastur bulk condition number for an iid matrix of this shape:
    # (1 + sqrt(gamma)) / (1 - sqrt(gamma)). Finite only for rectangular matrices
    # (gamma < 1); it diverges for square ones, so leave that ratio undefined and
    # rely on log_condition_number / stable_rank_ratio there instead.
    if aspect < 1.0 and math.isfinite(condition):
        root = math.sqrt(aspect)
        kappa_mp = (1.0 + root) / (1.0 - root)
        cond_mp_ratio = condition / kappa_mp
    else:
        cond_mp_ratio = float("nan")

    return dict(
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        sigma_min=sigma_min,
        spectral_gap=sigma_1 - sigma_2,
        relative_spectral_gap=(sigma_1 - sigma_2) / sigma_1,
        condition_number=condition,
        log_condition_number=log_condition,
        condition_number_mp_ratio=cond_mp_ratio,
        shannon_entropy=shannon,
        shannon_entropy_normalized=shannon_norm,
        renyi2_entropy=renyi2,
        renyi2_entropy_normalized=renyi2_norm,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank / k,
        stable_rank=stable_rank,
        stable_rank_ratio=stable_rank / k,
        aspect_ratio=aspect,
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


def _present(row: dict, field_name: str) -> bool:
    """True if the column exists and is non-empty (a computed value, incl. inf/nan)."""
    value = row.get(field_name, None)
    return value is not None and value != ""


def has_spectral(row: dict, need_rmt: bool = False) -> bool:
    """True if the row already carries the (normalized) spectral metrics.

    Presence (a written cell) is used rather than finiteness, because some
    spectral values can legitimately be inf/nan (e.g. condition number of a
    rank-deficient matrix). ``stable_rank_ratio`` and (when requested) the RMT
    condition-number ratio are checked as well, so rows written before those
    columns existed are cheaply recomputed to add them without discarding an
    already-computed sensitivity value.
    """
    if not (_present(row, "shannon_entropy") and _present(row, "stable_rank_ratio")):
        return False
    if need_rmt and not _present(row, "condition_number_rmt_ratio"):
        return False
    return True


def has_sensitivity(row: dict) -> bool:
    # Sensitivity NaN means "not computed" (e.g. a spectral-only run), so this
    # requires a finite value rather than mere presence.
    return _is_finite_field(row, "perplexity_sensitivity")


def row_is_complete(row: dict, need_sensitivity: bool, need_rmt: bool = False) -> bool:
    """A checkpointed row is complete only if the requested metrics are present."""
    if not has_spectral(row, need_rmt):
        return False
    if need_sensitivity and not has_sensitivity(row):
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
# Shape-normalized / scale-free metrics, so depth trends are comparable across
# layer types of different dimensions (raw columns remain in the CSV).
METRICS_FOR_DEPTH = [
    "shannon_entropy_normalized", "renyi2_entropy_normalized",
    "effective_rank_ratio", "stable_rank_ratio",
    "relative_spectral_gap", "log_condition_number",
    "condition_number_rmt_ratio", "perplexity_sensitivity",
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
    need_rmt = config.rmt_samples > 0

    # Per-metric, per-layer checkpoint plan. A layer is (re)processed if its
    # spectral metrics are missing (need_spec) or sensitivity is requested but
    # absent (need_sen). Spectral is cheap to recompute; sensitivity is not, so a
    # stored sensitivity value is reused whenever we are only refreshing spectral.
    plan = []
    for name, param in matrices:
        row = existing.get(name)
        if config.force_recompute or row is None:
            need_spec, need_sen = True, need_sensitivity
        else:
            need_spec = not has_spectral(row, need_rmt)
            need_sen = need_sensitivity and not has_sensitivity(row)
        if need_spec or need_sen:
            plan.append((name, param, need_spec, need_sen))

    skipped = total - len(plan)
    n_spec = sum(1 for _, _, need_spec, _ in plan if need_spec)
    n_sens = sum(1 for _, _, _, need_sen in plan if need_sen)
    print(f"\nFound {total} weight matrices; {skipped} complete in checkpoint, "
          f"{len(plan)} to compute (spectral: {n_spec}, sensitivity: {n_sens}).")
    print(f"Checkpoint / results CSV: {csv_path}")

    # Results keyed by name, seeded from the checkpoint so skipped rows survive
    # the atomic rewrite. Order follows the model's parameter order.
    order = [n for n, _ in matrices]
    order += [n for n in existing if n not in set(order)]
    results: dict[str, dict] = dict(existing)

    def _getf(row, key):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return float("nan")

    def _round(x, nd):
        return round(x, nd) if isinstance(x, float) and math.isfinite(x) else x

    # Random-matrix condition-number baselines, cached per (rows, cols) shape so
    # each distinct shape's Monte-Carlo estimate is computed at most once.
    rmt_cache: dict[tuple[int, int], float] = {}

    def rmt_condition_ratio(cond, m, n):
        if not need_rmt or not math.isfinite(cond):
            return float("nan")
        key = (m, n)
        if key not in rmt_cache:
            t0 = time.perf_counter()
            rmt_cache[key] = expected_condition_number(m, n, config.rmt_samples, config.rmt_seed)
            print(f"    RMT baseline for {m}x{n}: "
                  f"E[kappa]~{rmt_cache[key]:.4g} "
                  f"({config.rmt_samples} samples, {time.perf_counter() - t0:.1f}s)")
        base = rmt_cache[key]
        return cond / base if base and math.isfinite(base) else float("nan")

    # Baseline perplexity for the sensitivity probe (only if a layer needs it).
    eval_ids = None
    baseline_ppl = float("nan")
    generator = torch.Generator(device="cpu")
    if n_sens > 0:
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
    for i, (name, param, need_spec, need_sen) in enumerate(plan, start=1):
        row = existing.get(name)
        layer_type, depth = parse_layer_info(name)
        rows_, cols_ = int(param.shape[0]), int(param.shape[1])
        print(f"\n[{i}/{len(plan)}] {name}  "
              f"(type={layer_type}, depth={depth}, shape={rows_}x{cols_})")

        sm = spectral_metrics(param)
        cond_rmt_ratio = rmt_condition_ratio(sm["condition_number"], rows_, cols_)
        print(f"    spectral: gap*={sm['relative_spectral_gap']:.4g} "
              f"logcond={sm['log_condition_number']:.4g} "
              f"cond/RMT={cond_rmt_ratio:.4g} "
              f"H/logk={sm['shannon_entropy_normalized']:.4f} "
              f"H2/logk={sm['renyi2_entropy_normalized']:.4f} "
              f"eff_rank/k={sm['effective_rank_ratio']:.4f} "
              f"srank/k={sm['stable_rank_ratio']:.4f}")

        # Sensitivity: compute if needed, otherwise reuse the checkpointed value.
        if need_sen:
            generator.manual_seed(config.sensitivity_seed + i)
            window = min(config.sensitivity_window, eval_ids.size(1))
            sensitivity, ppl_pert = perplexity_sensitivity(
                model, param, eval_ids, device, window,
                baseline_ppl, config.epsilon, config.sensitivity_samples, generator,
            )
            base_used, eps_used, samples_used = baseline_ppl, config.epsilon, config.sensitivity_samples
            print(f"    sensitivity: dPPL/PPL per eps={config.epsilon} -> "
                  f"{sensitivity:.4g}  (PPL' = {ppl_pert:.4f})")
        elif row is not None and has_sensitivity(row):
            sensitivity, ppl_pert = _getf(row, "perplexity_sensitivity"), _getf(row, "ppl_perturbed")
            base_used, eps_used = _getf(row, "ppl_baseline"), _getf(row, "epsilon")
            samples_used = int(_getf(row, "sensitivity_samples")) if math.isfinite(_getf(row, "sensitivity_samples")) else config.sensitivity_samples
            print("    sensitivity: reused from checkpoint "
                  f"({sensitivity:.4g})")
        else:
            sensitivity, ppl_pert = float("nan"), float("nan")
            base_used, eps_used, samples_used = baseline_ppl, config.epsilon, config.sensitivity_samples

        result = LayerResult(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=config.model_id,
            param_name=name,
            layer_type=layer_type,
            depth=depth,
            rows=rows_,
            cols=cols_,
            num_params=rows_ * cols_,
            aspect_ratio=round(sm["aspect_ratio"], 6),
            sigma_1=round(sm["sigma_1"], 6),
            sigma_2=round(sm["sigma_2"], 6),
            sigma_min=round(sm["sigma_min"], 8),
            spectral_gap=round(sm["spectral_gap"], 6),
            condition_number=_round(sm["condition_number"], 4),
            shannon_entropy=round(sm["shannon_entropy"], 6),
            renyi2_entropy=round(sm["renyi2_entropy"], 6),
            effective_rank=round(sm["effective_rank"], 4),
            stable_rank=round(sm["stable_rank"], 4),
            shannon_entropy_normalized=round(sm["shannon_entropy_normalized"], 6),
            renyi2_entropy_normalized=round(sm["renyi2_entropy_normalized"], 6),
            effective_rank_ratio=round(sm["effective_rank_ratio"], 6),
            stable_rank_ratio=round(sm["stable_rank_ratio"], 6),
            relative_spectral_gap=round(sm["relative_spectral_gap"], 6),
            log_condition_number=_round(sm["log_condition_number"], 4),
            condition_number_mp_ratio=_round(sm["condition_number_mp_ratio"], 4),
            condition_number_rmt_ratio=_round(cond_rmt_ratio, 4),
            perplexity_sensitivity=_round(sensitivity, 6),
            ppl_baseline=_round(base_used, 4),
            ppl_perturbed=_round(ppl_pert, 4),
            epsilon=eps_used,
            sensitivity_samples=samples_used,
        )
        # Update in place and rewrite the whole CSV: this checkpoints after every
        # layer and updates existing rows (e.g. adding sensitivity) without ever
        # writing a duplicate param_name.
        results[name] = asdict(result)
        write_all_rows(csv_path, [results[n] for n in order if n in results])
        done_now = skipped + i
        elapsed = time.perf_counter() - start
        eta = elapsed / i * (len(plan) - i)
        print(f"    checkpointed ({done_now}/{total})  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    all_rows = read_all_rows(csv_path)
    print_depth_summary(all_rows)
    if config.make_plots:
        print("\nGenerating depth plots ...")
        make_depth_plots(all_rows, csv_path.parent)

    print(f"\nLayer analysis complete. Results in {csv_path}")
    return csv_path
