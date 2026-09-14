"""End-to-end smoke test using real torch/transformers objects.

Instantiates a tiny GPT-2 from its config (no HF download) and monkey-patches
the dataset fetch and model loader to synthetic in-memory objects, then runs:
  1. compute_perplexity end-to-end
  2. CompactifAI MPO compression (build_plan / compress_weight round-trip)
  3. Per-layer sweep pipeline (one layer at a time)
Everything below is real torch code -- only the network I/O is stubbed.
"""
import os, sys, time, random, string
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, "/home/user/qLLM")

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.tokenization_utils_base import BatchEncoding

torch.manual_seed(0)
random.seed(0)

# 1. Tiny GPT-2 with 2 blocks, hidden 64, vocab 512 -> ~200 K params
cfg = GPT2Config(vocab_size=512, n_positions=128, n_embd=64, n_layer=2, n_head=4,
                 n_inner=128, activation_function="gelu_new")
model = GPT2LMHeadModel(cfg); model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Tiny GPT-2: {n_params:,} params, {cfg.n_layer} blocks")

# 2. Character-level tokenizer via bytes (no HF fetch)
class ByteTok:
    pad_token_id = 0
    pad_token = "\0"
    eos_token_id = 0
    model_max_length = 10**8
    def __call__(s, text, return_tensors=None):
        ids = torch.tensor([[b % cfg.vocab_size for b in text.encode()[:5000]]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
tok = ByteTok()

# 3. Monkey-patch the loader used by the qllm modules
import qllm.benchmark as B
import qllm.compactifai_sweep as S
import qllm.layer_analysis as LA
B.load_model_and_tokenizer = lambda cfg_, dev: (model.to(dev), tok)
LA.load_model_and_tokenizer = lambda cfg_, dev: (model.to(dev), tok)
S.load_model_and_tokenizer  = lambda cfg_, dev: (model.to(dev), tok)

# Synthetic dataset (2 kB of pseudo-text)
class FakeDS:
    def __init__(s): s._t=" ".join("".join(random.choices(string.ascii_lowercase, k=random.randint(2,8))) for _ in range(400))
    def __getitem__(s,k): return [s._t]
def fake_load(*a, **k): return FakeDS()
B.load_dataset = fake_load
S.load_dataset = fake_load
LA.load_dataset = fake_load

# ---- (1) perplexity end-to-end
from qllm.benchmark import BenchmarkConfig, run_benchmark
print("\n=== 1. run_benchmark ===")
t0 = time.perf_counter()
res = run_benchmark(BenchmarkConfig(model_id="tiny-gpt2", device="cpu", dtype="float32",
    max_length=64, stride=64, max_eval_tokens=800, ppl_batch_size=4,
    prompt="hello world", gen_tokens=4, warmup=0))
print(f"perplexity = {res.perplexity:.3f}  gen = {res.generation_tokens_per_second:.1f} tok/s "
      f"({time.perf_counter()-t0:.1f}s)")
assert res.perplexity == res.perplexity  # not nan

# ---- (2) CompactifAI: MPO round-trip on a real weight
from qllm.compactifai import build_plan, compress_weight, mpo_param_count, full_rank_chi
print("\n=== 2. CompactifAI MPO round-trip ===")
W = model.transformer.h[0].mlp.c_fc.weight.data     # real trained weight matrix
plan = build_plan(W, n_sites=2)
print(f"weight shape {tuple(W.shape)}  factorization {plan.out_dims}x{plan.in_dims}  "
      f"max_useful_chi={plan.max_chi}  exact_rank={full_rank_chi(plan.out_dims, plan.in_dims)}")
# monotonicity check: lower chi -> larger reconstruction error
errs = []
for chi in [2, 4, 16, full_rank_chi(plan.out_dims, plan.in_dims)]:
    approx, params = compress_weight(W, plan, chi)
    e = float((W.float().cpu() - approx.float()).norm() / W.float().cpu().norm())
    errs.append((chi, params, e))
    print(f"  chi={chi:>3}  params={params:>5}  rel.err={e:.6f}")
assert errs[0][2] >= errs[1][2] >= errs[2][2] >= errs[3][2], "rel err must be monotone"
assert errs[-1][2] < 1e-5, "chi=full_rank must reconstruct exactly"
print("  monotone in chi, exact at full rank -> OK")

# ---- (3) per-layer sweep
from qllm.compactifai_sweep import CompactifaiConfig, run_per_layer_sweep, per_layer_csv_path, _read_rows
print("\n=== 3. Per-layer sweep (real model, one layer at a time) ===")
scratch = os.environ.get("QLLM_SMOKE_DIR", "/tmp/qllm-smoke")
import shutil; shutil.rmtree(scratch, ignore_errors=True)
t0 = time.perf_counter()
cfg2 = CompactifaiConfig(model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.\d+\.(attn|mlp)\.", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    chi_min=2, chi_max=32, num_chi=4,
    max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=512, ppl_batch_size=4,
    results_dir=scratch, make_plots=True)
run_per_layer_sweep(cfg2)

# ---- (4) per-layer sweep WITH healing (retrain the compressed layer)
print("\n=== 4. Per-layer sweep with healing (--heal) ===")
import shutil as _sh; _sh.rmtree(scratch + "-heal", ignore_errors=True)
cfg3 = CompactifaiConfig(model_id="tiny-gpt2", device="cpu", dtype="float32",
    include_pattern=r"h\.\d+\.(attn|mlp)\.", exclude_pattern=r"(wte|wpe|lm_head|ln|bias)",
    chi_min=2, chi_max=16, num_chi=2, num_depths=1,
    max_length=64, stride=64, per_layer_stride=64,
    per_layer_eval_tokens=256, ppl_batch_size=4,
    heal=True, heal_steps=8, heal_lr=5e-3, heal_tokens=512, heal_batch=2, heal_split="train",
    results_dir=scratch + "-heal", make_plots=False)
run_per_layer_sweep(cfg3)
hrows = _read_rows(per_layer_csv_path(cfg3))
assert hrows and all(r.get("perplexity_healed") not in (None, "", "nan") for r in hrows), \
    "healed perplexity missing"
print(f"healed {len(hrows)} (layer, chi) points; sample recovered% = "
      + ", ".join(r["heal_recovered_frac"] for r in hrows[:4]))
rows = _read_rows(per_layer_csv_path(cfg2))
print(f"\n{len(rows)} (layer, chi) rows written in {time.perf_counter()-t0:.1f}s")
from pathlib import Path
for p in Path(scratch).rglob("*"): 
    if p.is_file(): print(f"  {p.stat().st_size:>7}  {p.relative_to(scratch)}")
print("\nALL SMOKE STAGES PASSED")
