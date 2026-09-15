"""CLI for the hybrid PQC+TN vs. pure TN per-layer sweep.

Nothing here is specialized to a particular model or layer: the model is a
positional argument and the layers are chosen with ``--profile-depths`` /
``--num-depths`` and ``--layer-types``, so any single layer, any subset, or a
whole model is one invocation. ``--solve`` additionally reports N*, M* and
M*/N* on the just-measured layers, so a targeted comparison is one command.

Examples
--------
    # Both surfaces for a representative subset of layers
    python hybridize.py --preset standard

    # Two-qubit brickwalls of growing depth on one layer type, any model
    python hybridize.py MODEL_ID --gate-sizes 2 --circuit-depths 0 1 2 4 8 \
        --layer-types v_proj

    # One specific layer (a block index + a layer type), measured AND solved:
    python hybridize.py MODEL_ID --profile-depths 10 --layer-types v_proj \
        --gate-sizes 2 --circuit-depths 0 1 2 4 8 --solve --budgets 1.003

    # e.g. arXiv:2410.17397v2's layer is just those arguments on SmolLM2:
    #   MODEL_ID = HuggingFaceTB/SmolLM2-135M, --profile-depths 10,
    #   --layer-types v_proj  (its (576,192) self-attention projection)

Re-running resumes from the CSV checkpoint: (layer, method, chi, D) points
already recorded are skipped.
"""

from __future__ import annotations

import argparse

from .compactifai_cli import FALLBACKS, PRESETS, DEFAULT_MODEL
from .compactifai_sweep import DEFAULT_EXCLUDE, DEFAULT_INCLUDE
from .hybrid_planner import (
    parse_shape,
    plan_rows,
    print_plan,
    shapes_from_model,
)
from .hybrid_sweep import (
    HybridConfig,
    default_depths,
    run_hybrid_sweep,
    validate_gate_sizes,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure perplexity vs. memory for the pure tensor-network "
                    "layer and for the disentangler (PQC) + tensor-network "
                    "layer, on the same probe tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--threads", type=int, default=None)

    # Tensor network.
    parser.add_argument("--tensorization", default="balanced",
                        choices=["balanced", "qubit"],
                        help="MPO geometry for BOTH the classical and hybrid "
                             "curves: 'balanced' (2-site, cached) or 'qubit' "
                             "(the paper's per-qubit geometry -- 36 classical "
                             "params at chi=1 for the (576,192) layer).")
    parser.add_argument("--qubit-align", default="msb", choices=["msb", "lsb"])
    parser.add_argument("--mpo-sites", type=int, default=2)
    parser.add_argument("--chi-min", type=int, default=2)
    parser.add_argument("--chi-max", type=int, default=None)
    parser.add_argument("--num-chi", type=int, default=None)
    parser.add_argument("--chi", type=int, nargs="+", default=None,
                        help="Explicit bond dimensions (overrides log spacing).")

    # Quantum circuits.
    parser.add_argument("--circuit-depths", type=int, nargs="+", default=None,
                        help="Brickwall depths D to try (default: 0 1 2 4 8). "
                             "D=0 keeps the circuits at identity and measures "
                             "the qubit-padding overhead on its own.")
    parser.add_argument("--gate-sizes", type=int, nargs="+", default=[2],
                        help="Qubits per gate k to try (default: 2). This build "
                             "considers only two-qubit gates, so k must be 1 or "
                             "2 -- the hardware-realistic regime, and the one "
                             "where Q(D) ~ 4^k per gate stays affordable.")
    parser.add_argument("--disentangle-target-chi", type=int, default=1,
                        help="Bond dimension the circuits are optimized to "
                             "squeeze the layer into (the paper uses 1).")
    parser.add_argument("--disentangle-optimizer", default="explicit",
                        choices=["explicit", "gradient"],
                        help="Training scheme for the circuits: 'explicit' (the "
                             "paper's environment-SVD sweep, closed-form) or "
                             "'gradient' (Adam directly on the gate angles). Each "
                             "writes its own CSV.")
    parser.add_argument("--disentangle-sweeps", type=int, default=12,
                        help="explicit: maximum environment sweeps per optimization.")
    parser.add_argument("--disentangle-gd-steps", type=int, default=200,
                        help="gradient: Adam steps per optimization.")
    parser.add_argument("--disentangle-gd-lr", type=float, default=0.05,
                        help="gradient: Adam learning rate.")
    parser.add_argument("--disentangle-tol", type=float, default=1e-6)
    parser.add_argument("--disentangle-init", default="identity",
                        choices=["identity", "random"],
                        help="'identity' starts the hybrid exactly at the "
                             "classical (padded) layer; 'random' starts Haar.")
    parser.add_argument("--disentangle-target", default="adaptive",
                        choices=["adaptive", "fixed"],
                        help="'adaptive' re-truncates the current operator each "
                             "iteration; 'fixed' freezes the target at the "
                             "original's truncation (the paper's Eq. 4).")
    parser.add_argument("--restarts", type=int, default=1,
                        help="Keep the best of this many initializations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantum-param-counting", default="manifold",
                        choices=["manifold", "entries"],
                        help="Q(D) as independent gate angles (dim O(2^k), the "
                             "default) or as dense matrix entries (4^k).")
    parser.add_argument("--max-qubits", type=int, default=13,
                        help="Skip layers whose index needs a bigger register.")
    parser.add_argument("--no-classical", action="store_true",
                        help="Do not re-measure the pure-TN curve (only do this "
                             "if it is already in the checkpoint).")

    # Layer selection / probe.
    parser.add_argument("--include", default=DEFAULT_INCLUDE)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--num-depths", type=int, default=None)
    parser.add_argument("--profile-depths", type=int, nargs="+", default=None)
    parser.add_argument("--layer-types", nargs="+", default=None)
    parser.add_argument("--per-layer-eval-tokens", type=int, default=None)
    parser.add_argument("--per-layer-stride", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")

    parser.add_argument("--solve", action="store_true",
                        help="After the sweep, solve the memory budget on the "
                             "just-measured layers and print N*, M* and M*/N* "
                             "(the same report as analyze_budget.py, in one "
                             "command). Narrowed by --layer-types / "
                             "--profile-depths when given.")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[1.001, 1.01, 1.05],
                        help="Perplexity budgets for --solve, as a ratio to the "
                             "dense baseline (1.01 = at most 1%% worse).")
    parser.add_argument("--q-weights", type=float, nargs="+", default=[0.0, 1.0],
                        help="Quantum-parameter prices for --solve (0 = the "
                             "circuits are free QPU memory; 1 = an angle costs a "
                             "classical weight).")
    parser.add_argument("--no-pareto", action="store_true",
                        help="With --solve, skip the Pareto-front printout.")
    parser.add_argument("--pareto-limit", type=int, default=6,
                        help="Layers listed in the --solve Pareto printout.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not score anything: print which (k, D) can "
                             "undercut the classical MPO on parameter count "
                             "alone, and stop. Cheap way to prune the grid.")
    parser.add_argument("--shapes", nargs="+", default=None,
                        help="Plan from explicit shapes instead of a model, "
                             "e.g. --shapes v_proj:576x192 up_proj:1536x576 "
                             "(implies --dry-run; no model is loaded).")
    parser.add_argument("--reference-chi", type=int, default=None,
                        help="Classical bond dimension the plan must undercut "
                             "(default: the dense layer).")
    parser.add_argument("--q-weight", type=float, default=1.0,
                        help="Price of a quantum parameter, for --dry-run.")
    parser.add_argument("--results-dir", default="results/llms")
    parser.add_argument("--csv-name", default="hybrid_per_layer.csv")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    for key, fallback in FALLBACKS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, preset.get(key, fallback))
    if args.preset:
        print(f"Preset '{args.preset}': {args.num_depths} blocks, "
              f"{args.num_chi} bond dimensions, "
              f"{args.per_layer_eval_tokens} tokens per evaluation.")
    return args


def _run_plan(args, shapes) -> int:
    rows = plan_rows(
        shapes, gate_sizes=args.gate_sizes,
        depths=sorted(set(args.circuit_depths or default_depths())),
        reference_chi=args.reference_chi, q_weight=args.q_weight,
        n_sites=args.mpo_sites, counting=args.quantum_param_counting,
    )
    print_plan(rows, args.reference_chi, args.q_weight)
    live = [r for r in rows if not r["vetoed"]]
    print(f"\n{len(live)} of {len(rows)} (layer, k, D) configurations survive on "
          f"parameter count alone.")
    if live:
        ks = sorted({r["gate_size"] for r in live})
        ds = sorted({r["circuit_depth"] for r in live})
        print(f"Worth scoring: k in {ks}, D in {ds}.")
    else:
        print("None survive: at this price the circuits cost more than the layer "
              "they compress. Lower --q-weight, or use narrower gates.")
    return 0


def main(argv=None) -> int:
    args = apply_preset(parse_args(argv))
    try:
        args.gate_sizes = validate_gate_sizes(args.gate_sizes)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    if args.shapes:
        return _run_plan(args, [parse_shape(s) for s in args.shapes])

    config = HybridConfig(
        model_id=args.model, device=args.device, dtype=args.dtype,
        trust_remote_code=args.trust_remote_code, revision=args.revision,
        tensorization=args.tensorization, qubit_align=args.qubit_align,
        mpo_sites=args.mpo_sites, chi_min=args.chi_min, chi_max=args.chi_max,
        num_chi=args.num_chi, chi_values=args.chi,
        include_pattern=args.include, exclude_pattern=args.exclude,
        dataset=args.dataset, dataset_config=args.dataset_config,
        split=args.split, text_column=args.text_column,
        max_length=args.max_length, stride=args.stride,
        ppl_batch_size=args.ppl_batch_size,
        profile_depths=args.profile_depths, num_depths=args.num_depths,
        layer_types=args.layer_types,
        per_layer_eval_tokens=args.per_layer_eval_tokens or None,
        per_layer_stride=args.per_layer_stride,
        circuit_depths=args.circuit_depths or default_depths(),
        gate_sizes=args.gate_sizes,
        disentangle_target_chi=args.disentangle_target_chi,
        disentangle_optimizer=args.disentangle_optimizer,
        disentangle_gd_steps=args.disentangle_gd_steps,
        disentangle_gd_lr=args.disentangle_gd_lr,
        disentangle_sweeps=args.disentangle_sweeps,
        disentangle_tol=args.disentangle_tol,
        disentangle_init=args.disentangle_init,
        disentangle_target_mode=args.disentangle_target,
        disentangle_restarts=args.restarts,
        disentangle_seed=args.seed,
        quantum_param_counting=args.quantum_param_counting,
        max_qubits=args.max_qubits,
        run_classical=not args.no_classical,
        results_dir=args.results_dir, hybrid_csv_name=args.csv_name,
        force_recompute=args.recompute,
    )
    if args.dry_run:
        from .benchmark import BenchmarkConfig, load_model_and_tokenizer, resolve_device
        device = resolve_device(config.device)
        model, _tok = load_model_and_tokenizer(
            BenchmarkConfig(model_id=config.model_id, device=config.device,
                            dtype=config.dtype,
                            trust_remote_code=config.trust_remote_code,
                            revision=config.revision), device)
        return _run_plan(args, shapes_from_model(model, config))

    path = run_hybrid_sweep(config)

    if args.solve:
        from .budget_cli import filter_curves, report_curves
        from .budget_frontier import (
            filter_rows_by_tensorization,
            load_curves,
            read_rows,
        )
        # Report only rows of this run's geometry -- C(chi) is not comparable
        # across tensorizations, so never mix them in one N*/M* table.
        rows = filter_rows_by_tensorization(read_rows(path), config.tensorization)
        curves = filter_curves(
            load_curves(rows),
            layer_types=args.layer_types, depths=args.profile_depths)
        if not curves:
            print("\n(--solve: no layers to report after filtering.)")
        else:
            report_curves(curves, args.budgets, sorted(set(args.q_weights)),
                          pareto=not args.no_pareto,
                          pareto_limit=args.pareto_limit)
            print(f"\nFull figure suite: python analyze_budget.py {path}")
    else:
        print(f"\nNext: analyse the budget with\n"
              f"    python analyze_budget.py {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
