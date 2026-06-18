"""Command-line interface for the modular GNN edge-flow workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch

from .data import build_mosaic_data, selected_harmonics
from .evaluate import split_masks
from .experiment import run_experiment, run_sweep
from .graph_io import load_graph_from_args
from .plotting import plot_sweep_summary
from .utils import (
    install_numpy_pickle_compat,
    resolve_device,
    set_seed,
    write_csv,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(__file__).resolve().parents[5]


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--config", default=str(DATA_ROOT / "emb1" / "config.json"))
    ap.add_argument("--graph", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "renders" / "gnn_edge_dc"))

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-delta", type=float, default=1e-3)
    ap.add_argument("--lambda-h1", type=float, default=1.0)
    ap.add_argument("--lambda-h2", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--jitter", type=float, default=1e-18)

    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument("--torch-threads", type=int, default=1)

    ap.add_argument(
        "--flow-components",
        choices=("dc", "dc-h1", "dc-h1-h2"),
        default="dc",
        help="flow components used for edge features/losses",
    )
    ap.add_argument(
        "--include-harmonic-features",
        action="store_true",
        help="deprecated alias for --flow-components dc-h1-h2",
    )

    ap.add_argument(
        "--sweep",
        action="store_true",
        help="run K/hidden/lambda_delta/seed masked-validation sweep",
    )
    ap.add_argument("--K-values", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--hidden-dim-values", nargs="*", type=int, default=[32, 64, 128])
    ap.add_argument("--lambda-delta-values", nargs="*", type=float, default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])

    ap.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    ap.set_defaults(use_tqdm=True)

    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the modular GNN edge-flow workflow."""
    args = parse_args(argv)

    if bool(args.include_harmonic_features) and args.flow_components == "dc":
        args.flow_components = "dc-h1-h2"

    args.harmonics = list(selected_harmonics(args.flow_components))

    set_seed(args.seed)

    try:
        torch.set_num_threads(max(int(args.torch_threads), 1))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    install_numpy_pickle_compat()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    data = build_mosaic_data(graph, flow_components=args.flow_components)
    device = resolve_device(args.device)

    print(f"Loaded {graph_path}")
    print(f"Built mosaic graph data: {len(data.node_ids)} nodes, {len(data.edge_ids)} edges")
    print(f"Valid DC edge observations: {int(data.valid_mask.sum())}")

    if args.harmonics:
        counts = {
            f"H{h}": int(data.harmonic_valid_mask[:, i].sum().item())
            for i, h in enumerate(args.harmonics)
        }
        print(f"Harmonic components: {args.flow_components}; valid observations: {counts}")
    else:
        print("Harmonic components: dc only")

    print(f"Training on {device}")

    write_json(
        out_root / "run_config.json",
        {
            "args": vars(args),
            "graph": str(graph_path),
            "n_nodes": len(data.node_ids),
            "n_edges": len(data.edge_ids),
            "n_valid_edges": int(data.valid_mask.sum()),
        },
    )

    if args.sweep:
        summaries = run_sweep(data, graph, args, device, out_root)
        write_csv(out_root / "sweep_summary.csv", summaries)
        plot_sweep_summary(summaries, out_root)
        print(f"Done. Sweep outputs written to {out_root}")
        return 0

    all_train = data.valid_mask.clone()
    no_val = torch.zeros_like(data.valid_mask)

    run_experiment(
        data,
        args,
        device,
        out_root / "no_cross_validation",
        "no_cv",
        all_train,
        no_val,
        graph=graph,
    )

    train_mask, val_mask = split_masks(data, args.val_fraction, args.seed)

    run_experiment(
        data,
        args,
        device,
        out_root / "masked_edge_validation_15pct",
        "masked_15pct",
        train_mask,
        val_mask,
        graph=graph,
    )

    print(f"Done. Outputs written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
