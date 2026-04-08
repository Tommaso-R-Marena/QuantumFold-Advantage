#!/usr/bin/env python3
"""QuantumFold-Advantage — main experiment runner.

Trains both a quantum-enhanced and a classical-only model on the same
data, evaluates both on a held-out test set, and runs a full
statistical comparison.

Usage:
    python scripts/run_experiment.py                   # synthetic demo
    python scripts/run_experiment.py --epochs 50       # longer run
    python scripts/run_experiment.py --no-quantum      # classical only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hybrid_dataset import create_dataloaders, generate_synthetic_proteins
from src.evaluation.metrics import evaluate_structure
from src.evaluation.statistical_tests import (
    compare_quantum_classical,
    format_comparison_report,
)
from src.evaluation.visualization import (
    plot_metric_comparison,
    plot_per_protein_improvement,
    plot_training_curves,
)
from src.models.quantumfold_advantage import (
    create_classical_model,
    create_quantum_model,
)
from src.training.trainer import QuantumFoldTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")


def parse_args():
    p = argparse.ArgumentParser(description="QuantumFold-Advantage experiment")
    p.add_argument("--n-proteins", type=int, default=60)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-pair", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-evoformer", type=int, default=2)
    p.add_argument("--n-struct-iter", type=int, default=2)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-circuit-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quantum-lr", type=float, default=5e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-quantum", action="store_true", help="Skip quantum model")
    p.add_argument("--output-dir", type=str, default="results")
    return p.parse_args()


@torch.no_grad()
def evaluate_model_on_test(model, test_loader, device):
    """Run model on test set and compute per-protein metrics."""
    model.eval()
    all_metrics = {"rmsd": [], "tm_score": [], "gdt_ts": [], "gdt_ha": [], "lddt": []}

    for batch in test_loader:
        aa_idx = batch["aa_idx"].to(device)
        physchem = batch["physchem"].to(device)
        mask = batch["mask"].to(device)
        coords_true = batch["coords"]

        pred = model(aa_idx, physchem, mask=mask)

        for i in range(aa_idx.size(0)):
            length = int(mask[i].sum().item())
            p = pred["coords_ca"][i, :length].cpu().numpy()
            t = coords_true[i, :length, 1, :].numpy()
            m = evaluate_structure(p, t)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    return {k: np.array(v) for k, v in all_metrics.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Data
    logger.info("Generating synthetic protein data...")
    seqs, coords = generate_synthetic_proteins(args.n_proteins, seed=args.seed)
    train_ld, val_ld, test_ld = create_dataloaders(
        seqs, coords, max_len=args.max_len, batch_size=args.batch_size, seed=args.seed
    )
    logger.info(f"Train: {len(train_ld.dataset)}, Val: {len(val_ld.dataset)}, Test: {len(test_ld.dataset)}")

    model_kwargs = dict(
        d_model=args.d_model,
        d_pair=args.d_pair,
        n_heads=args.n_heads,
        n_evoformer_blocks=args.n_evoformer,
        n_structure_iterations=args.n_struct_iter,
        max_seq_len=args.max_len,
        n_qubits=args.n_qubits,
        n_circuit_layers=args.n_circuit_layers,
    )

    histories = {}
    test_results = {}

    # --- Classical model ---
    logger.info("=== Training CLASSICAL model ===")
    classical = create_classical_model(**model_kwargs).to(device)
    trainer_c = QuantumFoldTrainer(
        classical, lr=args.lr, quantum_lr=args.quantum_lr,
        patience=args.patience, checkpoint_dir=str(out / "checkpoints"),
    )
    histories["classical"] = trainer_c.train(train_ld, val_ld, epochs=args.epochs)
    test_results["classical"] = evaluate_model_on_test(classical, test_ld, device)

    # --- Quantum model ---
    if not args.no_quantum:
        logger.info("=== Training QUANTUM model ===")
        quantum = create_quantum_model(**model_kwargs).to(device)
        trainer_q = QuantumFoldTrainer(
            quantum, lr=args.lr, quantum_lr=args.quantum_lr,
            patience=args.patience, checkpoint_dir=str(out / "checkpoints"),
        )
        histories["quantum"] = trainer_q.train(train_ld, val_ld, epochs=args.epochs)
        test_results["quantum"] = evaluate_model_on_test(quantum, test_ld, device)

    # --- Statistical comparison ---
    if "quantum" in test_results and "classical" in test_results:
        logger.info("Running statistical comparison...")
        comparison = compare_quantum_classical(
            test_results["quantum"], test_results["classical"]
        )
        report = format_comparison_report(comparison)
        print(report)
        (out / "comparison_report.txt").write_text(report)

        # Plots
        for metric in ["rmsd", "tm_score", "gdt_ts"]:
            plot_metric_comparison(
                test_results["quantum"][metric],
                test_results["classical"][metric],
                metric,
                str(out / f"comparison_{metric}.png"),
            )
        plot_training_curves(histories["quantum"], histories["classical"],
                            str(out / "training_curves.png"))

        n_test = len(test_results["quantum"]["tm_score"])
        names = [f"P{i}" for i in range(n_test)]
        plot_per_protein_improvement(
            names,
            test_results["quantum"]["tm_score"],
            test_results["classical"]["tm_score"],
            output_path=str(out / "per_protein_tm.png"),
        )

    # Save raw results
    for model_name, metrics in test_results.items():
        np.savez(out / f"{model_name}_metrics.npz", **metrics)

    logger.info(f"All results saved to {out}/")


if __name__ == "__main__":
    main()
