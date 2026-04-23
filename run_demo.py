#!/usr/bin/env python3
"""Run a minimal end-to-end QuantumFold demo on a short synthetic sequence.

This script intentionally avoids external APIs and hardware-specific backends.
It trains for a few quick steps on synthetic proteins and runs inference on a
short sequence (<=10 residues). If PennyLane is available, it uses the hybrid
quantum model on the ``default.qubit`` simulator; otherwise it falls back to
the classical model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import importlib.util

from src.model import create_model
from src.train import evaluate_model, train_model


def _load_data_module():
    data_py = Path(__file__).resolve().parent / "src" / "data.py"
    spec = importlib.util.spec_from_file_location("qf_data_module", data_py)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantumFold-Advantage quick demo")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs for quick demo")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--output-dir", type=str, default="outputs/demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        choices=["classical", "quantum"],
        default="quantum",
        help="Try quantum by default; automatically falls back if PennyLane unavailable.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_demo")
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_module = _load_data_module()

    # Keep residue count <= 10 per request.
    sequences, coordinates = data_module.generate_synthetic_data(
        num_samples=8,
        min_length=10,
        max_length=10,
        seed=args.seed,
    )

    train_dataset = data_module.ProteinDataset(sequences[:6], coordinates[:6])
    test_dataset = data_module.ProteinDataset(sequences[6:], coordinates[6:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model
    model = create_model(model_type=model_type, input_dim=20, hidden_dim=32, output_dim=3, n_qubits=4)

    # If quantum requested but PennyLane unavailable, model will disable quantum internally.
    model = model.to(device)
    logger.info("Training %s model on %s", model_type, device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=1e-3,
    )
    metrics = evaluate_model(model, test_loader, device)

    # Inference on one short synthetic protein (<=10 residues)
    sample = test_dataset[0]
    sample_batch = sample["sequence"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_coords = model(sample_batch).squeeze(0).cpu().tolist()

    result = {
        "sequence": sequences[6],
        "sequence_length": len(sequences[6]),
        "metrics": metrics,
        "history": history,
        "predicted_coordinates": pred_coords,
        "quantum_backend": "default.qubit (simulator fallback)",
    }

    out_file = output_dir / "demo_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Demo complete. Results saved to %s", out_file)
    logger.info("Sequence length: %d", result["sequence_length"])
    logger.info("Test RMSD: %.4f | TM-score: %.4f", metrics["rmsd"], metrics["tm_score"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
