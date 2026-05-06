"""CASP benchmark loader and evaluator.

This module provides a lightweight CASP benchmark path that can fetch a real
CASP target from RCSB over HTTP only (no pre-downloaded binary artifacts).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm

from .benchmarks import BenchmarkMetrics

# CASP target to experimental PDB mapping (small curated subset)
CASP_TO_PDB: Dict[str, str] = {
    "T1049": "6XG2",
    "T1024": "6W2N",
    "T1030": "6XC4",
}

THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


class CASPDataset:
    """CASP target loader using direct HTTP calls to the RCSB PDB API."""

    def __init__(
        self,
        casp_version: int = 14,
        data_dir: str = "data/casp",
        download: bool = True,
        target_ids: Optional[list[str]] = None,
    ):
        del data_dir  # intentionally unused: no local PDB persistence required
        if casp_version not in {14, 15, 16}:
            raise ValueError(f"CASP version must be 14, 15, or 16, got {casp_version}")
        self.casp_version = casp_version
        self.targets = target_ids or ["T1049"]
        self.structures: Dict[str, np.ndarray] = {}
        self.sequences: Dict[str, str] = {}

        if download:
            self.download_targets()

    def _parse_pdb_text(self, pdb_text: str) -> Tuple[str, np.ndarray]:
        sequence = []
        coords = []
        seen_residues = set()
        chain_id = None

        for line in StringIO(pdb_text):
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            current_chain = line[21].strip()
            if chain_id is None:
                chain_id = current_chain
            if current_chain != chain_id:
                continue

            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = (current_chain, resseq, icode)
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)

            resname = line[17:20].strip().upper()
            sequence.append(THREE_TO_ONE.get(resname, "X"))
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

        if not coords:
            raise ValueError("No CA coordinates parsed from PDB text")

        return "".join(sequence), np.asarray(coords, dtype=np.float32)

    def download_targets(self) -> None:
        """Fetch mapped CASP targets from RCSB via HTTP."""
        for target in tqdm(self.targets, desc=f"Fetching CASP{self.casp_version}"):
            pdb_id = CASP_TO_PDB.get(target)
            if pdb_id is None:
                continue
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                sequence, coords = self._parse_pdb_text(response.text)
                self.sequences[target] = sequence
                self.structures[target] = coords
            except requests.RequestException:
                # Keep benchmarking runnable in restricted/offline environments.
                continue

    def get_target(self, target_id: str) -> Tuple[str, np.ndarray]:
        if target_id not in self.structures:
            raise ValueError(f"Target {target_id} not available: {list(self.structures.keys())}")
        return self.sequences[target_id], self.structures[target_id]

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self):
        for target_id in self.structures:
            yield target_id, self.sequences[target_id], self.structures[target_id]


class CASPBenchmark:
    """Benchmark a model + embedder on HTTP-fetched CASP targets."""

    def __init__(self, model, embedder, device: str = "cpu"):
        self.model = model
        self.embedder = embedder
        self.device = device
        self.evaluator = BenchmarkMetrics()
        self.model.to(device)
        self.model.eval()

    def benchmark_casp(self, casp_version: int = 14, max_targets: Optional[int] = None) -> Dict:
        dataset = CASPDataset(casp_version=casp_version, download=True)
        results = {"per_target": {}, "tm_scores": [], "rmsd_scores": [], "gdt_ts_scores": []}

        evaluated = 0
        for target_id, sequence, true_coords in dataset:
            if max_targets is not None and evaluated >= max_targets:
                break

            with torch.no_grad():
                emb_dict = self.embedder([sequence])
                embeddings = emb_dict["embeddings"].to(self.device)
                outputs = self.model(embeddings)
                pred_coords = outputs["coordinates"][0].detach().cpu().numpy()

            min_len = min(len(pred_coords), len(true_coords))
            pred_coords = pred_coords[:min_len]
            true_coords = true_coords[:min_len]
            metrics = self.evaluator.calculate_all(predicted=pred_coords, ground_truth=true_coords)

            results["tm_scores"].append(metrics["tm_score"])
            results["rmsd_scores"].append(metrics["rmsd"])
            results["gdt_ts_scores"].append(metrics["gdt_ts"])
            results["per_target"][target_id] = {"length": min_len, **metrics}
            evaluated += 1

        if evaluated > 0:
            results["summary"] = {
                "n_targets": evaluated,
                "tm_score_mean": float(np.mean(results["tm_scores"])),
                "rmsd_mean": float(np.mean(results["rmsd_scores"])),
                "gdt_ts_mean": float(np.mean(results["gdt_ts_scores"])),
            }
        return results

    def save_results(self, results: Dict, output_file: str) -> None:
        import json

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
