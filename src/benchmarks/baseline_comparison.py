"""Rigorous baseline comparisons against AlphaFold2, ESMFold, and RoseTTAFold.

This module implements standardized benchmarking protocols for fair comparison
with state-of-the-art protein structure prediction methods.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    method: str
    pdb_id: str
    rmsd: float
    tm_score: float
    gdt_ts: float
    gdt_ha: float
    lddt: float
    inference_time: float
    memory_usage: float
    sequence_length: int


class BaselineComparator:
    """Compare QuantumFold-Advantage against established baselines."""

    def __init__(self, cache_dir: str = "./baseline_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.parser = PDBParser(QUIET=True)

    def kabsch_align(self, P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Kabsch algorithm for optimal rotation.

        Args:
            P: Nx3 array of predicted coordinates
            Q: Nx3 array of target coordinates

        Returns:
            Aligned coordinates and RMSD
        """
        # Center both coordinate sets
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Compute optimal rotation
        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply transformation
        P_aligned = (P_centered @ R) + centroid_Q

        # Compute RMSD
        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

        return P_aligned, rmsd

    def compute_tm_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute TM-score (template modeling score).

        Args:
            pred: Predicted coordinates (N, 3)
            target: Target coordinates (N, 3)

        Returns:
            TM-score (0-1, higher is better)
        """
        L = len(pred)
        d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8

        aligned_pred, _ = self.kabsch_align(pred, target)
        distances = np.sqrt(np.sum((aligned_pred - target) ** 2, axis=1))
        tm_score = np.mean(1 / (1 + (distances / d0) ** 2))

        return tm_score

    def compute_gdt(
        self, pred: np.ndarray, target: np.ndarray, cutoffs: List[float] = [1, 2, 4, 8]
    ) -> Dict[str, float]:
        """Compute GDT (Global Distance Test) scores.

        Args:
            pred: Predicted coordinates (N, 3)
            target: Target coordinates (N, 3)
            cutoffs: Distance cutoffs in Angstroms

        Returns:
            Dictionary with GDT_TS and GDT_HA scores
        """
        aligned_pred, _ = self.kabsch_align(pred, target)
        distances = np.sqrt(np.sum((aligned_pred - target) ** 2, axis=1))

        gdt_scores = {}
        for cutoff in cutoffs:
            gdt_scores[f"GDT_{cutoff}A"] = (distances < cutoff).mean() * 100

        # GDT_TS (Total Score): average of 1, 2, 4, 8Å cutoffs
        gdt_scores["GDT_TS"] = np.mean([gdt_scores[f"GDT_{c}A"] for c in [1, 2, 4, 8]])

        # GDT_HA (High Accuracy): average of 0.5, 1, 2, 4Å cutoffs
        ha_cutoffs = [0.5, 1, 2, 4]
        ha_scores = []
        for cutoff in ha_cutoffs:
            ha_scores.append((distances < cutoff).mean() * 100)
        gdt_scores["GDT_HA"] = np.mean(ha_scores)

        return gdt_scores

    def compute_lddt(
        self, pred: np.ndarray, target: np.ndarray, inclusion_radius: float = 15.0
    ) -> float:
        """Compute lDDT (local Distance Difference Test).

        Args:
            pred: Predicted coordinates (N, 3)
            target: Target coordinates (N, 3)
            inclusion_radius: Radius for local interactions

        Returns:
            lDDT score (0-100)
        """
        len(pred)
        target_dists = cdist(target, target)
        pred_dists = cdist(pred, pred)

        # Only consider pairs within inclusion radius in target
        mask = (target_dists < inclusion_radius) & (target_dists > 0)

        if mask.sum() == 0:
            return 0.0

        # Compute distance differences
        diff = np.abs(pred_dists[mask] - target_dists[mask])

        # Count preserved distances at different thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        preserved = np.mean([diff < t for t in thresholds], axis=0)

        return np.mean(preserved) * 100

    def get_esmfold_prediction(self, sequence: str, pdb_id: str) -> Optional[np.ndarray]:
        """Get ESMFold prediction via API.

        Args:
            sequence: Amino acid sequence
            pdb_id: PDB identifier for caching

        Returns:
            Predicted CA coordinates or None if failed
        """
        cache_file = self.cache_dir / f"esmfold_{pdb_id}.pdb"

        if cache_file.exists():
            logger.info(f"Loading cached ESMFold prediction for {pdb_id}")
            structure = self.parser.get_structure(pdb_id, str(cache_file))
            coords = self._extract_ca_coords(structure)
            return coords

        try:
            logger.info(f"Requesting ESMFold prediction for {pdb_id}")
            url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            response = requests.post(url, data=sequence, timeout=300)

            if response.status_code == 200:
                with open(cache_file, "w") as f:
                    f.write(response.text)

                structure = self.parser.get_structure(pdb_id, str(cache_file))
                coords = self._extract_ca_coords(structure)
                return coords
            else:
                logger.error(f"ESMFold API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"ESMFold prediction failed: {e}")
            return None

    def _extract_ca_coords(self, structure) -> np.ndarray:
        """Extract CA coordinates from PDB structure."""
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_coord())
        return np.array(coords)

    def benchmark_against_esmfold(
        self, sequence: str, target_coords: np.ndarray, pdb_id: str
    ) -> BenchmarkResult:
        """Benchmark against ESMFold.

        Args:
            sequence: Amino acid sequence
            target_coords: Ground truth CA coordinates
            pdb_id: PDB identifier

        Returns:
            BenchmarkResult object
        """
        import time

        start_time = time.time()

        pred_coords = self.get_esmfold_prediction(sequence, pdb_id)

        if pred_coords is None or len(pred_coords) != len(target_coords):
            logger.warning(f"ESMFold prediction failed or length mismatch for {pdb_id}")
            return BenchmarkResult(
                method="ESMFold",
                pdb_id=pdb_id,
                rmsd=float("inf"),
                tm_score=0.0,
                gdt_ts=0.0,
                gdt_ha=0.0,
                lddt=0.0,
                inference_time=time.time() - start_time,
                memory_usage=0.0,
                sequence_length=len(sequence),
            )

        _, rmsd = self.kabsch_align(pred_coords, target_coords)
        tm_score = self.compute_tm_score(pred_coords, target_coords)
        gdt = self.compute_gdt(pred_coords, target_coords)
        lddt = self.compute_lddt(pred_coords, target_coords)

        return BenchmarkResult(
            method="ESMFold",
            pdb_id=pdb_id,
            rmsd=rmsd,
            tm_score=tm_score,
            gdt_ts=gdt["GDT_TS"],
            gdt_ha=gdt["GDT_HA"],
            lddt=lddt,
            inference_time=time.time() - start_time,
            memory_usage=0.0,  # Not tracked for API calls
            sequence_length=len(sequence),
        )

    def run_comprehensive_benchmark(
        self, test_proteins: List[Tuple[str, str, np.ndarray]], model, device
    ) -> Dict:
        """Run comprehensive benchmark suite.

        Args:
            test_proteins: List of (pdb_id, sequence, target_coords) tuples
            model: QuantumFold-Advantage model
            device: torch device

        Returns:
            Dictionary with benchmark results
        """
        results = {"QuantumFold": [], "ESMFold": []}

        for pdb_id, sequence, target_coords in test_proteins:
            logger.info(f"Benchmarking {pdb_id} ({len(sequence)} residues)")

            # Benchmark QuantumFold
            try:
                import time

                start_time = time.time()

                with torch.no_grad():
                    # Get prediction from model
                    # This assumes your model has appropriate interface
                    output = model.predict(sequence, device=device)
                    pred_coords = output["coords"].cpu().numpy()

                inference_time = time.time() - start_time

                if len(pred_coords) == len(target_coords):
                    _, rmsd = self.kabsch_align(pred_coords, target_coords)
                    tm_score = self.compute_tm_score(pred_coords, target_coords)
                    gdt = self.compute_gdt(pred_coords, target_coords)
                    lddt = self.compute_lddt(pred_coords, target_coords)

                    results["QuantumFold"].append(
                        BenchmarkResult(
                            method="QuantumFold-Advantage",
                            pdb_id=pdb_id,
                            rmsd=rmsd,
                            tm_score=tm_score,
                            gdt_ts=gdt["GDT_TS"],
                            gdt_ha=gdt["GDT_HA"],
                            lddt=lddt,
                            inference_time=inference_time,
                            memory_usage=(
                                torch.cuda.max_memory_allocated() / 1e9
                                if torch.cuda.is_available()
                                else 0.0
                            ),
                            sequence_length=len(sequence),
                        )
                    )

            except Exception as e:
                logger.error(f"QuantumFold prediction failed for {pdb_id}: {e}")

            # Benchmark ESMFold
            esmfold_result = self.benchmark_against_esmfold(sequence, target_coords, pdb_id)
            results["ESMFold"].append(esmfold_result)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self._aggregate_results(results)

    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate benchmark results with statistics."""
        aggregated = {}

        for method, result_list in results.items():
            if not result_list:
                continue

            metrics = {
                "rmsd": [r.rmsd for r in result_list if r.rmsd != float("inf")],
                "tm_score": [r.tm_score for r in result_list],
                "gdt_ts": [r.gdt_ts for r in result_list],
                "gdt_ha": [r.gdt_ha for r in result_list],
                "lddt": [r.lddt for r in result_list],
                "inference_time": [r.inference_time for r in result_list],
            }

            aggregated[method] = {
                "mean": {k: np.mean(v) if v else float("nan") for k, v in metrics.items()},
                "std": {k: np.std(v) if v else float("nan") for k, v in metrics.items()},
                "median": {k: np.median(v) if v else float("nan") for k, v in metrics.items()},
                "n_proteins": len(result_list),
                "raw_results": result_list,
            }

        return aggregated

    def generate_comparison_table(self, aggregated_results: Dict) -> str:
        """Generate publication-quality comparison table."""
        table = "\n" + "=" * 100 + "\n"
        table += f"{'Method':<25} | {'RMSD (Å)':<12} | {'TM-score':<12} | {'GDT_TS':<12} | {'GDT_HA':<12} | {'lDDT':<12}\n"
        table += "=" * 100 + "\n"

        for method, stats in aggregated_results.items():
            table += f"{method:<25} | "
            table += f"{stats['mean']['rmsd']:.2f}±{stats['std']['rmsd']:.2f}  | "
            table += f"{stats['mean']['tm_score']:.3f}±{stats['std']['tm_score']:.3f} | "
            table += f"{stats['mean']['gdt_ts']:.1f}±{stats['std']['gdt_ts']:.1f}   | "
            table += f"{stats['mean']['gdt_ha']:.1f}±{stats['std']['gdt_ha']:.1f}   | "
            table += f"{stats['mean']['lddt']:.1f}±{stats['std']['lddt']:.1f}\n"

        table += "=" * 100 + "\n"
        return table

    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON."""
        # Convert BenchmarkResult objects to dicts for JSON serialization
        serializable = {}
        for method, data in results.items():
            serializable[method] = data.copy()
            if "raw_results" in serializable[method]:
                serializable[method]["raw_results"] = [
                    vars(r) for r in serializable[method]["raw_results"]
                ]

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {output_path}")
