"""Benchmarking utilities for comparing QuantumFold with AlphaFold-3.

This module provides comprehensive evaluation metrics and comparison tools
for assessing protein structure prediction quality.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import json
from pathlib import Path

try:
    from Bio.PDB import PDBParser, Superimposer
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    print("Warning: Biopython not installed. Install with: pip install biopython")


@dataclass
class StructureMetrics:
    """Container for protein structure evaluation metrics."""
    rmsd: float  # Root mean square deviation (Å)
    tm_score: float  # Template modeling score
    gdt_ts: float  # Global distance test - total score
    gdt_ha: float  # Global distance test - high accuracy
    lddt: float  # Local distance difference test
    clash_score: float  # Steric clash score
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rmsd': self.rmsd,
            'tm_score': self.tm_score,
            'gdt_ts': self.gdt_ts,
            'gdt_ha': self.gdt_ha,
            'lddt': self.lddt,
            'clash_score': self.clash_score
        }


class ProteinStructureEvaluator:
    """Evaluator for protein structure prediction quality.
    
    Implements multiple metrics used in CASP and structure prediction
    competitions to provide comprehensive assessment.
    """
    
    def __init__(self, distance_thresholds: Optional[List[float]] = None):
        """
        Args:
            distance_thresholds: Distance cutoffs for GDT calculation (Å)
        """
        self.distance_thresholds = distance_thresholds or [1.0, 2.0, 4.0, 8.0]
    
    def calculate_rmsd(
        self,
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
        align: bool = True
    ) -> float:
        """Calculate RMSD between predicted and true coordinates.
        
        Args:
            coords_pred: Predicted coordinates (N, 3)
            coords_true: True coordinates (N, 3)
            align: Whether to align structures first
            
        Returns:
            RMSD value in Angstroms
        """
        if align:
            coords_pred = self._align_structures(coords_pred, coords_true)
        
        diff = coords_pred - coords_true
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        return float(rmsd)
    
    def calculate_tm_score(
        self,
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> float:
        """Calculate TM-score for structure comparison.
        
        TM-score is length-independent and ranges from 0 to 1,
        where >0.5 indicates similar fold and >0.6 indicates same topology.
        
        Args:
            coords_pred: Predicted coordinates (N, 3)
            coords_true: True coordinates (N, 3)
            sequence_length: Length of protein sequence (defaults to N)
            
        Returns:
            TM-score value
        """
        if sequence_length is None:
            sequence_length = len(coords_true)
        
        # Align structures
        coords_pred_aligned = self._align_structures(coords_pred, coords_true)
        
        # TM-score normalization factor
        d0 = 1.24 * (sequence_length - 15) ** (1/3) - 1.8
        d0 = max(d0, 0.5)  # Minimum d0 value
        
        # Calculate distances
        distances = np.sqrt(np.sum((coords_pred_aligned - coords_true) ** 2, axis=1))
        
        # TM-score formula
        tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
        
        return float(tm_score)
    
    def calculate_gdt(
        self,
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Tuple[float, float]:
        """Calculate GDT_TS and GDT_HA scores.
        
        Global Distance Test measures the percentage of residues within
        distance thresholds after optimal superposition.
        
        Args:
            coords_pred: Predicted coordinates (N, 3)
            coords_true: True coordinates (N, 3)
            thresholds: Distance thresholds in Angstroms
            
        Returns:
            Tuple of (GDT_TS, GDT_HA) scores (0-100 scale)
        """
        if thresholds is None:
            thresholds = self.distance_thresholds
        
        # Align structures
        coords_pred_aligned = self._align_structures(coords_pred, coords_true)
        
        # Calculate distances
        distances = np.sqrt(np.sum((coords_pred_aligned - coords_true) ** 2, axis=1))
        
        # Calculate percentage under each threshold
        percentages = []
        for threshold in thresholds:
            pct = 100.0 * np.mean(distances < threshold)
            percentages.append(pct)
        
        # GDT_TS: average of 1, 2, 4, 8 Å thresholds
        gdt_ts = np.mean(percentages)
        
        # GDT_HA: average of 0.5, 1, 2, 4 Å thresholds
        ha_thresholds = [0.5, 1.0, 2.0, 4.0]
        ha_percentages = []
        for threshold in ha_thresholds:
            pct = 100.0 * np.mean(distances < threshold)
            ha_percentages.append(pct)
        gdt_ha = np.mean(ha_percentages)
        
        return float(gdt_ts), float(gdt_ha)
    
    def calculate_lddt(
        self,
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
        cutoff: float = 15.0,
        thresholds: Optional[List[float]] = None
    ) -> float:
        """Calculate Local Distance Difference Test (lDDT) score.
        
        lDDT measures local structure quality by examining distance
        differences between nearby residues.
        
        Args:
            coords_pred: Predicted coordinates (N, 3)
            coords_true: True coordinates (N, 3)
            cutoff: Distance cutoff for considering residue pairs
            thresholds: Difference thresholds for scoring
            
        Returns:
            lDDT score (0-1 scale)
        """
        if thresholds is None:
            thresholds = [0.5, 1.0, 2.0, 4.0]
        
        n_residues = len(coords_true)
        
        # Calculate all pairwise distances
        dist_true = np.sqrt(np.sum((coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=2))
        dist_pred = np.sqrt(np.sum((coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=2))
        
        # Only consider pairs within cutoff in true structure
        mask = (dist_true < cutoff) & (dist_true > 0)  # Exclude self-distances
        
        if not np.any(mask):
            return 0.0
        
        # Calculate distance differences
        dist_diff = np.abs(dist_true - dist_pred)
        
        # Score each pair
        scores = []
        for threshold in thresholds:
            preserved = (dist_diff[mask] < threshold).astype(float)
            scores.append(np.mean(preserved))
        
        # Average over thresholds
        lddt = np.mean(scores)
        
        return float(lddt)
    
    def calculate_clash_score(
        self,
        coords: np.ndarray,
        clash_threshold: float = 2.0
    ) -> float:
        """Calculate steric clash score.
        
        Args:
            coords: Atomic coordinates (N, 3)
            clash_threshold: Minimum allowed distance between non-bonded atoms (Å)
            
        Returns:
            Clash score (number of clashes per 100 residues)
        """
        n_atoms = len(coords)
        
        # Calculate all pairwise distances
        distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
        
        # Count clashes (excluding neighbors and self)
        clash_mask = (distances < clash_threshold) & (distances > 0)
        # Exclude adjacent residues (assuming sequential atoms)
        for i in range(n_atoms):
            if i > 0:
                clash_mask[i, i-1] = False
            if i < n_atoms - 1:
                clash_mask[i, i+1] = False
        
        n_clashes = np.sum(clash_mask) // 2  # Divide by 2 to avoid double counting
        
        # Normalize by number of residues
        clash_score = (n_clashes / n_atoms) * 100
        
        return float(clash_score)
    
    def evaluate_structure(
        self,
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> StructureMetrics:
        """Comprehensive evaluation of predicted structure.
        
        Args:
            coords_pred: Predicted coordinates
            coords_true: True coordinates
            sequence_length: Protein sequence length
            
        Returns:
            StructureMetrics object with all scores
        """
        rmsd = self.calculate_rmsd(coords_pred, coords_true)
        tm_score = self.calculate_tm_score(coords_pred, coords_true, sequence_length)
        gdt_ts, gdt_ha = self.calculate_gdt(coords_pred, coords_true)
        lddt = self.calculate_lddt(coords_pred, coords_true)
        clash_score = self.calculate_clash_score(coords_pred)
        
        return StructureMetrics(
            rmsd=rmsd,
            tm_score=tm_score,
            gdt_ts=gdt_ts,
            gdt_ha=gdt_ha,
            lddt=lddt,
            clash_score=clash_score
        )
    
    def _align_structures(
        self,
        coords_mobile: np.ndarray,
        coords_target: np.ndarray
    ) -> np.ndarray:
        """Align mobile coordinates to target using Kabsch algorithm.
        
        Args:
            coords_mobile: Coordinates to align
            coords_target: Reference coordinates
            
        Returns:
            Aligned mobile coordinates
        """
        # Center both coordinate sets
        mobile_center = coords_mobile.mean(axis=0)
        target_center = coords_target.mean(axis=0)
        
        coords_mobile_centered = coords_mobile - mobile_center
        coords_target_centered = coords_target - target_center
        
        # Calculate rotation matrix using Kabsch algorithm
        correlation_matrix = coords_mobile_centered.T @ coords_target_centered
        U, _, Vt = np.linalg.svd(correlation_matrix)
        
        # Handle reflection case
        if np.linalg.det(U @ Vt) < 0:
            Vt[-1, :] *= -1
        
        rotation_matrix = U @ Vt
        
        # Apply transformation
        coords_aligned = (coords_mobile_centered @ rotation_matrix.T) + target_center
        
        return coords_aligned


class BenchmarkComparison:
    """Compare QuantumFold predictions with AlphaFold-3 and other baselines."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.evaluator = ProteinStructureEvaluator()
        self.results = []
    
    def compare_predictions(
        self,
        protein_id: str,
        coords_true: np.ndarray,
        coords_quantumfold: np.ndarray,
        coords_alphafold: Optional[np.ndarray] = None,
        coords_baseline: Optional[np.ndarray] = None,
        sequence_length: Optional[int] = None
    ) -> Dict[str, StructureMetrics]:
        """Compare multiple prediction methods.
        
        Args:
            protein_id: Identifier for the protein
            coords_true: Ground truth coordinates
            coords_quantumfold: QuantumFold predictions
            coords_alphafold: AlphaFold-3 predictions (optional)
            coords_baseline: Classical baseline predictions (optional)
            sequence_length: Sequence length for TM-score
            
        Returns:
            Dictionary mapping method names to metrics
        """
        comparison = {}
        
        # Evaluate QuantumFold
        comparison['QuantumFold'] = self.evaluator.evaluate_structure(
            coords_quantumfold, coords_true, sequence_length
        )
        
        # Evaluate AlphaFold-3 if provided
        if coords_alphafold is not None:
            comparison['AlphaFold-3'] = self.evaluator.evaluate_structure(
                coords_alphafold, coords_true, sequence_length
            )
        
        # Evaluate baseline if provided
        if coords_baseline is not None:
            comparison['Baseline'] = self.evaluator.evaluate_structure(
                coords_baseline, coords_true, sequence_length
            )
        
        # Store results
        result_entry = {
            'protein_id': protein_id,
            'timestamp': time.time(),
            'comparison': {k: v.to_dict() for k, v in comparison.items()}
        }
        self.results.append(result_entry)
        
        return comparison
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_{int(time.time())}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary statistics of benchmark results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        methods = list(self.results[0]['comparison'].keys())
        metrics = ['rmsd', 'tm_score', 'gdt_ts', 'gdt_ha', 'lddt']
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Number of proteins evaluated: {len(self.results)}\n")
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 60)
            for method in methods:
                values = [r['comparison'][method][metric] for r in self.results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {method:20s}: {mean_val:8.4f} ± {std_val:6.4f}")
        
        print("\n" + "="*80)
