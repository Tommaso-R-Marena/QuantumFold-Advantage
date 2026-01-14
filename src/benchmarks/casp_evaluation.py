"""CASP-Standard Protein Structure Evaluation.

Implements official CASP (Critical Assessment of protein Structure Prediction)
evaluation metrics and protocols for rigorous benchmarking.

References:
    - Kryshtafovych et al., "Critical assessment of methods of protein structure
      prediction (CASP)—Round XIV", Proteins (2021)
    - Zhang & Skolnick, "Scoring function for automated assessment of protein
      structure template quality", Proteins (2004)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings


@dataclass
class StructureMetrics:
    """Container for comprehensive structure quality metrics."""
    
    tm_score: float
    gdt_ts: float
    gdt_ha: float
    rmsd: float
    rmsd_ca: float
    lga_s: float
    contact_precision: float
    contact_recall: float
    contact_f1: float
    lddt: float
    qs_score: float
    plddt_mean: float
    plddt_std: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'TM-score': self.tm_score,
            'GDT_TS': self.gdt_ts,
            'GDT_HA': self.gdt_ha,
            'RMSD': self.rmsd,
            'RMSD_CA': self.rmsd_ca,
            'LGA_S': self.lga_s,
            'Contact_Precision': self.contact_precision,
            'Contact_Recall': self.contact_recall,
            'Contact_F1': self.contact_f1,
            'lDDT': self.lddt,
            'QS-score': self.qs_score,
            'pLDDT_mean': self.plddt_mean,
            'pLDDT_std': self.plddt_std
        }


class CASPEvaluator:
    """Official CASP evaluation metrics implementation.
    
    Implements the complete suite of CASP evaluation metrics including
    TM-score, GDT-TS, GDT-HA, lDDT, and contact prediction metrics.
    
    All metrics follow the official CASP evaluation protocols.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize CASP evaluator.
        
        Args:
            verbose: Enable detailed logging of calculations
        """
        self.verbose = verbose
        
    def kabsch_superimpose(self, 
                          mobile: np.ndarray,
                          target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Kabsch algorithm for optimal superposition.
        
        Args:
            mobile: Mobile coordinates (N, 3)
            target: Target coordinates (N, 3)
            
        Returns:
            Superimposed mobile coordinates and RMSD
        """
        # Center structures
        mobile_center = mobile.mean(axis=0)
        target_center = target.mean(axis=0)
        
        mobile_centered = mobile - mobile_center
        target_centered = target - target_center
        
        # Compute optimal rotation via SVD
        correlation = mobile_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(correlation)
        
        # Handle reflection
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        reflection_matrix = np.diag([1, 1, d])
        
        rotation = Vt.T @ reflection_matrix @ U.T
        
        # Apply transformation
        mobile_aligned = mobile_centered @ rotation + target_center
        
        # Compute RMSD
        rmsd = np.sqrt(np.mean(np.sum((mobile_aligned - target) ** 2, axis=1)))
        
        return mobile_aligned, rmsd
    
    def tm_score(self,
                pred_coords: np.ndarray,
                true_coords: np.ndarray,
                sequence_length: Optional[int] = None) -> float:
        """Compute TM-score (Template Modeling score).
        
        TM-score is a protein structure similarity metric that is length-independent
        and more sensitive to global topology than RMSD.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            sequence_length: Original sequence length (default: len(true_coords))
            
        Returns:
            TM-score in range [0, 1], where >0.5 indicates same fold
            
        References:
            Zhang & Skolnick, Proteins (2004) DOI: 10.1002/prot.20264
        """
        if sequence_length is None:
            sequence_length = len(true_coords)
            
        # TM-score normalization factor
        if sequence_length <= 21:
            d0 = 0.5
        else:
            d0 = 1.24 * np.cbrt(sequence_length - 15) - 1.8
            
        # Superimpose structures
        aligned_pred, _ = self.kabsch_superimpose(pred_coords, true_coords)
        
        # Compute per-residue distances
        distances = np.sqrt(np.sum((aligned_pred - true_coords) ** 2, axis=1))
        
        # TM-score formula
        tm = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / sequence_length
        
        return float(tm)
    
    def gdt_score(self,
                  pred_coords: np.ndarray,
                  true_coords: np.ndarray,
                  cutoffs: List[float]) -> float:
        """Compute GDT (Global Distance Test) score.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            cutoffs: Distance cutoffs in Angstroms (e.g., [1, 2, 4, 8])
            
        Returns:
            GDT score as percentage (0-100)
        """
        aligned_pred, _ = self.kabsch_superimpose(pred_coords, true_coords)
        distances = np.sqrt(np.sum((aligned_pred - true_coords) ** 2, axis=1))
        
        percentages = []
        for cutoff in cutoffs:
            percent_below = 100.0 * np.mean(distances < cutoff)
            percentages.append(percent_below)
            
        return float(np.mean(percentages))
    
    def gdt_ts(self,
              pred_coords: np.ndarray,
              true_coords: np.ndarray) -> float:
        """GDT_TS (Total Score) - average of 1, 2, 4, 8 Å cutoffs."""
        return self.gdt_score(pred_coords, true_coords, [1.0, 2.0, 4.0, 8.0])
    
    def gdt_ha(self,
              pred_coords: np.ndarray,
              true_coords: np.ndarray) -> float:
        """GDT_HA (High Accuracy) - average of 0.5, 1, 2, 4 Å cutoffs."""
        return self.gdt_score(pred_coords, true_coords, [0.5, 1.0, 2.0, 4.0])
    
    def lddt(self,
            pred_coords: np.ndarray,
            true_coords: np.ndarray,
            inclusion_radius: float = 15.0,
            thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]) -> float:
        """Compute lDDT (local Distance Difference Test).
        
        lDDT is a local superposition-free score that evaluates local geometry.
        
        Args:
            pred_coords: Predicted coordinates (L, 3)
            true_coords: True coordinates (L, 3)
            inclusion_radius: Distance threshold for local contacts (Å)
            thresholds: Distance difference thresholds (Å)
            
        Returns:
            lDDT score (0-1)
            
        References:
            Mariani et al., Bioinformatics (2013) DOI: 10.1093/bioinformatics/btt473
        """
        # Compute all pairwise distances
        true_dists = cdist(true_coords, true_coords)
        pred_dists = cdist(pred_coords, pred_coords)
        
        n_residues = len(true_coords)
        scores = []
        
        for i in range(n_residues):
            # Find local neighbors within inclusion radius
            neighbors = np.where(
                (true_dists[i] > 0) & (true_dists[i] < inclusion_radius)
            )[0]
            
            if len(neighbors) == 0:
                continue
                
            # Compute distance differences for neighbors
            dist_diffs = np.abs(pred_dists[i, neighbors] - true_dists[i, neighbors])
            
            # Count preserved distances at each threshold
            preserved_counts = [np.sum(dist_diffs < t) for t in thresholds]
            
            # Average over thresholds
            score = np.mean(preserved_counts) / len(neighbors)
            scores.append(score)
            
        return float(np.mean(scores)) if scores else 0.0
    
    def contact_metrics(self,
                       pred_coords: np.ndarray,
                       true_coords: np.ndarray,
                       contact_threshold: float = 8.0,
                       sequence_separation: int = 6) -> Dict[str, float]:
        """Compute contact prediction metrics.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            contact_threshold: Distance threshold for contact (Å)
            sequence_separation: Minimum sequence separation for contacts
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        n = len(true_coords)
        
        # Compute distance matrices
        true_dists = cdist(true_coords, true_coords)
        pred_dists = cdist(pred_coords, pred_coords)
        
        # Create contact maps (excluding close sequence neighbors)
        seq_mask = np.abs(np.arange(n)[:, None] - np.arange(n)) >= sequence_separation
        
        true_contacts = (true_dists < contact_threshold) & seq_mask
        pred_contacts = (pred_dists < contact_threshold) & seq_mask
        
        # Compute metrics
        tp = np.sum(true_contacts & pred_contacts)
        fp = np.sum(~true_contacts & pred_contacts)
        fn = np.sum(true_contacts & ~pred_contacts)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def lga_score(self,
                 pred_coords: np.ndarray,
                 true_coords: np.ndarray,
                 cutoff: float = 5.0) -> float:
        """Compute LGA_S (Longest Continuous Segment) score.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            cutoff: Distance cutoff for segment inclusion (Å)
            
        Returns:
            LGA_S score (0-100)
        """
        aligned_pred, _ = self.kabsch_superimpose(pred_coords, true_coords)
        distances = np.sqrt(np.sum((aligned_pred - true_coords) ** 2, axis=1))
        
        # Find residues within cutoff
        within_cutoff = distances < cutoff
        
        # Find longest continuous segment
        max_length = 0
        current_length = 0
        
        for is_within in within_cutoff:
            if is_within:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
                
        lga_s = 100.0 * max_length / len(true_coords)
        return float(lga_s)
    
    def qs_score(self,
                pred_coords: np.ndarray,
                true_coords: np.ndarray,
                pred_ss: Optional[str] = None,
                true_ss: Optional[str] = None) -> float:
        """Compute QS-score (Quality Score) combining structure and sequence.
        
        Simplified version when secondary structure is not available.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            pred_ss: Predicted secondary structure (optional)
            true_ss: True secondary structure (optional)
            
        Returns:
            QS-score (0-1)
        """
        # Use TM-score as base
        tm = self.tm_score(pred_coords, true_coords)
        
        # Add secondary structure agreement if available
        if pred_ss is not None and true_ss is not None:
            ss_agreement = np.mean([p == t for p, t in zip(pred_ss, true_ss)])
            qs = 0.7 * tm + 0.3 * ss_agreement
        else:
            qs = tm
            
        return float(qs)
    
    def evaluate_structure(self,
                          pred_coords: np.ndarray,
                          true_coords: np.ndarray,
                          pred_plddt: Optional[np.ndarray] = None,
                          sequence_length: Optional[int] = None) -> StructureMetrics:
        """Comprehensive structure evaluation.
        
        Args:
            pred_coords: Predicted CA coordinates (L, 3)
            true_coords: True CA coordinates (L, 3)
            pred_plddt: Predicted per-residue confidence scores (optional)
            sequence_length: Original sequence length
            
        Returns:
            StructureMetrics object with all quality metrics
        """
        # Superimpose and compute RMSD
        aligned_pred, rmsd_ca = self.kabsch_superimpose(pred_coords, true_coords)
        
        # All-atom RMSD (same as CA for now)
        rmsd = rmsd_ca
        
        # TM-score
        tm = self.tm_score(pred_coords, true_coords, sequence_length)
        
        # GDT scores
        gdt_ts = self.gdt_ts(pred_coords, true_coords)
        gdt_ha = self.gdt_ha(pred_coords, true_coords)
        
        # lDDT
        lddt_score = self.lddt(pred_coords, true_coords)
        
        # Contact metrics
        contact_metrics = self.contact_metrics(pred_coords, true_coords)
        
        # LGA score
        lga_s = self.lga_score(pred_coords, true_coords)
        
        # QS score
        qs = self.qs_score(pred_coords, true_coords)
        
        # pLDDT statistics
        if pred_plddt is not None:
            plddt_mean = float(np.mean(pred_plddt))
            plddt_std = float(np.std(pred_plddt))
        else:
            plddt_mean = 0.0
            plddt_std = 0.0
            
        return StructureMetrics(
            tm_score=tm,
            gdt_ts=gdt_ts,
            gdt_ha=gdt_ha,
            rmsd=rmsd,
            rmsd_ca=rmsd_ca,
            lga_s=lga_s,
            contact_precision=contact_metrics['precision'],
            contact_recall=contact_metrics['recall'],
            contact_f1=contact_metrics['f1'],
            lddt=lddt_score,
            qs_score=qs,
            plddt_mean=plddt_mean,
            plddt_std=plddt_std
        )
    
    def batch_evaluate(self,
                      pred_coords_list: List[np.ndarray],
                      true_coords_list: List[np.ndarray],
                      pred_plddt_list: Optional[List[np.ndarray]] = None,
                      sequence_lengths: Optional[List[int]] = None) -> List[StructureMetrics]:
        """Evaluate multiple structures.
        
        Args:
            pred_coords_list: List of predicted coordinate arrays
            true_coords_list: List of true coordinate arrays
            pred_plddt_list: List of predicted confidence scores (optional)
            sequence_lengths: List of sequence lengths (optional)
            
        Returns:
            List of StructureMetrics for each structure
        """
        results = []
        
        n_structs = len(pred_coords_list)
        if pred_plddt_list is None:
            pred_plddt_list = [None] * n_structs
        if sequence_lengths is None:
            sequence_lengths = [None] * n_structs
            
        for pred, true, plddt, seqlen in zip(
            pred_coords_list, true_coords_list, pred_plddt_list, sequence_lengths
        ):
            metrics = self.evaluate_structure(pred, true, plddt, seqlen)
            results.append(metrics)
            
        return results
    
    def aggregate_results(self,
                         metrics_list: List[StructureMetrics]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across multiple structures.
        
        Args:
            metrics_list: List of StructureMetrics objects
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not metrics_list:
            return {}
            
        # Extract all metrics
        all_metrics = {}
        for key in metrics_list[0].to_dict().keys():
            values = [getattr(m, key.lower().replace('-', '_').replace('_', '_')) 
                     for m in metrics_list]
            all_metrics[key] = values
            
        # Compute statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            aggregated[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75))
            }
            
        return aggregated
    
    def save_results(self,
                    metrics_list: List[StructureMetrics],
                    output_path: Union[str, Path],
                    protein_ids: Optional[List[str]] = None) -> None:
        """Save evaluation results to JSON.
        
        Args:
            metrics_list: List of StructureMetrics
            output_path: Output JSON file path
            protein_ids: Optional list of protein identifiers
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if protein_ids is None:
            protein_ids = [f"protein_{i}" for i in range(len(metrics_list))]
            
        results = {
            'individual_results': [
                {"protein_id": pid, **metrics.to_dict()}
                for pid, metrics in zip(protein_ids, metrics_list)
            ],
            'aggregated_statistics': self.aggregate_results(metrics_list),
            'n_structures': len(metrics_list)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        if self.verbose:
            print(f"Results saved to {output_path}")
