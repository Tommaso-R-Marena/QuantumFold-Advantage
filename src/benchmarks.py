"""Benchmarking suite for comparing against AlphaFold-3 and baseline methods.

Implements comprehensive evaluation metrics and comparison protocols.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    tm_score: float
    rmsd: float
    gdt_ts: float
    gdt_ha: float
    lddt: float
    inference_time: float
    memory_usage: float
    protein_id: str
    sequence_length: int


class StructureMetrics:
    """Compute protein structure quality metrics."""
    
    @staticmethod
    def rmsd(pred_coords: np.ndarray, 
             true_coords: np.ndarray,
             align: bool = True) -> float:
        """Calculate Root Mean Square Deviation.
        
        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]
            align: Whether to align structures first
            
        Returns:
            RMSD value in Angstroms
        """
        assert pred_coords.shape == true_coords.shape
        
        if align:
            pred_coords = StructureMetrics._align_structures(
                pred_coords, true_coords
            )
        
        diff = pred_coords - true_coords
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        
        return float(rmsd)
    
    @staticmethod
    def _align_structures(mobile: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """Align structures using Kabsch algorithm."""
        # Center structures
        mobile_center = mobile.mean(axis=0)
        target_center = target.mean(axis=0)
        
        mobile_centered = mobile - mobile_center
        target_centered = target - target_center
        
        # Compute rotation matrix
        H = mobile_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        
        # Ensure right-handed coordinate system
        d = np.linalg.det(Vt.T @ U.T)
        if d < 0:
            Vt[-1, :] *= -1
        
        R = Vt.T @ U.T
        
        # Apply transformation
        mobile_aligned = mobile_centered @ R + target_center
        
        return mobile_aligned
    
    @staticmethod
    def tm_score(pred_coords: np.ndarray,
                 true_coords: np.ndarray,
                 sequence_length: Optional[int] = None) -> float:
        """Calculate TM-score (Template Modeling score).
        
        TM-score is a more robust metric than RMSD, normalized by protein length.
        Range: [0, 1], where >0.5 indicates similar fold, >0.6 indicates same fold.
        
        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]
            sequence_length: Length for normalization (default: N)
            
        Returns:
            TM-score value
        """
        if sequence_length is None:
            sequence_length = len(pred_coords)
        
        # Align structures
        pred_aligned = StructureMetrics._align_structures(pred_coords, true_coords)
        
        # Calculate distances
        distances = np.sqrt(np.sum((pred_aligned - true_coords) ** 2, axis=1))
        
        # TM-score normalization factor
        d0 = 1.24 * (sequence_length - 15) ** (1/3) - 1.8
        
        # Calculate TM-score
        tm = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
        
        return float(tm)
    
    @staticmethod
    def gdt_ts(pred_coords: np.ndarray,
               true_coords: np.ndarray) -> float:
        """Calculate GDT_TS (Global Distance Test - Total Score).
        
        Average of percentages of residues within distance thresholds.
        Thresholds: 1Å, 2Å, 4Å, 8Å
        
        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]
            
        Returns:
            GDT_TS score [0, 100]
        """
        pred_aligned = StructureMetrics._align_structures(pred_coords, true_coords)
        distances = np.sqrt(np.sum((pred_aligned - true_coords) ** 2, axis=1))
        
        thresholds = [1.0, 2.0, 4.0, 8.0]
        percentages = []
        
        for threshold in thresholds:
            pct = 100.0 * np.mean(distances < threshold)
            percentages.append(pct)
        
        gdt_ts = np.mean(percentages)
        return float(gdt_ts)
    
    @staticmethod
    def gdt_ha(pred_coords: np.ndarray,
               true_coords: np.ndarray) -> float:
        """Calculate GDT_HA (Global Distance Test - High Accuracy).
        
        Similar to GDT_TS but with stricter thresholds.
        Thresholds: 0.5Å, 1Å, 2Å, 4Å
        """
        pred_aligned = StructureMetrics._align_structures(pred_coords, true_coords)
        distances = np.sqrt(np.sum((pred_aligned - true_coords) ** 2, axis=1))
        
        thresholds = [0.5, 1.0, 2.0, 4.0]
        percentages = []
        
        for threshold in thresholds:
            pct = 100.0 * np.mean(distances < threshold)
            percentages.append(pct)
        
        gdt_ha = np.mean(percentages)
        return float(gdt_ha)
    
    @staticmethod
    def lddt(pred_coords: np.ndarray,
             true_coords: np.ndarray,
             cutoff: float = 15.0) -> float:
        """Calculate lDDT (local Distance Difference Test).
        
        Measures local structure quality by comparing distances
        between nearby residues.
        
        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]
            cutoff: Distance cutoff for considering residue pairs
            
        Returns:
            lDDT score [0, 1]
        """
        n_residues = len(pred_coords)
        
        # Calculate distance matrices
        pred_dists = np.linalg.norm(
            pred_coords[:, None, :] - pred_coords[None, :, :], axis=2
        )
        true_dists = np.linalg.norm(
            true_coords[:, None, :] - true_coords[None, :, :], axis=2
        )
        
        # Consider only pairs within cutoff in true structure
        mask = (true_dists < cutoff) & (true_dists > 0)
        
        if not np.any(mask):
            return 0.0
        
        # Calculate distance differences
        dist_diffs = np.abs(pred_dists[mask] - true_dists[mask])
        
        # Count preserved distances at different thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        preserved = [np.mean(dist_diffs < t) for t in thresholds]
        
        lddt_score = np.mean(preserved)
        return float(lddt_score)


class ProteinBenchmark:
    """Main benchmarking suite for protein structure prediction."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = StructureMetrics()
        self.results: List[BenchmarkResult] = []
    
    def evaluate_model(self,
                      model: torch.nn.Module,
                      test_data: List[Tuple[torch.Tensor, np.ndarray, str]],
                      model_name: str = "QuantumFold",
                      device: str = "cuda") -> List[BenchmarkResult]:
        """Evaluate model on test dataset.
        
        Args:
            model: PyTorch model to evaluate
            test_data: List of (features, true_coords, protein_id)
            model_name: Name for results
            device: Device for inference
            
        Returns:
            List of benchmark results
        """
        model.eval()
        model.to(device)
        results = []
        
        with torch.no_grad():
            for features, true_coords, protein_id in test_data:
                # Measure inference time and memory
                start_time = time.time()
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Inference
                features = features.unsqueeze(0).to(device)
                pred_coords = model(features)
                pred_coords = pred_coords.squeeze(0).cpu().numpy()
                
                inference_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.max_memory_allocated() / 1e9
                else:
                    memory_usage = 0.0
                
                # Calculate metrics
                seq_len = len(true_coords)
                
                result = BenchmarkResult(
                    model_name=model_name,
                    tm_score=self.metrics.tm_score(pred_coords, true_coords),
                    rmsd=self.metrics.rmsd(pred_coords, true_coords),
                    gdt_ts=self.metrics.gdt_ts(pred_coords, true_coords),
                    gdt_ha=self.metrics.gdt_ha(pred_coords, true_coords),
                    lddt=self.metrics.lddt(pred_coords, true_coords),
                    inference_time=inference_time,
                    memory_usage=memory_usage,
                    protein_id=protein_id,
                    sequence_length=seq_len
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def compare_with_alphafold3(self,
                               alphafold_predictions: Dict[str, np.ndarray],
                               ground_truth: Dict[str, np.ndarray]) -> Dict:
        """Compare results with AlphaFold-3 predictions.
        
        Args:
            alphafold_predictions: Dict mapping protein_id to AF3 predictions
            ground_truth: Dict mapping protein_id to true structures
            
        Returns:
            Comparison statistics
        """
        comparison = {
            'quantumfold': {'tm_scores': [], 'rmsds': []},
            'alphafold3': {'tm_scores': [], 'rmsds': []}
        }
        
        for result in self.results:
            pid = result.protein_id
            
            if pid in alphafold_predictions and pid in ground_truth:
                # QuantumFold scores
                comparison['quantumfold']['tm_scores'].append(result.tm_score)
                comparison['quantumfold']['rmsds'].append(result.rmsd)
                
                # AlphaFold-3 scores
                af3_pred = alphafold_predictions[pid]
                true_coords = ground_truth[pid]
                
                af3_tm = self.metrics.tm_score(af3_pred, true_coords)
                af3_rmsd = self.metrics.rmsd(af3_pred, true_coords)
                
                comparison['alphafold3']['tm_scores'].append(af3_tm)
                comparison['alphafold3']['rmsds'].append(af3_rmsd)
        
        # Calculate statistics
        stats = {
            'quantumfold_mean_tm': np.mean(comparison['quantumfold']['tm_scores']),
            'alphafold3_mean_tm': np.mean(comparison['alphafold3']['tm_scores']),
            'quantumfold_mean_rmsd': np.mean(comparison['quantumfold']['rmsds']),
            'alphafold3_mean_rmsd': np.mean(comparison['alphafold3']['rmsds']),
            'tm_improvement': np.mean(comparison['quantumfold']['tm_scores']) - 
                            np.mean(comparison['alphafold3']['tm_scores']),
            'n_proteins': len(comparison['quantumfold']['tm_scores'])
        }
        
        return stats
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_file = self.output_dir / filename
        
        results_dict = [
            {
                'model_name': r.model_name,
                'protein_id': r.protein_id,
                'sequence_length': r.sequence_length,
                'tm_score': r.tm_score,
                'rmsd': r.rmsd,
                'gdt_ts': r.gdt_ts,
                'gdt_ha': r.gdt_ha,
                'lddt': r.lddt,
                'inference_time': r.inference_time,
                'memory_usage': r.memory_usage
            }
            for r in self.results
        ]
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        if not self.results:
            return "No results to report."
        
        report = ["=" * 60]
        report.append("QUANTUMFOLD BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        tm_scores = [r.tm_score for r in self.results]
        rmsds = [r.rmsd for r in self.results]
        gdt_ts_scores = [r.gdt_ts for r in self.results]
        
        report.append(f"Total proteins evaluated: {len(self.results)}")
        report.append("")
        report.append("Overall Metrics:")
        report.append(f"  TM-score:  {np.mean(tm_scores):.4f} ± {np.std(tm_scores):.4f}")
        report.append(f"  RMSD:      {np.mean(rmsds):.4f} ± {np.std(rmsds):.4f} Å")
        report.append(f"  GDT_TS:    {np.mean(gdt_ts_scores):.2f} ± {np.std(gdt_ts_scores):.2f}")
        report.append("")
        
        # Performance statistics
        times = [r.inference_time for r in self.results]
        memories = [r.memory_usage for r in self.results]
        
        report.append("Performance:")
        report.append(f"  Inference time: {np.mean(times):.3f} ± {np.std(times):.3f} s")
        report.append(f"  Memory usage:   {np.mean(memories):.2f} ± {np.std(memories):.2f} GB")
        report.append("")
        
        # Best and worst predictions
        best_idx = np.argmax(tm_scores)
        worst_idx = np.argmin(tm_scores)
        
        report.append("Best prediction:")
        best = self.results[best_idx]
        report.append(f"  Protein: {best.protein_id}")
        report.append(f"  TM-score: {best.tm_score:.4f}")
        report.append("")
        
        report.append("Worst prediction:")
        worst = self.results[worst_idx]
        report.append(f"  Protein: {worst.protein_id}")
        report.append(f"  TM-score: {worst.tm_score:.4f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
