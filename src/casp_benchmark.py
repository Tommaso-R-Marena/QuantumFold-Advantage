"""CASP (Critical Assessment of protein Structure Prediction) benchmark dataset loader.

Provides automated downloading, parsing, and evaluation on CASP14, CASP15, and CASP16 datasets.

CASP is the gold standard for protein structure prediction benchmarking:
- CASP14 (2020): ~100 targets, AlphaFold2's breakthrough
- CASP15 (2022): ~100 targets, including multimers
- CASP16 (2024): Latest targets

References:
    - CASP: https://predictioncenter.org/
    - Kryshtafovych et al., Proteins (2021) DOI: 10.1002/prot.26237
"""

import os
import urllib.request
import gzip
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
try:
    from Bio import PDB
    from Bio.PDB import PDBParser, MMCIFParser
except ImportError:
    warnings.warn("Biopython not installed. Install with: pip install biopython")
    PDB = None

import torch
from tqdm import tqdm

from .benchmarks import ProteinStructureEvaluator


class CASPDataset:
    """CASP dataset loader and manager.
    
    Handles downloading, parsing, and organizing CASP targets.
    
    Args:
        casp_version: Version to load (14, 15, or 16)
        data_dir: Directory to store downloaded data
        download: Whether to download if not present
    """
    
    # CASP data sources (using PDB and official releases)
    CASP14_TARGETS = [
        'T1024', 'T1026', 'T1027', 'T1028', 'T1029', 'T1030', 'T1031', 'T1032',
        'T1033', 'T1034', 'T1035', 'T1036', 'T1037', 'T1038', 'T1039', 'T1040',
        'T1041', 'T1042', 'T1043', 'T1044', 'T1045', 'T1046', 'T1047', 'T1048',
        'T1049', 'T1050', 'T1051', 'T1052', 'T1053', 'T1054', 'T1055', 'T1056',
        'T1057', 'T1058', 'T1059', 'T1060', 'T1061', 'T1062', 'T1063', 'T1064',
        'T1065', 'T1066', 'T1067', 'T1068', 'T1069', 'T1070', 'T1071', 'T1072',
        'T1073', 'T1074', 'T1075', 'T1076', 'T1077', 'T1078', 'T1079', 'T1080',
        'T1081', 'T1082', 'T1083', 'T1084', 'T1085', 'T1086', 'T1087', 'T1088',
        'T1089', 'T1090', 'T1091', 'T1092', 'T1093', 'T1094', 'T1095', 'T1096',
        'T1097', 'T1098', 'T1099', 'T1100', 'T1101', 'T1102', 'T1103', 'T1104',
        'T1105', 'T1106', 'T1107', 'T1108', 'T1109', 'T1110', 'T1111', 'T1112',
        'T1113', 'T1114', 'T1115', 'T1116', 'T1117', 'T1118', 'T1119', 'T1120',
        'T1121', 'T1122', 'T1123', 'T1124'  # 100+ targets
    ]
    
    # CASP15 targets (2022)
    CASP15_TARGETS = [
        'T1124', 'T1125', 'T1126', 'T1127', 'T1128', 'T1129', 'T1130', 'T1131',
        'T1132', 'T1133', 'T1134', 'T1135', 'T1136', 'T1137', 'T1138', 'T1139',
        'T1140', 'T1141', 'T1142', 'T1143', 'T1144', 'T1145', 'T1146', 'T1147',
        'T1148', 'T1149', 'T1150', 'T1151', 'T1152', 'T1153', 'T1154', 'T1155',
        # Add more as needed
    ]
    
    # CASP16 targets (2024) - subset available
    CASP16_TARGETS = [
        'T1200', 'T1201', 'T1202', 'T1203', 'T1204', 'T1205', 'T1206', 'T1207',
        'T1208', 'T1209', 'T1210', 'T1211', 'T1212', 'T1213', 'T1214', 'T1215',
    ]
    
    def __init__(
        self,
        casp_version: int = 14,
        data_dir: str = 'data/casp',
        download: bool = True
    ):
        if casp_version not in [14, 15, 16]:
            raise ValueError(f"CASP version must be 14, 15, or 16, got {casp_version}")
        
        self.casp_version = casp_version
        self.data_dir = Path(data_dir) / f'casp{casp_version}'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Select targets
        if casp_version == 14:
            self.targets = self.CASP14_TARGETS
        elif casp_version == 15:
            self.targets = self.CASP15_TARGETS
        else:
            self.targets = self.CASP16_TARGETS
        
        self.structures = {}
        self.sequences = {}
        
        if download:
            self.download_targets()
        
        self.load_targets()
    
    def download_targets(self):
        """Download CASP target structures from PDB.
        
        Note: This downloads native structures from PDB.
        For official CASP targets, visit: https://predictioncenter.org/
        """
        print(f"Downloading CASP{self.casp_version} targets...")
        
        # Create mapping file if it doesn't exist
        mapping_file = self.data_dir / 'target_pdb_mapping.txt'
        
        if not mapping_file.exists():
            # Common CASP14 target to PDB mappings (subset)
            casp14_mapping = {
                'T1024': '6W2N', 'T1026': '6VYN', 'T1027': '6W41',
                'T1028': '6W70', 'T1029': '6WQ2', 'T1030': '6XC4',
                'T1031': '6XQU', 'T1032': '6Y1X', 'T1033': '6Y2F',
                'T1034': '6YGO', 'T1035': '6ZJ1', 'T1036': '6ZS5',
                'T1037': '7JLL', 'T1038': '7JTL', 'T1040': '7BQI',
                'T1041': '7CAN', 'T1042': '7JFI', 'T1043': '7JHQ',
                'T1044': '7JTQ', 'T1045': '7K0G', 'T1046': '6X1Q',
                'T1047': '6X3I', 'T1048': '6XDP', 'T1049': '6XG2',
                'T1050': '6Y5E', 'T1064': '6Y9M', 'T1065': '6YAF',
                'T1078': '7JWV', 'T1080': '7K01', 'T1083': '6XGB',
                'T1084': '6Y3E', 'T1086': '6YS0', 'T1087': '7BWC',
                'T1088': '7JLN', 'T1089': '7JTM', 'T1091': '7JU3',
            }
            
            # Save mapping
            with open(mapping_file, 'w') as f:
                for target, pdb in casp14_mapping.items():
                    f.write(f"{target}\t{pdb}\n")
            
            print(f"Created PDB mapping file: {mapping_file}")
            print("Note: Add more mappings to this file for additional targets")
        
        # Load mapping
        target_to_pdb = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.strip():
                    target, pdb = line.strip().split('\t')
                    target_to_pdb[target] = pdb
        
        # Download PDB files
        downloaded = 0
        for target in tqdm(self.targets, desc=f"Downloading CASP{self.casp_version}"):
            if target not in target_to_pdb:
                continue
            
            pdb_id = target_to_pdb[target]
            pdb_file = self.data_dir / f"{target}_{pdb_id}.pdb"
            
            if pdb_file.exists():
                downloaded += 1
                continue
            
            try:
                # Download from PDB
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                urllib.request.urlretrieve(url, pdb_file)
                downloaded += 1
            except Exception as e:
                print(f"  Failed to download {target} ({pdb_id}): {e}")
        
        print(f"Downloaded {downloaded}/{len(self.targets)} structures")
        print(f"\nTo add more targets:")
        print(f"  1. Find PDB codes at https://predictioncenter.org/")
        print(f"  2. Add to {mapping_file}")
        print(f"  3. Run again to download")
    
    def load_targets(self):
        """Load downloaded target structures."""
        if PDB is None:
            print("Biopython not installed. Cannot load structures.")
            return
        
        parser = PDBParser(QUIET=True)
        
        pdb_files = list(self.data_dir.glob('*.pdb'))
        print(f"Loading {len(pdb_files)} structures...")
        
        for pdb_file in pdb_files:
            target_id = pdb_file.stem.split('_')[0]  # Extract T#### from filename
            
            try:
                structure = parser.get_structure(target_id, pdb_file)
                
                # Extract CA coordinates
                coords = []
                sequence = []
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.has_id('CA'):
                                ca_atom = residue['CA']
                                coords.append(ca_atom.coord)
                                # Get residue name
                                try:
                                    from Bio.PDB.Polypeptide import three_to_one
                                    aa = three_to_one(residue.resname)
                                    sequence.append(aa)
                                except:
                                    sequence.append('X')
                        break  # Only first chain
                    break  # Only first model
                
                if coords:
                    self.structures[target_id] = np.array(coords)
                    self.sequences[target_id] = ''.join(sequence)
            
            except Exception as e:
                print(f"  Failed to parse {pdb_file}: {e}")
        
        print(f"Successfully loaded {len(self.structures)} structures")
    
    def get_target(self, target_id: str) -> Tuple[str, np.ndarray]:
        """Get sequence and coordinates for a target.
        
        Args:
            target_id: CASP target ID (e.g., 'T1024')
        
        Returns:
            (sequence, coordinates) tuple
        """
        if target_id not in self.structures:
            raise ValueError(f"Target {target_id} not found. Available: {list(self.structures.keys())[:10]}...")
        
        return self.sequences[target_id], self.structures[target_id]
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __iter__(self):
        for target_id in self.structures:
            yield target_id, self.sequences[target_id], self.structures[target_id]


class CASPBenchmark:
    """Comprehensive CASP benchmarking framework.
    
    Args:
        model: PyTorch model to evaluate
        embedder: Protein embedder (e.g., ESM2Embedder)
        device: Device for computation
    """
    
    def __init__(self, model, embedder, device='cuda'):
        self.model = model
        self.embedder = embedder
        self.device = device
        self.evaluator = ProteinStructureEvaluator()
        
        self.model.to(device)
        self.model.eval()
    
    def benchmark_casp(self, casp_version: int = 14, max_targets: Optional[int] = None) -> Dict:
        """Run benchmark on CASP dataset.
        
        Args:
            casp_version: CASP version (14, 15, or 16)
            max_targets: Maximum number of targets to evaluate (for quick testing)
        
        Returns:
            Dictionary with comprehensive results
        """
        print(f"\n{'='*80}")
        print(f"CASP{casp_version} Benchmark")
        print(f"{'='*80}\n")
        
        # Load dataset
        dataset = CASPDataset(casp_version=casp_version)
        
        if len(dataset) == 0:
            print("No targets loaded. Please check dataset.")
            return {}
        
        results = {
            'tm_scores': [],
            'rmsd_scores': [],
            'gdt_ts_scores': [],
            'per_target': {}
        }
        
        targets_evaluated = 0
        
        # Evaluate each target
        for target_id, sequence, true_coords in tqdm(dataset, desc=f"CASP{casp_version}"):
            if max_targets and targets_evaluated >= max_targets:
                break
            
            try:
                # Generate embeddings
                with torch.no_grad():
                    emb_dict = self.embedder([sequence])
                    embeddings = emb_dict['embeddings'].to(self.device)
                    
                    # Predict structure
                    outputs = self.model(embeddings)
                    pred_coords = outputs['coordinates'][0].cpu().numpy()
                    
                    # Truncate to same length (in case of length mismatch)
                    min_len = min(len(pred_coords), len(true_coords))
                    pred_coords = pred_coords[:min_len]
                    true_coords = true_coords[:min_len]
                    
                    # Calculate metrics
                    tm_score = self.evaluator.calculate_tm_score(pred_coords, true_coords)
                    rmsd = self.evaluator.calculate_rmsd(pred_coords, true_coords)
                    gdt_ts = self.evaluator.calculate_gdt_ts(pred_coords, true_coords)
                    
                    # Store results
                    results['tm_scores'].append(tm_score)
                    results['rmsd_scores'].append(rmsd)
                    results['gdt_ts_scores'].append(gdt_ts)
                    
                    results['per_target'][target_id] = {
                        'tm_score': float(tm_score),
                        'rmsd': float(rmsd),
                        'gdt_ts': float(gdt_ts),
                        'length': min_len
                    }
                    
                    targets_evaluated += 1
            
            except Exception as e:
                print(f"\nError evaluating {target_id}: {e}")
                continue
        
        # Compute statistics
        if results['tm_scores']:
            results['summary'] = {
                'n_targets': targets_evaluated,
                'tm_score_mean': float(np.mean(results['tm_scores'])),
                'tm_score_std': float(np.std(results['tm_scores'])),
                'tm_score_median': float(np.median(results['tm_scores'])),
                'rmsd_mean': float(np.mean(results['rmsd_scores'])),
                'rmsd_std': float(np.std(results['rmsd_scores'])),
                'gdt_ts_mean': float(np.mean(results['gdt_ts_scores'])),
                'gdt_ts_std': float(np.std(results['gdt_ts_scores'])),
            }
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"CASP{casp_version} Results Summary")
            print(f"{'='*80}")
            print(f"Targets evaluated: {targets_evaluated}")
            print(f"\nTM-score: {results['summary']['tm_score_mean']:.4f} ± {results['summary']['tm_score_std']:.4f}")
            print(f"RMSD:     {results['summary']['rmsd_mean']:.2f} ± {results['summary']['rmsd_std']:.2f} Å")
            print(f"GDT-TS:   {results['summary']['gdt_ts_mean']:.4f} ± {results['summary']['gdt_ts_std']:.4f}")
            print(f"{'='*80}\n")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save benchmark results to JSON."""
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
