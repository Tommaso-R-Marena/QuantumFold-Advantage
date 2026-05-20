from __future__ import annotations

from typing import Dict, List

import pandas as pd


def compute_rna_metrics(pred_pdb, native_pdb) -> Dict:
    return {
        "rmsd_all_atom": 0.0,
        "rmsd_backbone_p": 0.0,
        "tm_score_rna": 0.0,
        "inf": 0.0,
        "deformation_index": 0.0,
        "secondary_structure_f1": 0.0,
        "clash_score": 0.0,
    }


class RNAPuzzlesBenchmark:
    def __init__(self):
        self.puzzles: List[int] = []

    def load_puzzles(self, puzzles=[1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 20, 21]):
        self.puzzles = list(puzzles)

    def evaluate_predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"puzzle": p, "method": "QuantumFold", "rmsd": 0.0, "baseline": "Rosetta"}
                for p in self.puzzles
            ]
        )
