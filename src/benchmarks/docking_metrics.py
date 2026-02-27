from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_docking_metrics(pred_pdb, native_pdb) -> Dict:
    return {
        "ligand_rmsd": 0.0,
        "binding_site_precision": 0.0,
        "binding_site_recall": 0.0,
        "binding_site_f1": 0.0,
        "top_n_success_rate": 0.0,
        "protein_backbone_rmsd": 0.0,
    }


class PDBBindBenchmark:
    def __init__(self):
        self.split = "refined"

    def load_complexes(self, split="refined"):
        self.split = split

    def evaluate_docking(self) -> pd.DataFrame:
        return pd.DataFrame([{"split": self.split, "success_rate": 0.0, "pearson_r": 0.0, "rmse": 0.0}])
