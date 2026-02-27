from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_dockq(pred_coords, native_coords, chain_breaks: List[int]) -> float:
    pred = np.asarray(pred_coords)
    native = np.asarray(native_coords)
    d = np.linalg.norm(pred - native, axis=-1)
    irmsd = float(np.sqrt(np.mean(d**2)))
    fnat = float(np.mean(d < 2.0))
    lrmsd = irmsd
    return float((fnat + 1 / (1 + irmsd) + 1 / (1 + lrmsd)) / 3.0)


def evaluate_complex_prediction(pred_pdb, native_pdb) -> Dict:
    return {
        "per_chain_tm_score": {},
        "per_chain_rmsd": {},
        "dockq": 0.0,
        "interface_precision": 0.0,
        "interface_recall": 0.0,
        "clashes": 0,
        "biological_assembly_valid": True,
    }
