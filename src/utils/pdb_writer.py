from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from Bio.PDB import PDBParser

AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def save_pdb(
    coords: np.ndarray,
    sequence: str,
    filename: str,
    chain_breaks: Optional[List[int]] = None,
) -> Path:
    """Write C-alpha-only PDB file."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape (N,3)")
    if len(sequence) < len(arr):
        raise ValueError("sequence length must be >= number of coordinates")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    breaks = set(chain_breaks or [])
    chain_idx = 0
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    lines = ["HEADER    QUANTUMFOLD PREDICTION\n"]
    atom_idx = 1
    res_id = 1
    for i, (x, y, z) in enumerate(arr):
        if i in breaks:
            lines.append("TER\n")
            chain_idx += 1
            res_id = 1
        aa = sequence[i]
        resname = AA3.get(aa, "GLY")
        chain = chain_ids[min(chain_idx, len(chain_ids) - 1)]
        lines.append(
            f"ATOM  {atom_idx:5d}  CA  {resname:>3s} {chain}{res_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
        )
        atom_idx += 1
        res_id += 1
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


def load_pdb_coords(pdb_path: Path) -> torch.Tensor:
    """Load CA coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())

    return torch.tensor(coords, dtype=torch.float32)
