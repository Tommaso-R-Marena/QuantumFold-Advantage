from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import torch
from Bio.PDB import PDBParser, PPBuilder
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class ChainData:
    chain_id: str
    sequence: str
    coordinates: np.ndarray
    start_residue: int
    coords: Optional[torch.Tensor] = None


@dataclass
class ProteinComplex:
    chains: List[ChainData]
    inter_chain_contacts: Optional[torch.Tensor] = None
    stoichiometry: Optional[Dict[str, int]] = None

    def get_chain_breaks(self) -> List[int]:
        breaks: List[int] = []
        cumsum = 0
        for chain in self.chains:
            cumsum += len(chain.sequence)
            breaks.append(cumsum)
        return breaks[:-1]

    def get_interface_residues(self, distance_cutoff: float = 8.0) -> List[List[int]]:
        if not self.chains:
            return []
        coords = [c.coordinates for c in self.chains if len(c.coordinates) > 0]
        if not coords:
            return [[] for _ in self.chains]
        interfaces = [set() for _ in self.chains]
        for i in range(len(self.chains)):
            for j in range(i + 1, len(self.chains)):
                ci = self.chains[i].coordinates
                cj = self.chains[j].coordinates
                if len(ci) == 0 or len(cj) == 0:
                    continue
                dist = np.linalg.norm(ci[:, None, :] - cj[None, :, :], axis=-1)
                ii, jj = np.where(dist < distance_cutoff)
                interfaces[i].update(ii.tolist())
                interfaces[j].update(jj.tolist())
        return [sorted(list(x)) for x in interfaces]


class ComplexDataset(Dataset):
    def __init__(self, pdb_ids: List[str], cache_dir: str = "./data/complexes"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.complexes = self._load_complexes(pdb_ids)

    def _download_pdb(self, pdb_id: str) -> Path:
        dest = self.cache_dir / f"{pdb_id.upper()}.pdb"
        if dest.exists():
            return dest
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.write_text(r.text)
        return dest

    def _chain_sequence(self, chain) -> str:
        ppb = PPBuilder()
        peptides = ppb.build_peptides(chain)
        if peptides:
            return "".join(str(p.get_sequence()) for p in peptides)
        # fallback map from residue names
        map3 = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
            "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
            "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
        }
        seq = []
        for res in chain:
            if "CA" in res and res.get_resname() in map3:
                seq.append(map3[res.get_resname()])
        return "".join(seq)

    def _load_complexes(self, pdb_ids: List[str]) -> List[ProteinComplex]:
        parser = PDBParser(QUIET=True)
        out: List[ProteinComplex] = []
        for pdb_id in pdb_ids:
            try:
                pdb_path = self._download_pdb(pdb_id)
                structure = parser.get_structure(pdb_id, str(pdb_path))
                model = next(iter(structure))
                chains: List[ChainData] = []
                start = 0
                for chain in model:
                    coords = []
                    for residue in chain:
                        if "CA" in residue:
                            coords.append(residue["CA"].get_coord())
                    coords_arr = np.asarray(coords, dtype=np.float32)
                    seq = self._chain_sequence(chain)
                    if len(seq) == 0 or len(coords_arr) == 0:
                        continue
                    l = min(len(seq), len(coords_arr))
                    seq = seq[:l]
                    coords_arr = coords_arr[:l]
                    chains.append(
                        ChainData(chain_id=chain.id, sequence=seq, coordinates=coords_arr, start_residue=start)
                    )
                    start += l
                if len(chains) < 2:
                    continue
                all_coords = np.concatenate([c.coordinates for c in chains], axis=0)
                d = np.linalg.norm(all_coords[:, None, :] - all_coords[None, :, :], axis=-1)
                contacts = torch.tensor((d < 8.0) & (d > 1e-6), dtype=torch.bool)
                stoich: Dict[str, int] = {}
                for c in chains:
                    stoich[c.chain_id] = stoich.get(c.chain_id, 0) + 1
                out.append(ProteinComplex(chains=chains, inter_chain_contacts=contacts, stoichiometry=stoich))
            except Exception:
                continue
        return out

    def __len__(self):
        return len(self.complexes)

    def __getitem__(self, idx):
        comp = self.complexes[idx]
        sequence = "".join(c.sequence for c in comp.chains)
        chain_breaks = comp.get_chain_breaks()
        native_coords = torch.tensor(np.concatenate([c.coordinates for c in comp.chains], axis=0), dtype=torch.float32)
        contacts = comp.inter_chain_contacts if comp.inter_chain_contacts is not None else torch.zeros((len(sequence), len(sequence)), dtype=torch.bool)
        return sequence, chain_breaks, native_coords, contacts
    """Container for multi-chain complexes."""

    chains: List[ChainData]
    inter_chain_contacts: Optional[torch.Tensor] = None
    stoichiometry: Optional[Dict[str, int]] = None
    biological_assembly: Optional[int] = None

    def get_chain_breaks(self) -> List[int]:
        breaks: List[int] = []
        running = 0
        for chain in self.chains[:-1]:
            running += len(chain.sequence)
            breaks.append(running)
        return breaks

    def get_interface_residues(self, distance_cutoff: float = 8.0) -> List[List[int]]:
        if self.inter_chain_contacts is None:
            return [[] for _ in self.chains]
        chain_breaks = [0] + self.get_chain_breaks() + [self.inter_chain_contacts.shape[0]]
        out: List[List[int]] = []
        for i in range(len(self.chains)):
            start, end = chain_breaks[i], chain_breaks[i + 1]
            contacts = self.inter_chain_contacts[start:end]
            interface_mask = contacts.any(dim=1)
            out.append(torch.where(interface_mask)[0].tolist())
        return out


class ComplexDataset(Dataset):
    """Lightweight complex dataset scaffold."""

    def __init__(
        self,
        root: str = "./data/complexes",
        sources: Sequence[str] = ("RCSB", "PDBbind"),
        max_chains: int = 4,
        min_interface_contacts: int = 10,
    ):
        self.root = Path(root)
        self.sources = list(sources)
        self.max_chains = max_chains
        self.min_interface_contacts = min_interface_contacts
        self.samples: List[ProteinComplex] = []

    def add_complex(self, complex_sample: ProteinComplex) -> None:
        if len(complex_sample.chains) <= self.max_chains:
            self.samples.append(complex_sample)

    def filter_by_type(
        self,
        complex_type: Sequence[str] = (
            "homo-oligomer",
            "hetero-oligomer",
            "antibody-antigen",
            "enzyme-substrate",
        ),
    ) -> List[ProteinComplex]:
        allowed = set(complex_type)
        return [
            c
            for c in self.samples
            if c.stoichiometry and ("homo-oligomer" in allowed or len(c.stoichiometry) > 1)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ProteinComplex:
        return self.samples[idx]
