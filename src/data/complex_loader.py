from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class ChainData:
    chain_id: str
    sequence: str
    coords: Optional[torch.Tensor] = None


@dataclass
class ProteinComplex:
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
