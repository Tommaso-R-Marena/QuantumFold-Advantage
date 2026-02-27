from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RNATokenEmbedder(nn.Module):
    """Embed RNA sequences with structural context."""

    def __init__(self, use_pretrained: bool = True, model_name: str = "rna-fm", d_model: int = 384):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.model_name = model_name
        self.vocab = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
        self.embedding = nn.Embedding(len(self.vocab), d_model)

    def forward(self, sequence: str) -> torch.Tensor:
        ids = torch.tensor([self.vocab.get(x, 4) for x in sequence], dtype=torch.long)
        return self.embedding(ids)


class RNAStructuralFeatures(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_features(self, sequence: str) -> Dict[str, torch.Tensor]:
        l = len(sequence)
        gc = torch.tensor([(sequence[i] in {"G", "C"}) for i in range(l)], dtype=torch.float32)
        return {
            "base_pair_probs": torch.zeros(l, l),
            "secondary_structure": torch.zeros(l),
            "conservation": torch.zeros(l),
            "chemical_modifications": torch.zeros(l),
            "gc_content_windows": gc,
        }
