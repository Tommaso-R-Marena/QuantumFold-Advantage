"""Pre-trained protein language model embeddings.

Integrates state-of-the-art protein language models:
- ESM-2 (Evolutionary Scale Modeling) from Meta AI
- ProtT5 from Rostlab
- Evolutionary features from MSA
- Geometric features (torsion angles, frames)
- Contact predictions

References:
    - ESM-2: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure
             with a language model" Science (2023) DOI: 10.1126/science.ade2574
    - ProtT5: Elnaggar et al., "ProtTrans: Toward Understanding the Language of Life
              Through Self-Supervised Learning" IEEE TPAMI (2022)
    - MSA Transformer: Rao et al., "MSA Transformer" ICML (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

try:
    import esm
except ImportError:
    warnings.warn("ESM not installed. Install with: pip install fair-esm")
    esm = None

try:
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:
    warnings.warn("Transformers not installed. Install with: pip install transformers")
    T5EncoderModel = None
    T5Tokenizer = None


class ESM2Embedder(nn.Module):
    """ESM-2 protein language model embeddings.
    
    Provides pre-trained representations that capture:
    - Sequence conservation
    - Structural information
    - Functional properties
    - Co-evolutionary couplings
    
    Args:
        model_name: ESM-2 model variant
                   ('esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                    'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D')
        repr_layers: Which layers to extract representations from
        freeze: Whether to freeze pre-trained weights
    """
    
    def __init__(
        self,
        model_name: str = 'esm2_t33_650M_UR50D',
        repr_layers: List[int] = [-1],
        freeze: bool = True
    ):
        super().__init__()
        
        if esm is None:
            raise ImportError("ESM not installed. Run: pip install fair-esm")
        
        self.model_name = model_name
        self.repr_layers = repr_layers
        self.freeze = freeze
        
        # Load pre-trained model
        print(f"Loading ESM-2 model: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get embedding dimension
        self.embed_dim = self.model.embed_dim
        
        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        print(f"ESM-2 model loaded. Embedding dimension: {self.embed_dim}")
    
    def forward(
        self,
        sequences: Union[List[str], List[Tuple[str, str]]],
        return_contacts: bool = False
    ) -> Dict[str, Tensor]:
        """Extract ESM-2 embeddings.
        
        Args:
            sequences: List of protein sequences or (id, sequence) tuples
            return_contacts: Whether to return predicted contacts
        
        Returns:
            Dictionary with:
                - 'embeddings': Per-residue embeddings (batch, seq_len, embed_dim)
                - 'mean_embedding': Sequence-level embedding (batch, embed_dim)
                - 'contacts': Contact predictions if requested (batch, seq_len, seq_len)
        """
        # Format sequences
        if isinstance(sequences[0], str):
            sequences = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        
        # Convert to batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        
        # Forward pass
        with torch.no_grad() if self.freeze else torch.enable_grad():
            results = self.model(
                batch_tokens,
                repr_layers=self.repr_layers,
                return_contacts=return_contacts
            )
        
        # Extract representations
        # results['representations'] is a dict: {layer: (batch, seq_len, embed_dim)}
        layer_idx = self.repr_layers[0]
        embeddings = results['representations'][layer_idx]
        
        # Remove start/end tokens
        embeddings = embeddings[:, 1:-1, :]  # (batch, seq_len, embed_dim)
        
        # Sequence-level embedding (mean pooling)
        mean_embedding = embeddings.mean(dim=1)  # (batch, embed_dim)
        
        output = {
            'embeddings': embeddings,
            'mean_embedding': mean_embedding
        }
        
        if return_contacts:
            # Contact predictions
            contacts = results['contacts']
            contacts = contacts[:, 1:-1, 1:-1]  # Remove special tokens
            output['contacts'] = contacts
        
        return output


class ProtT5Embedder(nn.Module):
    """ProtT5 protein language model embeddings.
    
    Based on T5 architecture trained on protein sequences.
    
    Args:
        model_name: ProtT5 model variant
        freeze: Whether to freeze pre-trained weights
    """
    
    def __init__(
        self,
        model_name: str = 'Rostlab/prot_t5_xl_uniref50',
        freeze: bool = True
    ):
        super().__init__()
        
        if T5EncoderModel is None:
            raise ImportError("Transformers not installed. Run: pip install transformers")
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load model and tokenizer
        print(f"Loading ProtT5 model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        
        self.embed_dim = self.model.config.d_model
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        print(f"ProtT5 model loaded. Embedding dimension: {self.embed_dim}")
    
    def forward(self, sequences: List[str]) -> Dict[str, Tensor]:
        """Extract ProtT5 embeddings.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            Dictionary with embeddings
        """
        # Add spaces between amino acids (ProtT5 requirement)
        sequences_spaced = [' '.join(list(seq)) for seq in sequences]
        
        # Tokenize
        tokens = self.tokenizer(
            sequences_spaced,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Forward pass
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model(**tokens)
        
        # Extract embeddings
        embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)
        
        # Mean pooling (excluding padding)
        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        mean_embedding = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return {
            'embeddings': embeddings,
            'mean_embedding': mean_embedding
        }


class EvolutionaryFeatureExtractor(nn.Module):
    """Extract evolutionary features from MSA.
    
    Computes:
    - Position-specific scoring matrices (PSSM)
    - Conservation scores
    - Co-evolution couplings
    - Gap statistics
    """
    
    def __init__(self):
        super().__init__()
        self.aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}
    
    def compute_pssm(
        self,
        msa: np.ndarray,
        pseudocount: float = 0.001
    ) -> np.ndarray:
        """Compute Position-Specific Scoring Matrix.
        
        Args:
            msa: Multiple sequence alignment (n_seqs, seq_len, 21) one-hot
            pseudocount: Pseudocount for smoothing
        
        Returns:
            PSSM (seq_len, 21)
        """
        # Frequency matrix
        freq = msa.mean(axis=0) + pseudocount
        freq = freq / freq.sum(axis=1, keepdims=True)
        
        # Convert to log-odds
        background = np.ones(21) / 21
        pssm = np.log2(freq / background)
        
        return pssm
    
    def compute_conservation(
        self,
        msa: np.ndarray
    ) -> np.ndarray:
        """Compute per-position conservation (Shannon entropy).
        
        Args:
            msa: Multiple sequence alignment (n_seqs, seq_len, 21)
        
        Returns:
            Conservation scores (seq_len,)
        """
        freq = msa.mean(axis=0) + 1e-9
        entropy = -np.sum(freq * np.log2(freq), axis=1)
        
        # Normalize to [0, 1], where 1 is highly conserved
        max_entropy = np.log2(21)
        conservation = 1.0 - (entropy / max_entropy)
        
        return conservation
    
    def compute_coevolution(
        self,
        msa: np.ndarray,
        apc_correction: bool = True
    ) -> np.ndarray:
        """Compute co-evolution matrix using mutual information.
        
        Args:
            msa: Multiple sequence alignment
            apc_correction: Apply Average Product Correction
        
        Returns:
            Co-evolution matrix (seq_len, seq_len)
        """
        n_seqs, seq_len, n_aa = msa.shape
        
        # Compute pairwise mutual information
        mi_matrix = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Joint distribution
                joint = np.zeros((n_aa, n_aa))
                for k in range(n_seqs):
                    aa_i = np.argmax(msa[k, i])
                    aa_j = np.argmax(msa[k, j])
                    joint[aa_i, aa_j] += 1
                joint = joint / n_seqs + 1e-9
                
                # Marginal distributions
                p_i = msa[:, i].mean(axis=0) + 1e-9
                p_j = msa[:, j].mean(axis=0) + 1e-9
                
                # Mutual information
                mi = 0.0
                for a in range(n_aa):
                    for b in range(n_aa):
                        if joint[a, b] > 0:
                            mi += joint[a, b] * np.log2(joint[a, b] / (p_i[a] * p_j[b]))
                
                mi_matrix[i, j] = mi_matrix[j, i] = mi
        
        if apc_correction:
            # Average Product Correction to remove phylogenetic bias
            mean_mi_row = mi_matrix.mean(axis=1)
            mean_mi_col = mi_matrix.mean(axis=0)
            mean_mi_total = mi_matrix.mean()
            
            apc_matrix = np.outer(mean_mi_row, mean_mi_col) / (mean_mi_total + 1e-9)
            mi_matrix = mi_matrix - apc_matrix
        
        return mi_matrix
    
    def forward(self, msa: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all evolutionary features.
        
        Args:
            msa: Multiple sequence alignment
        
        Returns:
            Dictionary with features
        """
        return {
            'pssm': self.compute_pssm(msa),
            'conservation': self.compute_conservation(msa),
            'coevolution': self.compute_coevolution(msa)
        }


class GeometricFeatureExtractor(nn.Module):
    """Extract geometric features from coordinates.
    
    Computes:
    - Backbone torsion angles (phi, psi, omega)
    - Local coordinate frames
    - Distance maps
    - Orientation features
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_torsion_angles(
        self,
        coords: Tensor
    ) -> Tensor:
        """Compute backbone torsion angles.
        
        Args:
            coords: CA coordinates (batch, N, 3)
        
        Returns:
            Angles (batch, N, 3) for phi, psi, omega
        """
        batch_size, n_res, _ = coords.shape
        angles = torch.zeros(batch_size, n_res, 3, device=coords.device)
        
        for i in range(1, n_res - 2):
            # Vectors
            v1 = coords[:, i] - coords[:, i-1]
            v2 = coords[:, i+1] - coords[:, i]
            v3 = coords[:, i+2] - coords[:, i+1]
            
            # Normalize
            v1 = F.normalize(v1, dim=-1)
            v2 = F.normalize(v2, dim=-1)
            v3 = F.normalize(v3, dim=-1)
            
            # Phi angle (approximate)
            cos_phi = torch.sum(v1 * v2, dim=-1)
            angles[:, i, 0] = torch.acos(torch.clamp(cos_phi, -1, 1))
            
            # Psi angle (approximate)
            cos_psi = torch.sum(v2 * v3, dim=-1)
            angles[:, i, 1] = torch.acos(torch.clamp(cos_psi, -1, 1))
        
        return angles
    
    def compute_local_frames(
        self,
        coords: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute local coordinate frames.
        
        Args:
            coords: CA coordinates (batch, N, 3)
        
        Returns:
            (origins, rotation_matrices)
            origins: (batch, N-2, 3)
            rotation_matrices: (batch, N-2, 3, 3)
        """
        batch_size, n_res, _ = coords.shape
        
        origins = coords[:, :-2]
        
        # X-axis: CA(i) -> CA(i+1)
        x_axis = coords[:, 1:-1] - origins
        x_axis = F.normalize(x_axis, dim=-1)
        
        # Y-axis: perpendicular in plane with CA(i+2)
        v = coords[:, 2:] - origins
        y_axis = v - torch.sum(v * x_axis, dim=-1, keepdim=True) * x_axis
        y_axis = F.normalize(y_axis, dim=-1)
        
        # Z-axis: cross product
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        
        # Rotation matrix
        rotation_matrices = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        return origins, rotation_matrices
    
    def forward(
        self,
        coords: Tensor
    ) -> Dict[str, Tensor]:
        """Extract geometric features.
        
        Args:
            coords: CA coordinates (batch, N, 3)
        
        Returns:
            Dictionary with features
        """
        # Torsion angles
        angles = self.compute_torsion_angles(coords)
        
        # Local frames
        origins, frames = self.compute_local_frames(coords)
        
        # Distance matrix
        distances = torch.cdist(coords, coords)
        
        return {
            'angles': angles,
            'origins': origins,
            'frames': frames,
            'distances': distances
        }


class CombinedProteinEmbedding(nn.Module):
    """Combined embedding from multiple sources.
    
    Integrates:
    - Pre-trained language model (ESM-2 or ProtT5)
    - Evolutionary features
    - Geometric features
    
    Args:
        use_esm: Use ESM-2 embeddings
        use_prot_t5: Use ProtT5 embeddings
        use_evolutionary: Use MSA features
        projection_dim: Output dimension
    """
    
    def __init__(
        self,
        use_esm: bool = True,
        use_prot_t5: bool = False,
        use_evolutionary: bool = True,
        projection_dim: int = 512
    ):
        super().__init__()
        
        self.use_esm = use_esm
        self.use_prot_t5 = use_prot_t5
        self.use_evolutionary = use_evolutionary
        
        # Initialize embedders
        input_dim = 0
        
        if use_esm:
            self.esm_embedder = ESM2Embedder()
            input_dim += self.esm_embedder.embed_dim
        
        if use_prot_t5:
            self.prot_t5_embedder = ProtT5Embedder()
            input_dim += self.prot_t5_embedder.embed_dim
        
        if use_evolutionary:
            self.evo_extractor = EvolutionaryFeatureExtractor()
            input_dim += 21  # PSSM features
        
        # Projection layer
        self.projection = nn.Linear(input_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)
    
    def forward(
        self,
        sequences: List[str],
        msa: Optional[np.ndarray] = None
    ) -> Tensor:
        """Generate combined embeddings.
        
        Args:
            sequences: Protein sequences
            msa: Optional MSA for evolutionary features
        
        Returns:
            Combined embeddings (batch, seq_len, projection_dim)
        """
        embeddings = []
        
        # ESM-2
        if self.use_esm:
            esm_out = self.esm_embedder(sequences)
            embeddings.append(esm_out['embeddings'])
        
        # ProtT5
        if self.use_prot_t5:
            prot_t5_out = self.prot_t5_embedder(sequences)
            embeddings.append(prot_t5_out['embeddings'])
        
        # Evolutionary features
        if self.use_evolutionary and msa is not None:
            evo_features = self.evo_extractor(msa)
            pssm = torch.from_numpy(evo_features['pssm']).float()
            pssm = pssm.unsqueeze(0).expand(len(sequences), -1, -1)
            pssm = pssm.to(embeddings[0].device)
            embeddings.append(pssm)
        
        # Concatenate
        combined = torch.cat(embeddings, dim=-1)
        
        # Project
        output = self.projection(combined)
        output = self.norm(output)
        
        return output
