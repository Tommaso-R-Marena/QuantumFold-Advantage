"""Validation utilities for protein structures and predictions.

Provides:
- Structure validation (stereochemistry, clashes)
- Sequence-structure consistency
- Prediction quality checks
- Data sanity checks
- Statistical validation helpers
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class StructureValidator:
    """Validate protein structure quality."""

    # Standard amino acid distances (CA-CA in Angstroms)
    MIN_CA_DISTANCE = 2.8
    MAX_CA_DISTANCE = 4.5
    TYPICAL_CA_DISTANCE = 3.8

    # Clash detection
    CLASH_DISTANCE = 2.0

    @staticmethod
    def validate_coordinates(
        coordinates: Union[np.ndarray, torch.Tensor], sequence: Optional[str] = None
    ) -> Dict[str, any]:
        """Validate structure coordinates.

        Args:
            coordinates: CA coordinates (N, 3)
            sequence: Optional amino acid sequence

        Returns:
            Dictionary with validation results
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()

        results = {"valid": True, "warnings": [], "errors": [], "statistics": {}}

        # Check shape
        if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
            results["valid"] = False
            results["errors"].append(
                f"Invalid coordinate shape: {coordinates.shape}. Expected (N, 3)"
            )
            return results

        n_residues = len(coordinates)

        # Check for NaN or Inf
        if not np.isfinite(coordinates).all():
            results["valid"] = False
            results["errors"].append("Coordinates contain NaN or Inf values")
            return results

        # Check sequence length match
        if sequence is not None and len(sequence) != n_residues:
            results["valid"] = False
            results["errors"].append(
                f"Sequence length ({len(sequence)}) doesn't match coordinates ({n_residues})"
            )

        # Check bond lengths (consecutive CA distances)
        bond_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)

        # Statistics
        results["statistics"]["mean_bond_length"] = float(np.mean(bond_lengths))
        results["statistics"]["std_bond_length"] = float(np.std(bond_lengths))
        results["statistics"]["min_bond_length"] = float(np.min(bond_lengths))
        results["statistics"]["max_bond_length"] = float(np.max(bond_lengths))

        # Check for broken chains (bonds too long)
        broken_bonds = np.where(bond_lengths > StructureValidator.MAX_CA_DISTANCE)[0]
        if len(broken_bonds) > 0:
            results["warnings"].append(
                f"Found {len(broken_bonds)} potentially broken bonds (>{StructureValidator.MAX_CA_DISTANCE}Å)"
            )

        # Check for compressed bonds (too short)
        compressed_bonds = np.where(bond_lengths < StructureValidator.MIN_CA_DISTANCE)[0]
        if len(compressed_bonds) > 0:
            results["warnings"].append(
                f"Found {len(compressed_bonds)} compressed bonds (<{StructureValidator.MIN_CA_DISTANCE}Å)"
            )

        # Check for clashes (atoms too close)
        distances = squareform(pdist(coordinates))
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        # Ignore sequential neighbors (i, i+1)
        for i in range(n_residues - 1):
            distances[i, i + 1] = np.inf
            distances[i + 1, i] = np.inf

        clashes = np.where(distances < StructureValidator.CLASH_DISTANCE)
        n_clashes = len(clashes[0]) // 2  # Divide by 2 because matrix is symmetric

        if n_clashes > 0:
            results["warnings"].append(f"Found {n_clashes} atomic clashes")

        results["statistics"]["n_clashes"] = n_clashes

        # Check overall structure compactness
        center = np.mean(coordinates, axis=0)
        max_radius = np.max(np.linalg.norm(coordinates - center, axis=1))
        results["statistics"]["radius_of_gyration"] = float(max_radius)

        return results

    @staticmethod
    def validate_ramachandran(coordinates: np.ndarray, tolerance: float = 0.02) -> Dict[str, float]:
        """Validate Ramachandran angles.

        Args:
            coordinates: CA coordinates (N, 3)
            tolerance: Tolerance for outlier detection (fraction)

        Returns:
            Dictionary with validation metrics
        """
        # This is a simplified version
        # Full implementation would calculate phi/psi angles
        # and check against Ramachandran plot distributions

        results = {"favored_fraction": 0.0, "allowed_fraction": 0.0, "outlier_fraction": 0.0}

        # Placeholder for now
        # In production, use Bio.PDB or similar for proper calculation
        warnings.warn(
            "Full Ramachandran validation not implemented. " "Use Bio.PDB for complete validation."
        )

        return results

    @staticmethod
    def check_structure_quality(
        coordinates: np.ndarray, sequence: str, confidence: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """Comprehensive structure quality check.

        Args:
            coordinates: CA coordinates (N, 3)
            sequence: Amino acid sequence
            confidence: Optional per-residue confidence scores

        Returns:
            Quality assessment dictionary
        """
        assessment = {
            "overall_quality": "unknown",
            "coordinate_validation": None,
            "confidence_statistics": None,
        }

        # Validate coordinates
        coord_results = StructureValidator.validate_coordinates(coordinates, sequence)
        assessment["coordinate_validation"] = coord_results

        # Analyze confidence if provided
        if confidence is not None:
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.detach().cpu().numpy()

            assessment["confidence_statistics"] = {
                "mean": float(np.mean(confidence)),
                "median": float(np.median(confidence)),
                "min": float(np.min(confidence)),
                "max": float(np.max(confidence)),
                "std": float(np.std(confidence)),
                "high_confidence_fraction": float(np.mean(confidence > 70)),
                "very_high_confidence_fraction": float(np.mean(confidence > 90)),
            }

        # Overall quality assessment
        if not coord_results["valid"]:
            assessment["overall_quality"] = "invalid"
        elif len(coord_results["errors"]) > 0:
            assessment["overall_quality"] = "poor"
        elif len(coord_results["warnings"]) > 3:
            assessment["overall_quality"] = "fair"
        elif confidence is not None and np.mean(confidence) > 70:
            assessment["overall_quality"] = "good"
        elif confidence is not None and np.mean(confidence) > 90:
            assessment["overall_quality"] = "excellent"
        else:
            assessment["overall_quality"] = "acceptable"

        return assessment


class DataValidator:
    """Validate input data for training and inference."""

    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def validate_sequence(sequence: str, strict: bool = True) -> Tuple[bool, str]:
        """Validate amino acid sequence.

        Args:
            sequence: Amino acid sequence
            strict: If True, only allow standard 20 amino acids

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequence:
            return False, "Empty sequence"

        if not isinstance(sequence, str):
            return False, f"Sequence must be string, got {type(sequence)}"

        # Check length
        if len(sequence) < 10:
            return False, f"Sequence too short ({len(sequence)} residues, minimum 10)"

        if len(sequence) > 10000:
            return False, f"Sequence too long ({len(sequence)} residues, maximum 10000)"

        # Check characters
        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - DataValidator.VALID_AA

        if strict and invalid_chars:
            return False, f"Invalid amino acids: {invalid_chars}"

        return True, ""

    @staticmethod
    def validate_embeddings(
        embeddings: Union[np.ndarray, torch.Tensor],
        sequence_length: int,
        expected_dim: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Validate embedding tensor.

        Args:
            embeddings: Embedding tensor (L, D) or (B, L, D)
            sequence_length: Expected sequence length
            expected_dim: Expected embedding dimension

        Returns:
            Tuple of (is_valid, error_message)
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # Check shape
        if len(embeddings.shape) not in [2, 3]:
            return False, f"Invalid embedding shape: {embeddings.shape}"

        # Check sequence length
        seq_len_axis = 0 if len(embeddings.shape) == 2 else 1
        if embeddings.shape[seq_len_axis] != sequence_length:
            return False, (
                f"Sequence length mismatch: embeddings have {embeddings.shape[seq_len_axis]}, "
                f"expected {sequence_length}"
            )

        # Check embedding dimension
        if expected_dim is not None:
            emb_dim = embeddings.shape[-1]
            if emb_dim != expected_dim:
                return False, (
                    f"Embedding dimension mismatch: got {emb_dim}, expected {expected_dim}"
                )

        # Check for NaN/Inf
        if not np.isfinite(embeddings).all():
            return False, "Embeddings contain NaN or Inf values"

        return True, ""

    @staticmethod
    def validate_batch(
        sequences: List[str],
        coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
        max_length_diff: int = 50,
    ) -> Tuple[bool, str]:
        """Validate a batch of sequences and coordinates.

        Args:
            sequences: List of sequences
            coordinates: Optional coordinates (B, L, 3)
            max_length_diff: Maximum allowed length difference in batch

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequences:
            return False, "Empty batch"

        # Validate each sequence
        for i, seq in enumerate(sequences):
            is_valid, msg = DataValidator.validate_sequence(seq)
            if not is_valid:
                return False, f"Sequence {i}: {msg}"

        # Check length variation
        lengths = [len(seq) for seq in sequences]
        length_diff = max(lengths) - min(lengths)

        if length_diff > max_length_diff:
            return False, (
                f"Batch has large length variation ({length_diff} residues). "
                f"Consider using bucketing or padding."
            )

        # Validate coordinates if provided
        if coordinates is not None:
            if isinstance(coordinates, torch.Tensor):
                coordinates = coordinates.detach().cpu().numpy()

            if len(coordinates) != len(sequences):
                return False, (
                    f"Batch size mismatch: {len(coordinates)} coordinates, "
                    f"{len(sequences)} sequences"
                )

        return True, ""


def sanitize_predictions(
    predictions: Dict[str, torch.Tensor], sequence: str
) -> Dict[str, torch.Tensor]:
    """Sanitize model predictions.

    Args:
        predictions: Model output dictionary
        sequence: Input sequence

    Returns:
        Sanitized predictions
    """
    sanitized = {}
    seq_len = len(sequence)

    # Sanitize coordinates
    if "coordinates" in predictions:
        coords = predictions["coordinates"]

        # Ensure correct shape
        if coords.shape[-2] != seq_len:
            warnings.warn(f"Coordinate shape mismatch. Expected {seq_len}, got {coords.shape[-2]}")

        # Replace NaN/Inf with zeros
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        sanitized["coordinates"] = coords

    # Sanitize confidence scores
    if "plddt" in predictions:
        plddt = predictions["plddt"]

        # Clamp to valid range [0, 100]
        plddt = torch.clamp(plddt, 0, 100)

        sanitized["plddt"] = plddt

    # Copy other keys
    for key, value in predictions.items():
        if key not in sanitized:
            sanitized[key] = value

    return sanitized


class ValidationError(ValueError):
    """Raised when validation checks fail."""


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: tuple) -> torch.Tensor:
    """Validate tensor shape; use None as wildcard in expected_shape."""
    if tensor.ndim != len(expected_shape):
        raise ValidationError(f"Expected {len(expected_shape)} dims, got {tensor.ndim}")
    for i, expected in enumerate(expected_shape):
        if expected is not None and tensor.shape[i] != expected:
            raise ValidationError(f"Dimension {i} expected {expected}, got {tensor.shape[i]}")
    return tensor


def validate_tensor_dtype(
    tensor: torch.Tensor, expected_dtype: torch.dtype, auto_convert: bool = False
) -> torch.Tensor:
    """Validate tensor dtype with optional auto-conversion."""
    if tensor.dtype == expected_dtype:
        return tensor
    if auto_convert:
        return tensor.to(expected_dtype)
    raise ValidationError(f"Expected dtype {expected_dtype}, got {tensor.dtype}")


def validate_range(value, min_val=None, max_val=None):
    """Validate scalar or tensor values are within [min_val, max_val]."""
    if isinstance(value, torch.Tensor):
        if min_val is not None and torch.any(value < min_val):
            raise ValidationError(f"Values below minimum {min_val}")
        if max_val is not None and torch.any(value > max_val):
            raise ValidationError(f"Values above maximum {max_val}")
    else:
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value {value} below minimum {min_val}")
        if max_val is not None and value > max_val:
            raise ValidationError(f"Value {value} above maximum {max_val}")
    return value


def check_numerical_stability(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has no NaN/Inf values."""
    if torch.isnan(tensor).any():
        raise ValidationError("Tensor contains NaN values")
    if torch.isinf(tensor).any():
        raise ValidationError("Tensor contains Inf values")
    return tensor


def safe_divide(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Numerically safe division."""
    return numerator / (denominator + eps)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe logarithm."""
    return torch.log(torch.clamp(x, min=eps))


def validate_tensor(tensor: torch.Tensor, expected_shape=None, expected_dtype=None) -> torch.Tensor:
    """Combined tensor validation helper."""
    if expected_shape is not None:
        validate_tensor_shape(tensor, expected_shape)
    if expected_dtype is not None:
        tensor = validate_tensor_dtype(tensor, expected_dtype)
    return check_numerical_stability(tensor)


def validate_type(value, expected_type):
    """Validate Python type."""
    if not isinstance(value, expected_type):
        raise ValidationError(f"Expected type {expected_type}, got {type(value)}")
    return value


def validate_config(config: dict, required_keys=None):
    """Validate config dictionary contains required keys."""
    required_keys = required_keys or []
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Missing config key: {key}")
    return config


def validate_protein_sequence(sequence: str) -> str:
    """Validate standard amino acid sequence."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    if not sequence or any(c.upper() not in valid for c in sequence):
        raise ValidationError("Invalid protein sequence")
    return sequence


def validate_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Validate coordinate tensor shape (..., 3)."""
    if coords.shape[-1] != 3:
        raise ValidationError("Coordinates must end with dimension 3")
    return check_numerical_stability(coords)


def clip_gradients(parameters, max_norm: float = 1.0) -> float:
    """Clip gradients and return total norm."""
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
