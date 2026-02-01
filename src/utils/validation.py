"""Input validation utilities for robust error handling.

Provides:
- Type checking and validation
- Range checking for numerical inputs
- Shape validation for tensors
- Configuration validation
- Defensive programming utilities
"""

import torch
import numpy as np
from typing import Any, Optional, Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation failures."""
    pass


def validate_tensor(
    tensor: Any,
    name: str = "tensor",
    dtype: Optional[torch.dtype] = None,
    shape: Optional[Tuple] = None,
    ndim: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> torch.Tensor:
    """Validate tensor with comprehensive checks.
    
    Args:
        tensor: Input to validate
        name: Name for error messages
        dtype: Expected dtype
        shape: Expected shape (None for any dimension)
        ndim: Expected number of dimensions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        
    Returns:
        Validated tensor
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if it's a tensor
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        else:
            raise ValidationError(f"{name} must be a tensor, got {type(tensor)}")
    
    # Check dtype
    if dtype is not None and tensor.dtype != dtype:
        raise ValidationError(
            f"{name} must have dtype {dtype}, got {tensor.dtype}"
        )
    
    # Check shape
    if shape is not None:
        if len(tensor.shape) != len(shape):
            raise ValidationError(
                f"{name} must have {len(shape)} dimensions, got {len(tensor.shape)}"
            )
        for i, (actual, expected) in enumerate(zip(tensor.shape, shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"{name} dimension {i} must be {expected}, got {actual}"
                )
    
    # Check ndim
    if ndim is not None and tensor.ndim != ndim:
        raise ValidationError(
            f"{name} must have {ndim} dimensions, got {tensor.ndim}"
        )
    
    # Check for NaN
    if not allow_nan and torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    # Check for inf
    if not allow_inf and torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Check value range
    if min_value is not None and tensor.min() < min_value:
        raise ValidationError(
            f"{name} minimum value {tensor.min():.6f} is less than {min_value}"
        )
    
    if max_value is not None and tensor.max() > max_value:
        raise ValidationError(
            f"{name} maximum value {tensor.max():.6f} exceeds {max_value}"
        )
    
    return tensor


def validate_range(
    value: Union[int, float],
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True
) -> Union[int, float]:
    """Validate numerical value is within range.
    
    Args:
        value: Value to validate
        name: Name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        inclusive: Whether bounds are inclusive
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is out of range
    """
    if min_value is not None:
        if inclusive and value < min_value:
            raise ValidationError(f"{name}={value} must be >= {min_value}")
        elif not inclusive and value <= min_value:
            raise ValidationError(f"{name}={value} must be > {min_value}")
    
    if max_value is not None:
        if inclusive and value > max_value:
            raise ValidationError(f"{name}={value} must be <= {max_value}")
        elif not inclusive and value >= max_value:
            raise ValidationError(f"{name}={value} must be < {max_value}")
    
    return value


def validate_type(
    value: Any,
    name: str,
    expected_type: Union[type, Tuple[type, ...]]
) -> Any:
    """Validate value has expected type.
    
    Args:
        value: Value to validate
        name: Name for error messages
        expected_type: Expected type or tuple of types
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If type doesn't match
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be of type {expected_type}, got {type(value)}"
        )
    return value


def validate_config(config: dict, required_keys: List[str]) -> dict:
    """Validate configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        Validated config
        
    Raises:
        ValidationError: If required keys are missing
    """
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValidationError(
            f"Configuration missing required keys: {missing_keys}"
        )
    return config


def validate_protein_sequence(sequence: str) -> str:
    """Validate protein sequence contains only valid amino acids.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        Validated uppercase sequence
        
    Raises:
        ValidationError: If sequence contains invalid characters
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.upper().strip()
    
    if not sequence:
        raise ValidationError("Protein sequence cannot be empty")
    
    invalid_chars = set(sequence) - valid_amino_acids
    if invalid_chars:
        raise ValidationError(
            f"Invalid amino acids in sequence: {invalid_chars}"
        )
    
    return sequence


def validate_coordinates(
    coords: torch.Tensor,
    num_atoms: Optional[int] = None,
    allow_nan: bool = False
) -> torch.Tensor:
    """Validate 3D coordinates tensor.
    
    Args:
        coords: Coordinates tensor (N, 3) or (B, N, 3)
        num_atoms: Expected number of atoms
        allow_nan: Whether to allow NaN values
        
    Returns:
        Validated coordinates
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    coords = validate_tensor(
        coords,
        name="coordinates",
        dtype=torch.float32,
        ndim=None,  # Can be 2D or 3D
        allow_nan=allow_nan,
        allow_inf=False
    )
    
    # Check last dimension is 3 (x, y, z)
    if coords.shape[-1] != 3:
        raise ValidationError(
            f"Coordinates must have 3 dimensions (x,y,z), got {coords.shape[-1]}"
        )
    
    # Check number of atoms if specified
    if num_atoms is not None:
        actual_atoms = coords.shape[-2]
        if actual_atoms != num_atoms:
            raise ValidationError(
                f"Expected {num_atoms} atoms, got {actual_atoms}"
            )
    
    return coords


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float = 1e-10
) -> torch.Tensor:
    """Safe division that avoids division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator
        
    Returns:
        Result of division
    """
    return numerator / (denominator + epsilon)


def clip_gradients(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = True
) -> torch.Tensor:
    """Clip gradients with validation.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
        error_if_nonfinite: Raise error if gradients are non-finite
        
    Returns:
        Total gradient norm
        
    Raises:
        ValidationError: If gradients contain NaN/Inf and error_if_nonfinite=True
    """
    parameters = [p for p in parameters if p.grad is not None]
    
    if error_if_nonfinite:
        for p in parameters:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                raise ValidationError(
                    "Gradients contain NaN or Inf values. "
                    "This may indicate numerical instability."
                )
    
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=False
    )
    
    return total_norm
