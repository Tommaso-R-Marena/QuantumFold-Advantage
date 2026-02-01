"""Input validation and error handling utilities.

Provides robust validation functions for:
- Tensor shapes and dtypes
- Parameter ranges
- File operations
- Device compatibility
- Numerical stability checks
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_tensor_shape(
    tensor: Tensor,
    expected_shape: Tuple[Union[int, None], ...],
    name: str = "tensor",
    allow_batch: bool = True,
) -> None:
    """Validate tensor shape matches expected dimensions.

    Args:
        tensor: Input tensor to validate
        expected_shape: Expected shape (None for any dimension)
        name: Tensor name for error messages
        allow_batch: If True, allows additional batch dimension

    Raises:
        ValidationError: If shape doesn't match

    Example:
        >>> x = torch.randn(32, 64, 128)
        >>> validate_tensor_shape(x, (None, 64, 128), "features")
    """
    actual_shape = tuple(tensor.shape)

    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        actual_shape = actual_shape[1:]

    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"{name}: Expected {len(expected_shape)} dimensions, "
            f"got {len(actual_shape)} (shape: {tensor.shape})"
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{name}: Dimension {i} mismatch. Expected {expected}, "
                f"got {actual} (full shape: {tensor.shape})"
            )


def validate_tensor_dtype(
    tensor: Tensor,
    expected_dtype: torch.dtype,
    name: str = "tensor",
    auto_convert: bool = False,
) -> Tensor:
    """Validate and optionally convert tensor dtype.

    Args:
        tensor: Input tensor
        expected_dtype: Expected dtype
        name: Tensor name for error messages
        auto_convert: If True, converts dtype automatically

    Returns:
        Validated (and possibly converted) tensor

    Raises:
        ValidationError: If dtype doesn't match and auto_convert=False
    """
    if tensor.dtype != expected_dtype:
        if auto_convert:
            return tensor.to(expected_dtype)
        else:
            raise ValidationError(f"{name}: Expected dtype {expected_dtype}, got {tensor.dtype}")
    return tensor


def validate_device(
    tensor: Tensor,
    expected_device: torch.device,
    name: str = "tensor",
    auto_move: bool = False,
) -> Tensor:
    """Validate and optionally move tensor to expected device.

    Args:
        tensor: Input tensor
        expected_device: Expected device
        name: Tensor name for error messages
        auto_move: If True, moves tensor automatically

    Returns:
        Validated (and possibly moved) tensor

    Raises:
        ValidationError: If device doesn't match and auto_move=False
    """
    if tensor.device != expected_device:
        if auto_move:
            return tensor.to(expected_device)
        else:
            raise ValidationError(f"{name}: Expected device {expected_device}, got {tensor.device}")
    return tensor


def validate_range(
    value: Union[float, int, Tensor],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
    inclusive: bool = True,
) -> None:
    """Validate value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (None for no limit)
        max_val: Maximum allowed value (None for no limit)
        name: Value name for error messages
        inclusive: If True, use <= and >=; otherwise < and >

    Raises:
        ValidationError: If value is out of range
    """
    if isinstance(value, Tensor):
        value = value.item() if value.numel() == 1 else value

    if isinstance(value, Tensor):
        # Validate all elements
        if min_val is not None:
            if inclusive:
                invalid = value < min_val
            else:
                invalid = value <= min_val
            if invalid.any():
                raise ValidationError(
                    f"{name}: Found values less than {'=' if inclusive else ''} {min_val}"
                )

        if max_val is not None:
            if inclusive:
                invalid = value > max_val
            else:
                invalid = value >= max_val
            if invalid.any():
                raise ValidationError(
                    f"{name}: Found values greater than {'=' if inclusive else ''} {max_val}"
                )
    else:
        # Scalar validation
        if min_val is not None:
            if (inclusive and value < min_val) or (not inclusive and value <= min_val):
                raise ValidationError(
                    f"{name}: Value {value} less than {'=' if inclusive else ''} {min_val}"
                )

        if max_val is not None:
            if (inclusive and value > max_val) or (not inclusive and value >= max_val):
                raise ValidationError(
                    f"{name}: Value {value} greater than {'=' if inclusive else ''} {max_val}"
                )


def validate_file_exists(path: Union[str, Path], name: str = "file") -> Path:
    """Validate file exists and is readable.

    Args:
        path: File path
        name: File name for error messages

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"{name}: File not found: {path}")
    if not path.is_file():
        raise ValidationError(f"{name}: Not a file: {path}")
    if not path.stat().st_size > 0:
        raise ValidationError(f"{name}: File is empty: {path}")
    return path


def validate_directory(
    path: Union[str, Path],
    create: bool = False,
    name: str = "directory",
) -> Path:
    """Validate directory exists or create it.

    Args:
        path: Directory path
        create: If True, creates directory if it doesn't exist
        name: Directory name for error messages

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If directory doesn't exist and create=False
    """
    path = Path(path)
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValidationError(f"{name}: Directory not found: {path}")
    elif not path.is_dir():
        raise ValidationError(f"{name}: Not a directory: {path}")
    return path


def check_numerical_stability(tensor: Tensor, name: str = "tensor") -> None:
    """Check tensor for numerical issues (NaN, Inf).

    Args:
        tensor: Tensor to check
        name: Tensor name for error messages

    Raises:
        ValidationError: If NaN or Inf values found
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise ValidationError(
            f"{name}: Found {nan_count} NaN values. "
            f"Shape: {tensor.shape}, dtype: {tensor.dtype}"
        )

    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise ValidationError(
            f"{name}: Found {inf_count} Inf values. "
            f"Shape: {tensor.shape}, dtype: {tensor.dtype}"
        )


def validate_probability_distribution(
    probs: Tensor,
    name: str = "probabilities",
    dim: int = -1,
    tolerance: float = 1e-5,
) -> None:
    """Validate tensor represents valid probability distribution.

    Args:
        probs: Probability tensor
        name: Tensor name for error messages
        dim: Dimension along which probabilities should sum to 1
        tolerance: Tolerance for sum check

    Raises:
        ValidationError: If not a valid probability distribution
    """
    # Check range [0, 1]
    if (probs < 0).any():
        raise ValidationError(f"{name}: Found negative probabilities")
    if (probs > 1).any():
        raise ValidationError(f"{name}: Found probabilities > 1")

    # Check sum to 1
    sums = probs.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
        raise ValidationError(
            f"{name}: Probabilities don't sum to 1. "
            f"Min sum: {sums.min():.6f}, Max sum: {sums.max():.6f}"
        )


def validate_batch_consistency(
    tensors: List[Tensor],
    names: Optional[List[str]] = None,
    check_device: bool = True,
    check_dtype: bool = False,
) -> None:
    """Validate batch of tensors have consistent batch size and optionally device/dtype.

    Args:
        tensors: List of tensors to validate
        names: Optional names for error messages
        check_device: If True, validates all tensors on same device
        check_dtype: If True, validates all tensors have same dtype

    Raises:
        ValidationError: If tensors are inconsistent
    """
    if not tensors:
        return

    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    # Check batch sizes
    batch_sizes = [t.shape[0] if t.ndim > 0 else 1 for t in tensors]
    if len(set(batch_sizes)) > 1:
        size_info = ", ".join([f"{n}: {s}" for n, s in zip(names, batch_sizes)])
        raise ValidationError(f"Inconsistent batch sizes: {size_info}")

    # Check devices
    if check_device:
        devices = [t.device for t in tensors]
        if len(set([str(d) for d in devices])) > 1:
            device_info = ", ".join([f"{n}: {d}" for n, d in zip(names, devices)])
            raise ValidationError(f"Inconsistent devices: {device_info}")

    # Check dtypes
    if check_dtype:
        dtypes = [t.dtype for t in tensors]
        if len(set(dtypes)) > 1:
            dtype_info = ", ".join([f"{n}: {d}" for n, d in zip(names, dtypes)])
            raise ValidationError(f"Inconsistent dtypes: {dtype_info}")


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    epsilon: float = 1e-10,
) -> Tensor:
    """Safe division with epsilon to prevent division by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator

    Returns:
        Result of division
    """
    return numerator / (denominator + epsilon)


def safe_log(tensor: Tensor, epsilon: float = 1e-10) -> Tensor:
    """Safe logarithm with epsilon to prevent log(0).

    Args:
        tensor: Input tensor
        epsilon: Small value to add before log

    Returns:
        Log of tensor
    """
    return torch.log(tensor + epsilon)


def clamp_tensor(
    tensor: Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    inplace: bool = False,
) -> Tensor:
    """Clamp tensor values to specified range with validation.

    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        inplace: If True, modifies tensor in-place

    Returns:
        Clamped tensor
    """
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValidationError(f"min_val ({min_val}) > max_val ({max_val})")

    if inplace:
        return tensor.clamp_(min=min_val, max=max_val)
    else:
        return tensor.clamp(min=min_val, max=max_val)
