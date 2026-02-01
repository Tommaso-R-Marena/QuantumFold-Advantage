"""Tests for validation utilities."""

import pytest
import torch
from src.utils.validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_range,
    check_numerical_stability,
    ValidationError,
    safe_divide,
    safe_log,
)


class TestTensorValidation:
    def test_validate_shape_success(self):
        x = torch.randn(32, 64, 128)
        validate_tensor_shape(x, (None, 64, 128))

    def test_validate_shape_failure(self):
        x = torch.randn(32, 64, 128)
        with pytest.raises(ValidationError):
            validate_tensor_shape(x, (None, 32, 128))

    def test_validate_dtype_success(self):
        x = torch.randn(10, dtype=torch.float32)
        result = validate_tensor_dtype(x, torch.float32)
        assert result.dtype == torch.float32

    def test_validate_dtype_auto_convert(self):
        x = torch.randn(10, dtype=torch.float64)
        result = validate_tensor_dtype(x, torch.float32, auto_convert=True)
        assert result.dtype == torch.float32


class TestRangeValidation:
    def test_range_valid(self):
        validate_range(0.5, min_val=0.0, max_val=1.0)

    def test_range_invalid_min(self):
        with pytest.raises(ValidationError):
            validate_range(-0.1, min_val=0.0, max_val=1.0)

    def test_range_invalid_max(self):
        with pytest.raises(ValidationError):
            validate_range(1.1, min_val=0.0, max_val=1.0)

    def test_tensor_range(self):
        x = torch.tensor([0.2, 0.5, 0.8])
        validate_range(x, min_val=0.0, max_val=1.0)


class TestNumericalStability:
    def test_nan_detection(self):
        x = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValidationError):
            check_numerical_stability(x)

    def test_inf_detection(self):
        x = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValidationError):
            check_numerical_stability(x)

    def test_safe_divide(self):
        numerator = torch.ones(10)
        denominator = torch.zeros(10)
        result = safe_divide(numerator, denominator)
        assert torch.isfinite(result).all()

    def test_safe_log(self):
        x = torch.zeros(10)
        result = safe_log(x)
        assert torch.isfinite(result).all()
