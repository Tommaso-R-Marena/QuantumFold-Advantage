"""Tests for training utilities."""

import pytest
import torch

try:
    from src.advanced_training import FrameAlignedPointError, StructureAwareLoss, TrainingConfig

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training modules not available")
class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.use_amp is True

    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = TrainingConfig(
            epochs=50,
            batch_size=16,
            learning_rate=1e-4,
        )
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training modules not available")
class TestFrameAlignedPointError:
    """Tests for FAPE loss."""

    def test_initialization(self):
        """Test FAPE initialization."""
        fape = FrameAlignedPointError()
        assert fape.clamp_distance == 10.0
        assert fape.loss_unit_distance == 10.0

    def test_forward_pass(self, sample_coordinates):
        """Test FAPE loss computation."""
        fape = FrameAlignedPointError()

        pred_coords = sample_coordinates
        true_coords = sample_coordinates + torch.randn_like(sample_coordinates) * 0.1

        loss = fape(pred_coords, true_coords)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss).any()

    def test_with_mask(self, sample_coordinates):
        """Test FAPE with mask."""
        fape = FrameAlignedPointError()

        pred_coords = sample_coordinates
        true_coords = sample_coordinates + torch.randn_like(sample_coordinates) * 0.1
        mask = torch.ones(
            sample_coordinates.shape[0], sample_coordinates.shape[1], dtype=torch.bool
        )

        loss = fape(pred_coords, true_coords, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training modules not available")
class TestStructureAwareLoss:
    """Tests for combined structure loss."""

    def test_initialization(self):
        """Test loss initialization."""
        config = TrainingConfig()
        loss_fn = StructureAwareLoss(config)
        assert loss_fn.config == config

    def test_forward_pass(self, sample_coordinates):
        """Test loss computation."""
        config = TrainingConfig()
        loss_fn = StructureAwareLoss(config)

        pred_coords = sample_coordinates
        true_coords = sample_coordinates + torch.randn_like(sample_coordinates) * 0.1

        losses = loss_fn(pred_coords, true_coords)

        assert "total" in losses
        assert "fape" in losses
        assert "rmsd" in losses
        assert all(v.item() >= 0 for v in losses.values())
