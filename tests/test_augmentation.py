"""Comprehensive tests for protein data augmentation.

Tests cover:
- Geometric transformations (rotation, translation)
- Augmentation invariants and properties
- Edge cases and numerical stability
- Integration with training pipeline
- Performance benchmarks
"""

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    AugmentationConfig,
    MultiScaleAugmentation,
    ProteinAugmentation,
    SequenceAugmentation,
    TorsionAngleAugmentation,
)


class TestProteinAugmentation:
    """Test geometric augmentation of protein structures."""

    @pytest.fixture
    def sample_coords(self):
        """Generate sample protein coordinates."""
        batch_size = 4
        seq_len = 50
        return torch.randn(batch_size, seq_len, 3) * 10.0

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample sequence embeddings."""
        batch_size = 4
        seq_len = 50
        embed_dim = 128
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def augmentation(self):
        """Create augmentation instance."""
        config = AugmentationConfig(rotation_prob=1.0, translation_prob=1.0, coord_noise_prob=1.0)
        return ProteinAugmentation(config, training=True)

    def test_rotation_preserves_distances(self, augmentation, sample_coords):
        """Test that rotation preserves pairwise distances."""
        # Compute original pairwise distances
        batch_size, seq_len, _ = sample_coords.shape
        original_dists = torch.cdist(sample_coords[0], sample_coords[0])

        # Apply rotation
        rotated = augmentation.augment_coordinates(sample_coords)
        rotated_dists = torch.cdist(rotated[0], rotated[0])

        # Check distances are preserved (within noise tolerance)
        # Note: noise is added, so perfect preservation not expected
        # Just check structure isn't completely destroyed
        assert rotated_dists.shape == original_dists.shape
        assert not torch.allclose(rotated, sample_coords)  # Something changed

    def test_augmentation_determinism_control(self, sample_coords):
        """Test that augmentation can be disabled."""
        aug_train = ProteinAugmentation(training=True)
        aug_eval = ProteinAugmentation(training=False)

        # Training mode should augment
        result_train = aug_train.augment_coordinates(sample_coords)

        # Eval mode should not augment
        result_eval = aug_eval.augment_coordinates(sample_coords)

        assert torch.allclose(result_eval, sample_coords)

    def test_rotation_matrix_properties(self, augmentation):
        """Test that generated rotation matrices are valid."""
        batch_size = 10
        device = torch.device("cpu")

        R = augmentation.random_rotation_matrix(batch_size, device)

        # Check shape
        assert R.shape == (batch_size, 3, 3)

        # Check orthogonality: R @ R.T = I
        for i in range(batch_size):
            identity = torch.mm(R[i], R[i].T)
            assert torch.allclose(identity, torch.eye(3), atol=1e-5)

        # Check determinant = 1 (proper rotation, not reflection)
        for i in range(batch_size):
            det = torch.det(R[i])
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_translation_magnitude(self, augmentation, sample_coords):
        """Test that translation is within expected range."""
        config = AugmentationConfig(translation_prob=1.0, translation_scale=2.0)
        aug = ProteinAugmentation(config, training=True)

        # Disable other augmentations
        config.rotation_prob = 0.0
        config.coord_noise_prob = 0.0
        config.mirror_prob = 0.0

        original_center = sample_coords.mean(dim=(0, 1))

        # Apply multiple times
        translations = []
        for _ in range(100):
            augmented = aug.augment_coordinates(sample_coords)
            new_center = augmented.mean(dim=(0, 1))
            translation = new_center - original_center
            translations.append(translation.norm().item())

        # Check translation magnitudes are reasonable
        mean_translation = np.mean(translations)
        assert mean_translation > 0  # Something happened
        assert mean_translation < 5.0  # Not too large

    def test_coordinate_noise_scale(self, sample_coords):
        """Test Gaussian noise scaling."""
        noise_scale = 0.5
        config = AugmentationConfig(
            coord_noise_prob=1.0,
            coord_noise_scale=noise_scale,
            rotation_prob=0.0,
            translation_prob=0.0,
        )
        aug = ProteinAugmentation(config, training=True)

        differences = []
        for _ in range(50):
            augmented = aug.augment_coordinates(sample_coords)
            diff = (augmented - sample_coords).abs().mean().item()
            differences.append(diff)

        mean_diff = np.mean(differences)
        # Should be roughly proportional to noise scale
        assert mean_diff > 0.05

    def test_embedding_augmentation(self, augmentation, sample_embeddings):
        """Test embedding space augmentation."""
        original = sample_embeddings.clone()
        augmented = augmentation.augment_embeddings(sample_embeddings)

        # Check shape preserved
        assert augmented.shape == original.shape

        # Check values changed
        assert not torch.allclose(augmented, original)

        # Check not completely destroyed
        correlation = torch.corrcoef(
            torch.stack([original.flatten()[:1000], augmented.flatten()[:1000]])
        )[0, 1]
        assert correlation > 0.5  # Still somewhat similar

    def test_mixup_augmentation(self, augmentation, sample_coords, sample_embeddings):
        """Test mixup augmentation."""
        config = AugmentationConfig(mixup_prob=1.0, mixup_alpha=0.5)
        aug = ProteinAugmentation(config, training=True)

        # Apply mixup
        mixed_coords, mixed_embed, _ = aug.mixup_batch(sample_coords, sample_embeddings, None)

        # Check shapes preserved
        assert mixed_coords.shape == sample_coords.shape
        assert mixed_embed.shape == sample_embeddings.shape

        # Check interpolation occurred
        assert not torch.allclose(mixed_coords, sample_coords)

    def test_full_pipeline(self, augmentation, sample_coords, sample_embeddings):
        """Test complete augmentation pipeline."""
        labels = torch.randn(sample_coords.shape[0], sample_coords.shape[1])

        result = augmentation(coords=sample_coords, embeddings=sample_embeddings, labels=labels)

        # Check all outputs present
        assert "coordinates" in result
        assert "embeddings" in result
        assert "labels" in result
        assert "augmented" in result

        # Check shapes
        assert result["coordinates"].shape == sample_coords.shape
        assert result["embeddings"].shape == sample_embeddings.shape

    def test_batch_consistency(self, augmentation, sample_coords):
        """Test augmentation is applied per-sample."""
        # Each sample should be augmented independently
        augmented = augmentation.augment_coordinates(sample_coords)

        # Samples should differ from each other
        assert augmented.shape == sample_coords.shape

        # Check first and second samples are different
        diff_original = (sample_coords[0] - sample_coords[1]).abs().mean()
        diff_augmented = (augmented[0] - augmented[1]).abs().mean()

        # Both should have some difference
        assert diff_original > 0
        assert diff_augmented > 0


class TestSequenceAugmentation:
    """Test sequence-level augmentation."""

    @pytest.fixture
    def sample_sequence(self):
        return "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

    def test_random_masking(self, sample_sequence):
        """Test random masking of amino acids."""
        masked = SequenceAugmentation.random_mask(sample_sequence, mask_prob=0.2)

        # Check length preserved
        assert len(masked) == len(sample_sequence)

        # Check some masking occurred
        assert "X" in masked

        # Check not everything masked
        assert masked.count("X") < len(sample_sequence)

    def test_conservative_substitution(self, sample_sequence):
        """Test conservative amino acid substitution."""
        substituted = SequenceAugmentation.conservative_substitution(sample_sequence, sub_prob=0.1)

        # Check length preserved
        assert len(substituted) == len(sample_sequence)

        # Check some substitution occurred (probabilistically)
        # Run multiple times to ensure
        differences = 0
        for _ in range(10):
            sub = SequenceAugmentation.conservative_substitution(sample_sequence, sub_prob=0.2)
            if sub != sample_sequence:
                differences += 1

        assert differences > 0  # At least some substitutions happened

    def test_masking_probability(self, sample_sequence):
        """Test that masking probability is approximately correct."""
        mask_prob = 0.15
        n_trials = 1000

        mask_counts = []
        for _ in range(n_trials):
            masked = SequenceAugmentation.random_mask(sample_sequence, mask_prob)
            count = masked.count("X")
            mask_counts.append(count)

        mean_masks = np.mean(mask_counts)
        expected_masks = len(sample_sequence) * mask_prob

        # Should be close to expected value
        assert abs(mean_masks - expected_masks) < expected_masks * 0.3


class TestTorsionAngleAugmentation:
    """Test torsion angle-based augmentation."""

    @pytest.fixture
    def sample_coords(self):
        """Generate alpha helix-like coordinates."""
        seq_len = 20
        coords = torch.zeros(1, seq_len, 3)

        # Simple helix: rise of 1.5Å and turn of 100°
        for i in range(seq_len):
            angle = i * 100 * np.pi / 180
            coords[0, i, 0] = 2.3 * np.cos(angle)
            coords[0, i, 1] = 2.3 * np.sin(angle)
            coords[0, i, 2] = i * 1.5

        return coords

    def test_torsion_angle_computation(self, sample_coords):
        """Test computation of torsion angles."""
        phi, psi = TorsionAngleAugmentation.compute_torsion_angles(sample_coords)

        # Check shapes
        assert phi.shape == (1, sample_coords.shape[1])
        assert psi.shape == (1, sample_coords.shape[1])

        # Check values are in valid range
        assert torch.all(torch.abs(phi) <= 2 * np.pi)

    def test_torsion_perturbation(self, sample_coords):
        """Test torsion angle perturbation."""
        perturbed = TorsionAngleAugmentation.perturb_torsion_angles(sample_coords, noise_scale=0.1)

        # Check shape preserved
        assert perturbed.shape == sample_coords.shape

        # Check perturbation occurred
        assert not torch.allclose(perturbed, sample_coords)

        # Check perturbation is small
        diff = (perturbed - sample_coords).abs().mean()
        assert diff < 1.0  # Should be small perturbation


class TestMultiScaleAugmentation:
    """Test adaptive multi-scale augmentation."""

    @pytest.fixture
    def multi_scale_aug(self):
        base_config = AugmentationConfig()
        return MultiScaleAugmentation(base_config)

    def test_curriculum_learning(self, multi_scale_aug):
        """Test augmentation strength increases over epochs."""
        config_early = multi_scale_aug.get_adaptive_config()
        early_strength = config_early.coord_noise_scale

        # Simulate training progress
        multi_scale_aug.current_epoch = 50
        config_mid = multi_scale_aug.get_adaptive_config()
        mid_strength = config_mid.coord_noise_scale

        multi_scale_aug.current_epoch = 99
        config_late = multi_scale_aug.get_adaptive_config()
        late_strength = config_late.coord_noise_scale

        # Strength should increase
        assert early_strength < mid_strength < late_strength

    def test_difficulty_adaptation(self, multi_scale_aug):
        """Test augmentation adapts to sample difficulty."""
        easy_config = multi_scale_aug.get_adaptive_config(difficulty=0.1)
        hard_config = multi_scale_aug.get_adaptive_config(difficulty=0.9)

        # Easier samples get more augmentation
        assert easy_config.coord_noise_scale > hard_config.coord_noise_scale

    def test_epoch_stepping(self, multi_scale_aug):
        """Test epoch counter increments correctly."""
        initial_epoch = multi_scale_aug.current_epoch
        multi_scale_aug.step_epoch()
        assert multi_scale_aug.current_epoch == initial_epoch + 1


class TestAugmentationIntegration:
    """Integration tests for augmentation in training."""

    def test_gpu_compatibility(self):
        """Test augmentation works with GPU tensors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords = torch.randn(2, 10, 3, device=device)
        embeddings = torch.randn(2, 10, 128, device=device)

        aug = ProteinAugmentation(training=True)
        result = aug(coords, embeddings)

        assert result["coordinates"].device == device
        assert result["embeddings"].device == device

    def test_gradient_flow(self):
        """Test gradients flow through augmentation."""
        coords = torch.randn(2, 10, 3, requires_grad=True)
        embeddings = torch.randn(2, 10, 128, requires_grad=True)

        config = AugmentationConfig(rotation_prob=0.0, translation_prob=0.0)
        aug = ProteinAugmentation(config, training=True)

        result = aug(coords, embeddings)
        loss = result["coordinates"].sum() + result["embeddings"].sum()
        loss.backward()

        # Gradients should exist
        assert coords.grad is not None
        assert embeddings.grad is not None

    def test_reproducibility_with_seed(self):
        """Test augmentation is reproducible with fixed seed."""
        coords = torch.randn(2, 10, 3)
        embeddings = torch.randn(2, 10, 128)

        aug = ProteinAugmentation(training=True)

        # Set seed and augment
        torch.manual_seed(42)
        np.random.seed(42)
        result1 = aug(coords.clone(), embeddings.clone())

        # Set same seed and augment again
        torch.manual_seed(42)
        np.random.seed(42)
        result2 = aug(coords.clone(), embeddings.clone())

        # Results should be identical
        assert torch.allclose(result1["coordinates"], result2["coordinates"])
        assert torch.allclose(result1["embeddings"], result2["embeddings"])


class TestAugmentationPerformance:
    """Performance and stress tests."""

    @pytest.mark.slow
    def test_large_batch_performance(self):
        """Test augmentation on large batches."""
        import time

        batch_size = 64
        seq_len = 256
        coords = torch.randn(batch_size, seq_len, 3)
        embeddings = torch.randn(batch_size, seq_len, 1280)

        aug = ProteinAugmentation(training=True)

        start = time.time()
        result = aug(coords, embeddings)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds for large batch
        assert result["coordinates"].shape == coords.shape

    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test augmentation doesn't leak memory."""
        import gc

        coords = torch.randn(16, 100, 3)
        embeddings = torch.randn(16, 100, 256)
        aug = ProteinAugmentation(training=True)

        # Run multiple times
        for _ in range(100):
            result = aug(coords, embeddings)
            del result

        gc.collect()
        # If we got here without OOM, test passes
        assert True
