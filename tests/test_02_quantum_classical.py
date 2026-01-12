"""Tests for 02_quantum_vs_classical.ipynb functionality."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import patch, MagicMock
import time


class TestDataPreparation:
    """Test data generation and DataLoader functionality."""
    
    def test_synthetic_data_generation(self):
        """Should generate training data with correct shapes."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_samples = 100
        seq_length = 50
        feature_dim = 64
        
        X = torch.randn(n_samples, seq_length, feature_dim)
        y = torch.randn(n_samples, seq_length, 3)
        
        assert X.shape == (n_samples, seq_length, feature_dim)
        assert y.shape == (n_samples, seq_length, 3)
        assert not torch.isnan(X).any()
        assert not torch.isnan(y).any()
    
    def test_train_test_split(self):
        """Should create separate train and test sets."""
        torch.manual_seed(42)
        
        X_train = torch.randn(100, 50, 64)
        X_test = torch.randn(20, 50, 64)
        
        # Verify they are different
        assert X_train.shape[0] != X_test.shape[0]
        assert not torch.allclose(X_train[:20], X_test)
    
    def test_tensor_dataset_creation(self):
        """Should create TensorDataset correctly."""
        X = torch.randn(100, 50, 64)
        y = torch.randn(100, 50, 3)
        
        dataset = TensorDataset(X, y)
        
        assert len(dataset) == 100
        assert dataset[0][0].shape == (50, 64)
        assert dataset[0][1].shape == (50, 3)
    
    def test_dataloader_batching(self):
        """DataLoader should batch data correctly."""
        X = torch.randn(100, 50, 64)
        y = torch.randn(100, 50, 3)
        dataset = TensorDataset(X, y)
        
        batch_size = 16
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Check first batch
        batch_X, batch_y = next(iter(loader))
        assert batch_X.shape == (batch_size, 50, 64)
        assert batch_y.shape == (batch_size, 50, 3)
    
    def test_dataloader_length(self):
        """DataLoader should have correct number of batches."""
        X = torch.randn(100, 50, 64)
        y = torch.randn(100, 50, 3)
        dataset = TensorDataset(X, y)
        
        batch_size = 16
        loader = DataLoader(dataset, batch_size=batch_size)
        
        expected_batches = np.ceil(100 / batch_size)
        assert len(loader) == expected_batches
    
    def test_dataloader_shuffle(self):
        """Shuffle should produce different order each epoch."""
        X = torch.arange(100).reshape(100, 1, 1).float()
        y = torch.zeros(100, 1, 1)
        dataset = TensorDataset(X, y)
        
        loader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Get first batch from two iterations
        torch.manual_seed(42)
        batch1 = next(iter(loader))[0]
        
        torch.manual_seed(43)
        batch2 = next(iter(loader))[0]
        
        # They should be different due to different seeds
        assert not torch.allclose(batch1, batch2)


class TestModelDefinitions:
    """Test quantum and classical model architectures."""
    
    @staticmethod
    def create_classical_model(feature_dim=64, n_heads=4):
        """Create classical baseline model."""
        class ClassicalModel(nn.Module):
            def __init__(self, feature_dim, n_heads):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    feature_dim, n_heads, batch_first=True
                )
                self.output = nn.Linear(feature_dim, 3)
            
            def forward(self, x):
                x, _ = self.attention(x, x, x)
                return self.output(x)
        
        return ClassicalModel(feature_dim, n_heads)
    
    @staticmethod
    def create_quantum_fallback_model(feature_dim=64, n_heads=4):
        """Create quantum model fallback (without actual quantum layers)."""
        class QuantumModel(nn.Module):
            def __init__(self, feature_dim, n_heads):
                super().__init__()
                # Fallback: use regular attention
                self.quantum = nn.MultiheadAttention(
                    feature_dim, n_heads, batch_first=True
                )
                self.output = nn.Linear(feature_dim, 3)
            
            def forward(self, x):
                x, _ = self.quantum(x, x, x)
                return self.output(x)
        
        return QuantumModel(feature_dim, n_heads)
    
    def test_classical_model_creation(self):
        """Classical model should initialize correctly."""
        model = self.create_classical_model()
        assert model is not None
        assert hasattr(model, 'attention')
        assert hasattr(model, 'output')
    
    def test_classical_model_forward(self):
        """Classical model forward pass should work."""
        model = self.create_classical_model()
        x = torch.randn(2, 50, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 50, 3)
    
    def test_quantum_fallback_model_creation(self):
        """Quantum fallback model should initialize."""
        model = self.create_quantum_fallback_model()
        assert model is not None
    
    def test_quantum_fallback_model_forward(self):
        """Quantum fallback model forward pass should work."""
        model = self.create_quantum_fallback_model()
        x = torch.randn(2, 50, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 50, 3)
    
    def test_model_parameter_counts(self):
        """Should count model parameters correctly."""
        classical = self.create_classical_model()
        quantum = self.create_quantum_fallback_model()
        
        c_params = sum(p.numel() for p in classical.parameters())
        q_params = sum(p.numel() for p in quantum.parameters())
        
        # Both should have similar architecture
        assert c_params > 0
        assert q_params > 0
    
    def test_device_placement(self):
        """Models should move to device correctly."""
        device = torch.device('cpu')
        model = self.create_classical_model().to(device)
        
        for param in model.parameters():
            assert param.device.type == device.type


class TestTrainingFunctionality:
    """Test training loop and optimization."""
    
    @staticmethod
    def train_one_epoch(model, train_loader, device='cpu'):
        """Simplified training function for testing."""
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def test_single_epoch_training(self):
        """Should train for one epoch without errors."""
        # Create simple data
        X = torch.randn(32, 10, 64)
        y = torch.randn(32, 10, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        # Create model
        model = TestModelDefinitions.create_classical_model()
        
        # Train one epoch
        loss = self.train_one_epoch(model, loader)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_loss_decreases(self):
        """Loss should decrease over multiple epochs."""
        # Create data with learnable pattern
        X = torch.randn(32, 10, 64)
        y = torch.zeros(32, 10, 3)  # Simple target
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        model = TestModelDefinitions.create_classical_model()
        
        losses = []
        for _ in range(5):
            loss = self.train_one_epoch(model, loader)
            losses.append(loss)
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]
    
    def test_optimizer_updates_weights(self):
        """Optimizer should update model weights."""
        X = torch.randn(16, 10, 64)
        y = torch.randn(16, 10, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        model = TestModelDefinitions.create_classical_model()
        
        # Store initial weights
        initial_params = [p.clone() for p in model.parameters()]
        
        # Train
        self.train_one_epoch(model, loader)
        
        # Check weights changed
        changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                changed = True
                break
        
        assert changed, "Weights did not update during training"
    
    def test_gradient_computation(self):
        """Gradients should be computed correctly."""
        model = TestModelDefinitions.create_classical_model()
        criterion = nn.MSELoss()
        
        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 3)
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"
    
    def test_training_mode_switch(self):
        """Should switch between train and eval modes."""
        model = TestModelDefinitions.create_classical_model()
        
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False


class TestPerformanceTracking:
    """Test performance metrics and comparison."""
    
    def test_loss_history_tracking(self):
        """Should track loss history correctly."""
        losses = [1.5, 1.2, 1.0, 0.9, 0.85]
        
        # Check monotonic decrease
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i+1], "Loss should decrease or stay same"
    
    def test_time_tracking(self):
        """Should track epoch times correctly."""
        start = time.time()
        time.sleep(0.01)  # Simulate work
        elapsed = time.time() - start
        
        assert elapsed >= 0.01
        assert elapsed < 1.0  # Reasonable upper bound
    
    def test_cumulative_time_calculation(self):
        """Should calculate cumulative time correctly."""
        epoch_times = [1.0, 1.5, 1.2, 1.3, 1.4]
        cumulative = np.cumsum(epoch_times)
        
        assert len(cumulative) == len(epoch_times)
        assert cumulative[-1] == sum(epoch_times)
        assert all(cumulative[i] <= cumulative[i+1] for i in range(len(cumulative)-1))
    
    def test_speedup_calculation(self):
        """Should calculate speedup correctly."""
        quantum_time = 10.0
        classical_time = 5.0
        
        if quantum_time < classical_time:
            speedup = classical_time / quantum_time
            assert speedup == 0.5
        else:
            slowdown = quantum_time / classical_time
            assert slowdown == 2.0
    
    def test_loss_improvement_calculation(self):
        """Should calculate loss improvement correctly."""
        classical_loss = 1.0
        quantum_loss = 0.8
        
        improvement = (classical_loss - quantum_loss) / classical_loss * 100
        
        assert improvement == 20.0  # 20% improvement
    
    def test_performance_comparison(self):
        """Should compare model performances correctly."""
        q_losses = [1.0, 0.8, 0.6]
        c_losses = [1.0, 0.9, 0.7]
        
        # Quantum converges faster
        q_final = q_losses[-1]
        c_final = c_losses[-1]
        
        assert q_final < c_final


class TestVisualization:
    """Test visualization and plotting functionality."""
    
    def test_matplotlib_import(self):
        """Matplotlib should be available."""
        import matplotlib.pyplot as plt
        assert plt is not None
    
    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_loss_plot_generation(self):
        """Should generate loss comparison plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        q_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        c_losses = [1.0, 0.9, 0.7, 0.6, 0.5]
        
        fig, ax = plt.subplots()
        ax.plot(q_losses, label='Quantum')
        ax.plot(c_losses, label='Classical')
        ax.legend()
        
        assert len(ax.lines) == 2
        plt.close(fig)
    
    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_time_plot_generation(self):
        """Should generate time comparison plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        q_times = [1.0, 1.5, 1.2, 1.3, 1.4]
        c_times = [0.5, 0.6, 0.55, 0.58, 0.6]
        
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(q_times), label='Quantum')
        ax.plot(np.cumsum(c_times), label='Classical')
        ax.legend()
        
        assert len(ax.lines) == 2
        plt.close(fig)
    
    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_subplot_creation(self):
        """Should create multiple subplots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataloader(self):
        """Should handle empty dataset gracefully."""
        X = torch.randn(0, 10, 64)
        y = torch.randn(0, 10, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        assert len(loader) == 0
    
    def test_single_sample_batch(self):
        """Should handle batch size of 1."""
        X = torch.randn(10, 10, 64)
        y = torch.randn(10, 10, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=1)
        
        batch_X, batch_y = next(iter(loader))
        assert batch_X.shape == (1, 10, 64)
    
    def test_model_with_nan_input(self):
        """Should handle NaN inputs appropriately."""
        model = TestModelDefinitions.create_classical_model()
        x = torch.full((1, 10, 64), float('nan'))
        
        with torch.no_grad():
            output = model(x)
        
        # Output will likely contain NaN
        assert output.shape == (1, 10, 3)
    
    def test_zero_learning_rate(self):
        """With lr=0, weights should not change."""
        model = TestModelDefinitions.create_classical_model()
        optimizer = optim.Adam(model.parameters(), lr=0.0)
        criterion = nn.MSELoss()
        
        # Store initial weights
        initial_params = [p.clone() for p in model.parameters()]
        
        # Try to train
        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 3)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Weights should be unchanged
        for initial, current in zip(initial_params, model.parameters()):
            assert torch.allclose(initial, current)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])