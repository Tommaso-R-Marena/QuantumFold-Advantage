"""Tests for data loading utilities."""

import pytest

try:
    from src.data.casp16_loader import CASP16Config, CASP16Loader, CASP16Target

    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data module not available")
class TestCASP16Config:
    """Tests for CASP16Config."""

    def test_config_attributes(self):
        """Test config has required attributes."""
        assert hasattr(CASP16Config, "BASE_URL")
        assert hasattr(CASP16Config, "CACHE_DIR")
        assert hasattr(CASP16Config, "TARGET_CATEGORIES")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data module not available")
class TestCASP16Target:
    """Tests for CASP16Target."""

    def test_initialization(self):
        """Test target initialization."""
        target = CASP16Target(
            target_id="T1104",
            sequence="MKTAYIAK",
        )
        assert target.target_id == "T1104"
        assert target.length == 8
        assert not target.has_structure

    def test_with_structure_path(self, tmp_path):
        """Test target with structure path."""
        structure_file = tmp_path / "test.pdb"
        structure_file.touch()

        target = CASP16Target(
            target_id="T1104",
            sequence="MKTAYIAK",
            structure_path=structure_file,
        )
        assert target.has_structure


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data module not available")
class TestCASP16Loader:
    """Tests for CASP16Loader."""

    def test_initialization(self, tmp_path):
        """Test loader initialization."""
        loader = CASP16Loader(
            cache_dir=tmp_path,
            download=False,
            verbose=False,
        )
        assert loader.cache_dir == tmp_path
        assert not loader.download

    def test_list_available_targets(self, tmp_path):
        """Test listing targets."""
        loader = CASP16Loader(
            cache_dir=tmp_path,
            download=False,
            verbose=False,
        )
        targets = loader.list_available_targets()
        assert isinstance(targets, list)
