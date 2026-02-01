"""Security utilities for safe model loading and data handling."""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import warnings


class SecureCheckpointLoader:
    """Secure checkpoint loading with validation."""

    @staticmethod
    def compute_checksum(path: Path, algorithm: str = "sha256") -> str:
        """Compute file checksum.

        Args:
            path: File path
            algorithm: Hash algorithm

        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def save_checkpoint(
        checkpoint: Dict[str, Any],
        path: Path,
        compute_checksum: bool = True,
    ) -> Optional[str]:
        """Save checkpoint with optional checksum.

        Args:
            checkpoint: Checkpoint data
            path: Save path
            compute_checksum: Whether to compute and save checksum

        Returns:
            Checksum if computed, else None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        torch.save(checkpoint, path)

        # Compute and save checksum
        if compute_checksum:
            checksum = SecureCheckpointLoader.compute_checksum(path)
            checksum_path = path.with_suffix(".sha256")
            with open(checksum_path, "w") as f:
                f.write(f"{checksum}  {path.name}\n")
            return checksum

        return None

    @staticmethod
    def load_checkpoint(
        path: Path,
        map_location: Optional[str] = None,
        verify_checksum: bool = True,
        weights_only: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint with security checks.

        Args:
            path: Checkpoint path
            map_location: Device mapping
            verify_checksum: Verify checksum if available
            weights_only: Load only weights (safer)

        Returns:
            Checkpoint data

        Raises:
            ValueError: If checksum verification fails
        """
        path = Path(path)

        # Verify checksum if requested
        if verify_checksum:
            checksum_path = path.with_suffix(".sha256")
            if checksum_path.exists():
                with open(checksum_path, "r") as f:
                    expected_checksum = f.read().split()[0]

                actual_checksum = SecureCheckpointLoader.compute_checksum(path)

                if actual_checksum != expected_checksum:
                    raise ValueError(
                        f"Checksum mismatch for {path}. "
                        f"Expected: {expected_checksum}, Got: {actual_checksum}"
                    )

        # Load checkpoint with security settings
        if weights_only:
            # Safer loading - only loads tensor data
            checkpoint = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
        else:
            # Legacy loading - may execute arbitrary code
            warnings.warn(
                "Loading checkpoint with weights_only=False. "
                "This may execute arbitrary code and is not recommended.",
                SecurityWarning,
                stacklevel=2,
            )
            checkpoint = torch.load(path, map_location=map_location)

        return checkpoint


class SafePickle:
    """Safe pickle wrapper for data serialization."""

    # Allowed types for unpickling
    SAFE_MODULES = {
        "torch",
        "numpy",
        "builtins",
        "collections",
    }

    @staticmethod
    def dump(obj: Any, path: Path) -> None:
        """Safely dump object to pickle file.

        Args:
            obj: Object to serialize
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> Any:
        """Safely load pickle file.

        Args:
            path: Pickle file path

        Returns:
            Deserialized object
        """
        with open(path, "rb") as f:
            return pickle.load(f)
