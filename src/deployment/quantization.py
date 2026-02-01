"""Model quantization for efficient deployment.

Supports:
- INT8 quantization
- FP16 conversion  
- Dynamic quantization
- Static quantization with calibration
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional
from pathlib import Path


class ModelQuantizer:
    """Quantize models for deployment."""

    @staticmethod
    def quantize_dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        modules_to_quantize: Optional[set] = None,
    ) -> nn.Module:
        """Apply dynamic quantization to model.

        Args:
            model: Model to quantize
            dtype: Target quantization dtype
            modules_to_quantize: Module types to quantize (default: Linear)

        Returns:
            Quantized model
        """
        if modules_to_quantize is None:
            modules_to_quantize = {nn.Linear}

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            modules_to_quantize,
            dtype=dtype,
        )
        return quantized_model

    @staticmethod
    def convert_to_fp16(model: nn.Module) -> nn.Module:
        """Convert model to FP16.

        Args:
            model: Model to convert

        Returns:
            FP16 model
        """
        return model.half()

    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: Tensor,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """Apply static quantization with calibration.

        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            backend: Quantization backend

        Returns:
            Statically quantized model
        """
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with data
        with torch.no_grad():
            model_prepared(calibration_data)
        
        # Convert to quantized version
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized

    @staticmethod
    def save_quantized(
        model: nn.Module,
        path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save quantized model.

        Args:
            model: Quantized model
            path: Save path
            metadata: Optional metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, path)

    @staticmethod
    def load_quantized(
        model_class: type,
        path: Path,
        **model_kwargs,
    ) -> nn.Module:
        """Load quantized model.

        Args:
            model_class: Model class
            path: Model path
            **model_kwargs: Model initialization arguments

        Returns:
            Loaded quantized model
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
