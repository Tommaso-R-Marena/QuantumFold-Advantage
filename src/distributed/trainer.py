"""Distributed training utilities.

Supports:
- DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- Multi-node training
- Gradient accumulation
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedTrainer:
    """Distributed training manager."""

    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
    ):
        self.backend = backend
        self.init_method = init_method
        self.is_initialized = False

    def setup(self, rank: int, world_size: int) -> None:
        """Initialize distributed training.

        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            rank=rank,
            world_size=world_size,
        )
        self.is_initialized = True

    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False

    @staticmethod
    def wrap_model_ddp(
        model: nn.Module,
        device_id: int,
        find_unused_parameters: bool = False,
    ) -> DDP:
        """Wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap
            device_id: GPU device ID
            find_unused_parameters: Find unused parameters

        Returns:
            DDP-wrapped model
        """
        model = model.to(device_id)
        ddp_model = DDP(
            model,
            device_ids=[device_id],
            find_unused_parameters=find_unused_parameters,
        )
        return ddp_model

    @staticmethod
    def wrap_model_fsdp(
        model: nn.Module,
        min_num_params: int = 1e6,
        mixed_precision: bool = False,
    ) -> FSDP:
        """Wrap model with Fully Sharded Data Parallel.

        Args:
            model: Model to wrap
            min_num_params: Minimum parameters for sharding
            mixed_precision: Use mixed precision

        Returns:
            FSDP-wrapped model
        """
        auto_wrap_policy = size_based_auto_wrap_policy
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            min_num_params=min_num_params,
        )
        return fsdp_model

    @staticmethod
    def get_rank() -> int:
        """Get current process rank."""
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def get_world_size() -> int:
        """Get total number of processes."""
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def is_main_process() -> bool:
        """Check if current process is main."""
        return DistributedTrainer.get_rank() == 0

    @staticmethod
    def barrier() -> None:
        """Synchronization barrier."""
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('mean', 'sum')

        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if op == "mean":
            tensor /= dist.get_world_size()

        return tensor
