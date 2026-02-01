"""Distributed training utilities for multi-GPU and multi-node training.

Supports:
- PyTorch DistributedDataParallel (DDP)
- Gradient accumulation
- Automatic distributed setup
- Synchronized batch normalization
- Gradient synchronization utilities
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = 'nccl', init_method: Optional[str] = None) -> bool:
    """Initialize distributed training environment.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method (e.g., 'env://', 'tcp://...')
        
    Returns:
        True if distributed training is initialized, False otherwise
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        logger.info(
            f"Initializing distributed training: "
            f"rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )
        
        if not dist.is_initialized():
            if init_method is None:
                init_method = 'env://'
            
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
            
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        logger.info("Distributed training initialized successfully")
        return True
    
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce operation across all processes.
    
    Args:
        tensor: Input tensor
        op: Reduce operation (SUM, PRODUCT, MIN, MAX)
        
    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """Gather tensors from all processes.
    
    Args:
        tensor: Input tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary with tensor values
        average: Whether to average (True) or sum (False)
        
    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = [input_dict[k] for k in names]
        
        # Stack values
        values = torch.stack(values, dim=0)
        
        # All reduce
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        # Unstack
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


class GradientAccumulator:
    """Handles gradient accumulation for memory efficiency.
    
    Allows training with larger effective batch sizes than fit in memory.
    
    Args:
        accumulation_steps: Number of steps to accumulate gradients
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated (not synchronized)."""
        return (self.current_step + 1) % self.accumulation_steps != 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter."""
        self.current_step = 0
    
    @contextmanager
    def no_sync_context(self, model):
        """Context manager for accumulation steps (no gradient sync).
        
        Usage:
            with accumulator.no_sync_context(model):
                # Forward and backward without gradient sync
                loss.backward()
        """
        if self.should_accumulate() and isinstance(model, DDP):
            with model.no_sync():
                yield
        else:
            yield


def convert_to_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False
) -> torch.nn.Module:
    """Convert model to DistributedDataParallel.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Enable gradient as bucket view (memory efficient)
        static_graph: Enable static graph optimization
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning model as-is")
        return model
    
    if isinstance(model, DDP):
        logger.warning("Model already wrapped in DDP")
        return model
    
    if device_ids is None and torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]
    
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph
    )
    
    logger.info(f"Model converted to DDP with device_ids={device_ids}")
    return ddp_model


@contextmanager
def distributed_zero_first():
    """Context manager to execute code on rank 0 first, then others.
    
    Useful for dataset preparation where only rank 0 should download/process.
    """
    if is_main_process():
        yield
    
    barrier()
    
    if not is_main_process():
        yield


def save_on_master(state_dict: dict, filepath: str):
    """Save state dict only on master process.
    
    Args:
        state_dict: Dictionary to save
        filepath: Path to save to
    """
    if is_main_process():
        torch.save(state_dict, filepath)
        logger.info(f"Saved to {filepath}")


def synchronize():
    """Synchronize all processes (alias for barrier)."""
    barrier()
