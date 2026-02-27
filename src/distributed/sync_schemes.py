from contextlib import nullcontext
from typing import ContextManager

import torch


class SyncController:
    """Interface for communication/synchronization behavior in training."""

    def __init__(self, sync_every: int, is_distributed: bool):
        if sync_every < 1:
            raise ValueError("--sync_every must be >= 1")
        self.sync_every = sync_every
        self.is_distributed = is_distributed
        self.syncs_this_epoch = 0

    def start_epoch(self, optimizer: torch.optim.Optimizer) -> None:
        self.syncs_this_epoch = 0
        optimizer.zero_grad(set_to_none=True)

    def backward_context(self, model: torch.nn.Module) -> ContextManager:
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def maybe_step(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        return False

    def finalize_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stop_requested: bool,
    ) -> bool:
        return False


class DeferredAllReduceSync(SyncController):
    """DDP gradient allreduce every k micro-steps via no_sync accumulation."""

    def __init__(self, sync_every: int, is_distributed: bool):
        super().__init__(sync_every=sync_every, is_distributed=is_distributed)
        self.micro_steps_since_sync = 0

    def start_epoch(self, optimizer: torch.optim.Optimizer) -> None:
        super().start_epoch(optimizer)
        self.micro_steps_since_sync = 0

    def _should_sync_next(self) -> bool:
        return ((self.micro_steps_since_sync + 1) % self.sync_every == 0)

    def backward_context(self, model: torch.nn.Module) -> ContextManager:
        if self.is_distributed and hasattr(model, "no_sync") and not self._should_sync_next():
            return model.no_sync()
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        # Keep gradient magnitude consistent with large-batch accumulation.
        (loss / self.sync_every).backward()

    def maybe_step(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        self.micro_steps_since_sync += 1
        if not (self.micro_steps_since_sync % self.sync_every == 0):
            return False
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.syncs_this_epoch += 1
        return True

    def finalize_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stop_requested: bool,
    ) -> bool:
        if stop_requested:
            return False
        remainder = self.micro_steps_since_sync % self.sync_every
        if remainder == 0:
            return False
        # Flush a partial accumulation window at end-of-epoch.
        scale = self.sync_every / remainder
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.syncs_this_epoch += 1
        return True


class LocalSGDSync(SyncController):
    """Skeleton for periodic parameter averaging (LocalSGD)."""

    def __init__(self, sync_every: int, is_distributed: bool):
        super().__init__(sync_every=sync_every, is_distributed=is_distributed)
        if not is_distributed:
            raise ValueError("local_sgd requires distributed training")
        raise NotImplementedError(
            "LocalSGD is not implemented yet. "
            "Use --sync_scheme deferred_allreduce for now."
        )


def build_sync_controller(
    scheme: str, sync_every: int, is_distributed: bool
) -> SyncController:
    if scheme == "deferred_allreduce":
        return DeferredAllReduceSync(sync_every=sync_every, is_distributed=is_distributed)
    if scheme == "local_sgd":
        return LocalSGDSync(sync_every=sync_every, is_distributed=is_distributed)
    raise ValueError(f"unsupported --sync_scheme: {scheme}")
