from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sllm.utils import ensure_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model_config,
        "train_config": train_config,
        "extra_state": extra_state or {},
    }
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
