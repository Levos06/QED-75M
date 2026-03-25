from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {name}")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision.lower() not in {"bf16", "fp16"}:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=get_dtype(precision))


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def setup_logger(name: str, output_dir: str | Path, filename_prefix: str) -> tuple[logging.Logger, Path]:
    logs_dir = ensure_dir(Path(output_dir) / "logs")
    log_path = logs_dir / f"{filename_prefix}_{timestamp_for_filename()}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tokens_per_step(micro_batch_size: int, grad_accum_steps: int, seq_len: int) -> int:
    return micro_batch_size * grad_accum_steps * seq_len


def cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def maybe_enable_tf32(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def require_cuda_bf16(precision: str) -> None:
    if precision.lower() != "bf16":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 режим требует CUDA-устройство.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Текущий GPU или драйвер не поддерживает BF16.")


def resolve_runtime_precision(device: torch.device, requested_precision: str) -> tuple[str, str | None]:
    normalized = requested_precision.lower()
    if device.type == "cuda":
        if normalized == "bf16":
            require_cuda_bf16(normalized)
        return normalized, None

    if normalized in {"bf16", "fp16"}:
        return "fp32", (
            f"Precision '{requested_precision}' is not supported on device '{device.type}' "
            "in this pipeline; falling back to fp32."
        )

    return normalized, None


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def cuda_memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
    return {
        "allocated_gb": allocated_gb,
        "reserved_gb": reserved_gb,
        "max_allocated_gb": max_allocated_gb,
        "max_reserved_gb": max_reserved_gb,
    }
