from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.checkpoint import load_checkpoint, save_checkpoint
from sllm.config import ModelConfig, SFTConfig, load_json, save_json
from sllm.data import FixedSFTDataset
from sllm.model import SLLMForCausalLM
from sllm.utils import (
    append_jsonl,
    autocast_context,
    cosine_lr,
    cuda_memory_snapshot,
    ensure_dir,
    format_number,
    get_device,
    iso_timestamp,
    maybe_enable_tf32,
    model_parameter_count,
    resolve_runtime_precision,
    set_optimizer_lr,
    set_seed,
    setup_logger,
    timestamp,
    tokens_per_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning for the sLLM.")
    parser.add_argument("--model-config", required=True, help="Path to model JSON config.")
    parser.add_argument("--train-config", required=True, help="Path to SFT JSON config.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional debug override.")
    return parser


def build_optimizer(model: torch.nn.Module, config: SFTConfig, device: torch.device):
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias"):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(model: SLLMForCausalLM, loader: DataLoader, device: torch.device, precision: str, max_batches: int):
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_context(device, precision):
            loss = model(**batch)["loss"]
        losses.append(loss.detach().float().item())
    model.train()
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return mean_loss, math.exp(min(mean_loss, 20))


def save_run_config(output_dir: Path, model_config: ModelConfig, train_config: SFTConfig) -> None:
    save_json(
        output_dir / "run_config.json",
        {
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    model_config = ModelConfig.from_dict(load_json(args.model_config))
    train_config = SFTConfig.from_dict(load_json(args.train_config))
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps

    set_seed(train_config.seed)
    device = get_device()
    maybe_enable_tf32(device)
    runtime_precision, precision_warning = resolve_runtime_precision(device, train_config.precision)
    train_config.precision = runtime_precision

    output_dir = ensure_dir(train_config.output_dir)
    checkpoint_dir = ensure_dir(train_config.checkpoint_dir)
    logger, log_path = setup_logger("sllm.train_sft", output_dir, "train_sft")
    metrics_path = Path(output_dir) / "logs" / f"{log_path.stem}.jsonl"
    logger.info("SFT training started")
    logger.info("Log file: %s", log_path)
    logger.info("Metrics JSONL: %s", metrics_path)
    logger.info("Arguments | model_config=%s train_config=%s max_steps_override=%s", args.model_config, args.train_config, args.max_steps)
    if precision_warning is not None:
        logger.warning(precision_warning)
    logger.info("Model config | %s", model_config.to_dict())
    logger.info("SFT config | %s", train_config.to_dict())
    append_jsonl(
        metrics_path,
        {
            "event": "run_started",
            "timestamp": iso_timestamp(),
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "args": {
                "model_config": args.model_config,
                "train_config": args.train_config,
                "max_steps_override": args.max_steps,
            },
        },
    )
    save_run_config(output_dir, model_config, train_config)

    train_dataset = FixedSFTDataset(train_config.dataset_path, split="train")
    val_dataset = FixedSFTDataset(train_config.dataset_path, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SLLMForCausalLM(model_config).to(device)
    if train_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = build_optimizer(model, train_config, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and train_config.precision.lower() == "fp16",
    )

    start_step = 0
    checkpoint_path = train_config.resume_from or train_config.init_from
    if checkpoint_path:
        payload = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"])
        if train_config.resume_from and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            logger.info("Resumed SFT | step=%s checkpoint=%s", start_step, checkpoint_path)
            append_jsonl(
                metrics_path,
                {
                    "event": "resumed",
                    "timestamp": iso_timestamp(),
                    "step": start_step,
                    "checkpoint": checkpoint_path,
                },
            )
        else:
            logger.info("Loaded initialization weights | checkpoint=%s", checkpoint_path)
            append_jsonl(
                metrics_path,
                {
                    "event": "initialized_from_checkpoint",
                    "timestamp": iso_timestamp(),
                    "checkpoint": checkpoint_path,
                },
            )

    model.train()
    tokens_step = tokens_per_step(
        train_config.micro_batch_size,
        train_config.grad_accum_steps,
        train_config.seq_len,
    )
    logger.info("Device summary | device=%s precision=%s compile_model=%s", device, train_config.precision, train_config.compile_model)
    logger.info("Model summary | parameters=%s", format_number(model_parameter_count(model)))
    logger.info(
        "Batch summary | seq_len=%s micro_batch_size=%s grad_accum_steps=%s tokens_per_step=%s",
        train_config.seq_len,
        train_config.micro_batch_size,
        train_config.grad_accum_steps,
        f"{tokens_step:,}",
    )
    logger.info(
        "Dataset summary | dataset_path=%s train_examples=%s val_examples=%s",
        train_config.dataset_path,
        len(train_dataset),
        len(val_dataset),
    )
    append_jsonl(
        metrics_path,
        {
            "event": "runtime_summary",
            "timestamp": iso_timestamp(),
            "device": str(device),
            "precision": train_config.precision,
            "compile_model": train_config.compile_model,
            "parameters": model_parameter_count(model),
            "seq_len": train_config.seq_len,
            "micro_batch_size": train_config.micro_batch_size,
            "grad_accum_steps": train_config.grad_accum_steps,
            "tokens_per_step": tokens_step,
            "dataset_path": train_config.dataset_path,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
        },
    )
    running_loss = 0.0
    log_start_time = time.perf_counter()
    train_iterator = iter(train_loader)
    last_grad_norm = float("nan")

    for step in range(start_step, train_config.max_steps):
        lr = cosine_lr(
            step=step,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            max_lr=train_config.learning_rate,
            min_lr=train_config.min_lr,
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(train_config.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
            with autocast_context(device, train_config.precision):
                loss = model(**batch)["loss"] / train_config.grad_accum_steps
            step_loss += loss.detach().float().item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if train_config.grad_clip and train_config.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            last_grad_norm = float(grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += step_loss

        if (step + 1) % train_config.log_interval == 0:
            elapsed = time.perf_counter() - log_start_time
            avg_loss = running_loss / train_config.log_interval
            tok_per_sec = (tokens_step * train_config.log_interval) / max(elapsed, 1e-6)
            memory = cuda_memory_snapshot(device)
            memory_suffix = ""
            if memory:
                memory_suffix = (
                    f" mem_alloc_gb={memory['allocated_gb']:.2f}"
                    f" mem_reserved_gb={memory['reserved_gb']:.2f}"
                    f" max_mem_alloc_gb={memory['max_allocated_gb']:.2f}"
                    f" max_mem_reserved_gb={memory['max_reserved_gb']:.2f}"
                )
            logger.info(
                "Train step | step=%s loss=%.4f lr=%.6f tok_per_sec=%s grad_norm=%.4f%s",
                step + 1,
                avg_loss,
                lr,
                f"{tok_per_sec:,.0f}",
                last_grad_norm,
                memory_suffix,
            )
            append_jsonl(
                metrics_path,
                {
                    "event": "train",
                    "timestamp": iso_timestamp(),
                    "step": step + 1,
                    "loss": avg_loss,
                    "lr": lr,
                    "tok_per_sec": tok_per_sec,
                    "grad_norm": last_grad_norm,
                    "tokens_seen": (step + 1) * tokens_step,
                    "elapsed_sec": elapsed,
                    "seq_len": train_config.seq_len,
                    "micro_batch_size": train_config.micro_batch_size,
                    "grad_accum_steps": train_config.grad_accum_steps,
                    **memory,
                },
            )
            running_loss = 0.0
            log_start_time = time.perf_counter()

        if (step + 1) % train_config.eval_interval == 0:
            val_loss, val_ppl = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                precision=train_config.precision,
                max_batches=train_config.eval_batches,
            )
            logger.info("Eval step | step=%s val_loss=%.4f perplexity=%.2f", step + 1, val_loss, val_ppl)
            append_jsonl(
                metrics_path,
                {
                    "event": "eval",
                    "timestamp": iso_timestamp(),
                    "step": step + 1,
                    "val_loss": val_loss,
                    "perplexity": val_ppl,
                    "eval_batches": train_config.eval_batches,
                },
            )

        if (step + 1) % train_config.save_interval == 0 or (step + 1) == train_config.max_steps:
            step_checkpoint_path = checkpoint_dir / f"step_{step + 1:07d}.pt"
            last_checkpoint_path = checkpoint_dir / "last.pt"
            save_checkpoint(
                step_checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                model_config=model_config.to_dict(),
                train_config=train_config.to_dict(),
                extra_state={"tokens_seen": (step + 1) * tokens_step},
            )
            save_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                model_config=model_config.to_dict(),
                train_config=train_config.to_dict(),
                extra_state={"tokens_seen": (step + 1) * tokens_step},
            )
            logger.info(
                "Checkpoint saved | step=%s step_checkpoint=%s last_checkpoint=%s",
                step + 1,
                step_checkpoint_path,
                last_checkpoint_path,
            )
            append_jsonl(
                metrics_path,
                {
                    "event": "checkpoint",
                    "timestamp": iso_timestamp(),
                    "step": step + 1,
                    "step_checkpoint": str(step_checkpoint_path),
                    "last_checkpoint": str(last_checkpoint_path),
                    "tokens_seen": (step + 1) * tokens_step,
                },
            )

    append_jsonl(
        metrics_path,
        {
            "event": "run_finished",
            "timestamp": iso_timestamp(),
            "final_step": train_config.max_steps,
            "tokens_seen": train_config.max_steps * tokens_step,
        },
    )


if __name__ == "__main__":
    main()
