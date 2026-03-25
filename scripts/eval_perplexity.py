from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.checkpoint import load_checkpoint
from sllm.config import ModelConfig, load_json
from sllm.data import SequentialEvalDataset
from sllm.model import SLLMForCausalLM
from sllm.utils import autocast_context, get_device, resolve_runtime_precision, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate perplexity on validation shards.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--model-config", required=False, help="Optional model config JSON path.")
    parser.add_argument("--data-dir", required=True, help="Validation root directory.")
    parser.add_argument("--seq-len", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--precision", default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, log_path = setup_logger("sllm.eval_perplexity", Path("outputs/eval"), "eval_perplexity")
    logger.info("Perplexity evaluation started")
    logger.info("Log file: %s", log_path)
    logger.info("Arguments | checkpoint=%s model_config=%s data_dir=%s seq_len=%s batch_size=%s batches=%s precision=%s", args.checkpoint, args.model_config, args.data_dir, args.seq_len, args.batch_size, args.batches, args.precision)
    device = get_device()
    runtime_precision, precision_warning = resolve_runtime_precision(device, args.precision)
    if precision_warning is not None:
        logger.warning(precision_warning)
    payload = load_checkpoint(args.checkpoint, map_location=device)
    if args.model_config:
        model_config = ModelConfig.from_dict(load_json(args.model_config))
    else:
        model_config = ModelConfig.from_dict(payload["model_config"])

    model = SLLMForCausalLM(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    dataset = SequentialEvalDataset(
        data_dir=args.data_dir,
        split="val",
        seq_len=args.seq_len,
        max_batches=args.batches * args.batch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, runtime_precision):
                loss = model(**batch)["loss"]
            losses.append(loss.detach().float().item())

    mean_loss = float(sum(losses) / max(1, len(losses)))
    perplexity = math.exp(min(mean_loss, 20))
    logger.info("Perplexity evaluation finished | val_loss=%.4f perplexity=%.2f", mean_loss, perplexity)
    print(f"val_loss={mean_loss:.4f}")
    print(f"perplexity={perplexity:.2f}")


if __name__ == "__main__":
    main()
