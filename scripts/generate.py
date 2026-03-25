from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.checkpoint import load_checkpoint
from sllm.config import ModelConfig, load_json
from sllm.model import SLLMForCausalLM
from sllm.utils import get_device, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.json.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model-config", required=False, help="Optional path to model config JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, log_path = setup_logger("sllm.generate", Path("outputs/generate"), "generate")
    logger.info("Generation started")
    logger.info("Log file: %s", log_path)
    logger.info(
        "Arguments | checkpoint=%s tokenizer_dir=%s max_new_tokens=%s temperature=%s top_k=%s model_config=%s",
        args.checkpoint,
        args.tokenizer_dir,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.model_config,
    )
    device = get_device()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer_dir) / "tokenizer.json"))
    tokenizer_meta = load_json(Path(args.tokenizer_dir) / "tokenizer_meta.json")
    specials = tokenizer_meta["special_tokens"]

    payload = load_checkpoint(args.checkpoint, map_location=device)
    if args.model_config:
        model_config = ModelConfig.from_dict(load_json(args.model_config))
    else:
        model_config = ModelConfig.from_dict(payload["model_config"])

    model = SLLMForCausalLM(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    prompt_ids = [int(specials["bos_token_id"])] + tokenizer.encode(
        args.prompt,
        add_special_tokens=False,
    ).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=int(specials["eos_token_id"]),
        )

    decoded = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    logger.info("Generation finished | prompt_tokens=%s output_tokens=%s", len(prompt_ids), output_ids.shape[1])
    print(decoded)


if __name__ == "__main__":
    main()
