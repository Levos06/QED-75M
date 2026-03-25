from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.config import load_json, save_json
from sllm.data import SFTShardWriter
from sllm.utils import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fixed-length SFT tensors.")
    parser.add_argument("--config", required=True, help="Path to SFT data JSON config.")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.json and metadata.")
    parser.add_argument("--output-dir", required=True, help="Directory to store processed SFT tensors.")
    parser.add_argument("--seq-len", type=int, default=2_048, help="Packed example length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    return parser


def load_tokenizer(tokenizer_dir: str | Path) -> tuple[Tokenizer, dict]:
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    metadata = load_json(tokenizer_dir / "tokenizer_meta.json")
    return tokenizer, metadata


def row_to_messages(row: dict, config: dict) -> list[dict[str, str]]:
    fmt = config.get("format", "messages")
    if fmt == "messages":
        messages = row.get(config.get("messages_field", "messages"))
        if not isinstance(messages, list):
            raise ValueError("Не найден список сообщений в SFT-датасете.")
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, list):
                parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                content = "\n".join(part for part in parts if part)
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content.strip()})
        return normalized

    if fmt == "prompt_response":
        prompt = row.get(config.get("prompt_field", "prompt"))
        response = row.get(config.get("response_field", "response"))
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("Не найдены поля prompt/response в SFT-датасете.")
        system_prompt = config.get("system_prompt")
        messages = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        messages.append({"role": "assistant", "content": response.strip()})
        return messages

    if fmt == "alpaca":
        instruction = row.get(config.get("instruction_field", "instruction"))
        input_text = row.get(config.get("input_field", "input"), "")
        output_text = row.get(config.get("output_field", "output"))
        if not isinstance(instruction, str) or not isinstance(output_text, str):
            raise ValueError("Не найдены поля instruction/output в Alpaca-подобном датасете.")
        prompt = instruction.strip()
        if isinstance(input_text, str) and input_text.strip():
            prompt = f"{prompt}\n\n{input_text.strip()}"
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output_text.strip()},
        ]

    raise ValueError(f"Unsupported SFT format: {fmt}")


def tokenize_messages(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    bos_id: int,
    eos_id: int,
) -> tuple[list[int], list[int]]:
    input_ids = [bos_id]
    labels = [-100]

    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        if not content:
            continue
        text = f"<|{role}|>\n{content}\n"
        piece = tokenizer.encode(text, add_special_tokens=False).ids
        if not piece:
            continue
        input_ids.extend(piece)
        if role == "assistant":
            labels.extend(piece)
        else:
            labels.extend([-100] * len(piece))

    input_ids.append(eos_id)
    labels.append(eos_id)
    return input_ids, labels


def pad_or_truncate(
    input_ids: list[int],
    labels: list[int],
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]
    if len(input_ids) < seq_len:
        pad_length = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_length
        labels = labels + [-100] * pad_length
    return input_ids, labels


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    tokenizer, tokenizer_meta = load_tokenizer(args.tokenizer_dir)
    specials = tokenizer_meta["special_tokens"]
    bos_id = int(specials["bos_token_id"])
    eos_id = int(specials["eos_token_id"])
    pad_id = int(specials["pad_token_id"])

    dataset = load_dataset(
        path=config["path"],
        name=config.get("config_name"),
        split=config.get("split", "train"),
        revision=config.get("revision"),
        streaming=bool(config.get("streaming", False)),
    )
    if config.get("shuffle", True):
        dataset = dataset.shuffle(seed=args.seed)

    val_examples = int(config.get("val_examples", 1_000))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger("sllm.prepare_sft_data", output_dir, "prepare_sft_data")
    logger.info("SFT data preparation started")
    logger.info("Log file: %s", log_path)
    logger.info(
        "Arguments | config=%s tokenizer_dir=%s output_dir=%s seq_len=%s seed=%s",
        args.config,
        args.tokenizer_dir,
        args.output_dir,
        args.seq_len,
        args.seed,
    )
    logger.info(
        "SFT source config | path=%s config_name=%s split=%s format=%s streaming=%s val_examples=%s max_train_examples=%s",
        config.get("path"),
        config.get("config_name"),
        config.get("split", "train"),
        config.get("format", "messages"),
        bool(config.get("streaming", False)),
        val_examples,
        config.get("max_train_examples"),
    )
    train_writer = SFTShardWriter(output_dir, prefix="train", seq_len=args.seq_len)
    val_writer = SFTShardWriter(output_dir, prefix="val", seq_len=args.seq_len)

    train_count = 0
    val_count = 0
    max_train_examples = config.get("max_train_examples")

    for row in dataset:
        messages = row_to_messages(row, config)
        if not messages:
            continue
        input_ids, labels = tokenize_messages(tokenizer, messages, bos_id=bos_id, eos_id=eos_id)
        input_ids, labels = pad_or_truncate(input_ids, labels, args.seq_len, pad_id=pad_id)

        if val_count < val_examples:
            val_writer.add_example(input_ids, labels)
            val_count += 1
        else:
            train_writer.add_example(input_ids, labels)
            train_count += 1

        total_examples = train_count + val_count
        if total_examples % 5_000 == 0:
            logger.info(
                "SFT progress | processed=%s train_examples=%s val_examples=%s",
                f"{total_examples:,}",
                f"{train_count:,}",
                f"{val_count:,}",
            )

        if max_train_examples is not None and train_count >= int(max_train_examples):
            break

    train_metadata = train_writer.finalize()
    val_metadata = val_writer.finalize()
    save_json(
        output_dir / "dataset_summary.json",
        {
            "config": config,
            "tokenizer_meta": tokenizer_meta,
            "train": train_metadata,
            "val": val_metadata,
        },
    )
    logger.info("SFT dataset saved | output_dir=%s", output_dir)
    logger.info("SFT summary | train_examples=%s val_examples=%s", f"{train_count:,}", f"{val_count:,}")
    logger.info("SFT metadata saved | path=%s", output_dir / "dataset_summary.json")


if __name__ == "__main__":
    main()
