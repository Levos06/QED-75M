from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.config import DataMixConfig, load_json, save_json
from sllm.utils import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for the sLLM pipeline.")
    parser.add_argument("--data-config", required=True, help="Path to data mixture JSON config.")
    parser.add_argument("--output-dir", required=True, help="Directory where tokenizer files will be stored.")
    parser.add_argument("--vocab-size", type=int, default=49_152, help="Target tokenizer vocabulary size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    return parser


def iter_source_texts(source, seed: int, limit: int) -> Iterator[str]:
    dataset = load_dataset(
        path=source.path,
        name=source.config_name,
        data_dir=source.data_dir,
        split=source.split,
        revision=source.revision,
        streaming=source.streaming,
    )
    if source.streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=source.shuffle_buffer)

    yielded = 0
    for row in dataset:
        text = row.get(source.text_field or "", None)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        yield text
        yielded += 1
        if yielded >= limit:
            return


def mixed_iterator(config: DataMixConfig, seed: int, logger) -> Iterator[str]:
    weight_map = config.normalized_weights()
    total_docs = config.tokenizer_sample_documents
    per_source = {
        source.name: max(1, int(total_docs * weight_map[source.name]))
        for source in config.sources
    }

    for index, source in enumerate(config.sources):
        limit = source.sample_documents or per_source[source.name]
        logger.info(
            "Tokenizer source start | name=%s path=%s data_dir=%s split=%s text_field=%s limit_docs=%s streaming=%s",
            source.name,
            source.path,
            source.data_dir,
            source.split,
            source.text_field,
            f"{limit:,}",
            source.streaming,
        )
        yield from iter_source_texts(source, seed + index, limit)


def main() -> None:
    args = build_parser().parse_args()
    data_config = DataMixConfig.from_dict(load_json(args.data_config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger("sllm.train_tokenizer", output_dir, "train_tokenizer")
    logger.info("Tokenizer training started")
    logger.info("Log file: %s", log_path)
    logger.info("Arguments | data_config=%s output_dir=%s vocab_size=%s seed=%s", args.data_config, args.output_dir, args.vocab_size, args.seed)
    logger.info("Tokenizer config | sample_documents=%s min_frequency=%s special_tokens=%s num_sources=%s", f"{data_config.tokenizer_sample_documents:,}", data_config.tokenizer_min_frequency, data_config.tokenizer_special_tokens, len(data_config.sources))

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=data_config.tokenizer_min_frequency,
        special_tokens=data_config.tokenizer_special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(mixed_iterator(data_config, args.seed, logger), trainer=trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    if bos_id is None or eos_id is None or pad_id is None:
        raise RuntimeError("Tokenizer special tokens were not created correctly.")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token_id": pad_id,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "unk_token_id": tokenizer.token_to_id("<unk>"),
        },
        "data_config": data_config.to_dict(),
    }
    save_json(output_dir / "tokenizer_meta.json", metadata)

    with (output_dir / "tokenizer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    logger.info("Tokenizer saved | path=%s", tokenizer_path)
    logger.info(
        "Tokenizer summary | vocab_size=%s pad_id=%s bos_id=%s eos_id=%s unk_id=%s",
        tokenizer.get_vocab_size(),
        pad_id,
        bos_id,
        eos_id,
        tokenizer.token_to_id("<unk>"),
    )
    logger.info("Tokenizer metadata saved | path=%s", output_dir / "tokenizer_meta.json")


if __name__ == "__main__":
    main()
