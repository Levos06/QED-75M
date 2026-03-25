from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sllm.config import DataMixConfig, load_json, save_json
from sllm.data import TokenShardWriter
from sllm.utils import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize and shard pretraining corpora.")
    parser.add_argument("--data-config", required=True, help="Path to data mixture JSON config.")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.json.")
    parser.add_argument("--output-dir", required=True, help="Root directory for train/val shards.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    return parser


def load_tokenizer(tokenizer_dir: str | Path) -> tuple[Tokenizer, dict]:
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    metadata = load_json(tokenizer_dir / "tokenizer_meta.json")
    return tokenizer, metadata


def iter_source_rows(source, seed: int):
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
    return iter(dataset)


TOKENIZE_BATCH_SIZE = 128


def allocate_token_targets(data_config: DataMixConfig, total_tokens: int) -> dict[str, int]:
    weights = data_config.normalized_weights()
    raw_targets = {
        source.name: total_tokens * weights[source.name]
        for source in data_config.sources
    }
    base_targets = {
        name: int(math.floor(value))
        for name, value in raw_targets.items()
    }
    remainder = total_tokens - sum(base_targets.values())
    ranked = sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    )
    for index in range(remainder):
        name = ranked[index % len(ranked)][0]
        base_targets[name] += 1
    return base_targets


def make_source_state(source, seed: int) -> dict:
    return {
        "source": source,
        "iterator": iter_source_rows(source, seed),
        "documents_used": 0,
        "train_tokens_written": 0,
        "val_tokens_written": 0,
        "exhausted": False,
        "token_queue": deque(),
    }


def refill_token_queue(state: dict, tokenizer: Tokenizer) -> None:
    if state["exhausted"]:
        return

    texts: list[str] = []
    while len(texts) < TOKENIZE_BATCH_SIZE:
        try:
            row = next(state["iterator"])
        except StopIteration:
            state["exhausted"] = True
            break

        text = row.get(state["source"].text_field or "", None)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        texts.append(text)

    if not texts:
        return

    encoded_batch = tokenizer.encode_batch(texts)
    for encoded in encoded_batch:
        token_ids = encoded.ids
        if token_ids:
            state["token_queue"].append(token_ids)


def next_valid_token_ids(state: dict, tokenizer: Tokenizer) -> list[int] | None:
    while True:
        if state["token_queue"]:
            state["documents_used"] += 1
            return state["token_queue"].popleft()
        if state["exhausted"]:
            return None
        refill_token_queue(state, tokenizer)


def choose_source_name(states: dict[str, dict], targets: dict[str, int], split: str, rng: random.Random) -> str | None:
    candidates = []
    for name, state in states.items():
        if state["exhausted"]:
            continue
        target = targets[name]
        if target <= 0:
            continue
        written = state[f"{split}_tokens_written"]
        if written >= target:
            continue
        progress = written / target
        candidates.append((progress, rng.random(), name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def interleave_split(
    split: str,
    writer: TokenShardWriter,
    states: dict[str, dict],
    targets: dict[str, int],
    tokenizer: Tokenizer,
    logger,
    rng: random.Random,
) -> int:
    total_target = sum(targets.values())
    total_written = 0
    emitted_documents = 0

    logger.info(
        "Interleave start | split=%s total_target_tokens=%s strategy=weighted_progress_balancing",
        split,
        f"{total_target:,}",
    )

    while total_written < total_target:
        source_name = choose_source_name(states, targets, split, rng)
        if source_name is None:
            raise RuntimeError(
                f"Недостаточно данных для заполнения split={split}. "
                "Все доступные источники исчерпаны до достижения целевого объема."
            )

        state = states[source_name]
        token_ids = next_valid_token_ids(state, tokenizer)
        if token_ids is None:
            logger.warning("Source exhausted early | split=%s source=%s", split, source_name)
            continue

        source_remaining = targets[source_name] - state[f"{split}_tokens_written"]
        split_remaining = total_target - total_written
        chunk = token_ids[: min(len(token_ids), source_remaining, split_remaining)]
        if not chunk:
            continue

        writer.add_tokens(chunk)
        state[f"{split}_tokens_written"] += len(chunk)
        total_written += len(chunk)
        emitted_documents += 1

        if emitted_documents % 10_000 == 0:
            logger.info(
                "Interleave progress | split=%s documents=%s total_tokens=%s/%s current_source=%s",
                split,
                f"{emitted_documents:,}",
                f"{total_written:,}",
                f"{total_target:,}",
                source_name,
            )

    logger.info(
        "Interleave done | split=%s documents=%s total_tokens=%s",
        split,
        f"{emitted_documents:,}",
        f"{total_written:,}",
    )
    return total_written


def main() -> None:
    args = build_parser().parse_args()
    data_config = DataMixConfig.from_dict(load_json(args.data_config))
    tokenizer, tokenizer_meta = load_tokenizer(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger("sllm.prepare_pretrain_data", output_dir, "prepare_pretrain_data")
    logger.info("Pretokenization started")
    logger.info("Log file: %s", log_path)
    logger.info("Arguments | data_config=%s tokenizer_dir=%s output_dir=%s seed=%s", args.data_config, args.tokenizer_dir, args.output_dir, args.seed)
    logger.info("Tokenizer meta | vocab_size=%s special_tokens=%s", tokenizer_meta.get("vocab_size"), tokenizer_meta.get("special_tokens"))
    logger.info("Mixing strategy | global interleaving with weighted progress balancing")
    logger.info("Tokenization strategy | encode_batch with batch_size=%s", TOKENIZE_BATCH_SIZE)

    weight_map = data_config.normalized_weights()
    train_targets = allocate_token_targets(data_config, data_config.train_tokens)
    val_targets = allocate_token_targets(data_config, data_config.val_tokens)
    dataset_summary: dict[str, dict] = {}
    states: dict[str, dict] = {}

    for index, source in enumerate(data_config.sources):
        states[source.name] = make_source_state(source, args.seed + index)
        logger.info(
            "Source registered | name=%s path=%s data_dir=%s split=%s text_field=%s weight=%.4f train_target=%s val_target=%s streaming=%s",
            source.name,
            source.path,
            source.data_dir,
            source.split,
            source.text_field,
            weight_map[source.name],
            f"{train_targets[source.name]:,}",
            f"{val_targets[source.name]:,}",
            source.streaming,
        )

    rng_val = random.Random(args.seed + 10_000)
    rng_train = random.Random(args.seed + 20_000)
    val_writer = TokenShardWriter(
        output_dir=val_dir,
        prefix="val",
        shard_size_tokens=max(1_000_000, min(data_config.shard_size_tokens, data_config.val_tokens)),
    )
    train_writer = TokenShardWriter(
        output_dir=train_dir,
        prefix="train",
        shard_size_tokens=data_config.shard_size_tokens,
    )

    total_val = interleave_split("val", val_writer, states, val_targets, tokenizer, logger, rng_val)
    total_train = interleave_split("train", train_writer, states, train_targets, tokenizer, logger, rng_train)

    train_shards = train_writer.finalize()
    val_shards = val_writer.finalize()

    for source in data_config.sources:
        state = states[source.name]
        dataset_summary[source.name] = {
            "path": source.path,
            "data_dir": source.data_dir,
            "split": source.split,
            "train_target_tokens": train_targets[source.name],
            "val_target_tokens": val_targets[source.name],
            "train_tokens_written": state["train_tokens_written"],
            "val_tokens_written": state["val_tokens_written"],
            "documents_used": state["documents_used"],
        }
        logger.info(
            "Source done | name=%s documents=%s train_tokens=%s/%s val_tokens=%s/%s",
            source.name,
            f"{state['documents_used']:,}",
            f"{state['train_tokens_written']:,}",
            f"{train_targets[source.name]:,}",
            f"{state['val_tokens_written']:,}",
            f"{val_targets[source.name]:,}",
        )

    save_json(
        output_dir / "dataset_summary.json",
        {
            "tokenizer": tokenizer_meta,
            "data_config": data_config.to_dict(),
            "mixing_strategy": "global_interleaving_weighted_progress_balancing",
            "train_target_tokens": data_config.train_tokens,
            "val_target_tokens": data_config.val_tokens,
            "train_tokens_written": total_train,
            "val_tokens_written": total_val,
            "train_shards": len(train_shards),
            "val_shards": len(val_shards),
            "sources": dataset_summary,
        },
    )
    logger.info(
        "Pretokenization finished | output_dir=%s total_train_tokens=%s total_val_tokens=%s train_shards=%s val_shards=%s",
        output_dir,
        f"{total_train:,}",
        f"{total_val:,}",
        len(train_shards),
        len(val_shards),
    )
    logger.info("Dataset summary saved | path=%s", output_dir / "dataset_summary.json")


if __name__ == "__main__":
    main()
