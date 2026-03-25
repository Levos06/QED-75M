from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from sllm.utils import ensure_dir


class TokenShardWriter:
    def __init__(self, output_dir: str | Path, prefix: str, shard_size_tokens: int) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.prefix = prefix
        self.shard_size_tokens = shard_size_tokens
        self.buffer: list[int] = []
        self.shard_index = 0
        self.shards: list[dict] = []

    def add_tokens(self, tokens: list[int]) -> None:
        self.buffer.extend(tokens)
        while len(self.buffer) >= self.shard_size_tokens:
            chunk = self.buffer[: self.shard_size_tokens]
            self.buffer = self.buffer[self.shard_size_tokens :]
            self._write_chunk(chunk)

    def finalize(self) -> list[dict]:
        if self.buffer:
            self._write_chunk(self.buffer)
            self.buffer = []
        manifest_path = self.output_dir / f"{self.prefix}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.shards, handle, indent=2, ensure_ascii=False)
        return self.shards

    def _write_chunk(self, chunk: list[int]) -> None:
        shard_name = f"{self.prefix}_{self.shard_index:05d}.bin"
        shard_path = self.output_dir / shard_name
        array = np.asarray(chunk, dtype=np.uint16)
        with shard_path.open("wb") as handle:
            array.tofile(handle)
        self.shards.append(
            {
                "path": shard_name,
                "num_tokens": int(array.shape[0]),
                "dtype": "uint16",
            }
        )
        self.shard_index += 1


class SFTShardWriter:
    def __init__(self, output_dir: str | Path, prefix: str, seq_len: int) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.prefix = prefix
        self.seq_len = seq_len
        self.num_examples = 0
        self.input_path = self.output_dir / f"{self.prefix}_input_ids.bin"
        self.label_path = self.output_dir / f"{self.prefix}_labels.bin"
        self.input_handle = self.input_path.open("wb")
        self.label_handle = self.label_path.open("wb")

    def add_example(self, input_ids: list[int], labels: list[int]) -> None:
        if len(input_ids) != self.seq_len or len(labels) != self.seq_len:
            raise ValueError("Packed SFT example must match fixed seq_len.")
        np.asarray(input_ids, dtype=np.uint16).tofile(self.input_handle)
        np.asarray(labels, dtype=np.int32).tofile(self.label_handle)
        self.num_examples += 1

    def finalize(self) -> dict:
        self.input_handle.close()
        self.label_handle.close()
        if self.num_examples == 0:
            raise RuntimeError("No SFT examples were written.")
        metadata = {
            "num_examples": self.num_examples,
            "seq_len": self.seq_len,
            "input_ids_path": self.input_path.name,
            "labels_path": self.label_path.name,
        }
        with (self.output_dir / f"{self.prefix}_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        return metadata


def load_shard_manifest(data_dir: str | Path, split: str) -> list[dict]:
    data_dir = Path(data_dir)
    manifest_paths = sorted(data_dir.glob(f"{split}_manifest.json"))
    if not manifest_paths and data_dir.name == split:
        manifest_paths = sorted(data_dir.glob("*_manifest.json"))
    if not manifest_paths and (data_dir / split).exists():
        manifest_paths = sorted((data_dir / split).glob("*_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"Shard manifest not found in {data_dir}.")

    shards: list[dict] = []
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        for item in items:
            item["absolute_path"] = str((manifest_path.parent / item["path"]).resolve())
            shards.append(item)
    if not shards:
        raise RuntimeError(f"Shard manifest {manifest_paths} is empty.")
    return shards


class RandomTokenDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.seed = seed
        self.shards = load_shard_manifest(data_dir, split)
        self.arrays = [np.memmap(item["absolute_path"], dtype=np.uint16, mode="r") for item in self.shards]
        capacities = [max(0, int(item["num_tokens"]) - seq_len - 1) for item in self.shards]
        valid_pairs = [(item, array, capacity) for item, array, capacity in zip(self.shards, self.arrays, capacities) if capacity > 0]
        if not valid_pairs:
            raise RuntimeError("No shard contains enough tokens for the selected sequence length.")
        self.shards, self.arrays, capacities = map(list, zip(*valid_pairs))
        weights = np.asarray(capacities, dtype=np.float64)
        self.probabilities = weights / weights.sum()
        self.capacities = capacities

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)

        while True:
            shard_index = int(rng.choice(len(self.arrays), p=self.probabilities))
            capacity = self.capacities[shard_index]
            start = int(rng.integers(0, capacity))
            array = self.arrays[shard_index]
            window = np.asarray(array[start : start + self.seq_len + 1], dtype=np.int64)
            input_ids = torch.from_numpy(window[:-1].copy()).long()
            labels = torch.from_numpy(window[1:].copy()).long()
            attention_mask = torch.ones(self.seq_len, dtype=torch.long)
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }


class SequentialEvalDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        max_batches: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.max_batches = max_batches
        self.shards = load_shard_manifest(data_dir, split)
        self.arrays = [np.memmap(item["absolute_path"], dtype=np.uint16, mode="r") for item in self.shards]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yielded = 0
        for array in self.arrays:
            max_start = len(array) - self.seq_len - 1
            if max_start <= 0:
                continue
            for start in range(0, max_start, self.seq_len):
                if yielded >= self.max_batches:
                    return
                window = np.asarray(array[start : start + self.seq_len + 1], dtype=np.int64)
                if len(window) < self.seq_len + 1:
                    break
                input_ids = torch.from_numpy(window[:-1].copy()).long()
                labels = torch.from_numpy(window[1:].copy()).long()
                attention_mask = torch.ones(self.seq_len, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
                yielded += 1


class FixedSFTDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str) -> None:
        dataset_dir = Path(dataset_dir)
        metadata_path = dataset_dir / f"{split}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.seq_len = int(metadata["seq_len"])
        self.num_examples = int(metadata["num_examples"])
        self.input_ids = np.memmap(
            dataset_dir / metadata["input_ids_path"],
            dtype=np.uint16,
            mode="r",
            shape=(self.num_examples, self.seq_len),
        )
        self.labels = np.memmap(
            dataset_dir / metadata["labels_path"],
            dtype=np.int32,
            mode="r",
            shape=(self.num_examples, self.seq_len),
        )

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(np.asarray(self.input_ids[index], dtype=np.int64).copy()).long()
        labels = torch.from_numpy(np.asarray(self.labels[index], dtype=np.int64).copy()).long()
        attention_mask = (input_ids != 0).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
