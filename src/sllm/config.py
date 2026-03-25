from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 49_152
    max_seq_len: int = 8_192
    d_model: int = 384
    n_layers: int = 32
    n_heads: int = 6
    ffn_hidden_dim: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    bias: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceConfig:
    name: str
    path: str
    split: str
    weight: float
    text_field: str | None = None
    config_name: str | None = None
    data_dir: str | None = None
    revision: str | None = None
    streaming: bool = True
    shuffle_buffer: int = 10_000
    sample_documents: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        return cls(**data)


@dataclass
class DataMixConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    tokenizer_sample_documents: int = 2_000_000
    tokenizer_min_frequency: int = 2
    tokenizer_special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    train_tokens: int = 10_000_000_000
    val_tokens: int = 20_000_000
    shard_size_tokens: int = 100_000_000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataMixConfig":
        sources = [SourceConfig.from_dict(item) for item in data.get("sources", [])]
        kwargs = {key: value for key, value in data.items() if key != "sources"}
        return cls(sources=sources, **kwargs)

    def normalized_weights(self) -> dict[str, float]:
        total = sum(source.weight for source in self.sources)
        return {source.name: source.weight / total for source in self.sources}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [source.__dict__ for source in self.sources]
        return payload


@dataclass
class TrainConfig:
    seed: int = 42
    train_dir: str = "data/pretokenized/train"
    val_dir: str = "data/pretokenized/val"
    output_dir: str = "outputs/pretrain"
    checkpoint_dir: str = "checkpoints/pretrain"
    init_from: str | None = None
    resume_from: str | None = None
    seq_len: int = 2_048
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    max_steps: int = 200_000
    warmup_steps: int = 2_000
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"
    num_workers: int = 0
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    save_interval: int = 1_000
    compile_model: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SFTConfig:
    seed: int = 42
    dataset_path: str = "data/sft/processed"
    output_dir: str = "outputs/sft"
    checkpoint_dir: str = "checkpoints/sft"
    init_from: str = "checkpoints/pretrain/last.pt"
    resume_from: str | None = None
    seq_len: int = 2_048
    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 20_000
    warmup_steps: int = 500
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"
    num_workers: int = 0
    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 50
    save_interval: int = 500
    compile_model: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SFTConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
