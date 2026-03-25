from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from sllm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[position_ids].unsqueeze(1).to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(dtype=x.dtype, device=x.device)
        return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.dropout = config.dropout

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(hidden_states))
        value = self._shape(self.v_proj(hidden_states))

        query = self.rotary(query, position_ids)
        key = self.rotary(key, position_ids)

        attn_mask = None
        is_causal = True
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool, device=query.device)
            if not torch.all(key_padding_mask):
                seq_len = query.size(-2)
                causal_mask = torch.ones(
                    (1, 1, seq_len, seq_len),
                    dtype=torch.bool,
                    device=query.device,
                ).tril()
                attn_mask = causal_mask & key_padding_mask
                is_causal = False

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(hidden_states.shape)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=config.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.input_norm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attn_norm(hidden_states))
        return hidden_states


class SLLMForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input length {seq_len} exceeds model context window {self.config.max_seq_len}."
            )

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            outputs = self(context)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, top_k)
                    cutoff = values[:, [-1]]
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return generated

    def export_config(self) -> dict:
        return asdict(self.config)
