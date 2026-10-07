"""Reference implementation for NOAI Singapore 2026 Programming Task 3."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot-product attention and return output and weights.

    Expected inputs have shape (batch, heads, query_length, key_dimension),
    with values ending in value_dimension. A mask is broadcast over score
    dimensions; nonzero/True entries are kept.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Keep the fill finite in every floating-point dtype, including fp16.
        neg = max(-1e9, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(mask == 0, neg)
    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Project Q/K/V, apply parallel attention heads, then combine them."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.W_q(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attended, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size, -1, self.d_model
        )
        return self.W_o(attended), weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with even-sine / odd-cosine columns."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(math.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model the odd-column slice has one fewer column.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("sequence length exceeds configured max_len")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class TransformerEncoderLayer(nn.Module):
    """Single self-attention and feed-forward encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_out))


class TransformerClassifier(nn.Module):
    """Encoder classifier with mean pooling, matching the task instructions."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        num_classes: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        return self.classifier(x.mean(dim=1))


def create_causal_mask(seq_len: int, device="cpu") -> torch.Tensor:
    """Return a broadcastable lower-triangular keep-mask (1, 1, T, T)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
