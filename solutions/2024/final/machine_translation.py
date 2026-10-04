"""Bahdanau-style German-to-English neural machine translation components.

The code is framework-only: provide the prepared token-id DataLoaders from the
official Multi30k notebook. No data is downloaded and no test labels are used
for fitting. Tensors follow the official time-major layout [time, batch].
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class TranslationConfig:
    input_dim: int
    output_dim: int
    pad_index: int
    sos_index: int
    eos_index: int
    encoder_embedding_dim: int = 256
    decoder_embedding_dim: int = 256
    encoder_hidden_dim: int = 512
    decoder_hidden_dim: int = 512
    dropout: float = 0.5
    maxout_hidden_dim: int = 256


class Encoder(nn.Module):
    """Bidirectional GRU encoder that retains every source annotation."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        pad_index: int = 1,
    ):
        super().__init__()
        self.pad_index = pad_index
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.bridge = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # src: [source length, batch]
        if src_lengths is None:
            src_lengths = (src != self.pad_index).sum(dim=0).cpu()
        else:
            src_lengths = torch.as_tensor(src_lengths, dtype=torch.long).cpu()
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.clamp(min=1), enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, total_length=src.shape[0])
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        initial_decoder_state = torch.tanh(self.bridge(final))
        return outputs, initial_decoder_state


class Attention(nn.Module):
    """Additive attention that softly aligns the decoder state to source states."""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int | None = None):
        super().__init__()
        attention_dim = attention_dim or decoder_hidden_dim
        self.encoder_projection = nn.Linear(2 * encoder_hidden_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs [src, batch, 2H], hidden [batch, D]
        scores = self.energy(
            torch.tanh(
                self.encoder_projection(encoder_outputs)
                + self.decoder_projection(hidden).unsqueeze(0)
            )
        ).squeeze(-1)
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=0)  # [src, batch]
        context = torch.sum(weights.unsqueeze(-1) * encoder_outputs, dim=0)
        return context, weights.T


class Decoder(nn.Module):
    """GRU decoder with Bahdanau attention and the paper's maxout output layer."""

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        attention: Attention,
        maxout_hidden_dim: int = 256,
        pad_index: int = 1,
    ):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim + 2 * encoder_hidden_dim, decoder_hidden_dim)
        self.state_output = nn.Linear(decoder_hidden_dim, 2 * maxout_hidden_dim)
        self.word_output = nn.Linear(embedding_dim, 2 * maxout_hidden_dim)
        self.context_output = nn.Linear(2 * encoder_hidden_dim, 2 * maxout_hidden_dim)
        self.vocabulary_output = nn.Linear(maxout_hidden_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token))
        context, attention = self.attention(hidden, encoder_outputs, source_mask)
        rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(0)
        output, next_hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        state = output.squeeze(0)
        combined = self.state_output(state) + self.word_output(embedded) + self.context_output(context)
        maxout = combined.reshape(combined.shape[0], -1, 2).max(dim=2).values
        logits = self.vocabulary_output(maxout)
        return logits, next_hidden.squeeze(0), attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_index: int, eos_index: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_index = pad_index
        self.eos_index = eos_index

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        src_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trg_len, batch_size = trg.shape
        src_mask = src != self.pad_index
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        output_dim = self.decoder.vocabulary_output.out_features
        outputs = encoder_outputs.new_zeros((trg_len, batch_size, output_dim))
        attentions = encoder_outputs.new_zeros((trg_len, batch_size, src.shape[0]))
        input_token = trg[0]
        for step in range(1, trg_len):
            logits, hidden, weights = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[step] = logits
            attentions[step] = weights
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = trg[step] if use_teacher else logits.argmax(dim=1)
        return outputs, attentions


def build_model(config: TranslationConfig) -> Seq2Seq:
    attention = Attention(config.encoder_hidden_dim, config.decoder_hidden_dim)
    encoder = Encoder(
        config.input_dim,
        config.encoder_embedding_dim,
        config.encoder_hidden_dim,
        config.decoder_hidden_dim,
        config.dropout,
        pad_index=config.pad_index,
    )
    decoder = Decoder(
        config.output_dim,
        config.decoder_embedding_dim,
        config.encoder_hidden_dim,
        config.decoder_hidden_dim,
        config.dropout,
        attention,
        maxout_hidden_dim=config.maxout_hidden_dim,
        pad_index=config.pad_index,
    )
    return Seq2Seq(encoder, decoder, pad_index=config.pad_index, eos_index=config.eos_index)


def _batch_tensors(
    batch: Mapping[str, torch.Tensor], source_key: str, target_key: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[source_key].to(device), batch[target_key].to(device)


def run_epoch(
    model: Seq2Seq,
    data_loader: Iterable[Mapping[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    source_key: str = "de_ids",
    target_key: str = "en_ids",
    optimizer: torch.optim.Optimizer | None = None,
    teacher_forcing_ratio: float = 0.0,
    clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, token_count = 0.0, 0
    for batch in data_loader:
        src, trg = _batch_tensors(batch, source_key, target_key, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
            step_logits = logits[1:].reshape(-1, logits.shape[-1])
            targets = trg[1:].reshape(-1)
            loss = criterion(step_logits, targets)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        valid_tokens = int((targets != model.pad_index).sum())
        total_loss += float(loss.detach()) * valid_tokens
        token_count += valid_tokens
    return total_loss / max(token_count, 1)


def fit(
    model: Seq2Seq,
    train_loader: Iterable[Mapping[str, torch.Tensor]],
    valid_loader: Iterable[Mapping[str, torch.Tensor]],
    pad_index: int,
    epochs: int = 20,
    device: str | torch.device | None = None,
    source_key: str = "de_ids",
    target_key: str = "en_ids",
    checkpoint_path: str | Path | None = None,
    patience: int = 5,
) -> dict[str, list[float]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    history = {"train_loss": [], "valid_loss": []}
    best_loss, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            source_key=source_key,
            target_key=target_key,
            optimizer=optimizer,
            teacher_forcing_ratio=0.5,
        )
        valid_loss = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            source_key=source_key,
            target_key=target_key,
            teacher_forcing_ratio=0.0,
        )
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    sos_index: int,
    max_length: int = 100,
) -> tuple[list[int], torch.Tensor]:
    """Decode one padded or unpadded source sentence and return attention."""
    if src.ndim == 1:
        src = src.unsqueeze(1)
    device = next(model.parameters()).device
    src = src.to(device)
    model.eval()
    mask = src != model.pad_index
    encoder_outputs, hidden = model.encoder(src)
    token = torch.tensor([sos_index], dtype=torch.long, device=device)
    output_ids: list[int] = []
    attention_rows = []
    for _ in range(max_length):
        logits, hidden, attention = model.decoder(token, hidden, encoder_outputs, mask)
        token = logits.argmax(dim=1)
        index = int(token.item())
        attention_rows.append(attention[0].detach().cpu())
        if index == model.eos_index:
            break
        output_ids.append(index)
    if attention_rows:
        attention_map = torch.stack(attention_rows)
    else:
        attention_map = torch.zeros((0, src.shape[0]), dtype=torch.float32)
    return output_ids, attention_map


def corpus_bleu(
    references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], max_order: int = 4
) -> float:
    """Smoothed corpus BLEU-4 on tokenized references and hypotheses."""
    if len(references) != len(hypotheses) or not references:
        return 0.0
    matches = [0] * max_order
    totals = [0] * max_order
    ref_len = hyp_len = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_len += len(reference)
        hyp_len += len(hypothesis)
        for order in range(1, max_order + 1):
            ref_counts = Counter(tuple(reference[i : i + order]) for i in range(len(reference) - order + 1))
            hyp_counts = Counter(tuple(hypothesis[i : i + order]) for i in range(len(hypothesis) - order + 1))
            matches[order - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            totals[order - 1] += max(len(hypothesis) - order + 1, 0)
    if hyp_len == 0:
        return 0.0
    log_precision = sum(math_log((matches[i] + 1) / (totals[i] + 1)) for i in range(max_order)) / max_order
    brevity_penalty = min(1.0, np.exp(1.0 - ref_len / hyp_len))
    return float(100.0 * brevity_penalty * np.exp(log_precision))


def math_log(value: float) -> float:
    return float(np.log(max(value, 1e-12)))


def plot_losses(history: Mapping[str, Sequence[float]], output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Training")
    ax.plot(history["valid_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Token cross-entropy", title="Translation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_attention(
    source_tokens: Sequence[str], target_tokens: Sequence[str], attention: torch.Tensor, output_path: str | Path | None = None
):
    import matplotlib.pyplot as plt

    values = attention.detach().cpu().numpy() if isinstance(attention, torch.Tensor) else np.asarray(attention)
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(source_tokens)), max(4, 0.35 * len(target_tokens))))
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(len(source_tokens)), source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)), target_tokens)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated target tokens")
    ax.set_title("Learned soft alignment")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig, ax
