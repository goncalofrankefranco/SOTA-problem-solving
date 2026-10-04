"""Fine-tuning and input preprocessing for Translation Stylization (2025 OAI)."""

from __future__ import annotations

import re

import torch
from torch.utils.data import DataLoader


def _split_keywords(keywords):
    if keywords is None:
        return []
    if isinstance(keywords, (list, tuple)):
        values = keywords
    else:
        values = str(keywords).split(",")
    return sorted({str(value).strip() for value in values if str(value).strip()},
                  key=len, reverse=True)


def process_example(en: str, keywords: str) -> str:
    """Mark source occurrences of technical terms for copy-preserving tuning.

    The markers are plain text (the tokenizer is left unchanged). Training uses
    this same preprocessing, with the original Polish reference as target.
    """
    text = str(en)
    terms = _split_keywords(keywords)
    if not terms:
        return text
    alternatives = "|".join(re.escape(term) for term in terms)
    pattern = re.compile(
        r"(?<!\w)(" + alternatives + r")(?!\w)", flags=re.IGNORECASE
    )
    return pattern.sub(
        lambda match: f"[KEEPTERM] {match.group(1)} [/KEEPTERM]", text
    )


def train_translation_model(
    train_dataset,
    tokenizer,
    model,
    *,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 192,
    device=None,
):
    """Fine-tune the supplied Marian model on training examples only.

    ``train_dataset`` is the Dataset loaded by the starter notebook, with
    English ``en``, Polish ``pl``, and comma-separated ``keywords`` columns.
    Validation translations are never used by this function.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def collate(rows):
        source_text = [
            process_example(row["en"], row.get("keywords", "")) for row in rows
        ]
        target_text = [row["pl"] for row in rows]
        source = tokenizer(
            source_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        target = tokenizer(
            text_target=target_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": labels,
        }

    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=generator,
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False

    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    model.eval()
    return model
