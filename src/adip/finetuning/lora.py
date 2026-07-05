"""PEFT LoRA fine-tuning of a small transformer classifier.

Heavy dependencies (torch, transformers, peft) live behind the ``[finetune]``
extra and are imported lazily. Two tracked comparisons come from the same
model: the frozen base encoder with only the classification head trained
(the "base" reference the roadmap asks for) and the LoRA-adapted version —
plus the deterministic baselines from :mod:`adip.finetuning.baselines`.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from adip.finetuning.baselines import classification_metrics
from adip.finetuning.dataset import LabeledChunk

DEFAULT_BASE_MODEL = "distilroberta-base"


def finetune_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("torch", "transformers", "peft")
    )


def train_lora_classifier(
    train: list[LabeledChunk],
    evaluation: list[LabeledChunk],
    base_model: str = DEFAULT_BASE_MODEL,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    epochs: int = 8,
    learning_rate: float = 2e-4,
    batch_size: int = 16,
    max_length: int = 128,
    device: str | None = None,
    seed: int = 13,
    head_only: bool = False,
    adapter_output: Path | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Train a sequence classifier and evaluate on the held-out documents.

    ``head_only=True`` freezes the encoder and trains just the classification
    head — the no-adaptation reference point. Otherwise LoRA adapters are
    attached with PEFT and trained together with the head.
    """
    if not finetune_available():
        raise ImportError(
            "torch, transformers, and peft are required for LoRA fine-tuning. "
            'Install the extra: pip install -e ".[finetune]"'
        )

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    labels = sorted({chunk.label for chunk in train})
    label_to_id = {label: index for index, label in enumerate(labels)}
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        local_files_only=local_files_only,
    )

    if head_only:
        for name, parameter in model.named_parameters():
            if "classifier" not in name:
                parameter.requires_grad = False
        approach = "head_only_frozen_base"
    else:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)
        approach = "lora"

    model.to(resolved_device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def encode(chunks: list[LabeledChunk]):
        batch = tokenizer(
            [chunk.text for chunk in chunks],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([label_to_id[chunk.label] for chunk in chunks])
        return batch

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    started = time.perf_counter()
    model.train()
    order = list(range(len(train)))
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        permutation = torch.randperm(len(order), generator=generator).tolist()
        for start in range(0, len(permutation), batch_size):
            batch_chunks = [train[order[i]] for i in permutation[start : start + batch_size]]
            batch = {key: value.to(resolved_device) for key, value in encode(batch_chunks).items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - started

    model.eval()
    predicted: list[str] = []
    with torch.no_grad():
        for start in range(0, len(evaluation), batch_size):
            batch_chunks = evaluation[start : start + batch_size]
            batch = encode(batch_chunks)
            batch.pop("labels")
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            logits = model(**batch).logits
            for index in logits.argmax(dim=-1).tolist():
                predicted.append(labels[index])

    if approach == "lora" and adapter_output is not None:
        adapter_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_output))

    return {
        "approach": approach,
        "base_model": base_model,
        "labels": labels,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r if approach == "lora" else None,
        "lora_alpha": lora_alpha if approach == "lora" else None,
        "train_seconds": train_seconds,
        "device": resolved_device,
        "adapter_path": str(adapter_output) if approach == "lora" and adapter_output else None,
        **classification_metrics([chunk.label for chunk in evaluation], predicted),
    }
