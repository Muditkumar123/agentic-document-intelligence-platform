"""Cross-encoder reranking on ONNX Runtime.

``export_cross_encoder`` turns a Hugging Face cross-encoder into an ONNX graph (and,
optionally, a dynamically quantized INT8 copy) next to its tokenizer files.
``score_onnx_pairs`` then scores (question, passage) pairs with ONNX Runtime on CPU:
the same tokenizer, truncation and logit handling as the PyTorch path in
``adip.rag.rerank``, so the two runtimes are interchangeable.

Exporting needs torch and transformers; scoring needs only onnxruntime and the
tokenizer (no torch). Install with ``pip install -e ".[onnx]"``.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from adip.rag.rerank import MAX_CROSS_ENCODER_TOKENS

FP32_FILENAME = "model.onnx"
INT8_FILENAME = "model.int8.onnx"
METADATA_FILENAME = "onnx_reranker.json"
DEFAULT_OPSET = 17


def export_cross_encoder(
    model_name: str,
    output_dir: Path,
    local_files_only: bool = True,
    quantize: bool = True,
    opset: int = DEFAULT_OPSET,
) -> dict[str, Any]:
    """Export ``model_name`` to ``output_dir``: FP32 ONNX, optional INT8 ONNX, tokenizer and metadata."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()

    sample = tokenizer(["what is a token"], ["a token is a piece of a word"], return_tensors="pt")
    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in sample]

    class LogitsOnly(torch.nn.Module):
        # Positional inputs in a fixed order, logits out: a stable graph signature for ONNX.
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *tensors):
            return self.wrapped(**dict(zip(input_names, tensors))).logits

    fp32_path = output_dir / FP32_FILENAME
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            LogitsOnly(model),
            tuple(sample[name] for name in input_names),
            str(fp32_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    tokenizer.save_pretrained(output_dir)

    files = {"fp32": FP32_FILENAME}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Weights to INT8, activations quantized on the fly: smaller and faster on CPU, no calibration data.
        quantize_dynamic(str(fp32_path), str(output_dir / INT8_FILENAME), weight_type=QuantType.QInt8)
        files["int8"] = INT8_FILENAME

    metadata = {
        "model_name": model_name,
        "opset": opset,
        "input_names": input_names,
        "max_length": int(min(getattr(tokenizer, "model_max_length", 512) or 512, MAX_CROSS_ENCODER_TOKENS)),
        "files": files,
    }
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@lru_cache(maxsize=4)
def load_onnx_cross_encoder(model_path: str, intra_op_threads: int = 0) -> dict[str, Any]:
    """An ONNX Runtime CPU session plus the tokenizer saved next to ``model_path``."""
    if importlib.util.find_spec("onnxruntime") is None:
        raise ImportError('onnxruntime is required for the ONNX reranker: pip install -e ".[onnx]"')
    import onnxruntime as ort
    from transformers import AutoTokenizer

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX reranker model not found: {path}")
    metadata = json.loads((path.parent / METADATA_FILENAME).read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads  # 0 = ONNX Runtime's default
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return {
        "session": session,
        "tokenizer": AutoTokenizer.from_pretrained(path.parent, local_files_only=True),
        "input_names": [item.name for item in session.get_inputs()],
        "max_length": int(metadata["max_length"]),
    }


def score_onnx_pairs(
    question: str,
    texts: list[str],
    model_path: str,
    batch_size: int = 16,
    intra_op_threads: int = 0,
) -> list[float]:
    """Relevance scores for (question, text) pairs, matching ``score_cross_encoder_pairs``."""
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("cross_encoder_batch_size must be greater than 0")
    bundle = load_onnx_cross_encoder(str(model_path), intra_op_threads)
    tokenizer, session = bundle["tokenizer"], bundle["session"]
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            [question] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=bundle["max_length"],
            return_tensors="np",
        )
        feed = {name: encoded[name].astype("int64") for name in bundle["input_names"]}
        (logits,) = session.run(["logits"], feed)
        batch_scores = logits[:, -1] if logits.ndim == 2 and logits.shape[-1] > 1 else logits.reshape(-1)
        scores.extend(float(score) for score in batch_scores)
    return scores
