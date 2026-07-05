# LoRA Fine-Tuning Experiment

ROADMAP Phase 7: demonstrate parameter-efficient fine-tuning with a tracked base-vs-adapted comparison, without overbuilding.

## Task and labels

**Chunk-category classification**: given a text chunk, predict which of the corpus's five document categories it belongs to (legal / academic / technical / security / finance). The labels come for free and are honest — every document's category is declared in [`data/eval/SOURCES.md`](../data/eval/SOURCES.md), the same authoritative table that documents corpus licensing. A model like this is directly useful to the platform: it can route queries to the right domain preset.

## Dataset discipline

- Documents are re-chunked at **~40 words** (the 800-word retrieval chunks would yield too few samples on this corpus); fragments under 80 characters are dropped; no overlap, so no two samples share text.
- The train/eval split is **document-level**: one full document per category is held out, and chunks from a document never appear on both sides. A chunk-level split would leak near-identical neighbouring text into eval and inflate every number.
- Result: **46 train / 16 eval** samples over 5 categories — deliberately demo-scale, and every claim below carries that caveat.

## Run it

```bash
pip install -e ".[finetune]"   # peft, torch, transformers, accelerate

conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_lora_experiment \
  --allow-model-download        # first run fetches distilroberta-base

# baselines only (no GPU/torch needed):
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_lora_experiment --skip-lora
```

Everything is tracked through the standard MLOps run record (MLflow-ready with `--enable-mlflow`): dataset sizes, LoRA hyperparameters, per-approach accuracy/macro-F1, and the full comparison report artifact.

## Results (2026-07-06, A100, distilroberta-base, r=8, alpha=16, 8 epochs)

| Approach | Accuracy | Macro F1 | Trainable params |
| --- | --- | --- | --- |
| Majority class | 0.250 | 0.080 | — |
| TF-IDF + logistic regression | 0.500 | 0.335 | — |
| Frozen base + head only | 0.500 | 0.300 | 0.72% |
| **LoRA (PEFT)** | **0.625** | **0.467** | **0.90% of 83M** |

Training time: ~1–2 seconds per variant on the A100.

## Honest reading

- **LoRA beats every baseline** on both metrics while touching under 1% of the model's parameters — that is the point of the experiment: the adapter learns something the frozen encoder's head cannot express, at negligible cost.
- **The margin is modest and the eval is small** (16 held-out chunks). With a document-level split the eval documents' surface vocabulary genuinely differs from training, which is why TF-IDF stalls at 0.500 — and why beating it means the adapted encoder generalized semantically rather than memorizing words.
- The frozen-head reference tying TF-IDF is itself informative: without adaptation, a pretrained encoder's fixed features are no better than a bag of words on this task.
- What would scale this into a production claim: more documents per category (the single biggest lever), a QLoRA variant for larger bases, and k-fold rotation over held-out documents to tighten the confidence interval. All three are deliberate non-goals at demo scale.
