import json

import pytest

from adip.finetuning.baselines import majority_baseline, tfidf_logreg_baseline
from adip.finetuning.dataset import (
    LabeledChunk,
    build_labeled_chunks,
    parse_sources_categories,
    split_by_document,
)


def write_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = {
        "law_one.md": ("legal", "Regulation personal data controller obligations lawful processing consent " * 20),
        "law_two.md": ("legal", "Regulation personal data controller erasure breach notification deadlines " * 20),
        "net_one.md": ("technical", "Protocol network request header response idempotent methods status codes " * 20),
        "net_two.md": ("technical", "Protocol network request header domain resolution records octets zones " * 20),
    }
    sources_lines = [
        "# Sources",
        "",
        "| Document | Category | Source | License |",
        "| --- | --- | --- | --- |",
    ]
    for filename, (category, text) in docs.items():
        (raw / filename).write_text(f"# {filename}\n\n{text}", encoding="utf-8")
        sources_lines.append(f"| `{filename}` | {category} | https://example.org | public |")
    sources = tmp_path / "SOURCES.md"
    sources.write_text("\n".join(sources_lines), encoding="utf-8")
    return raw, sources


def test_parse_sources_categories_reads_table(tmp_path):
    _raw, sources = write_corpus(tmp_path)

    categories = parse_sources_categories(sources)

    assert categories["law_one.md"] == "legal"
    assert categories["net_two.md"] == "technical"
    assert len(categories) == 4


def test_parse_sources_categories_rejects_empty(tmp_path):
    empty = tmp_path / "SOURCES.md"
    empty.write_text("# nothing tabular", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_sources_categories(empty)


def test_build_labeled_chunks_and_document_split(tmp_path):
    raw, sources = write_corpus(tmp_path)
    categories = parse_sources_categories(sources)

    chunks = build_labeled_chunks(raw, categories, chunk_size=30)
    train, evaluation = split_by_document(chunks, holdout_per_category=1, seed=13)

    assert len(chunks) > 8
    assert {chunk.label for chunk in chunks} == {"legal", "technical"}
    train_docs = {chunk.filename for chunk in train}
    eval_docs = {chunk.filename for chunk in evaluation}
    assert train_docs.isdisjoint(eval_docs)  # the leakage guard
    assert {chunk.label for chunk in evaluation} == {"legal", "technical"}


def test_split_by_document_refuses_single_document_categories():
    chunks = [LabeledChunk(text="only doc text", label="legal", filename="only.md")]
    with pytest.raises(ValueError, match="cannot hold out"):
        split_by_document(chunks, holdout_per_category=1)


def test_baselines_separate_obvious_categories(tmp_path):
    raw, sources = write_corpus(tmp_path)
    categories = parse_sources_categories(sources)
    chunks = build_labeled_chunks(raw, categories, chunk_size=30)
    train, evaluation = split_by_document(chunks, seed=13)

    majority = majority_baseline(train, evaluation)
    tfidf = tfidf_logreg_baseline(train, evaluation)

    assert 0.0 <= majority["accuracy"] <= 1.0
    assert tfidf["accuracy"] > majority["accuracy"]
    assert tfidf["macro_f1"] > 0.9  # lexically separable synthetic corpus


def test_lora_trainer_requires_the_extra():
    import importlib.util

    from adip.finetuning.lora import train_lora_classifier

    required = ("torch", "transformers", "peft")
    if all(importlib.util.find_spec(module) is not None for module in required):
        pytest.skip("finetune extra installed; lazy-import guard not exercisable")
    with pytest.raises(ImportError, match=r"pip install -e \"\.\[finetune\]\""):
        train_lora_classifier([], [])


def test_run_lora_experiment_baselines_only(tmp_path):
    from adip.mlops.run_lora_experiment import main

    raw, sources = write_corpus(tmp_path)
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--raw-dir", str(raw),
            "--sources", str(sources),
            "--skip-lora",
            "--report-output", str(report_path),
            "--run-dir", str(tmp_path / "runs"),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    approaches = [result["approach"] for result in report["results"]]
    assert approaches == ["majority_class", "tfidf_logistic_regression"]
    assert report["dataset"]["split"].startswith("document-level")
