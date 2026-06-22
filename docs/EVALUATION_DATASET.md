# Evaluation Dataset

This project includes a small domain-diverse evaluation corpus so retrieval and agent behavior can be measured before adding a large real dataset.

## Current Raw Documents

- `sample_research_note.md`: project overview, ingestion, chunking, metadata, and supported domains.
- `academic_rag_evaluation.md`: retrieval evaluation, hit rate@k, MRR, and academic focus areas.
- `agentops_trace_runbook.md`: agent workflow nodes and trace fields.
- `finance_risk_report.md`: finance-domain extraction and monitoring requirements.
- `legal_policy_brief.md`: legal-domain clause, obligation, deadline, and citation requirements.
- `llmops_prompt_registry.md`: prompt registry, prompt hashes, model profiles, and structured verifier output.
- `technical_deployment_runbook.md`: CUDA inspection, local model roles, and serving recommendations.
- `hard_negative_agentops_terms.md`: near-match agent vocabulary that lacks the exact trace fields.
- `hard_negative_llmops_terms.md`: near-match LLMOps vocabulary that lacks the full generation metadata list.
- `hard_negative_model_roles.md`: near-match model-role vocabulary that lacks the exact Qwen/DeepSeek role statements.

## Golden QA Set

Golden questions live in:

```text
data/reference/golden_qa.jsonl
```

Each row contains:

- `question`: the retrieval query.
- `expected_substrings`: text that should appear in at least one retrieved chunk.

Expected substrings are used instead of fixed chunk IDs because chunk IDs are checksum-based and change when source documents change.

## Real Public-Document Evaluation Corpus (CI quality gate)

The original corpus above (`data/raw/` + `data/reference/golden_qa.jsonl`) is small and project-authored, which made `hit_rate@k = 1.0` look potentially overfit. To make the metrics credible, the **CI quality gate** runs against a separate, fixed corpus of **real, externally authored public documents**:

- Corpus: `data/eval/raw/` — **18 documents across 5 categories** (legal, academic, technical, security, finance).
- Golden set: `data/eval/golden_qa.jsonl` — **45 questions**, each tagged with a `category`, including direct, synonym/paraphrase, and cross-cutting hard-negative questions.
- Sources & licensing: every document is a short excerpt from a genuine public source (GDPR / EU AI Act legal text, IETF RFCs, NIST publications, SEC / Investor.gov, arXiv abstracts), documented with URL and license in [`data/eval/SOURCES.md`](../data/eval/SOURCES.md). NIST/SEC are U.S. Government public-domain works; RFCs are reproducible under IETF Trust terms; legal texts are reusable with attribution.

This separation follows the principle that `data/raw/` is the **demo / user-upload** corpus, while `data/eval/` is the **fixed evaluation** corpus the gate is measured against.

**Measured baseline (TF-IDF, no faiss, no rerank, deterministic extractive writer):**

| Metric | Value |
| --- | --- |
| Retrieval hit_rate@k / MRR | 1.00 / 1.00 |
| Generation faithfulness | 0.595 |
| Generation grounded rate | 0.956 |
| Generation expected coverage | 0.795 |
| Generation citation coverage | 0.767 |
| Generation refusal rate | 0.000 |

**Honest reading of retrieval = 1.0:** the five domains are lexically distinct (GDPR vs RFC vs SEC vocabularies barely overlap), so TF-IDF separates them trivially and hit_rate stays 1.0 even per-category. That is an honest property of a clean corpus, not overfitting — so retrieval is treated as a smoke test, and the **discriminating gate lives in generation faithfulness (0.595)**, which varies and does not saturate. The retrieval *ranking* discrimination story is carried by the separate cross-encoder reranking benchmark on the hard-negative dataset (MRR 0.733 → 0.900).

The retrieval report also includes per-category slices `hit_rate_by_category` and `mrr_by_category` for diagnosis (the gate enforces robust overall aggregates, not noisy per-category numbers).

## What The Dataset Tests

- Whether the retriever can find project overview evidence.
- Whether retrieval metrics definitions are findable.
- Whether agent workflow nodes are findable.
- Whether AgentOps trace details are findable.
- Whether domain presets for finance and legal are findable.
- Whether LLMOps run metadata is findable.
- Whether local model roles for Qwen and DeepSeek are findable.

## Latest Benchmark Snapshot

After expanding the corpus and adding cross-encoder reranking:

- Raw documents: 10
- Processed chunks: 10
- Golden questions: 15
- Retrieval variants benchmarked: 6
- TF-IDF hit rate@3: 1.0
- TF-IDF MRR: about 0.733
- Dense LSA hit rate@3: 1.0
- Dense LSA MRR: about 0.756
- TF-IDF + lexical rerank hit rate@3: 1.0
- TF-IDF + lexical rerank MRR: about 0.767
- Dense LSA + lexical rerank hit rate@3: 1.0
- Dense LSA + lexical rerank MRR: about 0.767
- TF-IDF + cross-encoder rerank hit rate@3: 1.0
- TF-IDF + cross-encoder rerank MRR: about 0.900
- Dense LSA + cross-encoder rerank hit rate@3: 1.0
- Dense LSA + cross-encoder rerank MRR: about 0.900
- Best plain backend by MRR: `dense_lsa`
- Best overall variant by MRR: `tfidf_cross_encoder_rerank`

Cross-encoder reranking now gives the strongest ordering on the hard-negative dataset. It improves TF-IDF MRR by about 0.167 and Dense LSA MRR by about 0.144 while keeping hit rate@3 at 1.0. That means the correct evidence was already in the candidate set, and the reranker moved it closer to rank 1.

The current cross-encoder model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It is more expensive than lexical reranking because it scores the query and candidate chunk together, so it should be used on a small candidate set after fast first-stage retrieval.

The current dataset is still intentionally small. The next quality jump should come from adding 20-50 real documents and at least 50 golden questions, including synonym-heavy queries where dense retrieval and reranking have room to help.

## How To Extend The Dataset

1. Add real PDFs, Markdown files, or text files under `data/raw/`.
2. Run ingestion to refresh `data/processed/chunks.jsonl`.
3. Add golden questions to `data/reference/golden_qa.jsonl`.
4. Run `adip.mlops.run_retrieval_benchmark`.
5. Inspect `data/monitoring/retrieval_benchmark_report.json`.

## Good Golden Question Rules

- Ask about one specific fact at a time.
- Make the expected substring exact and distinctive.
- Cover all important document types.
- Include questions with synonyms to test dense retrieval.
- Include exact keyword questions to test TF-IDF.
- Include confusing near-match chunks to test reranking.
- Add negative or insufficient-evidence questions later for verifier testing.

## Generation Evaluation

The same golden set drives answer-quality (generation) evaluation, not just retrieval. `python -m adip.mlops.run_generation_eval` retrieves evidence, generates a grounded answer, and scores it for:

- faithfulness (grounding of the answer's tokens in the retrieved evidence),
- answer relevance (question coverage),
- expected coverage (did the answer surface the row's `expected_substrings`),
- citation coverage.

It is deterministic with the extractive baseline, so the numbers are reproducible in CI, and any hosted or local writer can be swapped in for model comparisons. On the **real public-document corpus** (`data/eval/`) the baseline scores about 0.60 faithfulness, 0.96 grounded rate, and 0.80 expected coverage — see the table above. The latest report is exposed at `GET /monitoring/generation-eval` and on the dashboard's Answer Quality tiles.

## Interview Talking Point

The dataset is small but intentionally structured. It lets the project test retrieval, citations, LLMOps, and AgentOps end to end before scaling to a larger document collection.
