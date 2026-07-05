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
- Golden set: `data/eval/golden_qa.jsonl` — **47 answerable questions** (each tagged with a `category`, including direct, synonym/paraphrase, cross-cutting hard-negative, and table-cell questions) plus **10 unanswerable questions** (`answerable: false`) used to measure abstention. Unanswerable rows are excluded from retrieval metrics and drive the generation refusal metrics.
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

### Abstention (knowing when to say "I don't know")

The 10 unanswerable questions measure whether the system **refuses** instead of confabulating. `generate_grounded_response` supports two evidence-gated abstention modes, both refusing before generating (for any provider) and scored by the same metrics:

- `gen_eval_refusal_precision` — of all refusals, the fraction on genuinely unanswerable questions (don't refuse real questions).
- `gen_eval_refusal_recall` — of all unanswerable questions, the fraction correctly refused.

**Score mode (lexical, the CI gate)** — refuse when the best retrieval score is below a threshold. Deterministic, so it runs in CI (`--abstention-threshold 0.10`). On the eval corpus: **precision 1.0, recall 0.5**. The limitation: it catches *off-domain* questions perfectly (near-zero scores), but *in-domain-but-unanswerable* questions ("What is the maximum fine under the EU AI Act?") retrieve a topically-related chunk scoring as high as a real question, so a lexical threshold can't separate them without false-refusing real questions.

**NLI mode (semantic, opt-in offline)** — refuse when a QNLI cross-encoder ("does this text answer this question?") scores every retrieved chunk below a threshold. Run with `--abstention-mode nli --abstention-threshold 0.2 --allow-nli-download` (needs `transformers` + a model download, so it is not in the deterministic CI gate). On the eval corpus it achieves **precision 1.0, recall 1.0** — it fully closes the gap, because the QNLI model scores the EU-AI-Act-fine question against the risk-tier text at 0.16 (correctly "doesn't answer") while real question/evidence pairs score 0.27–0.99. There is a clean separating gap (unanswerable max ≤ 0.121, answerable min 0.266), so threshold 0.2 perfectly separates the two on this corpus.

| Abstention mode | Precision | Recall | In CI? |
| --- | --- | --- | --- |
| Score (lexical, threshold 0.10) | 1.00 | 0.50 | yes (deterministic gate) |
| NLI / QNLI (threshold 0.2) | 1.00 | 1.00 | no (opt-in offline) |

The same `EntailmentScorer` interface lets any answerability model be swapped in; the pipeline and tests depend only on the callable, not on a specific model.

## What The Dataset Tests

- Whether the retriever can find project overview evidence.
- Whether retrieval metrics definitions are findable.
- Whether agent workflow nodes are findable.
- Whether AgentOps trace details are findable.
- Whether domain presets for finance and legal are findable.
- Whether LLMOps run metadata is findable.
- Whether local model roles for Qwen and DeepSeek are findable.

## Paraphrase Probe Set (`data/eval/paraphrase_probes.jsonl`)

The golden questions share vocabulary with the corpus, which is why retrieval saturates at 1.0. The probe set breaks that: **20 honest rewordings** of golden questions (one per row, `paraphrase_of` records the original) phrased the way a real user would ask — "Can I make a website delete everything it knows about me?" instead of "what is the right to be forgotten?". Same `expected_substrings`, same scoring, but the lexical bridge is gone.

Measured (top-k 5, no reranker):

| Variant | Probes hit@5 | Probes MRR |
| --- | --- | --- |
| tfidf | 0.850 | 0.742 |
| tfidf + keywords rewriter | 0.850 | 0.750 |
| tfidf + LLM rewriter | 0.950 | 0.817 |
| hybrid | 0.950 | 0.767 |
| hybrid + keywords rewriter | 0.950 | 0.750 |
| **hybrid + LLM rewriter** | **1.000** | **0.904** |

Three findings:

1. The probes are the first **unsaturated retrieval eval** in the project — TF-IDF drops to 0.85 hit rate when the words change.
2. **Hybrid retrieval's value is now measured**, not asserted: +0.10 hit rate over TF-IDF on paraphrases (the golden set could never show this).
3. **Deterministic keyword rewriting is flat** (morphology isn't the failure mode; semantics is), while **LLM multi-query rewriting** recovers the gap — perfect hit rate with hybrid — at ~2.3 s/query, so it is an offline/quality mode, not the CI path.

Regression check on the original golden set with the LLM rewriter: hit rate stays **1.000** on both backends (MRR dips slightly — tfidf 1.0 → 0.982, hybrid 0.978 → 0.945 — because fusing variant rankings can nudge the exact-match chunk off rank 1). Rewriting buys paraphrase robustness without losing any answers.

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

An optional **LLM-as-judge** pass (`--judge-model-name ...`) scores the same answers semantically and reports agreement with the lexical proxy (mean absolute gap + Pearson correlation) — see [LLMOPS.md](LLMOPS.md#llm-as-judge-optional-second-opinion). Judge metrics appear only when a judge ran, so CI stays deterministic.

## Interview Talking Point

The dataset is small but intentionally structured. It lets the project test retrieval, citations, LLMOps, and AgentOps end to end before scaling to a larger document collection.
