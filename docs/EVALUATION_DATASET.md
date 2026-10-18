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

- Corpus: `data/eval/raw/` — **19 documents across 5 categories** (legal, academic, technical, security, finance).
- Golden set: `data/eval/golden_qa.jsonl` — **47 answerable questions** (each tagged with a `category`, including direct, synonym/paraphrase, cross-cutting hard-negative, and table-cell questions) plus **10 unanswerable questions** (`answerable: false`) used to measure abstention. Unanswerable rows are excluded from retrieval metrics and drive the generation refusal metrics.
- Sources & licensing: every document is a short excerpt from a genuine public source (GDPR / EU AI Act legal text, IETF RFCs, NIST publications, SEC / Investor.gov, arXiv abstracts), documented with URL and license in [`data/eval/SOURCES.md`](../data/eval/SOURCES.md). NIST/SEC are U.S. Government public-domain works; RFCs are reproducible under IETF Trust terms; legal texts are reusable with attribution.

This separation follows the principle that `data/raw/` is the **demo / user-upload** corpus, while `data/eval/` is the **fixed evaluation** corpus the gate is measured against.

**Measured baseline (TF-IDF, no faiss, no rerank, deterministic extractive writer, abstention threshold 0.10 as in CI):**

| Metric | Value |
| --- | --- |
| Retrieval hit_rate@k / MRR | 1.00 / 1.00 |
| Generation faithfulness | 0.596 |
| Generation grounded rate | 0.942 |
| Generation expected coverage | 0.826 |
| Generation citation coverage | 0.684 |
| Generation refusal rate | 0.088 |

**Honest reading of retrieval = 1.0:** the five domains are lexically distinct (GDPR vs RFC vs SEC vocabularies barely overlap), so TF-IDF separates them trivially and hit_rate stays 1.0 even per-category. That is an honest property of a clean corpus, not overfitting — so retrieval is treated as a smoke test, and the **discriminating gate lives in generation faithfulness (0.596)**, which varies and does not saturate. The retrieval *ranking* discrimination story is carried by the separate cross-encoder reranking benchmark on the hard-negative dataset (MRR 0.733 → 0.900).

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

## Multi-hop Probes and the Retrieval Loop

Every golden question is answered by one passage of one document, and the eval documents are short excerpts (19 documents, about one chunk each), so they cannot show whether the agent can **combine evidence**. The multi-hop set adds both missing pieces.

**Corpus.** `data/eval/raw/` plus five **full-length** public documents in `data/eval/long/`: RFC 6749 (OAuth 2.0), RFC 7519 (JWT), RFC 8446 (TLS 1.3), NIST SP 800-207 (Zero Trust Architecture), and NIST AI 100-1 (AI RMF), about 103,000 words. That is 173 chunks instead of 19, so a query competes with whole documents of near-miss passages. Sources and licenses are in [`data/eval/SOURCES.md`](../data/eval/SOURCES.md). The CI gate corpus (`data/eval/raw/`) is unchanged.

**Probes** (`data/eval/multihop_probes.jsonl`, 16 questions). Each needs two facts from two documents, either a *comparison* ("How quickly must a breach be reported under the GDPR, and which NIST CSF function covers acting on an incident?") or a *bridge*, where the first answer leads to the second document ("Which protocol does OAuth 2.0 require ..., and how does version 1.3 of that protocol send data in its first flight?"). Every hop lists the document(s) that can supply it and the exact phrase it must contain, and each phrase was checked to exist in its source.

**Metrics** (`adip/agents/multihop_eval.py`), computed on the evidence the agent hands the writer, capped at `top_k` = 5 so no variant can win by retrieving more:

- **hop recall**: share of hops whose phrase is in a retrieved chunk of the right document.
- **both hops**: share of questions with every hop supported. This is the headline.
- **answer both**: the same check on the answer text (token overlap, as in `expected_coverage`).
- **searches**: retrieval calls per question.
- The same runs over the 47 single-hop golden questions check what the loop costs ordinary questions.

**Protocol.** The probes were written and the single-pass pipeline measured **before** the loop existed. The coverage threshold was then set by a rule fixed in advance (the 10th percentile of best-passage coverage over the 67 single-hop golden and paraphrase questions), never by looking at the probes. The `single_pass` variant reproduces the pre-loop pipeline exactly: identical evidence and answers on all 16 probes and all 47 golden questions.

**Results** (deterministic: extractive writer, rule-based decomposer; `python -m adip.mlops.run_multihop_eval`):

| Backend | Variant | Hop recall | Both hops | Answer both | Searches | Golden hit | Golden searches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | single_pass | 0.719 | 0.562 | 0.500 | 1.00 | 1.000 | 1.00 |
| TF-IDF | decompose | 0.875 | 0.750 | 0.688 | 3.00 | 1.000 | 1.09 |
| TF-IDF | retry | 0.719 | 0.562 | 0.500 | 1.00 | 1.000 | 1.00 |
| TF-IDF | **full** (default) | **0.875** | **0.750** | **0.688** | 3.06 | **1.000** | **1.09** |
| Hybrid | single_pass | 0.875 | 0.750 | 0.562 | 1.00 | 1.000 | 1.00 |
| Hybrid | **full** (default) | **0.906** | **0.812** | **0.688** | 3.06 | **1.000** | **1.09** |

"Answer both" is measured with the extractive writer's sentence choice described in finding 4. Before that change it was 0.438 (single_pass) and 0.500 (full) on both backends.

**With a local LLM** (Qwen/Qwen3-8B served by `adip.serving` on one A100, thinking off; TF-IDF; offline, not in CI):

| Variant | Decomposer | Writer | Hop recall | Both hops | Answer hop coverage | Answer both | Searches | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_pass | none | Qwen3-8B | 0.719 | 0.562 | 0.781 | 0.625 | 1.00 | 8.1 s |
| full | rules | Qwen3-8B | **0.875** | **0.750** | **0.844** | **0.688** | 3.06 | 8.3 s |
| llm_full | Qwen3-8B | Qwen3-8B | 0.844 | 0.688 | 0.844 | 0.688 | 2.75 | 22.6 s |

With the extractive writer, the Qwen decomposer split 9 of the 47 single-hop golden questions (rules: 2), averaging 1.45 searches per question against 1.09; hit rate stayed 1.0.

```bash
# terminal 1
CUDA_VISIBLE_DEVICES=1 python -m adip.serving server --model-profile qwen3_8b_default --port 8091 --device cuda:0
# terminal 2
python -m adip.mlops.run_multihop_eval --variants single_pass full --decomposer llm \
  --reasoning-provider openai_compatible --reasoning-model-name Qwen/Qwen3-8B \
  --reasoning-endpoint-url http://127.0.0.1:8091/v1 --reasoning-max-new-tokens 512 \
  --writer-model-name Qwen/Qwen3-8B --writer-endpoint-url http://127.0.0.1:8091/v1 --skip-golden
```

What this shows, including what it doesn't:

1. **Decomposition does the work.** Searching once per sub-question takes "both hops retrieved" from 9/16 to 12/16 on TF-IDF and from 12/16 to 13/16 on hybrid. Single-hop questions lose nothing: hit rate stays 1.0, because the whole question is still searched first, and only 2 of 47 are split (both really are two questions), so they average 1.09 searches.
2. **Coverage-triggered follow-ups barely fire at the pre-set threshold.** With the threshold at 0.3, a follow-up runs on 0 of 47 single-hop questions and 1 of 16 probes, so `retry` alone changes nothing. Whole-question coverage does not separate single-hop from multi-hop questions well, because single-hop paraphrases ("Can I make a website delete everything it knows about me?") also share few words with their answer. An earlier version that searched for rare missing words at a looser threshold fired on most questions, with useless queries such as "let" and "final". The follow-up path stays in, since it is cheap and gated by its own acceptance check, but this eval doesn't support claiming more for it.
3. **What still fails is the wrong section of the right document.** In three of the four TF-IDF misses, the sub-question's search lands in the right long RFC or NIST document but on a neighbouring chunk (an OAuth chunk about endpoints, not the sentence requiring TLS). In the fourth, the right chunk is fifth in its sub-question's results and falls outside the 5-slot evidence cap. Lexical coverage can't tell a neighbouring chunk from the right one. Per-sub-question cross-encoder reranking or an LLM verifier are the obvious next steps.
4. **The extractive writer was the answer-level bottleneck, and better sentence choice closes most of the gap.** For a question, the writer shows one sentence from each of its top 3 evidence chunks. In 5 of the 12 TF-IDF probes with both hops retrieved (mh07, mh08, mh12, mh14, mh16), the supporting chunk was in the top 2 but the chosen sentence missed the fact, so "answer both" moved only from 0.438 to 0.500. The sentence scorer now uses the shared `content_terms` (stopwords, plural folding) plus `-ed`/`-ing` folding, and weights each term by its IDF over the evidence sentences. It scores a sentence against the best single part of a multi-part question, prefers "We present ..." sentences only for summary requests ("tell me about this paper", briefs), and splits legal "(a) ...; (b) ..." lists into separate items. Before → after:
   - **TF-IDF full:** answer both 0.500 → 0.688, answer hop coverage 0.688 → 0.781. Counting only hops that were also retrieved, 7 → 10 of 16 probes (10 of the 12 with both hops retrieved). Fixed: mh08 (the "We present" bonus no longer beats the "152 layers" sentence), mh14 ("remove" now matches "removed"), and mh16 (a framing sentence no longer outscores the list of characteristics on the word "system", which the list spells "systems"). The full-variant score now equals Qwen3-8B's (0.688, finding 5), but Qwen's single-pass score is higher (0.625 against 0.500).
   - **Hybrid full:** 0.500 → 0.688; retrieved-and-covered 6 → 10 of 16 (mh02, mh08, mh14, mh16).
   - **Single pass:** TF-IDF 0.438 → 0.500 and hybrid 0.438 → 0.562. Retrieved-and-covered stays 6 of 16 on TF-IDF (the only TF-IDF changes are word-overlap coincidences, below) and goes 5 → 7 on hybrid (mh02, mh14).
   - **CI golden set** (this writer is the gate's writer): expected coverage 0.804 → 0.826, with two questions gaining (BERT "stand for and how is it pre-trained", GDPR erasure grounds; 0.0 → 0.5 each) and none losing. Mean faithfulness moves 0.595 → 0.596 (up on 13 questions, down on 19, because it measures token grounding of whichever sentence is shown). The other gate metrics are unchanged.
   - **Got worse:** mh12 in the single-pass variants (TF-IDF single_pass and retry, hybrid single_pass) loses its second hop. The old answer "covered" Article 17's "no longer necessary in relation to the purposes" with 3 of its 4 words: "longer" from the question, which the answer repeats, and "relation" and "purposes" from Article 5's items (a) and (b), the wrong document. Once the list is split, that Article 5 sentence is no longer shown. On hybrid, mh15 (full) and mh01 (single_pass) lose a hop that was never retrieved. In the other direction, mh01 (TF-IDF single_pass and retry) and mh15 (hybrid single_pass) gain one the same way. "Answer both" is a token-overlap check and moves with such coincidences; the retrieved-and-covered counts above don't.
   - **Still missed:** mh07 and mh12 on both backends, mh13 on hybrid. Chunk text has no line breaks, so a markdown heading is glued to the first sentence ("# NIST Cybersecurity Framework (CSF) 2.0 - Core Functions The NIST Cybersecurity Framework 2.0 organizes ..."), and its title words outscore "Respond: take action regarding a detected cybersecurity incident." The same happens with GDPR Articles 5 and 17. That is better fixed at ingestion, by keeping line breaks, than in the writer. In hybrid mh13 the DNS chunk is fifth in the evidence, and the extractive answer shows only the top 3.
   - **Tie-break:** with folding and IDF in place, two replacements for the old "length closest to 180 characters" tie-break were measured: earliest sentence (TF-IDF full answer both 0.562) and most terms beyond the question (0.688, but it loses mh03, and its mh01 gain is a coincidence). The length rule kept every fix with no losses, so it stays as the last tie-break.
   - **Caveat:** these rules were developed while reading the five failing probes, so the probe gains are in-sample. The golden set is the out-of-sample check, and it moved up slightly with no question lost. The gate now also holds `multihop_full_answer_full_coverage` at a 0.6 floor.
5. **With an LLM writer the evidence gain also reaches the answers.** Qwen3-8B reads all five evidence chunks, and the loop takes "answer covers both hops" from 0.625 to 0.688 and hop coverage from 0.781 to 0.844, for 0.2 s more (the writer dominates latency). Caveat: an 8B writer answers some hops from its own knowledge. Single-pass answer coverage (0.625) is higher than its evidence support (0.562), so with an LLM writer this metric is not purely a grounding measure; faithfulness and citation checks are the guard for that.
6. **The 8B LLM decomposer is not better than the rules here.** It gets 11/16 both-hops against 12/16. It left the two "statement. question" bridge probes whole, split simple questions that didn't need it ("What obligations does the EU AI Act impose on high-risk AI systems?"), and adds about 14 s per question (including the optional LLM verifier, which a configured reasoning model turns on). Rules stay the default; `decomposer=llm` remains for stronger planner models.

## Earlier Benchmark Snapshot (self-authored corpus)

On the original 10-document self-authored corpus, after adding cross-encoder reranking (superseded by the public-document corpus above, kept for the reranking story):

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

It is deterministic with the extractive baseline, so the numbers are reproducible in CI, and any hosted or local writer can be swapped in for model comparisons. On the **real public-document corpus** (`data/eval/`) the baseline scores about 0.60 faithfulness, 0.94 grounded rate, and 0.83 expected coverage — see the table above. The latest report is exposed at `GET /monitoring/generation-eval` and on the dashboard's Answer Quality tiles.

An optional **LLM-as-judge** pass (`--judge-model-name ...`) scores the same answers semantically and reports agreement with the lexical proxy (mean absolute gap + Pearson correlation) — see [LLMOPS.md](LLMOPS.md#llm-as-judge-optional-second-opinion). Judge metrics appear only when a judge ran, so CI stays deterministic.

## Interview Talking Point

The dataset is small but intentionally structured. It lets the project test retrieval, citations, LLMOps, and AgentOps end to end before scaling to a larger document collection.
