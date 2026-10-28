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
| Generation faithfulness | 0.583 |
| Generation grounded rate | 0.942 |
| Generation expected coverage | 0.841 |
| Generation citation coverage | 0.684 |
| Generation refusal rate | 0.088 |

**Honest reading of retrieval = 1.0:** the five domains are lexically distinct (GDPR vs RFC vs SEC vocabularies barely overlap), so TF-IDF separates them trivially and hit_rate stays 1.0 even per-category. That is an honest property of a clean corpus, not overfitting — so retrieval is treated as a smoke test, and the **discriminating gate lives in generation faithfulness (0.583)**, which varies and does not saturate. The retrieval *ranking* discrimination story is carried by the separate cross-encoder reranking benchmark on the hard-negative dataset (MRR 0.733 → 0.900).

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
| TF-IDF | single_pass | 0.719 | 0.562 | 0.562 | 1.00 | 1.000 | 1.00 |
| TF-IDF | decompose | 0.875 | 0.750 | 0.812 | 3.00 | 1.000 | 1.09 |
| TF-IDF | retry | 0.719 | 0.562 | 0.562 | 1.00 | 1.000 | 1.00 |
| TF-IDF | **full** (default) | **0.875** | **0.750** | **0.812** | 3.06 | **1.000** | **1.09** |
| Hybrid | single_pass | 0.875 | 0.750 | 0.688 | 1.00 | 1.000 | 1.00 |
| Hybrid | **full** (default) | **0.906** | **0.812** | **0.812** | 3.06 | **1.000** | **1.09** |

"Answer both" is measured with the extractive writer's sentence choice described in finding 4, on chunks that keep line breaks. Before the sentence choice was reworked it was 0.438 (single_pass) and 0.500 (full) on both backends; before chunks kept line breaks it was 0.500 and 0.688 on TF-IDF, 0.562 and 0.688 on hybrid.

**Run-to-run.** Repeated runs give identical reports, so a single before/after run is enough on either backend. Until the writer and coverage sums switched to `math.fsum`, one hybrid golden answer depended on the process's string-hash seed. The writer summed term weights over a Python set, whose order changes with `PYTHONHASHSEED`. For "What obligations does the EU AI Act impose on high-risk AI systems?", two sentences have equal totals, but under some seeds they differed in the last bit, so the length tie-break never ran and the heading sentence was shown instead of the one naming "risk-management systems" and "human oversight". On 1 of 16 seeds, hybrid golden expected coverage read 0.772 instead of 0.783, in every variant. Now all 16 seeds give the same TF-IDF and hybrid reports, with the values above, and the same CI generation report. Retrieval was already deterministic. Hybrid scores are bit-identical across 1 to 64 BLAS threads, and the exact fused-score tie at ranks 3 and 4 on that question (BM25 rank 7 + dense rank 3 against 3 + 7) always keeps chunk order.

**With a local LLM** (Qwen/Qwen3-8B served by `adip.serving` on one A100, thinking off, greedy decoding; TF-IDF; on chunks that keep line breaks; offline, not in CI). "Retrieved and covered" counts probes whose every hop is both retrieved and in the answer:

| Variant | Decomposer | Writer | Hop recall | Both hops | Answer hop coverage | Answer both | Retrieved and covered | Searches | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_pass | none | Qwen3-8B | 0.719 | 0.562 | 0.781 | 0.625 | 7/16 | 1.00 | 8.4 s |
| full | rules | Qwen3-8B | **0.875** | **0.750** | **0.875** | **0.750** | **9/16** | 3.06 | 8.0 s |
| llm_full | Qwen3-8B | Qwen3-8B | 0.844 | 0.688 | 0.844 | 0.688 | 8/16 | 2.75 | 21.5 s |

Before chunks kept line breaks, answer both was 0.625, 0.688 and 0.688, and hop coverage was 0.781, 0.844 and 0.844. A rerun on the old chunker (commit `9ec1aa2`, same server, same day) reproduced those numbers exactly, with identical evidence and sub-questions on every probe. Per probe, the only changes are on hops that were never retrieved, where the answer comes from the model:

- **full:** mh05 gains its zero-trust hop by word overlap, not by being right. The answer says the OAuth authorization server makes the final access decision. NIST SP 800-207 gives that decision to the policy engine. The answer's "responsible ... decision to grant" matches the expected phrase anyway. On the old chunks it named the resource server, also wrong, and didn't match.
- **single_pass:** mh02 gains "401 (Unauthorized)" (the old answer said 400). mh04 loses "absolute requirement": the old answer called MUST NOT "absolute and non-negotiable", the new one "a strong prohibition".
- **llm_full:** mh05 gains and mh04 loses, as above.

Counting only hops that were also retrieved, nothing changed: 7, 9 and 8 of 16 probes (20, 25 and 24 of 32 hops). Latency is within run-to-run noise on a GPU shared with another job (old-chunk rerun: 7.8, 8.2 and 21.0 s; first measurement: 8.1, 8.3 and 22.6 s).

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
4. **The extractive writer was the answer-level bottleneck; better sentence choice, then keeping line breaks in chunks, close most of the gap.** For a question, the writer shows one sentence from each of its top 3 evidence chunks. In 5 of the 12 TF-IDF probes with both hops retrieved (mh07, mh08, mh12, mh14, mh16), the supporting chunk was in the top 2 but the chosen sentence missed the fact, so "answer both" moved only from 0.438 to 0.500. The sentence scorer now uses the shared `content_terms` (stopwords, plural folding) plus `-ed`/`-ing` folding, and weights each term by its IDF over the evidence sentences. It scores a sentence against the best single part of a multi-part question, prefers "We present ..." sentences only for summary requests ("tell me about this paper", briefs), and splits legal "(a) ...; (b) ..." lists into separate items. Before → after:
   - **TF-IDF full:** answer both 0.500 → 0.688, answer hop coverage 0.688 → 0.781. Counting only hops that were also retrieved, 7 → 10 of 16 probes (10 of the 12 with both hops retrieved). Fixed: mh08 (the "We present" bonus no longer beats the "152 layers" sentence), mh14 ("remove" now matches "removed"), and mh16 (a framing sentence no longer outscores the list of characteristics on the word "system", which the list spells "systems"). The full-variant score then equalled Qwen3-8B's (0.688, finding 5), but Qwen's single-pass score was higher (0.625 against 0.500).
   - **Hybrid full:** 0.500 → 0.688; retrieved-and-covered 6 → 10 of 16 (mh02, mh08, mh14, mh16).
   - **Single pass:** TF-IDF 0.438 → 0.500 and hybrid 0.438 → 0.562. Retrieved-and-covered stays 6 of 16 on TF-IDF (the only TF-IDF changes are word-overlap coincidences, below) and goes 5 → 7 on hybrid (mh02, mh14).
   - **CI golden set** (this writer is the gate's writer): expected coverage 0.804 → 0.826, with two questions gaining (BERT "stand for and how is it pre-trained", GDPR erasure grounds; 0.0 → 0.5 each) and none losing. Mean faithfulness moves 0.595 → 0.596 (up on 13 questions, down on 19, because it measures token grounding of whichever sentence is shown). The other gate metrics are unchanged.
   - **Got worse:** mh12 in the single-pass variants (TF-IDF single_pass and retry, hybrid single_pass) loses its second hop. The old answer "covered" Article 17's "no longer necessary in relation to the purposes" with 3 of its 4 words: "longer" from the question, which the answer repeats, and "relation" and "purposes" from Article 5's items (a) and (b), the wrong document. Once the list is split, that Article 5 sentence is no longer shown. On hybrid, mh15 (full) and mh01 (single_pass) lose a hop that was never retrieved. In the other direction, mh01 (TF-IDF single_pass and retry) and mh15 (hybrid single_pass) gain one the same way. "Answer both" is a token-overlap check and moves with such coincidences; the retrieved-and-covered counts above don't.
   - **Tie-break:** with folding and IDF in place, two replacements for the old "length closest to 180 characters" tie-break were measured: earliest sentence (TF-IDF full answer both 0.562) and most terms beyond the question (0.688, but it loses mh03, and its mh01 gain is a coincidence). The length rule kept every fix with no losses, so it stays as the last tie-break.
   - **Still missed after that step:** mh07 and mh12 on both backends, mh13 on hybrid. Chunk text had no line breaks, so a markdown heading was glued to the first sentence ("# NIST Cybersecurity Framework (CSF) 2.0 - Core Functions The NIST Cybersecurity Framework 2.0 organizes ..."), and its title words outscored "Respond: take action regarding a detected cybersecurity incident." The same happened with GDPR Articles 5 and 17. In hybrid mh13 the DNS chunk is fifth in the evidence, and the extractive answer shows only the top 3.
   - **Second step: chunks keep line breaks.** The chunker keeps a blank line between paragraphs and a newline before each heading, list item and table row, and after a heading. Wrapped lines, such as a PDF's one break per visual line, are still joined. The writer splits candidates within lines and skips heading lines. Word windows are unchanged, so chunk boundaries and retrieval are too: every TF-IDF and hybrid score and ranking is identical, as are hop recall, both hops and searches. Chunk ids change, because they hash the text. Splitting alone lowered golden expected coverage from 0.826 to 0.783, for two reasons. The title words also appear in the line that introduces a list ("... into six core Functions:"), which then beat the items. And the glued heading had lent its words to the first sentence only. Three writer rules followed, each measured on both sets:
     - a heading's terms count for every sentence under it, since a heading says what its section is about;
     - a line ending in ":" ranks below sentences that state something, since it only introduces a list;
     - between sentences tied on that score, the one that matches more of the question in its own words wins. Without this, "safe" in the heading "Safe and Idempotent Methods" ties the two definitions under it.
   - **Second step, before → after:** TF-IDF full answer both 0.688 → 0.812. Retrieved-and-covered goes 10 → 11 of 16: mh07 and mh12 gained, mh03 lost. TF-IDF single pass 0.500 → 0.562 (6 → 7, the same three probes). Hybrid full 0.688 → 0.812 (10 → 12: mh07, mh12) and single pass 0.562 → 0.688 (7 → 9). On the CI golden set, expected coverage goes 0.826 → 0.841. Gains: storage limitation 0.0 → 1.0, erasure grounds 0.5 → 1.0, RECOMMENDED 0.5 → 1.0, CSF's six functions 0.17 → 0.33. Faithfulness drops 0.596 → 0.583 because the answers lost the heading words, which were trivially grounded: supported tokens per answer go 45.0 → 42.3, unsupported 29.4 → 29.1. The extractive full variant (0.812, 11 of 16 retrieved and covered) now scores above Qwen3-8B on the same chunks (0.750, 9 of 16; finding 5). LLM writers see the line breaks too, since the prompt inserts chunk text as is.
   - **Second step, got worse:** "How does NIST SP 800-207 define zero trust?" goes 1.0 → 0.0 and "What does the GDPR's 72-hour rule require?" goes 1.0 → 0.5. Both documents are a heading and one paragraph. With the heading's words no longer on the first sentence only, the defining sentence ties with another, and the "length closest to 180" tie-break picks the other. mh03 loses a hop that the old answer covered only by word overlap: it showed an RFC section title glued to its sentence ("Resource Owner Password Credentials The resource owner password credentials ... access token"), not the fact. In the multi-hop runs' golden check on hybrid, "What obligations does the EU AI Act impose on high-risk AI systems?" drops 1.0 → 0.5. With the heading's "act" and "systems" on every sentence, the GPAI sentence's "obligations" outweighs "High risk: ... requirements", because the NIST long-document chunks in that evidence use "high" often. That check's mean still rises (TF-IDF 0.804 → 0.833, hybrid 0.783 → 0.801). mh01 and mh02 gain an answer hop by word overlap, on hops that were never retrieved.
   - **Second step, tried and not kept:** showing a lead-in together with the items after it (golden 0.808, below demoting it). Preferring the sentence that opens its paragraph as a tie-break recovers both golden losses (golden 0.895) but costs multi-hop answers (hybrid full 12 → 11, single pass 9 → 8, TF-IDF single pass 7 → 6). It suits the one-paragraph golden excerpts, not long documents.
   - **Still missed:** in the full variants, with both hops retrieved, the answer misses one in mh03 (TF-IDF) and mh13 (hybrid, as above); the single-pass variants also miss mh06 (both backends), and mh05 and mh11 on hybrid. Indexes built before this change still work: their one-line chunk text is split as before, and only the lead-in rule changes a pick (5 of 57 golden answers, all GDPR legal lists; coverage rises on two).
   - **Caveat:** the first step's rules were developed while reading the five failing probes, so its probe gains are in-sample; the golden set was its out-of-sample check and moved up with no question lost. The second step's rules were developed while reading both the failing probes and the golden questions it hurt, so both sets are in-sample for it. The gate holds `multihop_full_answer_full_coverage` at a 0.7 floor (0.6 after the first step).
5. **With an LLM writer the evidence gain also reaches the answers.** Qwen3-8B reads all five evidence chunks. The loop takes "answer covers both hops" from 0.625 to 0.750 and hop coverage from 0.781 to 0.875, at the same latency (8.4 s and 8.0 s; the writer dominates latency). Counting only hops that were also retrieved, it goes from 7 to 9 of 16 probes. One probe in the 0.750, mh05, is a wrong answer that matches by word overlap (see above). Of the 12 full-variant probes with both hops retrieved, Qwen's answer misses a hop in three. In mh03 and mh11 the first hop is a premise that the question already states ("OAuth 2.0 describes access tokens as credentials."), and Qwen answers only what is asked. In mh13 it writes "63 octets", but the check wants "63 octets or less" and doesn't count numbers. All three answers are right about what was asked. The extractive writer quotes evidence sentences and misses only mh03, so on this token-overlap metric it scores above Qwen (11 against 9 of 16). Caveat: an 8B writer answers some hops from its own knowledge. Single-pass answer coverage (0.625) is higher than its evidence support (0.562), so with an LLM writer this metric is not purely a grounding measure; faithfulness and citation checks are the guard for that.
6. **The 8B LLM decomposer is not better than the rules here.** It gets 11/16 both-hops against 12/16, and 8 against 9 of 16 retrieved and covered. It left the two "statement. question" bridge probes whole (mh03, mh04), split simple questions that didn't need it ("What obligations does the EU AI Act impose on high-risk AI systems?"), and adds about 13.5 s per question (including the optional LLM verifier, which a configured reasoning model turns on). Rules stay the default; `decomposer=llm` remains for stronger planner models.

## Modern Retrieval: Neural Embeddings, a Reranker, and a Vector Store

The baseline retrievers are TF-IDF, LSA, and BM25: fast, deterministic, and CI-friendly, but lexical. This section adds the parts a current RAG stack would use and measures each one against the evaluations above.

**What was added**

- **Neural embeddings** (`adip/rag/embeddings.py`):
  - `bge-m3` (BAAI, 568M parameters, 1024 dimensions). Its weights load from Hugging Face's official safetensors conversion (`refs/pr/130`), because the main branch ships only a pickle-based `pytorch_model.bin`, which current transformers refuses on torch < 2.6.
  - `qwen3-embedding-0.6b` (Qwen, 1024 dimensions). Queries get its retrieval instruction through `encode_query`.
  - Any other Hugging Face id is passed through. Models load once per process, in fp16 on GPU, and only from the local cache unless `ADIP_ALLOW_MODEL_DOWNLOAD=1`.
- **Reranker** (`--reranker cross_encoder --rerank-model BAAI/bge-reranker-v2-m3`): in the agent loop every search stream is reranked against its own query, taking 15 candidates down to the top 5, so each sub-question chooses its section of a long document. The cross-encoder now sees up to the model's limit, capped at 1,024 tokens, instead of a fixed 512; chunks run to about 1,000 tokens, so the old cut hid half of each one.
- **Vector store** (`--vector-store qdrant`, `adip/rag/vector_store.py`): the dense vectors move out of the index archive into Qdrant, either embedded in the index folder or on a server via `ADIP_QDRANT_URL`. Search is nearest-neighbour with the document filter pushed down as a point-id filter, and hybrid fuses BM25 with Qdrant's top 100.

**Results.** Same corpus (173 chunks), 16 two-document probes, 47 single-hop golden questions, and 20 reworded questions (`paraphrase_probes.jsonl`, scored on one search each). Extractive writer, rule-based decomposer, `top_k` 5, one A100. "Loop" is the default agent (decomposition plus follow-ups); latency is per two-document question with the loop.

| Retrieval | Both hops, single pass → loop | Hop recall (loop) | Answer both (loop) | Golden hit / expected coverage | Reworded hit@5 / MRR | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF (default) | 0.562 → 0.750 | 0.875 | 0.812 | 1.000 / 0.833 | 0.750 / 0.627 | 45 ms |
| Dense, bge-m3 | 0.500 → 0.625 | 0.750 | 0.625 | 1.000 / 0.833 | 0.900 / 0.900 | 96 ms |
| Dense, Qwen3-Embedding-0.6B | 0.500 → **0.812** | 0.875 | 0.625 | 1.000 / **0.855** | **0.950** / 0.892 | 144 ms |
| Hybrid, BM25 + bge-m3 | **0.812** → 0.750 | 0.875 | 0.812 | 1.000 / 0.822 | 0.750 / 0.685 | 91 ms |
| Hybrid, BM25 + bge-m3, **Qdrant** | 0.812 → 0.750 | 0.875 | 0.812 | 1.000 / 0.822 | 0.750 / 0.685 | 104 ms |
| Hybrid, BM25 + Qwen3-Embedding | 0.750 → 0.812 | 0.906 | 0.750 | 1.000 / 0.844 | 0.700 / 0.650 | 149 ms |
| TF-IDF + bge reranker | 0.688 → **0.812** | **0.906** | **0.812** | 1.000 / 0.826 | 0.750 / 0.717 | 408 ms |
| Dense bge-m3 + bge reranker | 0.625 → **0.812** | **0.906** | 0.750 | 1.000 / 0.808 | 0.850 / 0.792 | 476 ms |
| Dense Qwen3 + bge reranker | 0.625 → **0.812** | **0.906** | **0.812** | 1.000 / 0.830 | 0.850 / 0.779 | 537 ms |
| Hybrid bge-m3 + bge reranker | 0.688 → **0.812** | **0.906** | **0.812** | 1.000 / 0.826 | 0.750 / 0.750 | 478 ms |
| Hybrid Qwen3 + bge reranker | 0.688 → **0.812** | **0.906** | **0.812** | 1.000 / 0.830 | 0.750 / 0.750 | 528 ms |

What the table shows, including what it doesn't:

1. **Neural embeddings fix reworded questions.** Dense retrieval finds the answer for 90% (bge-m3) to 95% (Qwen3-Embedding) of the reworded questions, against 75% for TF-IDF, with MRR rising from 0.63 to about 0.9. This is the clearest gain in the table.
2. **Alone, they can miss exact technical terms.** With bge-m3, the loop gets both hops on 10 of 16 (TF-IDF: 12), because phrases like "401 (Unauthorized)" or "CertificateVerify" are lexical targets. Qwen3-Embedding doesn't have this problem (13 of 16).
3. **An equal-weight hybrid gives most of the reworded gain back.** It drops to 0.70–0.75, because BM25's half of the fused ranking rewards word overlap, which reworded questions lack by design. Hybrid is best on single-pass two-document retrieval (bge-m3: 13 of 16). The fusion weight was not tuned on these sets.
4. **The reranker is the most consistent multi-hop gain.** With it, every first-stage retriever reaches 13 of 16 both-hops and 0.906 hop recall with the loop, for about 0.35–0.45 s more per question on an A100. It also lowers dense retrieval's reworded recall (0.90/0.95 → 0.85).
5. **The loop can hurt a strong single-pass retriever.** Hybrid bge-m3 gets 13 of 16 in one search, and the loop drops that to 12, because sub-question streams take evidence slots from the first search's results. With the reranker, the loop adds 2 questions on every retriever.
6. **Qdrant changes where vectors live, not what is found.** Its rows are identical to in-memory search, at about 13 ms more per question for 173 chunks. Its value is operational: vectors outside the archive, a server mode, and filtered nearest-neighbour search that scales past an in-memory matrix.
7. **With the extractive writer, dense evidence turns into fewer complete answers** (0.625 against 0.812). The writer still picks sentences by word overlap, so it does better with lexically retrieved chunks.

**Recommendation.** For a GPU deployment, use dense Qwen3-Embedding-0.6B for reworded-question robustness, and add the bge reranker where multi-document precision matters more than about 0.4 s. The defaults stay TF-IDF, for two reasons: the CI gate must stay hermetic (no model downloads), and the Render demo has neither torch nor a GPU.

**Caveats.**
- The sets are small: one question moves a column by 0.06 (16 probes) or 0.05 (20 reworded questions).
- Each row is a single run, with no per-configuration tuning.
- GPU fp16 inference can flip near-ties between runs.

```bash
# one A100; models load from the local Hugging Face cache
CUDA_VISIBLE_DEVICES=1 python -m adip.mlops.run_multihop_eval --variants single_pass full \
  --backend dense --embedding-model qwen3-embedding-0.6b \
  --reranker cross_encoder --rerank-model BAAI/bge-reranker-v2-m3   # optional
# add --vector-store qdrant to keep the vectors in Qdrant (needs the [vectordb] extra)
```

## Reranker on ONNX Runtime (CPU)

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) can be exported to ONNX and run on ONNX Runtime instead of PyTorch (`adip/rag/onnx_reranker.py`). Set `ADIP_RERANKER_ONNX=<path to model.onnx>` on the server and the API, the agent and the CLIs use it whenever they ask for the model it was exported from; in code, `rerank_results(..., cross_encoder_onnx=<path>)` does the same for one call. The export also writes a dynamically quantized INT8 copy: weights stored as 8-bit integers, activations quantized on the fly, no calibration data.

**Workload.** The 47 golden and 20 reworded questions each retrieve 15 candidates with TF-IDF over the 173-chunk multi-hop corpus. Every runtime scores all 1,005 (question, chunk) pairs in batches of 16, with chunks truncated at 512 tokens. CPU only: AMD EPYC 7452, the thread count fixed for both PyTorch and ONNX Runtime, three warm-up questions excluded. "Top-5 identical" counts questions whose reranked top 5 is in exactly the same order as PyTorch's.

| Threads | Runtime | Model size | p50 / p95 per question | Pairs/s | Speed-up (p50) | Max score diff vs PyTorch | Top-1 agrees | Top-5 identical | Hit@5 / MRR@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PyTorch FP32 | 91 MB | 3,485 / 3,509 ms | 4.3 | 1.00× | — | — | — | 0.940 / 0.910 |
| 1 | ONNX Runtime FP32 | 91 MB | 3,102 / 3,142 ms | 4.8 | 1.12× | 9.5×10⁻⁶ | 100% | 100% | 0.940 / 0.910 |
| 1 | ONNX Runtime INT8 | 23 MB | 2,835 / 2,930 ms | 5.2 | 1.23× | 1.1 | 97% | 46% | 0.940 / 0.906 |
| 4 | PyTorch FP32 | 91 MB | 1,515 / 1,571 ms | 10.0 | 1.00× | — | — | — | 0.940 / 0.910 |
| 4 | ONNX Runtime FP32 | 91 MB | 900 / 1,025 ms | 16.4 | 1.68× | 9.5×10⁻⁶ | 100% | 100% | 0.940 / 0.910 |
| 4 | ONNX Runtime INT8 | 23 MB | 864 / 934 ms | 17.0 | 1.75× | 1.1 | 97% | 46% | 0.940 / 0.906 |

What it shows:

1. **ONNX Runtime FP32 is a drop-in replacement.** Scores match PyTorch to 10⁻⁵ and every reranked list is identical, while it runs 1.12× faster on one thread and 1.68× faster on four.
2. **INT8 trades exact order for size.** The model is 4× smaller and slightly faster again, and the top result still matches PyTorch on 97% of questions. But only 46% of top-5 lists keep exactly the same order, because quantization shifts scores by up to 1.1. Hit@5 is unchanged (0.940); MRR@5 drops from 0.910 to 0.906.
3. **INT8's speed gain over ONNX FP32 is small on this CPU** (4–9%). Dynamic quantization mostly saves memory and disk here. The larger speed gains of INT8 depend on hardware with fast integer kernels, and static quantization with calibration data could recover more; neither was measured.

Reproduce (needs the `onnx` extra plus torch and transformers to export):

```bash
python -m adip.mlops.run_reranker_benchmark --model cross-encoder/ms-marco-MiniLM-L-6-v2 --onnx-dir data/models/onnx-reranker --threads 1,4
```

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

It is deterministic with the extractive baseline, so the numbers are reproducible in CI, and any hosted or local writer can be swapped in for model comparisons. On the **real public-document corpus** (`data/eval/`) the baseline scores about 0.58 faithfulness, 0.94 grounded rate, and 0.84 expected coverage — see the table above. The latest report is exposed at `GET /monitoring/generation-eval` and on the dashboard's Answer Quality tiles.

An optional **LLM-as-judge** pass (`--judge-model-name ...`) scores the same answers semantically and reports agreement with the lexical proxy (mean absolute gap + Pearson correlation) — see [LLMOPS.md](LLMOPS.md#llm-as-judge-optional-second-opinion). Judge metrics appear only when a judge ran, so CI stays deterministic.

## Interview Talking Point

The dataset is small but intentionally structured. It lets the project test retrieval, citations, LLMOps, and AgentOps end to end before scaling to a larger document collection.
