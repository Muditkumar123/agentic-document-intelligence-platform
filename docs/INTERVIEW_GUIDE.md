# Interview Guide

This guide explains what the project does, why each part exists, and how to talk about it in interviews.

## Short Pitch

I built an agentic document intelligence platform that ingests documents, chunks them with metadata, indexes them for retrieval, answers questions with citations, and tracks the whole workflow through MLOps, LLMOps, and AgentOps. The system supports baseline TF-IDF retrieval, dense retrieval, local LLM profiles such as Qwen and DeepSeek, prompt/version tracking, agent traces, and reproducible evaluation reports.

## Problem Statement

Long documents are hard to search, summarize, and verify manually. A normal chatbot can answer fluently but may hallucinate or lose source grounding. This project solves that by combining retrieval, citations, evidence verification, and operational tracking.

## Why This Project Is Not Just A Chatbot

- It has a document ingestion pipeline.
- It preserves source metadata for citations.
- It has multiple retrieval backends.
- It evaluates retrieval quality with golden questions.
- It tracks prompts, models, latency, token counts, and citation quality.
- It has an agent workflow with planner, retriever, verifier, writer, and citation checker.
- It logs MLOps/LLMOps/AgentOps artifacts for reproducibility and debugging.

## What We Built And Why

## 1. Ingestion Pipeline

What it does:

- Reads PDF, Markdown, and text files.
- Extracts document metadata.
- Splits long text into overlapping chunks.
- Saves chunks as JSONL.

Why we do it:

- LLMs and retrievers work better on smaller chunks than full documents.
- Metadata lets every answer cite exact source chunks.
- JSONL is easy to version, inspect, and process in pipelines.

Interview line:

> I made ingestion traceable by preserving filename, page number, checksum, and chunk IDs, so downstream RAG answers can be cited and audited.

## 2. TF-IDF Baseline Retrieval

What it does:

- Builds a sparse lexical index.
- Finds chunks with overlapping keywords.

Why we do it:

- It is fast, explainable, cheap, and dependency-light.
- It gives a baseline before using dense embeddings.
- It is often strong for exact terms, names, IDs, clauses, and technical phrases.

Interview line:

> I started with TF-IDF because every ML system needs a simple baseline. It helps prove whether more complex dense retrieval actually improves quality.

## 3. Dense Retrieval And FAISS

What it does:

- Builds dense embeddings for document chunks.
- Supports dependency-light LSA embeddings now.
- Can use sentence-transformers when installed.
- Uses FAISS automatically when available, otherwise falls back to NumPy search.

Why we do it:

- Dense retrieval can find semantically related chunks even when wording differs.
- FAISS makes vector search faster and scalable.
- Comparing dense retrieval with TF-IDF creates a real MLOps experiment.

Interview line:

> I added dense retrieval as a measurable upgrade, not as a blind replacement. The benchmark compares hit rate, MRR, latency, and index size against TF-IDF.

## 4. Retrieval Evaluation

What it does:

- Uses golden questions.
- Checks if expected chunks or expected text appear in the retrieved results.
- Computes hit rate@k and MRR.

Why we do it:

- RAG quality depends heavily on retrieval quality.
- Bad retrieval causes hallucination even with a good LLM.
- Metrics let us compare backend changes objectively.

Interview line:

> I evaluate retrieval before generation because if the right evidence is not retrieved, the LLM cannot produce a grounded answer.

## 5. Reranking

What it does:

- Retrieves a larger first-stage candidate set.
- Applies a second-stage lexical reranker.
- Supports an optional cross-encoder reranker through Hugging Face Transformers.
- Returns the reranked final top-k evidence chunks.
- Logs reranked variants in the retrieval benchmark.

Why we do it:

- First-stage retrievers are optimized for fast candidate recall.
- Rerankers are optimized for ordering the best evidence near the top.
- Reranking can improve MRR even when hit rate stays the same.
- Cross-encoders read the query and candidate chunk together, which is slower than vector search but often better for fine-grained relevance.
- On the current golden set, cross-encoder reranking improved TF-IDF MRR from about 0.733 to about 0.900.

Interview line:

> I use retrieval for candidate recall and reranking for precision. The benchmark compares plain retrieval, lexical reranking, and cross-encoder reranking so we can prove whether the second stage helps.

## 6. LLMOps Pipeline

What it does:

- Tracks prompt templates and hashes.
- Tracks model profile, provider, token counts, latency, citation coverage, and unsupported claims.
- Supports extractive baseline, Hugging Face local models, and OpenAI-compatible endpoints.

Why we do it:

- Prompt and model changes can silently change behavior.
- LLMOps makes generation reproducible and measurable.
- It gives a way to compare local models like Qwen and DeepSeek.

Interview line:

> I treat prompts and model choices as versioned artifacts, not hidden constants.

## 6.5. Application API

What it does:

- Exposes health, index rebuild, RAG query, and agent run endpoints through FastAPI.
- Returns cited evidence, reranker metadata, latency, LLMOps metadata, and AgentOps traces.
- Gives the project a clean integration boundary for a future UI or external client.
- Serves a lightweight dashboard for RAG, agent runs, index rebuilds, benchmark metrics, citations, traces, run history, raw JSON, and model selection.
- Supports separate writer and reasoning model controls, so Qwen can write while DeepSeek plans or verifies.

Why we do it:

- A service API makes the project demo-ready and UI-ready.
- It proves the pipelines are reusable outside one-off scripts.
- It gives external clients a clean contract for RAG and agent workflows.
- The history views make observability concrete by letting reviewers inspect previous MLOps runs and AgentOps traces.

Interview line:

> I added a FastAPI layer after the core pipelines were testable, so the project can be demonstrated as a service while still keeping ingestion, retrieval, LLMOps, and AgentOps logic reusable from the CLI and tests.

Dual-model interview line:

> I separate the writer model from the reasoning model: Qwen handles readable answer generation, while DeepSeek can be used for planner/verifier steps where deeper reasoning is useful.

## 7. Agentic Workflow

Workflow:

```text
intent_router -> planner -> retriever -> evidence_verifier -> writer -> citation_checker
```

What it does:

- Routes the task.
- Plans the work.
- Retrieves evidence.
- Verifies evidence sufficiency.
- Writes an answer or brief.
- Checks whether citations are visible.

Why we do it:

- Complex document tasks need multiple steps.
- Separating nodes makes behavior easier to debug.
- AgentOps traces show what happened at each step.

Interview line:

> I separated the agent into observable nodes so I can debug retrieval failures, verifier failures, and citation failures independently.

## 8. DeepSeek Reasoning Verifier

What it does:

- Uses a reasoning model profile for planning and evidence verification.
- Keeps raw model output for audit.
- Normalizes final verifier notes for scoring.

Why we do it:

- Reasoning models can help with multi-step evidence checks.
- Raw reasoning text is not always directly scoreable.
- Structured output normalization makes verifier quality measurable.

Interview line:

> DeepSeek is used where reasoning is useful: planning and verification. Qwen is better suited as a general local writer, while DeepSeek is a reasoning specialist.

## 9. MLOps Tracking

What it does:

- Writes run records for ingestion, retrieval evaluation, LLMOps smoke tests, and agent runs.
- Logs params, metrics, and artifacts.
- Supports optional MLflow and DVC.

Why we do it:

- Experiments must be reproducible.
- Metrics must be tied to the exact inputs and parameters.
- Artifacts help debug and demonstrate progress.

Interview line:

> Every pipeline stage logs parameters, metrics, and artifacts, so I can reproduce and compare experiments instead of relying on memory.

## 10. AgentOps Tracing

What it does:

- Logs each node in the workflow.
- Records timings, state summaries, final answer, verifier notes, and citation checks.

Why we do it:

- Agents are hard to debug without traces.
- Traces show where the workflow failed or became weak.
- They help explain agent behavior to reviewers.

Interview line:

> AgentOps is the observability layer for the agent. It shows how the answer was planned, what evidence was retrieved, and how citations were checked.

## Important Tradeoffs

- TF-IDF is simple and explainable but misses semantic matches.
- Dense retrieval finds semantic matches but can be harder to debug.
- FAISS improves search scalability but adds dependency and persistence complexity.
- Reranking can improve evidence ordering but adds latency and may not help when the dataset is already easy.
- Cross-encoder reranking is more accurate in many RAG systems, but it is slower because it scores each query-document pair jointly.
- Reasoning models are useful for verification but may need structured output parsing.
- Larger LLMs improve quality but increase latency, memory use, and deployment cost.
- Local models improve privacy/control but require GPU planning.

## What To Say If Asked About 40 GB GPU

Qwen3-8B and DeepSeek-R1-Distill-Qwen-14B fit on a 40 GB A100 in the project smoke tests. Qwen used around 15.6 GB, and DeepSeek 14B used around 28.3 GB allocated. For larger 32B models, I would use quantization or tensor parallel serving with vLLM/SGLang.

## What To Say If Asked About Storage

The cached Qwen model is about 16 GB and DeepSeek 14B is about 28 GB, so both fit comfortably within the available storage. The project also records index sizes because vector indexes can become important as document volume grows.

## How To Explain The Latest Dense Retrieval Step

We added dense retrieval and reranking so the project can compare lexical search, semantic search, and second-stage ordering.

- TF-IDF answers: "Which chunks share exact words with the query?"
- Dense retrieval answers: "Which chunks are semantically close to the query?"
- Reranking answers: "Of the retrieved candidates, which evidence should appear first?"
- FAISS answers: "How can we search dense vectors efficiently at scale?"
- MLOps benchmark answers: "Did the new retriever or reranker actually improve metrics?"

## Common Interview Questions And Strong Answers

Question: Why use RAG instead of putting the full document in the prompt?

Answer: RAG scales better, reduces context cost, improves source grounding, and lets us cite exact chunks. Full-context prompting is expensive and harder to evaluate for large document collections.

Question: How do you reduce hallucination?

Answer: I retrieve source chunks, force cited answers, evaluate citation coverage, count unsupported sentences, and use a verifier node before writing.

Question: How do you evaluate answer quality, not just retrieval?

Answer: I run a generation eval over the golden set that scores faithfulness (grounding in evidence), answer relevance, expected-fact coverage, and citation coverage, tracked as an MLOps run and surfaced on the dashboard. It is deterministic with the extractive baseline so it runs in CI, and the same harness can drive any hosted or local writer for model comparisons, with an LLM judge as a future upgrade.

Question: Your retrieval hit rate is 1.0 — isn't that overfit / not meaningful?

Answer: On the CI corpus it is 1.0, and I can explain exactly why: that corpus is 18 real public documents across five lexically-distinct domains (GDPR legal text, IETF RFCs, NIST, SEC, arXiv), so TF-IDF separates them trivially — it stays 1.0 even when sliced per category. That is an honest property of a clean, well-separated corpus, not the system memorizing answers. So I treat retrieval as a smoke test and put the **discriminating quality gate on generation faithfulness (~0.60)**, which actually varies and can regress. The retrieval *ranking* discrimination story is carried separately by the cross-encoder reranking benchmark on a hard-negative dataset, where reranking moves MRR from 0.733 to 0.900. I also deliberately switched the eval corpus from project-authored docs to real external sources precisely so the metrics can't be dismissed as "you wrote the documents and the questions."

Question: How does your RAG system know when to say "I don't know"?

Answer: It has an evidence-gated abstention step: before generating, it refuses when evidence is too weak, provider-agnostically, and I measure it with refusal precision/recall over unanswerable golden questions. I built two modes and the comparison is the interesting part. The lexical mode (best retrieval score below a threshold) is deterministic so it runs in CI, and it gets precision 1.0 but recall only 0.5 — it catches off-domain questions perfectly, but in-domain-but-unanswerable ones like "what's the max fine under the EU AI Act?" pull a topically-related chunk that scores as high as a real question, so a lexical threshold can't separate them. So I added a semantic mode: a QNLI cross-encoder that scores "does this text actually answer this question?". That scores the EU-AI-Act-fine question against the risk-tier text at 0.16 while real pairs score 0.27–0.99, opening a clean gap — and it lifts recall from 0.5 to 1.0 at precision 1.0. I keep the lexical mode as the CI gate because it's deterministic and hermetic, and the NLI mode is an opt-in offline eval behind the same `EntailmentScorer` interface and the same metrics, since it needs a model. I also gate CI on these metrics, so if abstention silently breaks, recall drops and the build fails.

Question: Why compare TF-IDF and dense retrieval?

Answer: TF-IDF is an explainable baseline. Dense retrieval should only be adopted if it improves retrieval metrics or solves semantic mismatch cases. The benchmark makes that decision measurable.

Question: What does MRR measure?

Answer: Mean reciprocal rank rewards systems that retrieve the correct evidence earlier. A hit at rank 1 is better than a hit at rank 5.

Question: What would you improve next?

Answer: I would add harder golden questions, test a cross-encoder reranker, add a FastAPI service, and build a small UI for uploading documents and inspecting traces.

Question: Where does MLOps appear in the project?

Answer: In tracked pipeline runs, parameter logging, metrics files, artifacts, DVC stages, Docker smoke tests, and optional MLflow integration.

Question: Where does LLMOps appear?

Answer: In prompt versioning, model profiles, provider adapters, token/latency/GPU metrics, citation checks, and generation reports.

Question: Where does AgentOps appear?

Answer: In node-level traces, run IDs, planner/verifier metadata, state summaries, final answers, and citation checker results.

## Debugging Stories

Concrete failure-and-fix stories. These answer "tell me about a hard bug you debugged" and "how do you handle unreliable model output." Full write-ups with root causes and regression tests live in [FIXES.md](FIXES.md).

### Story 1: A hosted answer dissolved into the model's own narration

- **Symptom**: With a hosted writer (Gemini), a long answer broke into the model's private narration — "Wait, let's make sure the continuation is seamless. / The last word was 'number'. / Continuation:" — followed by a duplicated paragraph.
- **Diagnosis**: The answer writer splits long answers across calls. When a response stops on `finish_reason=length`, it asks the model to continue and appended the reply verbatim. Gemini answers a "continue" turn with conversational narration instead of continuing silently, and that text was spliced straight into the answer.
- **Fix**: I treated the continuation as untrusted input — strip the narration and preamble, de-duplicate the seam, collapse restarted paragraphs, and fail safe when a continuation is empty. Then I locked the exact failure into a regression test.
- **Lesson**: LLM output is untrusted input. Stitching model text needs the same defensiveness as parsing user input.

### Story 2: Same question and retrieval, but completeness swung wildly with Max Tokens

- **Symptom**: "Tell me about this paper" with K=5 gave a one-and-a-half-sentence stub at 2048 max tokens, but a full five-section brief at 10000 — retrieval was identical in both runs.
- **Diagnosis**: Gemini 2.5 is a thinking model; it spends hidden reasoning tokens before writing, and those count against `max_tokens`. At 2048, roughly 1900 tokens went to reasoning and only ~150 were left for the answer. The continuation mechanism could not help, because each continuation call re-incurs the same thinking cost.
- **Fix**: I made truncation observable instead of silent — surface an actionable warning ("the model spent ~1900 of 2048 tokens on reasoning; raise Max Tokens or lower thinking") as a dashboard banner, inferring the reasoning spend from the token-usage gap since Gemini omits the explicit field. I also raised the default budget and added a **Writer Thinking** control (`reasoning_effort`). A live probe confirmed `none` disables Gemini's thinking entirely, cutting total tokens by ~56% for the same brief.
- **Lesson**: With reasoning models, the token budget is shared between hidden thinking and the visible answer. Make that split visible to the user instead of returning a confusing stub.

### Story 3: A reasoning model's hidden "thinking" leaked into answers — twice

- **Symptom**: A DeepSeek writer's answer opened with its private monologue ("Okay, so I need to ...") and a stray `</think>` before the real content.
- **Diagnosis**: Two gaps. First, DeepSeek-R1 distills emit reasoning with a closing `</think>` but no opening tag (the chat template injects the opener), which the stripper's pair / leading-orphan logic missed. Second — and the reason my first fix did not help — the **local** model runs through a different adapter that never stripped reasoning at all; only the hosted path did.
- **Fix**: Generalized the stripper to drop a stray closing tag, and applied stripping in the local adapter too, so every producer (hosted, local, extractive) is clean. Separately, the answer panel now renders Markdown + KaTeX math instead of raw text.
- **Lesson**: Sanitizing model output is a per-code-path job — a fix on one adapter is not a fix on the feature.

### Story 4: An isolated-venv check caught an undeclared dependency before it could redden CI

- **Symptom**: The test suite was green locally, but I needed the *first* CI run to pass on a clean machine, not just on mine.
- **Diagnosis**: Local development ran inside a fat conda env where `jinja2` happened to already be installed (pulled in transitively by torch/transformers/jupyter). But the project only declared `numpy` / `PyYAML` / `scikit-learn` as core dependencies, while a core module (`llmops/prompts.py`) imports `jinja2` at module load. CI installs *only* the declared dependencies, so test collection and the eval commands would have crashed with `ModuleNotFoundError: jinja2`. Base `fastapi` lists jinja2 only as an optional extra, so the `api` install did not cover it either.
- **Fix**: Before committing the workflow, I reproduced CI's environment exactly — a fresh `venv` with only `.[dev,api]` installed (no torch/faiss/transformers) — and ran the full suite plus the end-to-end eval pipeline there. It failed on the missing `jinja2`, so I declared `Jinja2` as a core dependency and re-verified the minimal environment green (115 tests, ingestion → retrieval → generation eval, all gate checks passing).
- **Lesson**: "Works on my machine" is usually an undeclared-dependency bug in waiting. Verify against the *minimal declared* environment, not your development environment — that is exactly what CI is, so build it locally first.

## Resume Bullets You Can Use

- Built an agentic document intelligence platform with ingestion, RAG, LLMOps, MLOps, and AgentOps tracing.
- Implemented TF-IDF, dense retrieval, and lexical reranking with benchmarked hit rate@k, MRR, latency, and index size.
- Added prompt/version tracking, citation coverage checks, and unsupported-claim evaluation for grounded LLM responses.
- Integrated local Qwen and DeepSeek model profiles for answer generation, planning, and evidence verification.
- Designed reproducible MLOps pipelines with tracked parameters, metrics, artifacts, DVC stages, and optional MLflow logging.
- Hardened the hosted-LLM answer path: sanitized model continuation output and surfaced reasoning-token truncation warnings, each covered by regression tests.
- Built a deterministic answer-quality evaluation (faithfulness, relevance, expected-fact coverage, citations) over a golden set, tracked as MLOps runs and surfaced on the dashboard.
- Set up a GitHub Actions CI pipeline that runs the test suite across Python 3.10–3.12 and enforces retrieval and answer-quality thresholds as automated quality gates, failing the build on metric regressions.
- Built a credibility-first evaluation corpus of real public documents (GDPR, IETF RFCs, NIST, SEC, arXiv) across five domains with 45 categorized golden questions, exposing per-category retrieval slices and gating CI on generation faithfulness rather than a saturated retrieval metric.
- Containerized the API + dashboard as a slim multi-stage Docker image (pinned deps, non-root, healthcheck, demo index baked in, `$PORT`-aware) with a one-click Render blueprint and a CI job that builds and smoke-tests the image on every push.
- Added evidence-gated abstention to the RAG writer in two modes — a deterministic lexical gate (in CI) and a QNLI answer-entailment check — measured with refusal precision/recall over unanswerable questions; the semantic check lifted recall from 0.5 to 1.0 at full precision.
