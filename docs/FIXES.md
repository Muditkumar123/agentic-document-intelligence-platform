# Fixes and Debugging Log

This log records concrete problems found in the platform and how they were resolved, with the tests that lock each fix in place. It consolidates the kind of notes that otherwise live in the "Notes" sections of the local smoke docs and the "Troubleshooting" section of [API_KEYS.md](API_KEYS.md), so a reviewer has one place to see how failures were diagnosed, not just what works.

Each entry uses the same shape: **Symptom**, **Root cause**, **Fix**, and **Verification**.

## 1. Hosted-model answers were garbled by the continuation splice

Status: Fixed.

### Symptom

With a hosted OpenAI-compatible writer (Gemini), asking "tell me about this paper" over the SIMON document returned an answer that dissolved into model narration and duplicated text:

```text
... varies depending on the specific cipher, number Wait, let's make sure the
continuation is seamless.
The last word was "number".
Continuation:
"of rounds, and input differences, meaning a model that performs well ... [1].

### 3. of rounds, or input differences, meaning a model that performs well ... [1].

### 3. Source Coverage
```

The visible answer contained the model's private "how do I continue" narration and a near-duplicate restarted paragraph.

### Root cause

The OpenAI-compatible adapter (`src/adip/llmops/models.py`) splits long answers across calls. When a response stops with `finish_reason == "length"` (the model hit `max_tokens`), the adapter asks the model to continue and appends the result. The old loop appended that continuation **verbatim**. Hosted models such as Gemini do not always continue silently: they answer a bare "continue" instruction with conversational narration ("Wait, let's make sure the continuation is seamless. / The last word was ... / Continuation:") and sometimes re-emit a near-duplicate of the last paragraph. All of that text was spliced straight into the answer.

### Fix

The continuation path now defends the splice in four coordinated ways:

1. **Stricter continuation instruction** (`CONTINUATION_INSTRUCTION`). The "continue" turn now explicitly forbids preamble, quotes, section restarts, and meta-commentary such as `Continuation:`.
2. **Continuation cleaning** (`clean_continuation_text`). Leading conversational/meta lines and wrapping quotes are stripped before splicing, while genuine content — including an inline `Continuation: <text>` — is kept. A cap (`_MAX_CONTINUATION_META_LINES`) prevents an over-broad pattern from ever eating real answer text.
3. **Seam de-duplication** (`append_continuation` / `_drop_seam_overlap`). A continuation prefix that verbatim repeats the tail of the answer is dropped, but only on a substantial match (at least 3 tokens and 12 characters), so common words like "the" or "of" are never trimmed.
4. **Near-duplicate line collapse** (`collapse_near_duplicate_lines`). After stitching, a long line that largely repeats a recent line (the restarted `### 3.` paragraph) is removed, while short headings such as `### 3. Source Coverage` are preserved.

The loop is also now **best-effort**: if a continuation call fails or cleans to empty, the adapter stops and returns the answer it already has instead of aborting the request or emitting filler.

After the fix, the same input reads as one continuous answer:

```text
... varies depending on the specific cipher, number of rounds, and input
differences, meaning a model that performs well on one configuration cannot be
hastily applied to others [1].

### 3. Source Coverage
```

### Verification

- New unit tests in `tests/test_serving.py` cover each helper: `test_clean_continuation_text_strips_meta_preamble_and_quotes`, `test_clean_continuation_text_keeps_inline_label_content`, `test_clean_continuation_text_preserves_plain_continuation`, `test_append_continuation_drops_verbatim_seam_overlap`, `test_append_continuation_keeps_non_overlapping_join`, and `test_collapse_near_duplicate_lines_removes_restarted_paragraph`.
- An end-to-end adapter test, `test_openai_compatible_adapter_sanitizes_garbled_continuation`, replays the exact Gemini failure against a stub server and asserts the narration and duplicate paragraph are gone while the real `### 3. Source Coverage` section survives.
- The pre-existing `test_openai_compatible_adapter_continues_when_finish_reason_is_length` still passes unchanged.

```bash
conda run -n crypto_env env PYTHONPATH=src pytest -q
# 98 passed
```

### Interview line

> Splicing a truncated hosted answer is not string concatenation; it is an untrusted-input problem. I made the continuation path strip model narration, de-duplicate the seam, collapse restarted paragraphs, and fail safe, then locked the exact failure into a regression test.

## 2. Thinking models returned truncated stubs because hidden reasoning ate the token budget

Status: Fixed (diagnostic, a higher default, and a `reasoning_effort` control).

### Symptom

Same question ("tell me about this paper"), same retrieval (K=5), same Gemini model — only the token budget changed:

- `Max Tokens = 2048` produced a one-and-a-half-sentence stub that stopped at "...to construct".
- `Max Tokens = 10000` produced a complete five-section research brief.

### Root cause

Gemini 2.5 (like DeepSeek-R1 and the o-series) is a *thinking* model: it generates hidden reasoning tokens **before** the visible answer, and those tokens count against `max_tokens`. With a 2048 budget, roughly 1900 tokens went to reasoning and only ~150 were left for prose, so the answer hit `finish_reason == "length"` almost immediately. Continuation (fix #1) can eventually stitch a complete answer, but only by re-incurring the thinking cost on every call, so it is slow and token-heavy; and on a small enough budget each call still truncates. Answer completeness therefore depends heavily on the token budget for these models, and the app gave no signal about why.

### Fix

Three changes:

1. **Make truncation visible (`build_answer_warning`).** When generation ends on `finish_reason == "length"`, the LLMOps result now carries a plain-language `answer_warning` that quantifies the hidden-reasoning spend and tells the user to raise Max Tokens or lower the model's thinking. The agent API surfaces it as `answer_warning`, and the dashboard renders it as an amber banner above the answer, on both live runs and history detail. Gemini's OpenAI-compatible usage omits `completion_tokens_details.reasoning_tokens`, so the estimate falls back to the `total_tokens - (prompt_tokens + completion_tokens)` gap.
2. **Raise the default headroom.** The agent `max_new_tokens` default moved from 1536 to 4096 (schema, dashboard Max Tokens box, and custom-model fallback), so the common case truncates far less often.
3. **Control the thinking (`reasoning_effort`).** A `reasoning_effort` setting (`auto`/`none`/`low`/`medium`/`high`) is threaded from the request through the agent runner to the OpenAI-compatible adapter, merged into the request body, so it works for dashboard-added models too. The Agent tab exposes it as a **Writer Thinking** dropdown. A live probe against Gemini confirmed `reasoning_effort: "none"` fully disables thinking on gemini-2.5-flash (0 reasoning tokens), whereas the `thinking_config`/`thinking_budget` JSON some sources suggest is **not** a valid field on this endpoint (HTTP 400). For our grounded writer this is the real fix: it is fed the evidence and should spend the budget writing, not thinking.

The warning degrades gracefully: if a provider does not report reasoning tokens, it falls back to a generic "reached the token limit — raise Max Tokens" message.

### Verification

- Unit tests for the message logic in `tests/test_serving.py`: `test_build_answer_warning_flags_reasoning_token_starvation`, `test_build_answer_warning_handles_length_without_reasoning_usage`, `test_build_answer_warning_is_none_when_answer_completes`, and `test_build_answer_warning_is_none_for_non_hosted_generation`.
- A pipeline test, `test_generate_grounded_response_attaches_truncation_warning` in `tests/test_llmops.py`, drives a stub server that returns `finish_reason=length` and asserts the warning is attached and exposed through `metadata()`.
- Service tests `test_agent_workflow_surfaces_answer_warning` and `test_agent_workflow_answer_warning_is_none_when_not_truncated` in `tests/test_api.py` confirm the field reaches the agent response.
- `reasoning_effort` threading is covered by `test_generate_grounded_response_forwards_reasoning_effort` (`tests/test_llmops.py`, asserts the field is sent for `none` and omitted for `auto`) and `test_build_answer_warning_infers_reasoning_from_usage_gap` (`tests/test_serving.py`).
- Live end-to-end against Gemini: the same brief at `max_tokens=2048` completed under both settings, but `reasoning_effort=none` used about 56% fewer total tokens (4757 vs 10887) with zero thinking and fewer continuation round-trips.

### Interview line

> A thinking model spends your token budget on hidden reasoning before it writes a word, so a "reasonable" 2K budget silently produced a stub. I surfaced the reasoning-token spend as an actionable warning and raised the default, turning a confusing empty-looking answer into a one-click fix.

## 3. Reasoning models leaked their hidden thinking into the answer

Status: Fixed.

### Symptom

Answers from a DeepSeek reasoning writer began with the model's private monologue — "Okay, so I need to create a research brief ... I'll structure each section accordingly." — followed by a stray `</think>` and only then the real answer.

### Root cause

Two separate gaps:

1. **Stripper missed the unopened-tag shape.** DeepSeek-R1 distills emit reasoning with a closing `</think>` but **no opening tag**, because the chat template already injected `<think>`. The completion is `reasoning ... </think> answer`. The original `strip_reasoning_blocks` only removed well-formed `<think>...</think>` pairs and a *leading* orphan close, so a mid-text `</think>` slipped through.
2. **The local adapter never stripped at all.** Only the hosted OpenAI-compatible path called `strip_reasoning_blocks` (via `extract_chat_completion_text`). The dashboard's "DeepSeek 14B" writer is a **local** model that runs through `TransformersTextGenerationAdapter`, which returned the decoded text verbatim — so it leaked regardless of the stripper.

### Fix

- `strip_reasoning_blocks` now also drops everything up to and including a stray `</think>` (covers reasoning with no opening tag).
- `TransformersTextGenerationAdapter` runs its decoded output through `strip_reasoning_blocks`, matching the hosted path. All three producers are now clean: hosted OpenAI-compatible, local Hugging Face, and the deterministic extractive baseline.

### Verification

- `tests/test_serving.py`: `test_strip_reasoning_blocks_removes_unopened_think_close` and `test_strip_reasoning_blocks_keeps_plain_answer`, plus the existing `test_extract_chat_completion_text_strips_reasoning_blocks`.
- The local Hugging Face path can't be unit-tested without loading a multi-billion-parameter model, so the stripping logic was proven directly against the leaked transcript.

### Interview line

> The first fix only covered the hosted path; the bug persisted because the *local* model adapter was a second code path that never stripped. The lesson: when sanitizing model output, cover every adapter, not just the one in front of you.

## 4. Answers displayed as raw Markdown (and unreadable math)

Status: Fixed.

### Symptom

The answer panel rendered literal `###`, `**bold**`, `1.` list markers, and `$2^{8.7}$` in a monospace box, so well-structured model output was hard to read.

### Fix

- A small, **XSS-safe** Markdown renderer (HTML-escape first, then transform) renders headings, bold/italic, ordered lists with correct `start` numbers, unordered lists, `---` rules, inline code, code blocks, and `http(s)` links into the dark-theme answer panel.
- **KaTeX** (auto-render) renders `$…$` and `$$…$$` math, so `$2^{8.7}$` shows as a real exponent. It is loaded from the jsdelivr CDN and degrades safely to literal text if the browser is offline.

### Verification

- Served-output checks confirm the answer container, renderer, and assets ship correctly; the renderer escapes all model text before adding tags, so output cannot inject markup.
- Status and error messages stay plain text by design (only the answer is rendered).

> Note: KaTeX is CDN-hosted, so math rendering needs browser internet. It can be vendored into `static/` for fully offline use.

## 5. The public API trusted client-supplied paths and unpickled its indexes

Status: Fixed.

### Symptom

Anyone who could reach the API, including visitors to the public Render demo, could:

- write an upload into any directory the process could write to (`raw_dir` form field on `POST /documents/upload`);
- delete `.pdf`/`.md`/`.txt` files in any directory (`raw_dir` on `DELETE /documents/{filename}`);
- ingest any folder on the server and make its text searchable (`input_path` on `POST /pipeline/rebuild-index`), and write chunk files, indexes, and agent traces to arbitrary locations (`chunks_path`, `index_path`, `trace_dir`);
- point `POST /rag/query` and `POST /agent/run` at any directory, whose `rag_index.pkl` the server would `pickle.load`;
- upload a file of any size, read whole into memory;
- delete the demo corpus or rebuild the index, with no authentication at all.

### Root cause

Two design shortcuts that are harmless on a laptop and dangerous on a network:

1. **The request chose the filesystem.** Every path was a request field with a default, so the dashboard could switch indexes by editing a text box. Nothing confined those paths to the project's data folder; only uploaded *filenames* were sanitized.
2. **Indexes were pickles.** `RagIndex.save` pickled the whole object, and unpickling executes whatever callables the file names. Combined with a client-chosen `index_path`, that is a classic remote-code-execution shape (no direct way to plant a malicious pickle was found, since uploads are restricted to document extensions, but the safety rested on that alone).

### Fix

- **Server-owned layout** (`adip/api/settings.py`). The server decides `data/raw`, `data/processed/chunks.jsonl`, the three `vector_index*` directories, and the trace directory. Clients pick an index by `backend` name. Request models use `extra="forbid"`, so a stale client sending `index_path` gets a 422, not a silent fallback.
- **Pickle-free index format** (`adip/rag/serialization.py`). An index is one `rag_index.npz`: numeric arrays plus a JSON manifest, loaded with `np.load(allow_pickle=False)`. Fitted vectorizers are stored as constructor parameters, vocabulary, and IDF weights (restored through scikit-learn's public `idf_` setter); the LSA projection is stored as its component matrix. Writes are atomic (temp file plus rename). A directory holding only an old `rag_index.pkl` raises a "rebuild the index" error; the pickle is never opened.
- **Write protection** (`adip/api/security.py`). Upload, delete, and rebuild require `Authorization: Bearer $ADIP_ADMIN_TOKEN` (compared with `hmac.compare_digest`). With no token configured they only accept loopback clients: the local dev server still works with no setup, and any network-exposed deployment fails closed. `render.yaml` generates a token; the dashboard has an Admin Token field, kept in memory only.
- **Upload limits.** `ADIP_MAX_UPLOAD_MB` (default 20). An ASGI middleware refuses an oversized declared `Content-Length` before anything is buffered and counts bytes for chunked bodies; the upload handler also reads at most one byte past the limit.

### Verification

- `tests/test_rag.py`: `test_saved_index_round_trips_with_identical_search_results` (tfidf, dense, hybrid give identical rankings and scores after a save/load), `test_saved_index_is_a_pickle_free_archive`, `test_load_index_refuses_legacy_pickle_without_unpickling` and `test_load_index_refuses_pickle_saved_under_the_new_filename` (a pickle payload that would write a marker file never runs), `test_saving_an_index_removes_a_legacy_pickle`, `test_load_index_rejects_unknown_format_version`.
- `tests/test_api.py`: `test_api_requests_reject_client_supplied_paths`, `test_api_upload_ignores_client_raw_dir`, `test_api_delete_ignores_client_raw_dir`, `test_write_endpoints_require_the_admin_token`, `test_write_endpoints_without_a_token_only_accept_loopback_clients`, `test_read_endpoints_stay_open_when_a_token_is_configured`, `test_upload_over_the_size_limit_is_rejected`, `test_request_body_over_the_limit_is_rejected_before_auth`, `test_streamed_request_body_without_content_length_is_counted`.
- End to end against a running server: 401 without a token, 422 for `index_path`, 413 for oversized uploads (from the handler and from the middleware), a `raw_dir` form field ignored, and loopback writes allowed but a forwarded remote client refused when no token is set.
- The suite passes on the pinned conda environment and on a clean venv with the latest scikit-learn, NumPy, and FastAPI.

Existing local indexes are `rag_index.pkl` files and must be rebuilt once (`python -m adip.rag.index ...`, or Rebuild Index in the dashboard). The Docker image builds its indexes at build time, so deployments need nothing extra.

### Interview line

> The API let the client choose filesystem paths and then unpickled whatever index sat at that path. I moved every path to server-side settings, replaced pickle with a plain-array format that cannot execute code on load, and put the write endpoints behind a token that fails closed when it is missing, with regression tests that try each attack.

## 6. The API would call any URL, take unlimited requests, download any model, and trust document text

Status: Fixed.

### Symptom

Four gaps left after fix #5:

- `/models/check` and `/agent/run` accepted any `endpoint_url` and made the server send a request there, with the caller's API key. A visitor could aim it at `169.254.169.254` or an internal service.
- No endpoint had a rate limit.
- `/rag/query` let a client set `allow_reranker_download` and name any `cross_encoder_model`, so the server could be made to download, or load from a path, a model nobody chose.
- Retrieved chunks were pasted into LLM prompts as plain text. A document saying "ignore previous instructions" read like the rest of the prompt.

### Root cause

These request fields were built for a single trusted user on localhost, and the public deployment inherited them unchanged.

### Fix

- **SSRF:** `model_endpoint_error` (`adip/api/security.py`) requires remote clients to use `https` hosts whose addresses are all public (`ipaddress.is_global`), and to send their own API key with any endpoint they name, so the server's key is never sent to it. The model client refuses redirects, which urllib would otherwise follow with the `Authorization` header. Loopback clients are exempt for local development.
- **Rate limits:** `RateLimiter` gives each client `ADIP_RATE_LIMIT_PER_MINUTE` requests (default 60) on the endpoints that do work, answering 429 with `Retry-After`. It's in memory and per process, documented as a shortcut, and capped at 10,000 tracked clients, least recently seen evicted first.
- **Model downloads:** the request field is gone. The reranker is one of two fixed models, and downloads need `ADIP_ALLOW_MODEL_DOWNLOAD=1` on the server. Local models load with `trust_remote_code=False`, so a model repository can't run its own code.
- **Prompt injection:** `qa_v2`, `brief_v2` and `verify_v2` fence each chunk in an `<evidence>` block, mark it untrusted, and neutralise evidence tags inside it and its citation, in any case or spacing (the `fence` filter in `adip/llmops/prompts.py`). The agent flags instruction-like chunks without dropping them.

### Verification

- Tests in `tests/test_api.py`:
  - public, private, link-local, loopback, IPv6-private and plain-`http` endpoints for remote and loopback clients;
  - `/models/check` and `/agent/run` refusing a metadata address before any connection;
  - a client endpoint without the client's own API key refused for the writer and the reasoner;
  - 429 after the budget, a sliding window, `0` disabling the limit, and the client cap evicting the least recently seen;
  - 422 for the download flag and for a path as the reranker model;
  - downloads off unless the server allows them.
- Tests in `tests/test_llmops.py`: all three prompts fence evidence; an injected `</evidence>`, in any case or spacing, or a quote in a citation cannot close its block; a 302 from the model endpoint is not followed and the redirect target never sees the key; both local-model loaders pass `trust_remote_code=False`.
- Tests in `tests/test_agent_loop.py`: an injected chunk is flagged and still used.

### Known limits

- DNS rebinding between the check and the connection; route model calls through an egress proxy to close it.
- The rate limit is per process, and behind a proxy that hides client addresses it is shared by all callers.
- Fencing and flagging reduce prompt injection; they don't eliminate it. The extractive default writer is unaffected because it sends nothing to a model.

## Related answer-quality safeguards already in place

These guards were added in earlier work and are catalogued here so the hosted-answer path is documented in one place. Each has its own test.

- **DeepSeek-style hidden reasoning is stripped.** `strip_reasoning_blocks` removes `<think>...</think>` blocks so private reasoning never reaches the visible answer. Test: `test_extract_chat_completion_text_strips_reasoning_blocks`.
- **Reasoning-only responses fail loudly.** If a provider returns only `reasoning_content` with no final answer, `extract_chat_completion_text` raises a clear, actionable error instead of showing an empty answer. Test: `test_extract_chat_completion_text_rejects_reasoning_only_output`.
- **Truncation with no content is explained.** A `finish_reason == "length"` response that carries no text raises guidance to raise Max Tokens or use a lower-overhead model. Test: `test_extract_chat_completion_text_explains_length_without_content`.
- **No retrieved evidence does not hallucinate.** For hosted writers with an empty evidence set, `should_use_no_evidence_fallback` routes to the deterministic extractive baseline, which states the evidence is insufficient. Test: `test_empty_evidence_uses_local_fallback_for_hosted_writer`.
- **Optional reasoning failures are non-fatal.** A reasoning/verifier model error (for example a hosted `402 Payment Required`) is recorded in verification notes and the agent continues with the normal cited writer. See the Troubleshooting section in [API_KEYS.md](API_KEYS.md).
- **Local Qwen prompt fidelity.** The Hugging Face adapter renders prompts through the tokenizer chat template, which fixed an early run that copied the prompt and missed the citation. See [QWEN3_LOCAL_SMOKE.md](QWEN3_LOCAL_SMOKE.md).
- **Reasoning prose is normalized before scoring.** Reasoning models spend early tokens on analysis; the raw output is kept for audit, then verifier notes are normalized through `structured_output.final_text` before citation scoring. See [DEEPSEEK14B_LOCAL_SMOKE.md](DEEPSEEK14B_LOCAL_SMOKE.md).

## How fixes are verified

Every fix in this log is backed by a test in `tests/`, so a regression re-triggers a failure. Run the full suite with:

```bash
conda run -n crypto_env env PYTHONPATH=src pytest -q
```

## Resume Signal

The platform treats LLM output as untrusted: hidden reasoning is stripped, truncated answers are stitched defensively, empty-evidence requests refuse instead of hallucinating, and every safeguard has a regression test.
