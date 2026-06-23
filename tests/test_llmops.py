import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adip.llmops.evaluation import evaluate_generation
from adip.llmops.generation_eval import aggregate_eval, score_answer
from adip.llmops.models import clean_evidence_text
from adip.llmops.pipeline import evaluate_abstention, generate_grounded_response, write_llmops_report
from adip.llmops.prompts import load_prompt_template
from adip.llmops.verifier import normalize_verifier_output
from adip.mlops.run_generation_eval import main as run_generation_eval_main
from adip.mlops.run_llmops_smoke import main as run_llmops_smoke_main
from adip.rag.retriever import build_index
from adip.config.model_profiles import load_model_profile
from adip.serving.backends import ExtractiveServingBackend
from adip.serving.openai_server import build_handler


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_test",
        "filename": "sample.md",
        "source_path": "/tmp/sample.md",
        "source_type": "md",
        "checksum": "abc123",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "metadata": {},
    }


def test_prompt_template_renders_evidence():
    template = load_prompt_template("qa")

    rendered = template.render(
        question="What is tracked?",
        domain_preset="general",
        focus_areas="claim, evidence",
        evidence=[{"text": "Runs track prompts.", "citation": "sample.md p.1 chunk c1"}],
    )

    assert template.version == "qa_v1"
    assert "What is tracked?" in rendered
    assert "sample.md p.1 chunk c1" in rendered
    assert len(template.template_hash) == 64


def test_verify_prompt_template_renders_evidence():
    template = load_prompt_template("verify")

    rendered = template.render(
        question="Is the claim supported?",
        domain_preset="general",
        focus_areas="claim, evidence",
        evidence=[{"text": "Agent traces are stored.", "citation": "sample.md p.1 chunk c1"}],
    )

    assert template.version == "verify_v1"
    assert "Is the claim supported?" in rendered
    assert "sample.md p.1 chunk c1" in rendered


def test_plan_prompt_template_renders_request():
    template = load_prompt_template("plan")

    rendered = template.render(
        question="Create a grounded brief.",
        domain_preset="academic",
        focus_areas="problem, method",
        evidence=[],
    )

    assert template.version == "plan_v1"
    assert "Create a grounded brief." in rendered
    assert "problem, method" in rendered


def test_brief_prompt_requests_complete_structured_output():
    template = load_prompt_template("brief")

    rendered = template.render(
        question="Create a research brief.",
        domain_preset="academic",
        focus_areas="problem, method",
        evidence=[{"text": "Evidence text.", "citation": "sample.md p.1 chunk c1"}],
    )

    assert "Create a complete research brief" in rendered
    assert "Use complete sentences" in rendered
    assert "Key Evidence" in rendered


def test_generate_grounded_response_returns_metrics_and_quality():
    retrieved = [
        {
            "rank": 1,
            "score": 0.9,
            "citation": "sample.md p.1 chunk chunk_llmops",
            "chunk": make_chunk("chunk_llmops", "LLMOps tracks prompt versions and generation latency."),
        }
    ]

    result = generate_grounded_response(
        question="What does LLMOps track?",
        task_type="qa",
        domain_preset="general",
        retrieved=retrieved,
        model_profile_id="extractive_baseline",
    )

    assert "LLMOps tracks prompt versions" in result.answer
    assert result.generation.model_name == "grounded-extractive-v1"
    assert result.model_profile["profile_id"] == "extractive_baseline"
    assert result.metrics()["llm_input_token_count"] > 0
    assert result.quality.citation_coverage == 1.0


def test_generate_grounded_response_uses_profile_api_key_env(tmp_path, monkeypatch):
    backend_profile = load_model_profile("extractive_baseline")
    backend = ExtractiveServingBackend(backend_profile)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(backend, backend_profile))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    profile_path = tmp_path / "model_profiles.yaml"
    profile_path.write_text(
        """
profiles:
  test_cloud:
    description: "Test OpenAI-compatible profile."
    role: test
    provider: openai_compatible
    model_name: grounded-extractive-v1
    context_window: 8192
    max_new_tokens: 128
    quantization: provider_managed
    local_files_only: false
    recommended_for:
      - tests
    serving:
      endpoint_env: TEST_CLOUD_BASE_URL
      api_key_env: TEST_CLOUD_API_KEY
      extra_body:
        test_flag: true
""",
        encoding="utf-8",
    )

    try:
        port = server.server_address[1]
        monkeypatch.setenv("TEST_CLOUD_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("TEST_CLOUD_API_KEY", "test-secret")
        result = generate_grounded_response(
            question="What does LLMOps track?",
            task_type="qa",
            domain_preset="general",
            retrieved=[
                {
                    "rank": 1,
                    "score": 0.9,
                    "citation": "sample.md p.1 chunk chunk_llmops",
                    "chunk": make_chunk(
                        "chunk_llmops",
                        "LLMOps tracks prompt versions and generation latency.",
                    ),
                }
            ],
            model_profile_id="test_cloud",
            model_profiles_path=profile_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.generation.model_provider == "openai_compatible"
    assert result.generation.raw["extra_body_keys"] == ["test_flag"]
    assert "LLMOps tracks prompt versions" in result.answer


def test_empty_evidence_uses_local_fallback_for_hosted_writer():
    result = generate_grounded_response(
        question="Give me a summary of this PDF file.",
        task_type="brief",
        domain_preset="academic",
        retrieved=[],
        provider="openai_compatible",
        model_name="hosted-model",
        endpoint_url="http://127.0.0.1:9/v1",
        api_key="test-key",
    )

    assert result.generation.model_provider == "local"
    assert "insufficient evidence" in result.answer


def test_generate_grounded_response_attaches_truncation_warning():
    calls = {"count": 0}

    class LengthHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls["count"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            content = "Partial brief about the SIMON cipher" if calls["count"] == 1 else ""
            payload = {
                "choices": [{"finish_reason": "length", "message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 1800,
                    "completion_tokens_details": {"reasoning_tokens": 1750},
                },
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), LengthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        result = generate_grounded_response(
            question="Tell me about this paper.",
            task_type="brief",
            domain_preset="academic",
            retrieved=[
                {
                    "rank": 1,
                    "score": 0.9,
                    "citation": "simon.pdf p.1 chunk c1",
                    "chunk": make_chunk("c1", "SIMON32/64 neural distinguisher evidence."),
                }
            ],
            provider="openai_compatible",
            model_name="thinking-writer",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
            max_new_tokens=2048,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.answer_warning is not None
    assert "1750" in result.answer_warning
    assert "Max Tokens" in result.answer_warning
    assert result.metadata()["answer_warning"] == result.answer_warning


def test_generate_grounded_response_forwards_reasoning_effort():
    bodies = []

    class CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            bodies.append(json.loads(self.rfile.read(length).decode("utf-8")))
            payload = {
                "choices": [{"finish_reason": "stop", "message": {"content": "Grounded answer [1]."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    retrieved = [
        {
            "rank": 1,
            "score": 0.9,
            "citation": "simon.pdf p.1 chunk c1",
            "chunk": make_chunk("c1", "SIMON evidence for the writer."),
        }
    ]

    try:
        port = server.server_address[1]
        endpoint = f"http://127.0.0.1:{port}/v1"
        common = dict(
            question="What is evaluated?",
            task_type="qa",
            domain_preset="general",
            retrieved=retrieved,
            provider="openai_compatible",
            model_name="thinking-writer",
            endpoint_url=endpoint,
            api_key="test-key",
        )
        generate_grounded_response(reasoning_effort="none", **common)
        generate_grounded_response(reasoning_effort="auto", **common)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert bodies[0].get("reasoning_effort") == "none"
    assert "reasoning_effort" not in bodies[1]


def test_brief_request_in_qa_mode_uses_readable_brief_format():
    retrieved = [
        {
            "rank": 1,
            "score": 0.9,
            "citation": "simon.pdf p.1 chunk chunk_title",
            "chunk": make_chunk(
                "chunk_title",
                "Deep Learning Assisted Differential Cryptanalysis for the Lightweight Cipher SIMON studies neural distinguishers for SIMON32/64.",
            ),
        },
        {
            "rank": 2,
            "score": 0.8,
            "citation": "simon.pdf p.3 chunk chunk_attack",
            "chunk": make_chunk(
                "chunk_attack",
                "The authors use high-accuracy neural distinguishers to perform a 15-round distinguishing attack and key-recovery attack on SIMON32/64.",
            ),
        },
    ]

    result = generate_grounded_response(
        question="Create a research brief about the SIMON differential cryptanalysis paper.",
        task_type="qa",
        domain_preset="academic",
        retrieved=retrieved,
        model_profile_id="extractive_baseline",
    )

    assert result.answer.startswith("Research Brief:")
    assert "Executive Summary:" in result.answer
    assert "Key Evidence:" in result.answer
    assert "Source Coverage:" in result.answer
    assert "Grounded Answer:" not in result.answer
    assert result.quality.citation_coverage == 1.0


def test_clean_evidence_text_removes_pdf_title_page_noise():
    raw = (
        "KSII TRANSACTIONS ON INTERNET AND INFORMATION SYSTEMS VOL. 15, NO. 2, Feb. 2021 600 "
        "Copyright ⓒ 2021 KSII Deep Learning Assisted Differential Cryptanalysis for the Lightweight Cipher SIMON "
        "Wenqiang Tian* and Bin Hu PLA SSF Information Engineering University [e-mail: test@example.com] "
        "*Corresponding author: Wenqiang Tian Received October 19, 2020; revised December 16, 2020; "
        "accepted January 19, 2021; published February 28, 2021 Abstract SIMON and SPECK are two families "
        "of lightweight block ciphers."
    )

    cleaned = clean_evidence_text(raw)

    assert cleaned.startswith("Abstract: SIMON and SPECK")
    assert "e-mail" not in cleaned
    assert "Copyright" not in cleaned


def test_generate_grounded_verification_returns_metrics():
    retrieved = [
        {
            "rank": 1,
            "score": 0.9,
            "citation": "sample.md p.1 chunk chunk_verify",
            "chunk": make_chunk("chunk_verify", "AgentOps records node traces for verification."),
        }
    ]

    result = generate_grounded_response(
        question="Does the project record AgentOps traces?",
        task_type="verify",
        domain_preset="general",
        retrieved=retrieved,
        model_profile_id="extractive_baseline",
    )

    assert "Verifier decision" in result.answer
    assert result.prompt.version == "verify_v1"
    assert result.structured_output["structured"] is True
    assert result.metrics()["llm_structured_output"] == 1.0
    assert result.metrics()["llm_citation_coverage"] == 1.0


def test_verifier_normalization_extracts_final_sections():
    evidence = [{"citation": "sample.md p.1 chunk c1", "text": "Metadata is preserved."}]
    raw_text = (
        "Let me think through the evidence first.\n\n"
        "Supported claims:\n"
        "- Metadata preservation is supported (sample.md p.1 chunk c1).\n\n"
        "Missing evidence:\n"
        "- None.\n\n"
        "Verifier decision:\n"
        "- Supported (sample.md p.1 chunk c1)."
    )

    output = normalize_verifier_output(raw_text, evidence)

    assert output.structured is True
    assert output.final_text.startswith("Supported claims:")
    assert "Let me think" not in output.final_text


def test_verifier_normalization_falls_back_with_citation():
    evidence = [{"citation": "sample.md p.1 chunk c1", "text": "Metadata is preserved."}]
    raw_text = "Okay, the evidence seems to support the metadata claim, but no sections are present."

    output = normalize_verifier_output(raw_text, evidence)
    report = evaluate_generation(output.final_text, evidence)

    assert output.structured is False
    assert output.normalization_reason == "structured_sections_missing"
    assert "top retrieved chunk" in output.final_text
    assert report.citation_coverage == 1.0


def test_generate_grounded_plan_returns_metrics():
    result = generate_grounded_response(
        question="Plan a document QA run.",
        task_type="plan",
        domain_preset="technical",
        retrieved=[],
        model_profile_id="extractive_baseline",
    )

    assert "Retrieve the most relevant document chunks" in result.answer
    assert result.prompt.version == "plan_v1"
    assert result.metrics()["llm_input_token_count"] > 0


def test_quality_report_detects_missing_citation():
    evidence = [{"citation": "sample.md p.1 chunk c1", "text": "Tracked evidence."}]

    report = evaluate_generation("This answer has no citation.", evidence)

    assert report.citation_count == 1
    assert report.visible_citation_count == 0
    assert report.citation_coverage == 0.0
    assert report.unsupported_sentence_count == 1


def test_write_llmops_report(tmp_path):
    retrieved = [
        {
            "rank": 1,
            "score": 0.9,
            "citation": "sample.md p.1 chunk chunk_report",
            "chunk": make_chunk("chunk_report", "Reports store prompt and model metadata."),
        }
    ]
    result = generate_grounded_response("What do reports store?", "qa", "general", retrieved)
    output_path = tmp_path / "llmops_report.json"

    write_llmops_report(result, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["prompt"]["version"] == "qa_v1"
    assert payload["generation"]["model_provider"] == "local"


def test_tracked_llmops_smoke_command(tmp_path):
    index_path = tmp_path / "vector_index"
    build_index(
        [
            make_chunk(
                "chunk_smoke",
                "The smoke test retrieves evidence before prompt-versioned generation.",
            )
        ]
    ).save(index_path)

    report_path = tmp_path / "monitoring" / "llmops_report.json"
    metrics_path = tmp_path / "monitoring" / "llmops_metrics.json"
    run_dir = tmp_path / "runs"

    exit_code = run_llmops_smoke_main(
        [
            "--index",
            str(index_path),
            "--question",
            "What does the smoke test retrieve?",
            "--task",
            "qa",
            "--model-profile",
            "extractive_baseline",
            "--report-output",
            str(report_path),
            "--metrics-output",
            str(metrics_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_records = list(run_dir.glob("*/run.json"))

    assert exit_code == 0
    assert report_path.exists()
    assert metrics["llm_citation_coverage"] == 1.0
    assert len(run_records) == 1


def test_score_answer_rewards_grounded_answer():
    evidence = [
        {
            "citation": "simon.pdf p.1 chunk c1",
            "text": "SIMON32/64 neural distinguisher achieves high accuracy on eight rounds.",
        }
    ]
    answer = "The SIMON32/64 neural distinguisher achieves high accuracy on eight rounds (simon.pdf p.1 chunk c1)."

    case = score_answer(
        "What does the SIMON distinguisher achieve?",
        answer,
        evidence,
        ["neural distinguisher"],
    )

    assert case.refusal is False
    assert case.faithfulness is not None and case.faithfulness >= 0.8
    assert case.grounded is True
    assert case.citation_coverage == 1.0
    assert case.expected_coverage == 1.0


def test_score_answer_flags_ungrounded_answer():
    evidence = [
        {"citation": "simon.pdf p.1 chunk c1", "text": "SIMON32/64 neural distinguisher achieves high accuracy."}
    ]
    answer = "Bitcoin transactions settle through proof-of-work consensus among distributed miners worldwide."

    case = score_answer("What does the distinguisher achieve?", answer, evidence)

    assert case.faithfulness is not None and case.faithfulness < 0.3
    assert case.grounded is False


def test_score_answer_treats_refusal_as_unscored_faithfulness():
    case = score_answer(
        "Unanswerable question about quantum teleportation?",
        "The indexed documents do not contain enough evidence to answer this question.",
        [],
    )

    assert case.refusal is True
    assert case.faithfulness is None
    assert case.grounded is False


def test_aggregate_eval_summarizes_cases():
    grounded = score_answer(
        "What does it achieve?",
        "neural distinguisher achieves accuracy (c1).",
        [{"citation": "c1", "text": "neural distinguisher achieves accuracy"}],
        ["neural distinguisher"],
    )
    refusal = score_answer(
        "Unanswerable?",
        "The indexed documents do not contain enough evidence.",
        [],
    )

    report = aggregate_eval([grounded, refusal])

    assert report.case_count == 2
    assert report.answered_count == 1
    assert report.refusal_rate == 0.5
    assert report.mean_faithfulness == 1.0
    assert report.grounded_rate == 1.0
    assert report.metrics()["gen_eval_refusal_rate"] == 0.5


def test_evaluate_abstention_disabled_when_threshold_none():
    assert evaluate_abstention([{"score": 0.01}], None) is None


def test_evaluate_abstention_passes_when_score_above_threshold():
    assert evaluate_abstention([{"score": 0.5}, {"score": 0.2}], 0.1) is None


def test_evaluate_abstention_refuses_on_weak_evidence():
    response = evaluate_abstention([{"score": 0.05}, {"score": 0.02}], 0.1)
    assert response is not None
    assert "insufficient evidence" in response.text.lower()
    assert response.raw["abstained"] is True
    assert response.raw["abstention_mode"] == "score"
    assert response.raw["confidence"] == 0.05


def test_evaluate_abstention_refuses_on_empty_evidence():
    response = evaluate_abstention([], 0.1)
    assert response is not None and response.raw["abstained"] is True


def test_evaluate_abstention_nli_mode_uses_entailment_scorer():
    # Fake NLI scorer: low entailment -> should refuse even though lexical score is high.
    evidence = [{"score": 0.9, "text": "off-topic chunk"}]
    response = evaluate_abstention(
        evidence, 0.5, entailment_scorer=lambda q, e: 0.1, question="unanswerable?"
    )
    assert response is not None
    assert response.raw["abstention_mode"] == "nli"
    assert response.raw["confidence"] == 0.1


def test_evaluate_abstention_nli_mode_proceeds_when_entailed():
    evidence = [{"score": 0.05, "text": "answering chunk"}]
    # High entailment -> answer even though lexical score is low.
    response = evaluate_abstention(
        evidence, 0.5, entailment_scorer=lambda q, e: 0.92, question="answerable?"
    )
    assert response is None


def test_refusal_precision_and_recall_track_abstention_quality():
    correct_refusal = score_answer("q", "insufficient evidence here", [], answerable=False)
    missed_unanswerable = score_answer(
        "q", "some grounded text (c1).", [{"citation": "c1", "text": "some grounded text"}], answerable=False
    )
    answered = score_answer(
        "q", "grounded text (c1).", [{"citation": "c1", "text": "grounded text"}], answerable=True
    )
    false_refusal = score_answer("q", "insufficient evidence", [], answerable=True)

    report = aggregate_eval([correct_refusal, missed_unanswerable, answered, false_refusal])

    assert report.unanswerable_count == 2
    assert report.refusal_precision == 0.5  # 2 refusals, 1 on a genuinely unanswerable question
    assert report.refusal_recall == 0.5  # 2 unanswerable, 1 correctly refused
    metrics = report.metrics()
    assert metrics["gen_eval_refusal_precision"] == 0.5
    assert metrics["gen_eval_refusal_recall"] == 0.5


def test_refusal_metrics_default_to_one_when_no_refusals_or_unanswerable():
    answered = score_answer(
        "q", "grounded text (c1).", [{"citation": "c1", "text": "grounded text"}], answerable=True
    )
    report = aggregate_eval([answered])
    assert report.refusal_precision == 1.0  # no false refusals possible
    assert report.refusal_recall == 1.0  # no unanswerable questions to catch


def test_tracked_generation_eval_command(tmp_path):
    index_path = tmp_path / "vector_index"
    build_index(
        [make_chunk("chunk_eval", "The platform retrieves evidence and writes cited grounded answers.")]
    ).save(index_path)
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "question": "What does the platform write?",
                "expected_substrings": ["cited grounded answers"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "gen_report.json"
    metrics_path = tmp_path / "gen_metrics.json"
    run_dir = tmp_path / "runs"

    exit_code = run_generation_eval_main(
        [
            "--index",
            str(index_path),
            "--golden",
            str(golden_path),
            "--top-k",
            "1",
            "--model-profile",
            "extractive_baseline",
            "--report-output",
            str(report_path),
            "--metrics-output",
            str(metrics_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metrics["gen_eval_case_count"] == 1.0
    assert metrics["gen_eval_mean_faithfulness"] >= 0.0
    assert report["report"]["case_count"] == 1
    assert len(list(run_dir.glob("*/run.json"))) == 1
