import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adip.config.model_profiles import load_model_profile
from adip.llmops.models import (
    GenerationRequest,
    GenerationResponse,
    OpenAICompatibleChatAdapter,
    append_continuation,
    build_answer_warning,
    clean_continuation_text,
    collapse_near_duplicate_lines,
    extract_chat_completion_text,
    strip_reasoning_blocks,
)
from adip.serving.backends import ExtractiveServingBackend
from adip.serving.environment import package_availability, resolve_hf_cache_root
from adip.serving.launch import build_launch_plan
from adip.serving.openai_server import build_handler, messages_to_prompt


def test_package_availability_reports_known_packages():
    packages = package_availability(["json", "definitely_missing_adip_package"])

    assert packages["json"] is True
    assert packages["definitely_missing_adip_package"] is False


def test_resolve_hf_cache_root_from_argument(tmp_path):
    assert resolve_hf_cache_root(tmp_path) == tmp_path


def test_build_qwen_launch_plan_contains_vllm_command():
    plan = build_launch_plan("qwen3_8b_default")

    assert plan.model_name == "Qwen/Qwen3-8B"
    assert "vllm serve Qwen/Qwen3-8B" in plan.commands["vllm"]
    assert "--tensor-parallel-size 1" in plan.commands["vllm"]


def test_build_deepseek_32b_launch_plan_uses_tensor_parallel():
    plan = build_launch_plan("deepseek_r1_distill_qwen_32b_stretch")

    assert "DeepSeek-R1-Distill-Qwen-32B" in plan.commands["vllm"]
    assert "--tensor-parallel-size 2" in plan.commands["vllm"]


def test_build_deepseek_cloud_launch_plan_uses_hosted_api_env():
    plan = build_launch_plan("deepseek_v4_pro_cloud")

    assert plan.recommended_runtime == "hosted_api"
    assert "hosted_api_env" in plan.commands
    assert "DEEPSEEK_API_KEY" in plan.commands["hosted_api_env"]
    assert "vllm" not in plan.commands


def test_extractive_serving_backend_generates_response():
    profile = load_model_profile("extractive_baseline")
    backend = ExtractiveServingBackend(profile)

    response = backend.generate("hello serving layer")

    assert response.model_name == "grounded-extractive-v1"
    assert "hello serving layer" in response.text
    assert response.input_token_count == 3


def test_messages_to_prompt_combines_roles():
    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "Be grounded."},
            {"role": "user", "content": "Summarize this."},
        ]
    )

    assert prompt == "system: Be grounded.\nuser: Summarize this."


def test_serving_response_is_json_serializable():
    profile = load_model_profile("extractive_baseline")
    backend = ExtractiveServingBackend(profile)
    response = backend.generate("serialize this")

    json.dumps(response.to_dict())


def test_openai_compatible_adapter_can_call_local_server():
    profile = load_model_profile("extractive_baseline")
    backend = ExtractiveServingBackend(profile)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(backend, profile))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        adapter = OpenAICompatibleChatAdapter(
            model_name="grounded-extractive-v1",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        response = adapter.generate(
            GenerationRequest(
                prompt="hello through adapter",
                question="hello",
                task_type="qa",
                domain_preset="general",
                evidence=[],
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.model_provider == "openai_compatible"
    assert "hello through adapter" in response.text
    assert response.output_token_count > 0


def test_openai_compatible_adapter_sends_api_headers():
    captured = {}

    class HeaderCaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["user_agent"] = self.headers.get("User-Agent")
            captured["accept"] = self.headers.get("Accept")
            captured["authorization"] = self.headers.get("Authorization")
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps(
                {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), HeaderCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        adapter = OpenAICompatibleChatAdapter(
            model_name="header-test-model",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        adapter.generate(
            GenerationRequest(
                prompt="hello",
                question="hello",
                task_type="qa",
                domain_preset="general",
                evidence=[],
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert captured["user_agent"] == "agentic-document-intelligence-platform/0.1.0"
    assert captured["accept"] == "application/json"
    assert captured["authorization"] == "Bearer test-key"


def test_openai_compatible_adapter_continues_when_finish_reason_is_length():
    calls = {"count": 0}

    class ContinuationHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls["count"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if calls["count"] == 1:
                payload = {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "The platform ingests documents and"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6},
                }
            else:
                payload = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": " preserves metadata."},
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ContinuationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        adapter = OpenAICompatibleChatAdapter(
            model_name="continuation-test-model",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        response = adapter.generate(
            GenerationRequest(
                prompt="brief",
                question="brief",
                task_type="brief",
                domain_preset="general",
                evidence=[],
                max_new_tokens=4,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.text == "The platform ingests documents and preserves metadata."
    assert response.raw["continuation_count"] == 1
    assert response.raw["finish_reasons"] == ["length", "stop"]
    assert response.input_token_count == 18
    assert response.output_token_count == 9


def test_clean_continuation_text_strips_meta_preamble_and_quotes():
    raw = (
        "Wait, let's make sure the continuation is seamless.\n"
        'The last word was "number".\n'
        "Continuation:\n"
        '"of rounds, and input differences, meaning a model that performs well [1]."'
    )

    cleaned = clean_continuation_text(raw)

    assert cleaned.startswith("of rounds, and input differences")
    assert "Wait" not in cleaned
    assert "Continuation:" not in cleaned
    assert "last word" not in cleaned
    assert not cleaned.startswith('"')


def test_clean_continuation_text_keeps_inline_label_content():
    assert clean_continuation_text("Continuation: of rounds and beyond [1].") == (
        "of rounds and beyond [1]."
    )


def test_clean_continuation_text_preserves_plain_continuation():
    assert clean_continuation_text(" preserves metadata.") == "preserves metadata."


def test_append_continuation_drops_verbatim_seam_overlap():
    text = "The attack reduces the data complexity to 2^28 and recovers the key."
    continuation = "recovers the key. It then verifies the distinguisher [1]."

    merged = append_continuation(text, continuation)

    assert merged.count("recovers the key") == 1
    assert merged.endswith("verifies the distinguisher [1].")


def test_append_continuation_keeps_non_overlapping_join():
    assert append_continuation("the specific cipher, number", "of rounds and inputs") == (
        "the specific cipher, number of rounds and inputs"
    )


def test_collapse_near_duplicate_lines_removes_restarted_paragraph():
    text = (
        "of rounds, and input differences, meaning a model that performs well on one "
        "configuration cannot be hastily applied to others [1].\n"
        "\n"
        "### 3. of rounds, or input differences, meaning a model that performs well on one "
        "configuration cannot be hastily applied to others [1].\n"
        "\n"
        "### 3. Source Coverage"
    )

    collapsed = collapse_near_duplicate_lines(text)

    assert collapsed.count("cannot be hastily applied to others") == 1
    assert "### 3. Source Coverage" in collapsed


def _generation_with_raw(raw):
    return GenerationResponse(
        text="partial answer",
        model_provider="openai_compatible",
        model_name="thinking-model",
        latency_ms=1.0,
        input_token_count=10,
        output_token_count=5,
        raw=raw,
    )


def test_build_answer_warning_flags_reasoning_token_starvation():
    generation = _generation_with_raw(
        {
            "finish_reason": "length",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 1900}},
        }
    )

    warning = build_answer_warning(generation, max_new_tokens=2048)

    assert warning is not None
    assert "1900" in warning
    assert "2048-token budget" in warning
    assert "Max Tokens" in warning


def test_build_answer_warning_handles_length_without_reasoning_usage():
    generation = _generation_with_raw({"finish_reason": "length", "usage": {}})

    warning = build_answer_warning(generation, max_new_tokens=512)

    assert warning is not None
    assert "cut off" in warning
    assert "Max Tokens" in warning


def test_build_answer_warning_is_none_when_answer_completes():
    assert build_answer_warning(_generation_with_raw({"finish_reason": "stop"})) is None


def test_build_answer_warning_is_none_for_non_hosted_generation():
    # Extractive/local adapters carry no finish_reason and never truncate.
    assert build_answer_warning(_generation_with_raw({"adapter": "deterministic_extractive"})) is None


def test_build_answer_warning_infers_reasoning_from_usage_gap():
    # Gemini's OpenAI-compatible usage omits reasoning_tokens; it is the gap between
    # total_tokens and (prompt_tokens + completion_tokens).
    generation = _generation_with_raw(
        {
            "finish_reason": "length",
            "usage": {"total_tokens": 608, "prompt_tokens": 12, "completion_tokens": 132},
        }
    )

    warning = build_answer_warning(generation, max_new_tokens=600)

    assert warning is not None
    assert "464" in warning  # 608 - 12 - 132
    assert "Max Tokens" in warning


def test_strip_reasoning_blocks_removes_unopened_think_close():
    # DeepSeek-R1 distills often emit reasoning with no opening tag (the template
    # injected it), leaving a stray "</think>" before the real answer.
    raw = "Okay, let me reason about the evidence.\nMore reasoning here.\n</think>\n\nThe key findings are clear [1]."

    assert strip_reasoning_blocks(raw) == "The key findings are clear [1]."


def test_strip_reasoning_blocks_keeps_plain_answer():
    assert strip_reasoning_blocks("A grounded answer with no reasoning [1].") == (
        "A grounded answer with no reasoning [1]."
    )


def test_openai_compatible_adapter_sanitizes_garbled_continuation():
    """Reproduces the leaked Gemini answer: a meta-narrated, restarted continuation."""
    calls = {"count": 0}

    class GarbledContinuationHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls["count"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if calls["count"] == 1:
                content = (
                    "## 2. Findings\n"
                    "The optimal network structure varies depending on the specific "
                    "cipher, number"
                )
                finish = "length"
            else:
                content = (
                    "Wait, let's make sure the continuation is seamless.\n"
                    'The last word was "number".\n'
                    "Continuation:\n"
                    "of rounds, and input differences, meaning a model that performs "
                    "well on one configuration cannot be hastily applied to others [1].\n"
                    "\n"
                    "### 3. of rounds, or input differences, meaning a model that performs "
                    "well on one configuration cannot be hastily applied to others [1].\n"
                    "\n"
                    "### 3. Source Coverage\n"
                    "* Document: SIMON.pdf"
                )
                finish = "stop"
            payload = {
                "choices": [{"finish_reason": finish, "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), GarbledContinuationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        adapter = OpenAICompatibleChatAdapter(
            model_name="garbled-continuation-model",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        response = adapter.generate(
            GenerationRequest(
                prompt="brief",
                question="tell me about this paper",
                task_type="brief",
                domain_preset="academic",
                evidence=[],
                max_new_tokens=64,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    text = response.text
    # Meta-commentary must not leak into the visible answer.
    assert "Wait" not in text
    assert "Continuation:" not in text
    assert "last word" not in text
    # The seam reads as one continuous sentence...
    assert "specific cipher, number of rounds" in text
    # ...the restarted near-duplicate paragraph is collapsed...
    assert text.count("cannot be hastily applied to others") == 1
    # ...and the genuine next section survives.
    assert "### 3. Source Coverage" in text
    assert response.raw["continuation_count"] == 1


def test_openai_compatible_adapter_reports_http_error_body():
    class ForbiddenHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps(
                {
                    "error": {
                        "message": "Model is not allowed for this project.",
                        "code": "model_permission_denied",
                    }
                }
            ).encode("utf-8")
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ForbiddenHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        adapter = OpenAICompatibleChatAdapter(
            model_name="restricted-model",
            endpoint_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        try:
            adapter.generate(
                GenerationRequest(
                    prompt="hello",
                    question="hello",
                    task_type="qa",
                    domain_preset="general",
                    evidence=[],
                )
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            message = ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "HTTP Error 403" in message
    assert "restricted-model" in message
    assert "model_permission_denied" in message


def test_extract_chat_completion_text_accepts_list_content_and_gemini_candidates():
    assert (
        extract_chat_completion_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello "},
                                {"type": "text", "text": "Gemini"},
                            ]
                        }
                    }
                ]
            }
        )
        == "Hello Gemini"
    )
    assert (
        extract_chat_completion_text(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Candidate text"},
                            ]
                        }
                    }
                ]
            }
        )
        == "Candidate text"
    )


def test_extract_chat_completion_text_reports_missing_content():
    try:
        extract_chat_completion_text({"choices": [{"message": {"role": "assistant"}}]})
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = ""

    assert "did not include assistant text" in message
    assert '"role": "assistant"' in message


def test_extract_chat_completion_text_strips_reasoning_blocks():
    text = extract_chat_completion_text(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>I should inspect the evidence.</think>\n\nFinal cited answer."
                    }
                }
            ]
        }
    )

    assert text == "Final cited answer."


def test_extract_chat_completion_text_rejects_reasoning_only_output():
    try:
        extract_chat_completion_text(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "Private reasoning without a final answer."
                        }
                    }
                ]
            }
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = ""

    assert "only included reasoning content" in message


def test_extract_chat_completion_text_explains_length_without_content():
    try:
        extract_chat_completion_text(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant"},
                    }
                ],
                "usage": {"completion_tokens": 0},
            }
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = ""

    assert "finish_reason=length" in message
    assert "Increase Max Tokens" in message
