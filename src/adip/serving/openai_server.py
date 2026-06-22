"""Minimal OpenAI-compatible HTTP server for local development."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from adip.config.model_profiles import ModelProfile
from adip.serving.backends import ServingBackend


def run_openai_compatible_server(
    backend: ServingBackend,
    profile: ModelProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    handler = build_handler(backend, profile)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {profile.profile_id} on http://{host}:{port}")
    print("Health: GET /health")
    print("Chat:   POST /v1/chat/completions")
    server.serve_forever()


def build_handler(backend: ServingBackend, profile: ModelProfile):
    class OpenAICompatibleHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404, "Not found")
                return
            self._send_json(
                {
                    "status": "ok",
                    "model_profile": profile.profile_id,
                    "model_name": backend.model_name,
                    "model_provider": backend.model_provider,
                }
            )

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_error(404, "Not found")
                return

            payload = self._read_json()
            prompt = messages_to_prompt(payload.get("messages", []))
            max_tokens = payload.get("max_tokens")
            response = backend.generate(prompt, max_new_tokens=max_tokens)
            self._send_json(
                {
                    "id": "chatcmpl-adip-local",
                    "object": "chat.completion",
                    "model": backend.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response.text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.input_token_count,
                        "completion_tokens": response.output_token_count,
                        "total_tokens": response.input_token_count + response.output_token_count,
                    },
                }
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw else {}

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OpenAICompatibleHandler


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()
