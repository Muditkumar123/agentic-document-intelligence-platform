# API Keys

The project supports hosted OpenAI-compatible model APIs without storing secrets in code.

## Safe Setup

Copy the template:

```bash
cp .env.example .env
```

Edit `.env` and set your real key:

```bash
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_API_KEY=your-real-key
```

`.env` is ignored by Git. Keep real keys out of commits, screenshots, notebooks, traces, and chat messages.

## DeepSeek Profiles

The cloud profiles use the OpenAI-compatible adapter:

- `deepseek_v4_flash_cloud`: faster hosted DeepSeek profile for normal QA and summaries.
- `deepseek_v4_pro_cloud`: stronger hosted DeepSeek profile for reasoning-heavy agent work.

The profiles read:

- endpoint from `DEEPSEEK_BASE_URL`, falling back to the profile default.
- API key from `DEEPSEEK_API_KEY`.

## Website Entry

You can also paste the key directly into the dashboard:

1. Open the `Agent` tab.
2. Choose `DeepSeek API Fast` or `DeepSeek API Pro`.
3. Paste the key into `API Key`.
4. Leave `API Endpoint` empty for the profile default, or provide an override.
5. Run the agent.

Dashboard-entered keys are session-only. They are sent with the current request, masked in AgentOps traces, and not written to `.env` or browser storage.

## Add Model In The Website

Use `Add Model` when the model is not already in the dropdown.

Fields:

- `Model Label`: the name shown in the dashboard dropdown.
- `Model API Name`: the provider model identifier, such as `deepseek-v4-flash`.
- `Model API Endpoint`: the chat completion URL. Leave empty to use the DeepSeek default.
- `Model API Key`: the key for this browser session.

After clicking `Add Model`, the model appears in both `Writer Model` and `Reasoning Model`. At minimum you need `Model API Name` and a key. The key is kept only in the current browser session.

## Troubleshooting

`HTTP Error 402: Payment Required` means the provider accepted the request path but the account cannot run the selected model. For DeepSeek, check that the API account has billing enabled, available credits, and access to the selected model.

If you see this during `evidence_verifier`, the reasoning model failed before the writer ran. The agent now treats optional reasoning failures as non-fatal, records the error in verification notes, and continues with the normal cited writer. You can also set `Reasoning Model` to `None` to avoid the verifier API call.

For Groq:

- `Model API Endpoint`: `https://api.groq.com/openai/v1/chat/completions`
- `Model API Name`: start with `llama-3.1-8b-instant`
- `Reasoning Model`: `None`

Use `Test Model` before `Add Model`. A Groq `403 Forbidden` usually means the key is valid but the project or organization is not allowed to use the selected model. Check Groq project model permissions and try `llama-3.1-8b-instant` first.

For Gemini:

- `Model API Endpoint`: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- `Model API Name`: start with `gemini-2.5-flash`
- `Reasoning Model`: `None`

Paste only the URL into `Model API Endpoint`. Do not paste labels such as `Model API Key:` into the endpoint field.

If a hosted model's answer contains its own narration such as `Continuation:` or `The last word was ...`, or repeats a paragraph, that was a continuation-splice bug in the answer writer. It is fixed; see [FIXES.md](FIXES.md) for the root cause and the regression tests.

If a Gemini or other "thinking" model returns a very short or cut-off answer, it most likely spent its token budget on hidden reasoning before writing. In the Agent tab, set **Writer Thinking** to **Off** (this sends `reasoning_effort: "none"`, which disables thinking on gemini-2.5-flash so the whole budget goes to the answer), or raise `Max Tokens`. The dashboard shows an amber warning above the answer when an answer is truncated, and the agent response carries the same note as `answer_warning`. See [FIXES.md](FIXES.md).

## Start The API

After `.env` is configured:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.api \
  --host 127.0.0.1 \
  --port 8010
```

Open:

```text
http://127.0.0.1:8010/
```

In the `Agent` tab, choose `DeepSeek API Fast` or `DeepSeek API Pro`. The runtime status line should show `API key ready` when the server can read the key.

## CLI Smoke Test

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "Create a cited research brief from the indexed document." \
  --task brief \
  --model-profile deepseek_v4_pro_cloud \
  --max-new-tokens 512
```

## Generic OpenAI-Compatible Endpoint

For local vLLM, SGLang, LiteLLM, or another compatible provider:

```bash
ADIP_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
ADIP_OPENAI_API_KEY=local-dev-key
```

Profiles can declare their own `endpoint_env` and `api_key_env` in `config/model_profiles.yaml`.
