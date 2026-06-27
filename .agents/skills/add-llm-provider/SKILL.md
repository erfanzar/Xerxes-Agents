---
name: add-llm-provider
description: Scaffold a new LLM provider entry in the Xerxes registry. Covers ProviderConfig, COSTS, prefix detection, default headers, and context limits.
version: 1.0.0
tags: [llm, provider, registry, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool]
---

# When to use

Use this skill when you need to add support for a new LLM provider to Xerxes-Agents. The provider must expose an OpenAI-compatible API (or an API that the existing `OpenAILLM` / `AnthropicLLM` / `GeminiLLM` adapter can speak to with minor tweaks).

Examples:
- A new inference provider (e.g., Fireworks, Together AI, Groq, AI21, etc.)
- A new local endpoint (e.g., vLLM, TGI, custom internal API)
- A new cloud provider with an OpenAI-compatible gateway

Do NOT use this for providers that require a completely new wire protocol (e.g., a new non-OpenAI, non-Anthropic, non-Gemini protocol). In that case, you need a new `BaseLLM` subclass in `llms/` too.

# How to use

## 1. Inspect the current registry

Read `src/python/xerxes/llms/registry.py` to understand the current `PROVIDERS`, `COSTS`, `_PREFIX_MAP`, `_PROVIDER_DEFAULT_HEADERS`, and `_MODEL_CONTEXT_LIMITS` structures.

## 2. Add a `ProviderConfig` entry to `PROVIDERS`

In `src/python/xerxes/llms/registry.py`, locate the `PROVIDERS` dict and add a new entry:

```python
"myprovider": ProviderConfig(
    name="myprovider",
    type="openai",          # or "anthropic", "gemini", etc. depending on adapter
    api_key_env="MYPROVIDER_API_KEY",
    base_url="https://api.myprovider.com/v1",
    context_limit=128_000,  # default for this provider
    models=[
        "myprovider-model-1",
        "myprovider-model-2",
    ],
    default_api_key=None,
),
```

**Rules:**
- `name` must match the dict key exactly.
- `type` must be one of the supported adapter types: `"openai"`, `"anthropic"`, `"gemini"`, `"custom"`.
- `api_key_env` is the environment variable name the user should set. If the provider requires no key (e.g., local Ollama), set it to `None`.
- `base_url` is the default base URL. If the provider is just a prefix alias for an existing provider (e.g., `groq` uses OpenAI-compatible API), set `base_url` to the provider's actual endpoint.

## 3. Add per-model pricing to `COSTS`

In the same file, locate the `COSTS` dict (USD per 1M tokens). Add:

```python
"myprovider-model-1": (0.50, 1.50),   # (input_cost_per_1M, output_cost_per_1M)
"myprovider-model-2": (5.00, 15.00),
```

**Rules:**
- Every model listed in `PROVIDERS[...].models` MUST have a `COSTS` entry, even if free (use `(0.0, 0.0)`).
- Costs are per **1 million tokens**.

## 4. Add prefix detection to `_PREFIX_MAP`

Locate `_PREFIX_MAP` (longest-match prefix → provider name). Add:

```python
"myprovider-": "myprovider",
"myprovider-model-": "myprovider",
```

**Rules:**
- Longest match wins. Be specific enough to avoid collisions with other providers.
- Include both the bare model prefix and the provider-prefixed form if users might write `provider/model` notation.
- If the provider has no distinctive prefix (e.g., a custom local endpoint), you can skip this step and rely on `detect_provider()` fallback logic, but it's better to add a prefix.

## 5. Add custom headers if needed

Locate `_PROVIDER_DEFAULT_HEADERS`. If the provider requires a custom User-Agent or other header (e.g., Kimi Code's `claude-code/1.0.0` spoof), add:

```python
"myprovider": {
    "User-Agent": "myprovider-client/1.0.0",
},
```

**Rules:**
- Only add headers if the provider documentation explicitly requires them.
- Headers are merged into every request made by the adapter.

## 6. Add context limit overrides if needed

Locate `_MODEL_CONTEXT_LIMITS`. If any model has a non-default context window, add:

```python
"myprovider-model-1": 256_000,
```

**Rules:**
- Only add overrides for models that deviate from the provider's default `context_limit` set in `ProviderConfig`.
- Use the exact model name string as the key.

## 7. Verify

After editing, run the following to confirm the provider is discoverable:

```bash
uv run python -c "
from xerxes.llms.registry import detect_provider, list_all_models
print(detect_provider('myprovider-model-1'))
print(list_all_models())
"
```

## 8. Run lint

```bash
uv run ruff check --fix src/python/xerxes/llms/registry.py
```

## 9. Add a test

If the provider has distinctive behavior (custom headers, special prefix, context limit overrides), add a test in `tests/llms/test_registry.py` following the existing pattern (e.g., `test_detect_provider_*`).

## Common pitfalls

- **Mismatched model names:** `PROVIDERS["myprovider"].models` and `COSTS` keys must be identical strings. Typos here cause `calc_cost()` to return `None` silently.
- **Missing COSTS entry:** Even free models need `(0.0, 0.0)`.
- **Prefix collisions:** `_PREFIX_MAP` is longest-match. If two providers share a prefix (e.g., `mistral` and `mistral-large`), the longer key wins. Check the existing map before adding.
- **Provider type mismatch:** If the provider is NOT OpenAI-compatible, you need a new `BaseLLM` subclass. See `llms/openai.py` for the OpenAI-compatible adapter pattern and `llms/anthropic.py` for a non-OpenAI adapter.
