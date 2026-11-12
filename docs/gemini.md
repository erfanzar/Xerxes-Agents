# Gemini

Xerxes supports Google's Gemini Developer API with an **API key**, using its native REST Generate Content transport. No SDK is required. Google OAuth, Google subscription sign-in, embedded OAuth client secrets, and subscription quota readback are not supported by this provider. Vertex AI is a separate provider (`google-vertex`) with different authentication.

## Add and select an account

In the desktop **Provider accounts** center, choose **Gemini (API key)** and enter a label, key, and model id. Saving stores the profile on this computer; it does not validate the key with Google or make a billable model call. Select the saved account/profile to use it for tasks. Multiple labels can hold different dedicated keys.

Keys remain in owner-only profile storage, not in account-list responses. Moving an account to an SSH workspace requires the separate **Sync to SSH** review and confirmation. Review the destination before approving a transfer. The remote runtime must advertise provider-account isolation (v0.8.35 or newer); older runtimes are refused rather than silently using their shared login. Updating a remote remains an explicit operation, never an automatic account-center restart.

In the terminal UI, run `/provider`, choose **+ Add new profile…**, give it a name, and choose `gemini`. Accept the native base URL:

```text
https://generativelanguage.googleapis.com/v1beta
```

Supply a key or, for an environment-backed profile, leave it blank. Choose a model from the discovered list, or enter the exact model id yourself. `gemini-2.5-flash` is an example, not a promise that every key can access it. The provider's current model list is authoritative.

Existing official URLs ending in `/v1beta/openai` are normalized to the native API root. A saved explicit provider choice is not silently migrated; if an old profile says `openai`, edit its provider type to `gemini`.

## CLI setup

For environment-backed use, configure `GEMINI_API_KEY` outside the repository. `GOOGLE_API_KEY` is accepted as a fallback; when both are set, `GEMINI_API_KEY` wins. Then write a setup configuration:

```sh
xerxes setup --provider gemini --model gemini-2.5-flash
```

Omit `--model` to use the first generation-capable model Google lists. If listing is unavailable, setup asks for an explicit `--model` rather than inventing a default. `--target /path/to/setup.yaml` writes elsewhere; an existing target is not overwritten. The default target is `~/.xerxes/config/setup.yaml` (or `$XERXES_HOME/config/setup.yaml`). This writes a setup file; it does not switch a running task's provider or restart a service.

`--api-key` is also supported, but command-line keys may appear in shell history and process listings. Prefer the app's password field or an environment variable. Setup files containing a key are owner-readable/writable only; do not commit them.

An explicitly selected account's key takes precedence over ambient keys. Isolated account clients with missing credentials fail rather than borrowing another account's environment key. The direct `GeminiClient` likewise treats an explicitly supplied empty key as missing, not as permission to fall back.

## Model discovery and requests

Model discovery calls native `GET /v1beta/models` with `x-goog-api-key`, not a query-string key. It strips `models/` from ids, retains positive input/output token limits and display names, and excludes entries that advertise generation methods but not `generateContent`. Pagination stays on the same endpoint and follows opaque page tokens only. One discovery is bounded to 8 seconds, 1 MiB across pages, and 20 pages. Redirects are refused. Public endpoint DNS/private-address checks are repeated before each page; explicitly trusted stored local endpoints retain their existing local allowance.

Generation uses `:generateContent`; streaming uses `:streamGenerateContent?alt=sse`. The same native adapter handles system instructions, tools and tool replies, supported images, thinking settings/signatures, cancellation, and usage. A stale `responses_api` setting does not move Gemini to the OpenAI Responses API.

A safety-blocked candidate or blocked prompt fails visibly instead of becoming an empty successful answer. Buffered tool calls from a blocked stream are not dispatched. Cancellation stops the reader, including while waiting for the next chunk. An unexpectedly truncated stream remains an error.

Reported usage separates fresh input from cached input, and includes reasoning tokens in total output while reporting them separately. Recorded usage is not an account balance or remaining-quota estimate. Provider HTTP errors retain status and Retry-After metadata for normal retry classification.

## Cloud, Hybrid, and Private

- **Cloud:** a Gemini profile can be the main model.
- **Hybrid:** Gemini can be the cloud lead; local-model agents do local work under the existing data-boundary protections. Selecting Gemini does not disable those protections.
- **Private:** the fully local guard refuses Google's cloud endpoint before making a request. Gemini selection never grants an exception.

Ambient vendor keys are withheld from custom destinations and non-HTTPS versions of the official origin. A custom Gemini-compatible proxy requires an explicit profile key (unless the operator deliberately enables the existing ambient-key override). Generation and discovery refuse redirects so a credential is not forwarded to another destination. Custom proxy paths are not automatically rewritten into Google's native API layout.

All repository verification for this integration uses fake keys and injected transports. No live-key validity, model availability, billing, or deployed-app behavior is implied by those tests.
