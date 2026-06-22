# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provider onboarding flow — interactive /provider slash command panel.

Extracted from daemon/server.py as a mixin to keep the server module navigable.
"""

from __future__ import annotations

import asyncio
import uuid

from ..bridge import profiles
from .gateway import EmitFn


class ProviderFlowMixin:
    """Interactive provider-profile onboarding panel methods.

    Handles the ``/provider`` slash command flow: add/edit/remove provider
    profiles, collect credentials, select models, and persist the result.
    Mixed into :class:`DaemonServer` so all methods retain access to
    ``self.config``, ``self.runtime``, and ``self._provider_flow`` state.
    """

    _PROVIDER_ADD_LABEL = "+ Add new profile…"
    _PROVIDER_EDIT_LABEL = "✎ Edit existing profile…"
    _PROVIDER_REMOVE_LABEL = "✗ Remove existing profile…"
    _PROVIDER_CANCEL_LABEL = "Cancel"
    _PROVIDER_TYPE_OPTIONS: tuple[str, ...] = (
        "auto",
        "openai",
        "openrouter",
        "anthropic",
        "ollama",
        "gemini",
        "deepseek",
        "groq",
        "together",
        "kimi",
        "kimi-code",
        "zhipu",
        "qwen",
        "minimax",
        "lmstudio",
        "custom",
    )

    async def _emit_provider_edit_panel(self, emit: EmitFn) -> None:
        """Ask which profile + field to edit + the new value (batched)."""
        from ..bridge import profiles

        names = [p.get("name", "") for p in profiles.list_profiles() if p.get("name")]
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "edit", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "profile",
                        "question": "Which profile to edit?",
                        "options": names,
                        "allow_free_form": False,
                    },
                    {
                        "id": "field",
                        "question": "Which field?",
                        "options": ["base_url", "api_key", "model", "name", "provider_type"],
                        "allow_free_form": False,
                    },
                    {
                        "id": "value",
                        "question": "New value:",
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_credentials_panel(self, emit: EmitFn, name: str, provider_type: str) -> None:
        """Stage 2 of the Add flow — ask base URL + API key.

        Pre-fills the base URL from :data:`xerxes.llms.registry.PROVIDERS` so
        users picking a well-known provider can usually just hit Enter and
        paste an API key. The model is collected separately in stage 3 after
        we try to enumerate ``/models`` from the actual endpoint.
        """
        from ..llms.registry import PROVIDERS

        default_url = ""
        default_model = ""
        if provider_type and provider_type not in {"auto", "custom"}:
            prov_cfg = PROVIDERS.get(provider_type)
            if prov_cfg is not None:
                # ``base_url`` is optional on the dataclass (anthropic and a
                # few others rely on the SDK default). Treat ``None`` as "no
                # suggestion" rather than letting it crash the f-string.
                default_url = prov_cfg.base_url or ""
                default_model = prov_cfg.models[0] if prov_cfg.models else ""

        url_question = "Base URL"
        if default_url:
            url_question += f" (press Enter for `{default_url}`)"
        url_question += ":"

        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_creds",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "default_url": default_url,
            "default_model": default_model,
        }
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "base_url",
                        "question": url_question,
                        "options": [],
                        "allow_free_form": True,
                    },
                    {
                        "id": "api_key",
                        "question": "API key (blank uses env var when available):",
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_main_panel(self, emit: EmitFn) -> None:
        """Send the top-level provider action question to the TUI."""
        from ..bridge import profiles

        plist = profiles.list_profiles()
        active_name = (profiles.get_active_profile() or {}).get("name", "")

        options: list[str] = []
        # Existing profiles come first — selecting one switches to it.
        for p in plist:
            name = p.get("name", "")
            marker = "  ← active" if name == active_name else ""
            model = p.get("model", "?")
            base = p.get("base_url", "")
            options.append(f"{name}  ({model} @ {base}){marker}")
        options.append(self._PROVIDER_ADD_LABEL)
        if plist:
            options.append(self._PROVIDER_EDIT_LABEL)
            options.append(self._PROVIDER_REMOVE_LABEL)
        options.append(self._PROVIDER_CANCEL_LABEL)

        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "main", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "action",
                        "question": "Provider profiles — pick a profile to switch, or choose an action:",
                        "options": options,
                        "allow_free_form": False,
                    }
                ],
            },
        )

    async def _advance_provider_flow(self, answers: dict[str, str], emit: EmitFn) -> None:
        """State machine that consumes ``question_response`` answers."""
        from ..bridge import profiles

        flow = self._provider_flow
        if flow is None:
            return
        step = flow.get("step", "")

        # Cancel sentinel everywhere — bail without touching disk.
        for v in answers.values():
            if v == self._PROVIDER_CANCEL_LABEL:
                self._provider_flow = None
                await self._emit_slash(emit, "Cancelled.")
                return

        if step == "main":
            choice = answers.get("action", "")
            if choice == self._PROVIDER_ADD_LABEL:
                await self._emit_provider_add_panel(emit)
                return
            if choice == self._PROVIDER_EDIT_LABEL:
                await self._emit_provider_edit_panel(emit)
                return
            if choice == self._PROVIDER_REMOVE_LABEL:
                await self._emit_provider_remove_panel(emit)
                return
            # Otherwise the user picked an existing profile row — pull the
            # name off the front of the rendered label.
            name = choice.split("  ", 1)[0].strip()
            self._provider_flow = None
            if not name:
                await self._emit_slash(emit, "No profile selected.")
                return
            if not profiles.set_active(name):
                await self._emit_slash(emit, f"Failed to switch to `{name}`.")
                return
            try:
                self.runtime.reload({})
                self._sync_runtime_to_connection_session(emit)
            except Exception:
                pass
            switched = profiles.get_active_profile() or {}
            await self._emit_slash(
                emit,
                f"Switched to `{name}` (model: `{switched.get('model', '?')}`).",
            )
            await self._emit_init_done(emit)
            return

        if step == "add_meta":
            # Stage 1 → 2: collect name + provider type, then pop the
            # credentials panel with provider-aware defaults.
            name = (answers.get("name") or "").strip()
            provider_type = (answers.get("provider_type") or "").strip() or "auto"
            if not name:
                self._provider_flow = None
                await self._emit_slash(emit, "Add cancelled — profile name is required.")
                return
            await self._emit_provider_credentials_panel(emit, name, provider_type)
            return

        if step == "add_creds":
            # Stage 2 → 3: capture base_url + api_key, then try to enumerate
            # models from the endpoint and present the result as a picker.
            name = flow.get("name", "")
            provider_type = flow.get("provider_type", "auto")
            default_url = flow.get("default_url", "")
            default_model = flow.get("default_model", "")
            base_url = (answers.get("base_url") or "").strip() or default_url
            api_key = (answers.get("api_key") or "").strip()
            if not base_url:
                self._provider_flow = None
                await self._emit_slash(
                    emit,
                    f"Add cancelled — base_url is required for `{provider_type}` (no registry default to fall back to).",
                )
                return
            await self._emit_provider_model_panel(
                emit,
                name=name,
                provider_type=provider_type,
                base_url=base_url,
                api_key=api_key,
                default_model=default_model,
            )
            return

        if step == "add_model":
            # Stage 3 (picker). The user either picked a real model, picked
            # the "type custom" sentinel, or typed a free-form id.
            picked = (answers.get("model") or "").strip()
            type_sentinel = flow.get("type_sentinel", "")
            if picked == type_sentinel:
                await self._emit_provider_custom_model_panel(
                    emit,
                    flow.get("name", ""),
                    flow.get("provider_type", "auto"),
                    flow.get("base_url", ""),
                    flow.get("api_key", ""),
                    flow.get("default_model", ""),
                )
                return
            # Empty picker answer = accept registry default.
            model = picked or flow.get("default_model", "")
            await self._finalize_provider_add(
                emit,
                flow.get("name", ""),
                flow.get("provider_type", "auto"),
                flow.get("base_url", ""),
                flow.get("api_key", ""),
                model,
            )
            return

        if step == "add_model_text":
            # Free-text fallback (user picked "type custom" or /models failed).
            model = (answers.get("model") or "").strip() or flow.get("default_model", "")
            await self._finalize_provider_add(
                emit,
                flow.get("name", ""),
                flow.get("provider_type", "auto"),
                flow.get("base_url", ""),
                flow.get("api_key", ""),
                model,
            )
            return

        if step == "edit":
            self._provider_flow = None
            target = (answers.get("profile") or "").strip()
            field = (answers.get("field") or "").strip()
            value = (answers.get("value") or "").strip()
            if not target or not field:
                await self._emit_slash(emit, "Edit cancelled — no profile or field selected.")
                return
            existing = next(
                (p for p in profiles.list_profiles() if p.get("name") == target),
                None,
            )
            if existing is None:
                await self._emit_slash(emit, f"No profile named `{target}` (it may have been removed).")
                return
            merged = dict(existing)
            # The user-facing field "provider_type" maps to the underlying
            # "provider" column on the profile dict.
            store_key = "provider" if field == "provider_type" else field
            if field == "name":
                # Rename: save under the new key and remove the old.
                new_name = value
                if not new_name:
                    await self._emit_slash(emit, "Edit cancelled — new name is empty.")
                    return
                profiles.save_profile(
                    new_name,
                    merged.get("base_url", ""),
                    merged.get("api_key", ""),
                    merged.get("model", ""),
                    merged.get("provider", ""),
                )
                profiles.delete_profile(target)
                target = new_name
            else:
                # ``auto`` for provider_type means "let the URL heuristic
                # pick" — store as empty so save_profile triggers the guess.
                stored_value = "" if field == "provider_type" and value == "auto" else value
                merged[store_key] = stored_value
                profiles.save_profile(
                    merged.get("name", ""),
                    merged.get("base_url", ""),
                    merged.get("api_key", ""),
                    merged.get("model", ""),
                    merged.get("provider", ""),
                )
            try:
                self.runtime.reload({})
                self._sync_runtime_to_connection_session(emit)
            except Exception:
                pass
            shown = "***redacted***" if field == "api_key" else value
            await self._emit_slash(emit, f"Updated `{target}`: `{field}` = `{shown}`.")
            await self._emit_init_done(emit)
            return

        if step == "remove":
            self._provider_flow = None
            target = (answers.get("profile") or "").strip()
            confirm = (answers.get("confirm") or "").strip().lower()
            if not target:
                await self._emit_slash(emit, "Remove cancelled.")
                return
            if confirm not in {"yes", "y"}:
                await self._emit_slash(emit, f"Remove cancelled — `{target}` was not deleted.")
                return
            if not profiles.delete_profile(target):
                await self._emit_slash(emit, f"Failed to remove `{target}`.")
                return
            try:
                self.runtime.reload({})
                self._sync_runtime_to_connection_session(emit)
            except Exception:
                pass
            await self._emit_slash(emit, f"Removed profile `{target}`.")
            await self._emit_init_done(emit)
            return

        # Unknown step — defensive reset.
        self._provider_flow = None

    async def _emit_provider_remove_panel(self, emit: EmitFn) -> None:
        """Ask which profile to remove + confirm."""
        from ..bridge import profiles

        names = [p.get("name", "") for p in profiles.list_profiles() if p.get("name")]
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "remove", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "profile",
                        "question": "Which profile to remove?",
                        "options": names,
                        "allow_free_form": False,
                    },
                    {
                        "id": "confirm",
                        "question": "Type `yes` to confirm deletion:",
                        "options": ["yes", "no"],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _finalize_provider_add(
        self,
        emit: EmitFn,
        name: str,
        provider_type: str,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        """Save the new profile, reload the runtime, and confirm to the user.

        Called from every terminal branch of the Add flow (model picker,
        free-text fallback). Handles the same provider-tag resolution and
        ``init_done`` refresh the old single-shot Add path used to do.
        """
        self._provider_flow = None
        saved_type = "" if provider_type == "auto" else provider_type
        if not model:
            await self._emit_slash(
                emit,
                f"Add cancelled — no model id given and `{provider_type}` has no default.",
            )
            return
        try:
            profiles.save_profile(name, base_url, api_key, model, saved_type)
        except Exception as exc:
            await self._emit_slash(emit, f"Failed to save profile: `{exc}`")
            return
        try:
            self.runtime.reload({})
            self._sync_runtime_to_connection_session(emit)
        except Exception:
            pass
        saved = next(
            (p for p in profiles.list_profiles() if p.get("name") == name),
            {},
        )
        resolved_type = saved.get("provider") or saved_type or "custom"
        await self._emit_slash(
            emit,
            f"Added profile `{name}` (type `{resolved_type}`, model `{model}` @ `{base_url}`) and switched to it.",
        )
        await self._emit_init_done(emit)

    async def _emit_provider_add_panel(self, emit: EmitFn) -> None:
        """Stage 1 of the Add flow — ask for name + provider type.

        Stage 2 (``_emit_provider_credentials_panel``) follows once the
        provider type is known so its base URL and default model can be
        pre-filled from :data:`xerxes.llms.registry.PROVIDERS`. Without that
        split the user would have to retype URLs we already know.
        """
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "add_meta", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "name",
                        "question": "New profile name (short slug, e.g. `kimi-code` or `openai-prod`):",
                        "options": [],
                        "allow_free_form": True,
                    },
                    {
                        "id": "provider_type",
                        "question": "Inference provider type:",
                        "options": list(self._PROVIDER_TYPE_OPTIONS),
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_model_panel(
        self,
        emit: EmitFn,
        *,
        name: str,
        provider_type: str,
        base_url: str,
        api_key: str,
        default_model: str,
    ) -> None:
        """Stage 3 of the Add flow — try ``GET /models``, then ask the user.

        Network call runs on a worker thread (so the daemon's event loop
        stays responsive) with a hard 3s timeout. Every failure path falls
        back to a free-form question with the registry default as a hint —
        the Add flow never gets stuck because of a flaky provider.
        """
        # Sentinel that lets the user open the free-text mode even when a
        # list of options is offered. Detected in ``_advance_provider_flow``.
        type_sentinel = "— Type a custom model id —"

        # Pull the model catalogue. ``profiles.fetch_models`` is synchronous
        # httpx; offload to a thread so we don't block other RPC handlers.
        models: list[str] = []
        fetch_error: str = ""
        try:
            models = await asyncio.to_thread(profiles.fetch_models, base_url, api_key)
        except Exception as exc:
            fetch_error = str(exc)
        # Order: registry default first (if it's in the list), then alphabetical.
        if default_model and default_model in models:
            models = [default_model] + [m for m in models if m != default_model]

        question_text = "Pick a model"
        if fetch_error:
            question_text += " (couldn't reach `/models` — type one manually)"
        elif not models:
            question_text += " (the endpoint returned no catalogue — type one)"
        elif default_model:
            question_text += f" (first option = registry default `{default_model}`)"
        question_text += ":"

        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_model",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "base_url": base_url,
            "api_key": api_key,
            "default_model": default_model,
            "type_sentinel": type_sentinel,
        }
        # Always allow free-form so users can paste a model id even when the
        # provider returns a list. Append the sentinel after real options so
        # picking it triggers the custom-text follow-up.
        options = list(models)
        if options:
            options.append(type_sentinel)
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "model",
                        "question": question_text,
                        "options": options,
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _slash_provider(self, args: str, emit: EmitFn) -> None:
        """Open the interactive provider panel (or quick-switch via ``/provider <name>``).

        Forms:
          * ``/provider``        — pops a TUI question panel listing every
            profile plus ``Add``/``Edit``/``Remove``/``Cancel`` actions.
          * ``/provider <name>`` — quick switch by name (no panel).

        The panel is built on top of the standard ``question_request`` wire
        event, so the same arrow-key + Enter UX as agent ``ask_user`` prompts
        applies. Multi-step actions (add/edit/remove) are batched into a
        single follow-up question_request with one entry per field.
        """
        from ..bridge import profiles

        target = args.strip()
        plist = profiles.list_profiles()

        # Inline quick-switch — bypass the panel entirely.
        if target:
            if not any(p.get("name") == target for p in plist):
                names = ", ".join(f"`{p.get('name', '')}`" for p in plist if p.get("name"))
                msg = f"No profile named `{target}`."
                if names:
                    msg += f" Available: {names}."
                await self._emit_slash(emit, msg)
                return
            if not profiles.set_active(target):
                await self._emit_slash(emit, f"Failed to switch to `{target}`.")
                return
            try:
                self.runtime.reload({})
                self._sync_runtime_to_connection_session(emit)
            except Exception:
                pass
            switched = profiles.get_active_profile() or {}
            await self._emit_slash(
                emit,
                f"Switched to `{target}` (model: `{switched.get('model', '?')}`).",
            )
            await self._emit_init_done(emit)
            return

        await self._emit_provider_main_panel(emit)

    async def _emit_provider_custom_model_panel(
        self, emit: EmitFn, name: str, provider_type: str, base_url: str, api_key: str, default_model: str
    ) -> None:
        """Fallback panel asking for a free-text model id (used after the user
        picks the "type custom" sentinel on the model picker)."""
        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_model_text",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "base_url": base_url,
            "api_key": api_key,
            "default_model": default_model,
        }
        question_text = "Model id"
        if default_model:
            question_text += f" (press Enter for `{default_model}`)"
        question_text += " — e.g. `gpt-4o`, `kimi-for-coding`:"
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "model",
                        "question": question_text,
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )
