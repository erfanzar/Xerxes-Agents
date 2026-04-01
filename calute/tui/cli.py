# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Command-line entrypoint for launching the Calute TUI.

This module provides the CLI argument parser, environment-based provider and
model detection, agent construction helpers, and the ``main()`` function
that wires everything together and hands control to the Textual UI.
"""

from __future__ import annotations

import argparse
import os
import sys
import typing as tp
from urllib.parse import urlparse

from .. import __version__
from ..calute import Calute
from ..llms import create_llm
from ..operators import OperatorRuntimeConfig
from ..runtime.features import RuntimeFeaturesConfig
from ..runtime.profiles import PromptProfile
from ..tools.standalone import ExecutePythonCode, ListDir, ReadFile, WriteFile
from ..types import Agent
from .app import launch_tui
from .model_discovery import discover_available_models
from .terminal_config import TerminalConfigStore, TerminalProfile

DEFAULT_AGENT_ID = "assistant"
DEFAULT_AGENT_NAME = "Calute"
DEFAULT_INSTRUCTIONS = (
    "You are Calute, a pragmatic terminal assistant. Answer directly when the request can be handled from"
    " the conversation alone. Use tools sparingly and only for live information, workspace inspection,"
    " execution, or verification. If the user explicitly asks you to search/look up/browse the web and a"
    " web search tool is available, use it instead of answering from memory. If the user gives a generic"
    " follow-up like `search the web`, `look it up`, or `find it`, infer the topic from the latest relevant"
    " user request instead of asking the same clarification again, then choose the web search tool if"
    " needed. Read tool descriptions and parameter docs carefully, and use the smallest correct tool"
    " sequence. If tools are available or prior tool results are already in the conversation, do not claim"
    " that you cannot browse or access current information. Treat search-result snippets as leads rather"
    " than verified facts unless you opened the source and confirmed them. Never simulate tool calls or emit"
    " tool/XML wrappers in normal answers. Put the final answer in plain assistant text, not in a scratchpad"
    " or reasoning field. Keep responses concise unless depth is needed."
)
DEFAULT_TOOLS = [
    ReadFile,
    WriteFile,
    ListDir,
    ExecutePythonCode,
]


def canonical_provider_name(provider: str | None) -> str:
    """Normalize provider aliases to their canonical provider names.

    Maps common shorthand names (e.g. ``"oai"``, ``"claude"``, ``"google"``,
    ``"local"``) to the canonical identifiers expected by the LLM factory.

    Args:
        provider: Raw provider string, which may be ``None``, an alias, or
            already canonical. Leading/trailing whitespace is stripped and the
            value is lowercased before lookup.

    Returns:
        The canonical provider name (e.g. ``"openai"``, ``"anthropic"``,
        ``"gemini"``, ``"ollama"``), or the lowercased input if no alias
        matches.

    Example:
        >>> canonical_provider_name("oai")
        'openai'
        >>> canonical_provider_name("anthropic")
        'anthropic'
    """
    normalized = (provider or "").strip().lower()
    aliases = {
        "oai": "openai",
        "claude": "anthropic",
        "google": "gemini",
        "local": "ollama",
    }
    return aliases.get(normalized, normalized)


def looks_openai_compatible_endpoint(base_url: str | None) -> bool:
    """Detect whether a base URL looks like an OpenAI-compatible ``/v1`` API.

    This is used during profile reconciliation to automatically switch the
    provider from ``ollama`` to ``openai`` when the user has configured an
    endpoint that follows the OpenAI ``/v1`` convention.

    Args:
        base_url: The endpoint URL to inspect, or ``None``.

    Returns:
        ``True`` if the URL path ends with ``/v1``, ``False`` otherwise or
        when *base_url* is falsy.

    Example:
        >>> looks_openai_compatible_endpoint("http://localhost:8080/v1")
        True
        >>> looks_openai_compatible_endpoint("http://localhost:11434")
        False
    """
    if not base_url:
        return False
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    return path == "/v1" or path.endswith("/v1")


def reconcile_terminal_profile(profile: TerminalProfile) -> TerminalProfile:
    """Heal inconsistent saved profiles before startup.

    Applies heuristic corrections to a ``TerminalProfile`` so that stale or
    contradictory fields do not cause runtime errors.  The primary fix today
    is detecting an Ollama profile whose ``base_url`` actually points at an
    OpenAI-compatible ``/v1`` endpoint, in which case the provider is silently
    switched to ``"openai"`` and cached model lists are cleared.

    Args:
        profile: The terminal profile to reconcile.  The object is mutated
            in-place and also returned for convenience.

    Returns:
        The same *profile* instance after any corrections have been applied.
    """
    canonical_provider = canonical_provider_name(profile.provider)
    if canonical_provider in {"ollama"} and looks_openai_compatible_endpoint(profile.base_url):
        profile.provider = "openai"
        profile.available_models = []
        return profile

    profile.provider = canonical_provider or profile.provider
    return profile


def detect_provider(env: tp.Mapping[str, str] | None = None) -> str:
    """Choose a sensible default LLM provider by inspecting environment variables.

    The detection order is:

    1. ``CALUTE_PROVIDER`` -- explicit override, canonicalized.
    2. ``OPENAI_API_KEY`` or ``OPENAI_BASE_URL`` -- selects ``"openai"``.
    3. ``ANTHROPIC_API_KEY`` -- selects ``"anthropic"``.
    4. ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` -- selects ``"gemini"``.
    5. ``OLLAMA_HOST`` or ``OLLAMA_BASE_URL`` -- selects ``"ollama"``.
    6. Falls back to ``"ollama"`` when no hint is found.

    Args:
        env: Mapping to read environment variables from.  Defaults to
            ``os.environ`` when ``None``.

    Returns:
        A canonical provider name string.
    """
    env = env or os.environ
    explicit = env.get("CALUTE_PROVIDER")
    if explicit:
        return canonical_provider_name(explicit)
    if env.get("OPENAI_API_KEY") or env.get("OPENAI_BASE_URL"):
        return "openai"
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    if env.get("OLLAMA_HOST") or env.get("OLLAMA_BASE_URL"):
        return "ollama"
    return "ollama"


def detect_model(provider: str, env: tp.Mapping[str, str] | None = None) -> str | None:
    """Resolve a model name from generic or provider-specific environment variables.

    Checks ``CALUTE_MODEL`` first (a universal override), then falls back to
    a provider-specific key such as ``OPENAI_MODEL`` or ``OLLAMA_MODEL``.

    Args:
        provider: Canonical or aliased provider name used to select the
            correct environment variable key.
        env: Mapping to read environment variables from.  Defaults to
            ``os.environ`` when ``None``.

    Returns:
        The model name string, or ``None`` if no relevant variable is set.
    """
    env = env or os.environ
    if env.get("CALUTE_MODEL"):
        return env["CALUTE_MODEL"]

    provider = canonical_provider_name(provider)
    keys = {
        "openai": "OPENAI_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
        "claude": "ANTHROPIC_MODEL",
        "gemini": "GEMINI_MODEL",
        "google": "GEMINI_MODEL",
        "ollama": "OLLAMA_MODEL",
        "local": "OLLAMA_MODEL",
    }
    key = keys.get(provider)
    return env.get(key) if key else None


def build_default_agent(
    *,
    model: str | None,
    agent_id: str = DEFAULT_AGENT_ID,
    instructions: str = DEFAULT_INSTRUCTIONS,
    include_tools: bool = True,
    sampling_params: dict[str, float | int] | None = None,
) -> Agent:
    """Create the default interactive terminal agent.

    Constructs an ``Agent`` pre-configured with the Calute terminal
    personality, the standard file-system and code-execution tools, and
    optional sampling overrides.

    Args:
        model: LLM model identifier passed to the agent, or ``None`` to
            accept whatever the LLM client defaults to.
        agent_id: Unique identifier for the agent within the orchestrator.
        instructions: System-level prompt that describes agent behaviour.
        include_tools: When ``True`` (default), the built-in ``ReadFile``,
            ``WriteFile``, ``ListDir``, and ``ExecutePythonCode`` tools are
            attached to the agent.
        sampling_params: Optional dictionary of sampling overrides
            (e.g. ``{"temperature": 0.7}``).  Applied via
            ``Agent.set_sampling_params`` when not ``None``.

    Returns:
        A fully constructed ``Agent`` instance ready for registration with a
        ``Calute`` executor.
    """
    agent = Agent(
        id=agent_id,
        name=DEFAULT_AGENT_NAME,
        model=model,
        instructions=instructions,
        functions=list(DEFAULT_TOOLS) if include_tools else [],
    )
    if sampling_params:
        agent.set_sampling_params(**sampling_params)
    return agent


def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` argument parser for the Calute CLI.

    Defines all flags accepted by the ``calute`` command, including provider
    selection, model overrides, profile management, and runtime options.

    Returns:
        A configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(prog="calute", description="Launch the Calute terminal UI.")
    parser.add_argument(
        "command",
        nargs="?",
        default="tui",
        choices=["tui"],
        help="Interface to launch. Currently only 'tui' is available.",
    )
    parser.add_argument("--provider", help="LLM provider to use. Defaults to env auto-detection.")
    parser.add_argument("--model", help="Model name. Defaults to provider-specific env or provider default.")
    parser.add_argument("--api-key", help="Override API key passed to the provider.")
    parser.add_argument("--base-url", help="Override provider base URL or OpenAI-compatible endpoint.")
    parser.add_argument("--profile-name", help="Saved terminal profile name. Defaults to last-used profile.")
    parser.add_argument("--list-profiles", action="store_true", help="Print saved profiles and exit.")
    parser.add_argument(
        "--list-models", action="store_true", help="Print models available from the resolved endpoint and exit."
    )
    parser.add_argument(
        "--choose-model", action="store_true", help="Prompt to choose a model when multiple models are available."
    )
    parser.add_argument("--agent-id", help="Agent ID shown inside the TUI.")
    parser.add_argument("--instructions", help="System instructions for the default agent.")
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in PromptProfile],
        default=None,
        help="Runtime prompt profile for the terminal assistant.",
    )
    parser.add_argument("--no-tools", action="store_true", help="Start the terminal assistant without built-in tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _print_profiles(store: TerminalConfigStore) -> int:
    """Print all saved terminal profiles as tab-separated rows to stdout.

    Each row contains: profile name, provider, model (or ``-``), and
    base URL (or ``-``).

    Args:
        store: The configuration store to load profiles from.

    Returns:
        Always ``0``, suitable for use as a CLI exit code.
    """
    profiles = store.list_profiles()
    if not profiles:
        print("No saved profiles.")
        return 0
    for profile in profiles:
        endpoint = profile.base_url or "-"
        model = profile.model or "-"
        print(f"{profile.name}\t{profile.provider}\t{model}\t{endpoint}")
    return 0


def choose_model_interactively(
    models: list[str],
    *,
    current: str | None = None,
    input_fn: tp.Callable[[str], str] = input,
    output: tp.TextIO = sys.stdout,
) -> str:
    """Prompt the user to select a model from a discovered list interactively.

    Displays a numbered menu of available models and repeatedly asks for
    input until a valid selection is made.  The user may enter a 1-based
    index number or the exact model name.  Pressing Enter with no input
    re-selects *current* (when it exists in *models*).

    Args:
        models: Non-empty list of model name strings to present.
        current: The currently active model, highlighted with ``(current)``
            in the menu and accepted as the default on empty input.
        input_fn: Callable used to read a line of user input.  Defaults to
            the built-in ``input`` for normal TTY use; override for testing.
        output: Writable text stream for menu output.  Defaults to
            ``sys.stdout``.

    Returns:
        The selected model name string.

    Raises:
        ValueError: If *models* is empty.
    """
    if not models:
        raise ValueError("No models available to choose from")

    prompt_lines = ["Available models:"]
    for index, model_name in enumerate(models, start=1):
        marker = " (current)" if model_name == current else ""
        prompt_lines.append(f"  {index}. {model_name}{marker}")
    prompt_lines.append("Select a model by number or exact name.")
    output.write("\n".join(prompt_lines) + "\n")

    while True:
        raw = input_fn("> ").strip()
        if not raw and current and current in models:
            return current
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        if raw in models:
            return raw
        output.write("Invalid model selection. Try again.\n")


def _resolve_profile(
    args: argparse.Namespace,
    store: TerminalConfigStore,
    env: tp.Mapping[str, str] | None = None,
) -> TerminalProfile:
    """Resolve CLI arguments plus saved state into a concrete terminal profile.

    Merges three sources of configuration in descending priority:

    1. Explicit CLI flags (``args``).
    2. A previously saved ``TerminalProfile`` looked up by name.
    3. Environment-variable detection helpers (``detect_provider``,
       ``detect_model``).

    The resulting profile is also passed through
    ``reconcile_terminal_profile`` before being returned.

    Args:
        args: Parsed CLI namespace from ``build_parser``.
        store: Persistent config store used to look up saved profiles.
        env: Optional environment mapping; defaults to ``os.environ``.

    Returns:
        A fully resolved and reconciled ``TerminalProfile``.
    """
    env = env or os.environ
    settings = store.load()
    profile_name = args.profile_name or settings.last_profile or "default"
    saved = settings.profiles.get(profile_name)

    provider = canonical_provider_name(args.provider or (saved.provider if saved else None) or detect_provider(env))
    detected_model = detect_model(provider, env)
    resolved_model = args.model or (saved.model if saved else None) or detected_model

    profile = TerminalProfile(
        name=profile_name,
        provider=provider,
        model=resolved_model,
        api_key=args.api_key if args.api_key is not None else (saved.api_key if saved else None),
        base_url=args.base_url if args.base_url is not None else (saved.base_url if saved else None),
        prompt_profile=args.profile or (saved.prompt_profile if saved else PromptProfile.FULL.value),
        agent_id=args.agent_id or (saved.agent_id if saved else DEFAULT_AGENT_ID),
        instructions=(
            args.instructions
            if args.instructions is not None
            else (saved.instructions if saved and saved.instructions else DEFAULT_INSTRUCTIONS)
        ),
        include_tools=not args.no_tools if args.no_tools else (saved.include_tools if saved else True),
        power_tools_enabled=saved.power_tools_enabled if saved else True,
        available_models=list(saved.available_models) if saved else [],
        sampling_params=dict(saved.sampling_params) if saved else {},
    )
    return reconcile_terminal_profile(profile)


def main(argv: list[str] | None = None) -> int:
    """Launch the Textual terminal UI from the command line.

    This is the top-level entry point used by the ``calute`` console script.
    It parses arguments, resolves profiles, discovers available models,
    constructs the LLM client and default agent, and finally starts the
    Textual application.

    Args:
        argv: Argument list to parse.  Defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        An integer exit code: ``0`` on success, ``1`` on generic errors,
        ``130`` on ``KeyboardInterrupt``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TerminalConfigStore()

    if args.list_profiles:
        return _print_profiles(store)

    try:
        terminal_profile = _resolve_profile(args, store)
        terminal_profile = reconcile_terminal_profile(terminal_profile)
        should_discover_models = args.list_models or args.choose_model or terminal_profile.model is None
        discovered_models = list(terminal_profile.available_models)
        if should_discover_models:
            try:
                discovered_models = discover_available_models(
                    terminal_profile.provider,
                    model=terminal_profile.model,
                    api_key=terminal_profile.api_key,
                    base_url=terminal_profile.base_url,
                )
            except Exception:
                discovered_models = list(terminal_profile.available_models)

        if discovered_models:
            terminal_profile.available_models = discovered_models
            if terminal_profile.model not in discovered_models:
                terminal_profile.model = None

        if args.list_models:
            for model_name in terminal_profile.available_models:
                print(model_name)
            return 0

        if terminal_profile.available_models:
            if args.choose_model and sys.stdin.isatty():
                terminal_profile.model = choose_model_interactively(
                    terminal_profile.available_models,
                    current=terminal_profile.model,
                )
            elif terminal_profile.model is None:
                terminal_profile.model = terminal_profile.available_models[0]

        llm_kwargs: dict[str, tp.Any] = {}
        if terminal_profile.model:
            llm_kwargs["model"] = terminal_profile.model
        if terminal_profile.api_key:
            llm_kwargs["api_key"] = terminal_profile.api_key
        if terminal_profile.base_url:
            llm_kwargs["base_url"] = terminal_profile.base_url

        llm = create_llm(terminal_profile.provider, **llm_kwargs)
        calute = Calute(
            llm=llm,
            runtime_features=RuntimeFeaturesConfig(
                enabled=True,
                default_prompt_profile=PromptProfile(terminal_profile.prompt_profile),
                operator=OperatorRuntimeConfig(
                    enabled=True,
                    power_tools_enabled=terminal_profile.power_tools_enabled,
                ),
            ),
        )
        agent_model = terminal_profile.model or getattr(getattr(llm, "config", None), "model", None)
        agent = build_default_agent(
            model=agent_model,
            agent_id=terminal_profile.agent_id,
            instructions=terminal_profile.instructions or DEFAULT_INSTRUCTIONS,
            include_tools=terminal_profile.include_tools,
            sampling_params=terminal_profile.sampling_params,
        )
        calute.register_agent(agent)
        store.upsert_profile(terminal_profile, make_default=True)
        launch_tui(calute, agent, profile=terminal_profile, config_store=store).launch()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"calute: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
