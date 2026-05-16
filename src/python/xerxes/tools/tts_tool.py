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
"""Text-to-speech dispatch with pluggable provider backends.

Provides a single :func:`speak` entry point with four built-in providers:

* ``edge`` — free Microsoft Edge TTS via the ``edge-tts`` package (default).
* ``elevenlabs`` — premium ``elevenlabs`` SDK + ``ELEVENLABS_API_KEY``.
* ``openai`` — OpenAI ``tts-1`` voices via the OpenAI SDK.
* ``neutts`` — local model invoked through the ``neutts`` binary on ``$PATH``.

Each provider is a callable conforming to :data:`TTSProvider`; new providers
are registered via :func:`register_provider`. Providers import their
dependencies lazily so the module loads cleanly even when no TTS package is
installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TTSProvider = Callable[[str, str, Path], Path]
# Signature: (text, voice, out_path) -> out_path

SUPPORTED_PROVIDERS: tuple[str, ...] = ("edge", "elevenlabs", "openai", "neutts")

_REGISTRY: dict[str, TTSProvider] = {}


def register_provider(name: str, fn: TTSProvider) -> None:
    """Register a custom provider implementation (or override a built-in)."""
    _REGISTRY[name] = fn


def _provider_edge_tts(text: str, voice: str, out_path: Path) -> Path:
    """Edge-TTS provider — synchronous wrapper over the async API."""
    try:
        import asyncio

        import edge_tts  # type: ignore
    except ImportError as exc:
        raise RuntimeError("edge-tts not installed; `pip install edge-tts`") from exc

    async def _run() -> None:
        comm = edge_tts.Communicate(text, voice or "en-US-AriaNeural")
        await comm.save(str(out_path))

    asyncio.run(_run())
    return out_path


def _provider_elevenlabs(text: str, voice: str, out_path: Path) -> Path:
    """Synthesize ``text`` with ElevenLabs ``eleven_turbo_v2_5`` model."""
    try:
        from elevenlabs import ElevenLabs  # type: ignore
    except ImportError as exc:
        raise RuntimeError("elevenlabs not installed; install xerxes-agent[tts-premium]") from exc
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    client = ElevenLabs(api_key=api_key)
    voice_id = voice or "Rachel"
    audio = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id="eleven_turbo_v2_5")
    out_path.write_bytes(b"".join(audio))
    return out_path


def _provider_openai(text: str, voice: str, out_path: Path) -> Path:
    """Synthesize ``text`` with the OpenAI ``tts-1`` endpoint."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai not installed") from exc
    client = OpenAI()
    voice_id = voice or "alloy"
    resp = client.audio.speech.create(model="tts-1", voice=voice_id, input=text)
    out_path.write_bytes(resp.read())  # type: ignore[no-any-return]
    return out_path


def _provider_neutts(text: str, voice: str, out_path: Path) -> Path:
    """Synthesize ``text`` by shelling out to the ``neutts`` CLI."""
    binary = shutil.which("neutts")
    if binary is None:
        raise RuntimeError("neutts binary not found on PATH")
    subprocess.run(
        [binary, "--text", text, "--voice", voice or "default", "--out", str(out_path)],
        check=True,
        capture_output=True,
    )
    return out_path


_REGISTRY.update(
    {
        "edge": _provider_edge_tts,
        "elevenlabs": _provider_elevenlabs,
        "openai": _provider_openai,
        "neutts": _provider_neutts,
    }
)


def _autodetect_provider() -> str | None:
    """Pick the first provider whose dependency is present."""
    import importlib.util

    if importlib.util.find_spec("edge_tts") is not None:
        return "edge"
    if importlib.util.find_spec("openai") is not None:
        return "openai"
    if importlib.util.find_spec("elevenlabs") is not None:
        return "elevenlabs"
    if shutil.which("neutts") is not None:
        return "neutts"
    return None


def speak(
    text: str,
    *,
    provider: str | None = None,
    voice: str = "",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Synthesize ``text`` to an audio file using the chosen provider.

    Args:
        text: Non-empty text to synthesize.
        provider: Provider identifier from :data:`SUPPORTED_PROVIDERS`.
            ``None`` triggers :func:`_autodetect_provider`.
        voice: Provider-specific voice identifier; empty string asks the
            backend to pick its default voice.
        out_path: Destination audio file. When ``None``, a temporary ``.mp3``
            is created via :func:`tempfile.mkstemp` and its path returned.

    Returns:
        Mapping with ``provider``, ``voice``, ``path`` (the synthesized file)
        and ``bytes`` (its on-disk size).

    Raises:
        ValueError: ``text`` is blank or ``provider`` is not registered.
        RuntimeError: No provider is available on this machine.
    """
    if not text.strip():
        raise ValueError("text must be non-empty")
    chosen = provider or _autodetect_provider()
    if chosen is None:
        raise RuntimeError(
            "No TTS provider available; install xerxes-agent[voice] (Edge-TTS) "
            "or xerxes-agent[tts-premium] (ElevenLabs)."
        )
    if chosen not in _REGISTRY:
        raise ValueError(f"unknown provider {chosen!r}; supported: {SUPPORTED_PROVIDERS}")
    if out_path is None:
        out_path = Path(tempfile.mkstemp(prefix="xerxes-tts-", suffix=".mp3")[1])
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fn = _REGISTRY[chosen]
    fn(text, voice, out_path)
    return {
        "provider": chosen,
        "voice": voice or "(default)",
        "path": str(out_path),
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


__all__ = ["SUPPORTED_PROVIDERS", "TTSProvider", "register_provider", "speak"]
