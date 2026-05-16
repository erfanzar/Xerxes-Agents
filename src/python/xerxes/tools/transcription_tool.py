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
"""Speech-to-text dispatch layer for the voice subsystem.

Provides a single :func:`transcribe` entry point with two interchangeable
backends:

* ``faster_whisper`` — local CPU inference via the ``faster-whisper`` package
  (no network), installed with the ``voice`` extra.
* ``openai`` — cloud ``whisper-1`` via the OpenAI SDK; requires
  ``OPENAI_API_KEY``.

When no backend is requested explicitly the module auto-detects the local
backend first to avoid surprise network calls. Third-party backends can be
plugged in via :func:`register_backend`.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS: tuple[str, ...] = ("faster_whisper", "openai")

TranscribeFn = Callable[[Path, str], str]

_BACKENDS: dict[str, TranscribeFn] = {}


def register_backend(name: str, fn: TranscribeFn) -> None:
    """Register or override a transcription backend keyed by ``name``."""
    _BACKENDS[name] = fn


def _backend_faster_whisper(path: Path, model: str) -> str:
    """Transcribe ``path`` with local ``faster-whisper`` using int8 CPU compute."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faster-whisper not installed; install xerxes-agent[voice]") from exc
    m = WhisperModel(model or "base", device="cpu", compute_type="int8")
    segments, _info = m.transcribe(str(path))
    return "".join(seg.text for seg in segments).strip()


def _backend_openai(path: Path, model: str) -> str:
    """Transcribe ``path`` via the OpenAI ``audio.transcriptions`` endpoint."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai not installed") from exc
    client = OpenAI()
    with path.open("rb") as fh:
        resp = client.audio.transcriptions.create(file=fh, model=model or "whisper-1")
    return resp.text  # type: ignore[no-any-return]


_BACKENDS.update({"faster_whisper": _backend_faster_whisper, "openai": _backend_openai})


def _autodetect() -> str | None:
    """Return the first installed backend name, preferring local over cloud."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster_whisper"
    if importlib.util.find_spec("openai") is not None:
        return "openai"
    return None


def transcribe(audio_path: str | Path, *, backend: str | None = None, model: str = "") -> dict[str, Any]:
    """Transcribe an audio file and return the recognized text plus metadata.

    Args:
        audio_path: Existing audio file (WAV/MP3/etc.) to transcribe.
        backend: Backend identifier from :data:`SUPPORTED_BACKENDS`. ``None``
            triggers auto-detection (local first).
        model: Backend-specific model name. Empty string asks the backend to
            pick its default (``base`` for ``faster_whisper``, ``whisper-1``
            for ``openai``).

    Returns:
        Mapping with keys ``backend``, ``model``, and ``text``.

    Raises:
        FileNotFoundError: ``audio_path`` does not exist.
        RuntimeError: No backend is installed or available.
        ValueError: ``backend`` is not a registered name.
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    chosen = backend or _autodetect()
    if chosen is None:
        raise RuntimeError(
            "No STT backend available; install xerxes-agent[voice] (faster-whisper) "
            "or set OPENAI_API_KEY for cloud Whisper."
        )
    if chosen not in _BACKENDS:
        raise ValueError(f"unknown STT backend {chosen!r}; supported: {SUPPORTED_BACKENDS}")
    fn = _BACKENDS[chosen]
    text = fn(p, model)
    return {"backend": chosen, "model": model or "(default)", "text": text}


__all__ = ["SUPPORTED_BACKENDS", "register_backend", "transcribe"]
