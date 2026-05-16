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
"""Vision dispatch for describing and OCR-ing images.

Wraps Claude Vision (``anthropic``) and GPT-4V (``openai``) behind a single
:func:`analyze_image` call. Backends are simple callables, so tests inject
fakes via :func:`register_backend` rather than monkey-patching SDK internals.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

VisionBackend = Callable[[str, str, str, dict[str, Any]], str]
# Signature: (b64_image, mime, prompt, params) -> response_text

_BACKENDS: dict[str, VisionBackend] = {}


def register_backend(name: str, fn: VisionBackend) -> None:
    """Register or override a vision backend keyed by ``name``."""
    _BACKENDS[name] = fn


def _read_image(path: Path) -> tuple[str, str]:
    """Load ``path`` and return ``(base64_data, mime_type)`` for transmission."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"not an image file: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii"), mime


def _backend_anthropic(b64: str, mime: str, prompt: str, params: dict[str, Any]) -> str:
    """Send the image + prompt to Anthropic's Messages API and return the text reply."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("anthropic SDK required for vision") from exc
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=params.get("model", "claude-opus-4-7"),
        max_tokens=params.get("max_tokens", 1024),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    if resp.content and resp.content[0].type == "text":
        return resp.content[0].text  # type: ignore[no-any-return]
    return ""


def _backend_openai(b64: str, mime: str, prompt: str, params: dict[str, Any]) -> str:
    """Send the image + prompt to OpenAI's chat completions and return the text reply."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai SDK required for vision") from exc
    client = OpenAI()
    resp = client.chat.completions.create(
        model=params.get("model", "gpt-4o"),
        max_tokens=params.get("max_tokens", 1024),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""  # type: ignore[no-any-return]


_BACKENDS.update({"anthropic": _backend_anthropic, "openai": _backend_openai})


def analyze_image(
    image_path: str | Path,
    *,
    prompt: str = "Describe this image in detail.",
    provider: str = "anthropic",
    **params: Any,
) -> dict[str, Any]:
    """Send an image and a natural-language prompt to a vision model.

    Args:
        image_path: Path to an existing image file with a recognized MIME
            type beginning with ``image/``.
        prompt: Instruction sent alongside the image.
        provider: Backend identifier (``anthropic`` or ``openai`` by default).
        **params: Provider-specific overrides such as ``model`` and
            ``max_tokens``; forwarded to the underlying backend.

    Returns:
        Mapping with ``provider``, ``model``, ``prompt``, and ``response``.

    Raises:
        FileNotFoundError: ``image_path`` does not exist.
        ValueError: File is not an image or ``provider`` is unknown.
    """
    path = Path(image_path)
    b64, mime = _read_image(path)
    if provider not in _BACKENDS:
        raise ValueError(f"unknown vision provider: {provider}")
    response = _BACKENDS[provider](b64, mime, prompt, params)
    return {
        "provider": provider,
        "model": params.get("model", "(default)"),
        "prompt": prompt,
        "response": response,
    }


__all__ = ["VisionBackend", "analyze_image", "register_backend"]
