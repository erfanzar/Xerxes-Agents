# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal agent tools — image generation, vision, and speech.

Three tools backed by the OpenAI-compatible image / chat / audio
endpoints:

- :class:`image_generate` — text-to-image via ``/v1/images/generations``.
- :class:`vision_analyze` — analyse an image with a vision-capable LLM.
- :class:`text_to_speech` — render text to MP3/WAV via ``/v1/audio/speech``.

All three accept an injected HTTP client through
:func:`set_media_client` so tests run fully offline. Production code
configures the real endpoint via :func:`configure_media`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import typing as tp
from dataclasses import dataclass

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)


@dataclass
class MediaConfig:
    """Endpoint + auth used by the three media tools."""

    base_url: str = ""
    api_key: str = ""
    image_model: str = "gpt-image-1"
    vision_model: str = "gpt-4o-mini"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"


_lock = threading.Lock()
_config = MediaConfig(
    base_url=os.environ.get("XERXES_MEDIA_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
    api_key=os.environ.get("XERXES_MEDIA_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
)
_http_client: tp.Any | None = None


def configure_media(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    image_model: str | None = None,
    vision_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
) -> MediaConfig:
    """Update the process-wide media tool configuration."""
    global _config
    with _lock:
        _config = MediaConfig(
            base_url=(base_url if base_url is not None else _config.base_url).rstrip("/"),
            api_key=api_key if api_key is not None else _config.api_key,
            image_model=image_model or _config.image_model,
            vision_model=vision_model or _config.vision_model,
            tts_model=tts_model or _config.tts_model,
            tts_voice=tts_voice or _config.tts_voice,
        )
        return _config


def get_media_config() -> MediaConfig:
    """Return the currently active media tool configuration."""
    with _lock:
        return _config


def set_media_client(client: tp.Any | None) -> None:
    """Install (or clear) an injected HTTP client for the media tools.

    The client should expose ``post(url, json=..., headers=...)`` and
    return either a dict, an object with ``json()``, or raw bytes. Set
    to ``None`` to revert to the default ``httpx`` path.
    """
    global _http_client
    with _lock:
        _http_client = client


def _post(url: str, *, json_body: dict[str, tp.Any], expect: str = "json") -> tp.Any:
    """POST *json_body* to *url* using the injected client or httpx.

    Args:
        url: Absolute endpoint URL.
        json_body: Request payload (serialised as JSON).
        expect: ``"json"`` to decode the response, or ``"bytes"`` to return raw bytes.

    Returns:
        Parsed JSON (dict) or raw bytes depending on *expect*.
    """
    cfg = get_media_config()
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if _http_client is not None:
        resp = _http_client.post(url, json=json_body, headers=headers)
        return _coerce(resp, expect=expect)
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for media tools") from exc
    resp = httpx.post(url, json=json_body, headers=headers, timeout=60.0)
    resp.raise_for_status()
    if expect == "bytes":
        return resp.content
    return resp.json()


def _coerce(resp: tp.Any, *, expect: str) -> tp.Any:
    """Normalise diverse response types into bytes or a dict as requested."""
    if expect == "bytes":
        if isinstance(resp, bytes | bytearray):
            return bytes(resp)
        return getattr(resp, "content", b"") or b""
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "json") and callable(resp.json):
        try:
            return resp.json()
        except Exception:
            pass
    body = getattr(resp, "text", None) or getattr(resp, "body", "") or ""
    if isinstance(body, bytes):
        body = body.decode()
    try:
        return json.loads(body)
    except Exception:
        return {"raw": body}


class image_generate(AgentBaseFn):
    """Generate an image from a text prompt."""

    @staticmethod
    def static_call(
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Call the OpenAI-compatible images endpoint and return base64 PNG(s).

        Use this when the user asks for an image, diagram, illustration,
        cover art, etc.

        Args:
            prompt: Plain-language description of the desired image.
            size: ``"WIDTHxHEIGHT"``. Common: ``"512x512"``,
                ``"1024x1024"``, ``"1792x1024"``.
            n: Number of variations (1-4 typically).
            model: Override the configured image model.

        Returns:
            ``{"model", "size", "count", "images": [{"b64", "format"}], "raw"}``.
        """
        cfg = get_media_config()
        body = {
            "model": model or cfg.image_model,
            "prompt": prompt,
            "size": size,
            "n": int(n),
            "response_format": "b64_json",
        }
        try:
            data = _post(f"{cfg.base_url}/images/generations", json_body=body)
        except Exception as exc:
            return {"error": str(exc), "model": body["model"]}
        items = data.get("data") or []
        images = [
            {"b64": item.get("b64_json", ""), "format": "png", "revised_prompt": item.get("revised_prompt", "")}
            for item in items
        ]
        return {
            "model": body["model"],
            "size": size,
            "count": len(images),
            "images": images,
        }


class vision_analyze(AgentBaseFn):
    """Analyse an image with a vision LLM and return a description."""

    @staticmethod
    def static_call(
        image_url: str | None = None,
        image_b64: str | None = None,
        question: str = "Describe this image in detail.",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Send an image + question to a vision-capable chat model.

        Either pass a remote ``image_url`` or an inline ``image_b64``
        payload (PNG/JPEG, no data URL prefix needed).

        Args:
            image_url: HTTPS URL of the image.
            image_b64: Base64-encoded image bytes (alternative to URL).
            question: What the agent wants to know about the image.
            model: Override the configured vision model.

        Returns:
            ``{"model", "answer", "raw"}``.
        """
        if not image_url and not image_b64:
            return {"error": "either image_url or image_b64 is required"}
        cfg = get_media_config()
        if image_b64 and not image_url:
            image_url = f"data:image/png;base64,{image_b64}"
        body = {
            "model": model or cfg.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        try:
            data = _post(f"{cfg.base_url}/chat/completions", json_body=body)
        except Exception as exc:
            return {"error": str(exc), "model": body["model"]}
        try:
            answer = data["choices"][0]["message"]["content"]
        except Exception:
            answer = ""
        return {"model": body["model"], "answer": answer or "", "raw": data}


class text_to_speech(AgentBaseFn):
    """Synthesise speech audio from text."""

    @staticmethod
    def static_call(
        text: str,
        voice: str | None = None,
        audio_format: str = "mp3",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Call the OpenAI-compatible audio/speech endpoint.

        Use this when the user explicitly asks the agent to "read",
        "speak", or "narrate" something out loud — the result is
        suitable for shipping over a channel adapter that supports
        audio (Telegram voice, Signal voice notes, BlueBubbles).

        Args:
            text: Text to render. Trimmed at provider's max length.
            voice: Voice id (e.g. ``"alloy"``, ``"verse"``); falls
                back to the configured default.
            audio_format: ``"mp3"`` (default), ``"wav"``, ``"opus"``,
                ``"aac"``. (Also accepted as ``format`` via
                ``**context_variables`` for backward compatibility.)
            model: Override the configured TTS model.

        Returns:
            ``{"model", "voice", "format", "audio_b64", "bytes"}``.
        """
        legacy_format = context_variables.pop("format", None)
        if legacy_format:
            audio_format = legacy_format
        cfg = get_media_config()
        body = {
            "model": model or cfg.tts_model,
            "input": text,
            "voice": voice or cfg.tts_voice,
            "format": audio_format,
        }
        try:
            audio = _post(f"{cfg.base_url}/audio/speech", json_body=body, expect="bytes")
        except Exception as exc:
            return {"error": str(exc), "model": body["model"]}
        if not isinstance(audio, bytes | bytearray):
            audio = bytes(audio or b"")
        return {
            "model": body["model"],
            "voice": body["voice"],
            "format": audio_format,
            "bytes": len(audio),
            "audio_b64": base64.b64encode(audio).decode(),
        }


__all__ = [
    "MediaConfig",
    "configure_media",
    "get_media_config",
    "image_generate",
    "set_media_client",
    "text_to_speech",
    "vision_analyze",
]
