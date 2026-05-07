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
"""Media generation and processing tools for image generation, vision analysis, and text-to-speech.

This module provides tools for generating images, analyzing images with vision models,
and converting text to speech. These tools enable agents to work with visual and audio media.

Example:
    >>> from xerxes.tools.media_tools import image_generate, vision_analyze, text_to_speech
    >>> image_generate.static_call(prompt="A sunset over mountains")
    >>> vision_analyze.static_call(image_url="https://example.com/image.jpg")
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
    """Configuration for media generation tools.

    Attributes:
        base_url: API base URL (defaults to OpenAI API).
        api_key: API key for authentication.
        image_model: Model for image generation.
        vision_model: Model for vision analysis.
        tts_model: Model for text-to-speech.
        tts_voice: Default voice for TTS.
    """

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
    """Configure media tool settings.

    Args:
        base_url: API base URL. Defaults to OPENAI_BASE_URL or OpenAI default.
        api_key: API key for authentication.
        image_model: Default model for image generation.
        vision_model: Default model for vision analysis.
        tts_model: Default model for text-to-speech.
        tts_voice: Default voice for TTS.

    Returns:
        Updated MediaConfig instance.

    Example:
        >>> configure_media(api_key="sk-...", image_model="dall-e-3")
    """
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
    """Get the current media configuration.

    Returns:
        The current MediaConfig instance.
    """
    with _lock:
        return _config


def set_media_client(client: tp.Any | None) -> None:
    """Set a custom HTTP client for media requests.

    Args:
        client: Custom HTTP client instance, or None to use default.
    """
    global _http_client
    with _lock:
        _http_client = client


def _post(url: str, *, json_body: dict[str, tp.Any], expect: str = "json") -> tp.Any:
    """Send POST request to media API.

    Args:
        url: Full URL to POST to.
        json_body: JSON body for the request.
        expect: Expected response type ('json' or 'bytes').

    Returns:
        Response data.
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
    """Coerce response to expected format.

    Args:
        resp: HTTP response object.
        expect: Expected type ('json' or 'bytes').

    Returns:
        Coerced response data.
    """
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
    """Generate images from text prompts using AI models.

    Creates images based on textual descriptions using configured image generation models.

    Example:
        >>> image_generate.static_call(prompt="A beautiful sunset over the ocean")
    """

    @staticmethod
    def static_call(
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Generate images from text prompts.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions. Defaults to "1024x1024".
            n: Number of images to generate. Defaults to 1.
            model: Override the default image generation model.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with generated images as base64-encoded data.
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
    """Analyze images using vision models.

    Provides visual understanding of images through AI vision capabilities.

    Example:
        >>> vision_analyze.static_call(
        ...     image_url="https://example.com/photo.jpg",
        ...     question="What is in this image?"
        ... )
    """

    @staticmethod
    def static_call(
        image_url: str | None = None,
        image_b64: str | None = None,
        question: str = "Describe this image in detail.",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Analyze images with vision model.

        Args:
            image_url: URL of the image to analyze.
            image_b64: Base64-encoded image data.
            question: Question or prompt about the image. Defaults to general description.
            model: Override the default vision model.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with vision analysis results.
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
    """Convert text to speech audio.

    Generates spoken audio from text using AI voice synthesis.

    Example:
        >>> text_to_speech.static_call(
        ...     text="Hello, this is a test of the text to speech system.",
        ...     voice="alloy"
        ... )
    """

    @staticmethod
    def static_call(
        text: str,
        voice: str | None = None,
        audio_format: str = "mp3",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Generate speech audio from text.

        Args:
            text: Text to convert to speech.
            voice: Voice to use. Common options: alloy, echo, fable, onyx, nova, shimmer.
            audio_format: Output format. Defaults to "mp3".
            model: Override the default TTS model.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with base64-encoded audio data.
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
