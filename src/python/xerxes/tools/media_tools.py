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
"""Media tools module for Xerxes.

Exports:
    - logger
    - MediaConfig
    - configure_media
    - get_media_config
    - set_media_client
    - image_generate
    - vision_analyze
    - text_to_speech"""

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
    """Media config.

    Attributes:
        base_url (str): base url.
        api_key (str): api key.
        image_model (str): image model.
        vision_model (str): vision model.
        tts_model (str): tts model.
        tts_voice (str): tts voice."""

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
    """Configure media.

    Args:
        base_url (str | None, optional): IN: base url. Defaults to None. OUT: Consumed during execution.
        api_key (str | None, optional): IN: api key. Defaults to None. OUT: Consumed during execution.
        image_model (str | None, optional): IN: image model. Defaults to None. OUT: Consumed during execution.
        vision_model (str | None, optional): IN: vision model. Defaults to None. OUT: Consumed during execution.
        tts_model (str | None, optional): IN: tts model. Defaults to None. OUT: Consumed during execution.
        tts_voice (str | None, optional): IN: tts voice. Defaults to None. OUT: Consumed during execution.
    Returns:
        MediaConfig: OUT: Result of the operation."""

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
    """Retrieve the media config.

    Returns:
        MediaConfig: OUT: Result of the operation."""

    with _lock:
        return _config


def set_media_client(client: tp.Any | None) -> None:
    """Set the media client.

    Args:
        client (tp.Any | None): IN: client. OUT: Consumed during execution."""

    global _http_client
    with _lock:
        _http_client = client


def _post(url: str, *, json_body: dict[str, tp.Any], expect: str = "json") -> tp.Any:
    """Internal helper to post.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
        json_body (dict[str, tp.Any]): IN: json body. OUT: Consumed during execution.
        expect (str, optional): IN: expect. Defaults to 'json'. OUT: Consumed during execution.
    Returns:
        tp.Any: OUT: Result of the operation."""

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
    """Internal helper to coerce.

    Args:
        resp (tp.Any): IN: resp. OUT: Consumed during execution.
        expect (str): IN: expect. OUT: Consumed during execution.
    Returns:
        tp.Any: OUT: Result of the operation."""

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
    """Image generate.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            prompt (str): IN: prompt. OUT: Consumed during execution.
            size (str, optional): IN: size. Defaults to '1024x1024'. OUT: Consumed during execution.
            n (int, optional): IN: n. Defaults to 1. OUT: Consumed during execution.
            model (str | None, optional): IN: model. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Vision analyze.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        image_url: str | None = None,
        image_b64: str | None = None,
        question: str = "Describe this image in detail.",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            image_url (str | None, optional): IN: image url. Defaults to None. OUT: Consumed during execution.
            image_b64 (str | None, optional): IN: image b64. Defaults to None. OUT: Consumed during execution.
            question (str, optional): IN: question. Defaults to 'Describe this image in detail.'. OUT: Consumed during execution.
            model (str | None, optional): IN: model. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Text to speech.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        text: str,
        voice: str | None = None,
        audio_format: str = "mp3",
        model: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            text (str): IN: text. OUT: Consumed during execution.
            voice (str | None, optional): IN: voice. Defaults to None. OUT: Consumed during execution.
            audio_format (str, optional): IN: audio format. Defaults to 'mp3'. OUT: Consumed during execution.
            model (str | None, optional): IN: model. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
