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
"""Image-generation dispatch for diffusion and DALL-E providers.

Wraps OpenAI image generation (``dall-e-3`` by default) and the FAL
runtime (``fal-ai/flux/schnell`` by default) behind a single
:func:`generate_image` entry point. Providers are simple callables, so tests
can register fakes through :func:`register_provider` to avoid real network
traffic. Generated images are persisted as PNG files to a caller-supplied or
temporary path.
"""

from __future__ import annotations

import base64
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

GenerateFn = Callable[[str, dict[str, Any]], bytes]

_PROVIDERS: dict[str, GenerateFn] = {}


def register_provider(name: str, fn: GenerateFn) -> None:
    """Register or override an image-generation provider keyed by ``name``."""
    _PROVIDERS[name] = fn


def _provider_openai(prompt: str, params: dict[str, Any]) -> bytes:
    """Generate raw PNG bytes via OpenAI's ``images.generate`` endpoint."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai SDK required") from exc
    client = OpenAI()
    resp = client.images.generate(
        model=params.get("model", "dall-e-3"),
        prompt=prompt,
        size=params.get("size", "1024x1024"),
        response_format="b64_json",
    )
    return base64.b64decode(resp.data[0].b64_json)  # type: ignore[no-any-return]


def _provider_fal(prompt: str, params: dict[str, Any]) -> bytes:
    """Submit a job to FAL and download the resulting image as bytes."""
    try:
        import fal_client  # type: ignore
    except ImportError as exc:
        raise RuntimeError("fal-client required; install xerxes-agent[creative]") from exc
    handler = fal_client.submit(
        params.get("model", "fal-ai/flux/schnell"),
        arguments={"prompt": prompt, **{k: v for k, v in params.items() if k != "model"}},
    )
    result = handler.get()
    # fal returns an http url; download for the caller.
    import httpx

    return httpx.get(result["images"][0]["url"]).content


_PROVIDERS.update({"openai": _provider_openai, "fal": _provider_fal})


def generate_image(
    prompt: str,
    *,
    provider: str = "openai",
    out_path: str | Path | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Generate an image from ``prompt`` and persist it to disk.

    Args:
        prompt: Non-empty natural-language description of the image.
        provider: Provider identifier; defaults to ``openai``.
        out_path: Destination file. When ``None`` a temporary ``.png`` is
            allocated via :func:`tempfile.mkstemp`.
        **params: Provider-specific overrides forwarded verbatim (``model``,
            ``size``, etc.).

    Returns:
        Mapping with ``provider``, the resolved ``path`` and the file size
        in ``bytes``.

    Raises:
        ValueError: ``prompt`` is blank or ``provider`` is unknown.
    """
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown image-gen provider: {provider}")
    data = _PROVIDERS[provider](prompt, params)
    if out_path is None:
        out_path = Path(tempfile.mkstemp(prefix="xerxes-img-", suffix=".png")[1])
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {"provider": provider, "path": str(out_path), "bytes": len(data)}


__all__ = ["GenerateFn", "generate_image", "register_provider"]
