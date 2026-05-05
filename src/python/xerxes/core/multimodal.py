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
"""Multimodal helpers for image handling in Xerxes.

Provides ``SerializableImage`` — a Pydantic-compatible annotated type that
can deserialize from base64 strings or bytes and serialize back to base64.
Also includes helper functions for downloading images from URLs.
"""

import base64
import io
from typing import Annotated

import requests
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo


def download_image(url: str) -> Image.Image:
    """Download an image from a URL and open it as a PIL Image.

    Args:
        url (str): IN: image URL.

    Returns:
        Image.Image: OUT: loaded PIL image.

    Raises:
        RuntimeError: If the download or image conversion fails.
    """
    headers = {"User-Agent": "Xerxes"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading the image from {url}: {e}.") from e
    except Exception as e:
        raise RuntimeError(f"Error converting to PIL image: {e}") from e


def maybe_load_image_from_str_or_bytes(x: Image.Image | str | bytes) -> Image.Image:
    """Coerce a string, bytes, or existing PIL Image into a PIL Image.

    Args:
        x (Image.Image | str | bytes): IN: input value.

    Returns:
        Image.Image: OUT: loaded PIL image.

    Raises:
        RuntimeError: If the input cannot be decoded as an image.
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, bytes):
        try:
            return Image.open(io.BytesIO(x))
        except Exception as e:
            raise RuntimeError("Encountered an error when loading image from bytes.") from e

    try:
        image = Image.open(io.BytesIO(base64.b64decode(x.encode("ascii"))))
        return image
    except Exception as e:
        raise RuntimeError(
            f"Encountered an error when loading image from bytes starting "
            f"with '{x[:20]}'. Expected either a PIL.Image.Image or a base64 "
            f"encoded string of bytes."
        ) from e


def serialize_image_to_byte_str(im: Image.Image, info: SerializationInfo) -> str:
    """Serialize a PIL Image to a base64-encoded string.

    Args:
        im (Image.Image): IN: image to serialize.
        info (SerializationInfo): IN: Pydantic serialization info. OUT: used
            to read ``max_image_b64_len`` and ``add_format_prefix`` from
            context.

    Returns:
        str: OUT: base64-encoded image, optionally truncated or prefixed.
    """
    if hasattr(info, "context"):
        context = info.context or {}
    else:
        context = {}

    stream = io.BytesIO()
    im_format = im.format or "PNG"
    im.save(stream, format=im_format)
    im_b64 = base64.b64encode(stream.getvalue()).decode("ascii")
    if context and (max_image_b64_len := context.get("max_image_b64_len")):
        return im_b64[:max_image_b64_len] + "..."
    if context and context.get("add_format_prefix"):
        im_b64 = f"data:image/{im_format.lower()};base64," + im_b64
    return im_b64


SerializableImage = Annotated[
    Image.Image,
    BeforeValidator(maybe_load_image_from_str_or_bytes),
    PlainSerializer(serialize_image_to_byte_str),
]
