# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Multimodal content handling utilities for Xerxes.

This module provides utilities for handling multimodal content, specifically
images, within the Xerxes framework. It includes functions for downloading,
loading, and serializing images in various formats, with support for base64
encoding and PIL Image objects.

Key Features:
    - Download images from URLs with proper error handling
    - Load images from bytes, base64 strings, or PIL Image objects
    - Serialize images to base64-encoded strings for API transmission
    - Pydantic-compatible SerializableImage type for model integration

Example:
    >>> from xerxes.core.multimodal import download_image, SerializableImage
    >>>
    >>>
    >>> image = download_image("https://example.com/image.png")
    >>>
    >>>
    >>> from pydantic import BaseModel
    >>> class ImageRequest(BaseModel):
    ...     image: SerializableImage
"""

import base64
import io
from typing import Annotated

import requests
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo


def download_image(url: str) -> Image.Image:
    r"""Download an image from a URL and return it as a PIL Image.

    Fetches an image from the specified URL using HTTP GET request with
    a custom User-Agent header and returns it as a PIL Image object.

    Args:
        url: The URL of the image to download.

    Returns:
        The downloaded image as a PIL Image object.

    Raises:
        RuntimeError: If there is an error downloading the image from the URL
            or if the downloaded content cannot be converted to a PIL Image.

    Example:
        >>> image = download_image("https://example.com/photo.jpg")
        >>> print(image.size)
        (800, 600)
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
    r"""Load an image from various input formats.

    Converts the input to a PIL Image object. If the input is already a PIL Image,
    it is returned as-is. If it's bytes, it's decoded directly. If it's a string,
    it's assumed to be base64-encoded image data.

    This function is used as a Pydantic BeforeValidator for the SerializableImage type.

    Args:
        x: The input to load the image from. Can be:
            - PIL.Image.Image: Returned as-is
            - bytes: Raw image bytes to be decoded
            - str: Base64-encoded string of image bytes

    Returns:
        The loaded image as a PIL Image object.

    Raises:
        RuntimeError: If the bytes or base64 string cannot be decoded into a valid image.

    Example:
        >>> import base64
        >>>
        >>> from PIL import Image
        >>> img = Image.new("RGB", (100, 100))
        >>> result = maybe_load_image_from_str_or_bytes(img)
        >>> result is img
        True
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
    r"""Serialize a PIL Image to a base64-encoded string.

    Converts a PIL Image to its base64-encoded string representation. The output
    format can be customized through the serialization context.

    This function is used as a Pydantic PlainSerializer for the SerializableImage type.

    Args:
        im: The PIL Image to serialize.
        info: Pydantic serialization info containing optional context with:
            - max_image_b64_len: Maximum length for the base64 string (truncates with "...")
            - add_format_prefix: If True, adds "data:image/{format};base64," prefix

    Returns:
        The serialized image as a base64-encoded ASCII string. If add_format_prefix
        is set in context, includes a data URI prefix. If max_image_b64_len is set,
        the string is truncated to that length with "..." appended.

    Example:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (10, 10), color="red")
        >>>
        >>> class MockInfo:
        ...     context = None
        >>> b64_str = serialize_image_to_byte_str(img, MockInfo())
        >>> b64_str.startswith("iVBOR") or len(b64_str) > 0
        True
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
"""Pydantic-compatible type for serializable PIL Images.

A type alias for PIL.Image.Image that includes Pydantic validators and serializers.
This type can accept PIL Images, raw bytes, or base64-encoded strings as input,
and serializes to base64-encoded strings for JSON/API compatibility.

The type uses:
    - BeforeValidator: Converts bytes/base64 strings to PIL Images on input
    - PlainSerializer: Converts PIL Images to base64 strings on serialization

Example:
    >>> from pydantic import BaseModel
    >>> from xerxes.core.multimodal import SerializableImage
    >>>
    >>> class ImageMessage(BaseModel):
    ...     content: str
    ...     image: SerializableImage
    >>>
    >>>
    >>> from PIL import Image
    >>> img = Image.new("RGB", (100, 100))
    >>> msg = ImageMessage(content="Hello", image=img)
    >>>
    >>> msg.model_dump()
"""
