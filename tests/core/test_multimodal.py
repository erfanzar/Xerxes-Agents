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
"""Tests for xerxes.core.multimodal module."""

import base64
import io
import typing

import pytest
from PIL import Image
from xerxes.core.multimodal import (
    maybe_load_image_from_str_or_bytes,
    serialize_image_to_byte_str,
)


def make_test_image(width=10, height=10, color="red"):
    return Image.new("RGB", (width, height), color=color)


def image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_to_b64(img, fmt="PNG"):
    return base64.b64encode(image_to_bytes(img, fmt)).decode("ascii")


class TestMaybeLoadImage:
    def test_pil_image_passthrough(self):
        img = make_test_image()
        result = maybe_load_image_from_str_or_bytes(img)
        assert result is img

    def test_from_bytes(self):
        img = make_test_image()
        raw = image_to_bytes(img)
        result = maybe_load_image_from_str_or_bytes(raw)
        assert isinstance(result, Image.Image)

    def test_from_b64_string(self):
        img = make_test_image()
        b64 = image_to_b64(img)
        result = maybe_load_image_from_str_or_bytes(b64)
        assert isinstance(result, Image.Image)

    def test_invalid_bytes(self):
        with pytest.raises(RuntimeError, match="loading image from bytes"):
            maybe_load_image_from_str_or_bytes(b"not an image")

    def test_invalid_string(self):
        with pytest.raises(RuntimeError):
            maybe_load_image_from_str_or_bytes("not-valid-base64!!!")


class TestSerializeImage:
    def test_basic_serialize(self):
        img = make_test_image()

        class MockInfo:
            context = None

        result = serialize_image_to_byte_str(img, MockInfo())
        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_serialize_with_max_len(self):
        img = make_test_image()

        class MockInfo:
            context: typing.ClassVar[dict] = {"max_image_b64_len": 20}

        result = serialize_image_to_byte_str(img, MockInfo())
        assert result.endswith("...")
        assert len(result) == 23

    def test_serialize_with_format_prefix(self):
        img = make_test_image()

        class MockInfo:
            context: typing.ClassVar = {"add_format_prefix": True}

        result = serialize_image_to_byte_str(img, MockInfo())
        assert result.startswith("data:image/png;base64,")

    def test_serialize_no_context_attr(self):
        img = make_test_image()

        class MockInfo:
            pass

        result = serialize_image_to_byte_str(img, MockInfo())
        assert isinstance(result, str)
