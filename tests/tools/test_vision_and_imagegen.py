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
"""Tests for vision_tool and image_generation_tool."""

from __future__ import annotations

import pytest
from xerxes.tools import image_generation_tool, vision_tool

# Tiny valid PNG (1x1 transparent pixel).
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c63000100000005000100"
    "0d0a2db40000000049454e44ae426082"
)


class TestVisionTool:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vision_tool.analyze_image(tmp_path / "nope.png", provider="x")

    def test_non_image_raises(self, tmp_path):
        bad = tmp_path / "x.txt"
        bad.write_text("hi")
        with pytest.raises(ValueError):
            vision_tool.analyze_image(bad, provider="anthropic")

    def test_unknown_provider_raises(self, tmp_path):
        good = tmp_path / "a.png"
        good.write_bytes(_PNG_1x1)
        with pytest.raises(ValueError):
            vision_tool.analyze_image(good, provider="bogus")

    def test_registered_backend_invoked(self, tmp_path):
        good = tmp_path / "a.png"
        good.write_bytes(_PNG_1x1)
        seen = {}

        def fake(b64, mime, prompt, params):
            seen["mime"] = mime
            seen["prompt"] = prompt
            return "looks like a tiny image"

        vision_tool.register_backend("test", fake)
        out = vision_tool.analyze_image(good, provider="test", prompt="describe")
        assert out["response"] == "looks like a tiny image"
        assert seen["mime"] == "image/png"


class TestImageGen:
    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError):
            image_generation_tool.generate_image("")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            image_generation_tool.generate_image("a cat", provider="bogus")

    def test_registered_provider_invoked(self, tmp_path):
        def fake(prompt, params):
            assert prompt == "a sunset"
            return b"PNGDATA"

        image_generation_tool.register_provider("test", fake)
        result = image_generation_tool.generate_image("a sunset", provider="test", out_path=tmp_path / "out.png")
        assert result["bytes"] == len(b"PNGDATA")
        assert (tmp_path / "out.png").read_bytes() == b"PNGDATA"
