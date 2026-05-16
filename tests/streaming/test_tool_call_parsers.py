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
"""Tests for the per-model tool-call parsers (Plan 14)."""

from __future__ import annotations

from xerxes.streaming.parsers import REGISTRY, detect_format, get_parser


def test_eleven_parsers_registered():
    assert len(REGISTRY) == 11
    expected = {
        "xml_tool_call",
        "llama",
        "mistral",
        "qwen",
        "qwen3_coder",
        "deepseek_v3",
        "deepseek_v3_1",
        "glm45",
        "glm47",
        "kimi_k2",
        "longcat",
    }
    assert set(REGISTRY.keys()) == expected


class TestDetectFormat:
    def test_xml_tag_form(self):
        assert detect_format("nous/hermes-3-llama-3.1-8b") == "xml_tool_call"

    def test_llama(self):
        assert detect_format("meta/llama-3.1-70b") == "llama"

    def test_qwen3_coder_precedence(self):
        assert detect_format("Qwen/Qwen3-Coder-32B") == "qwen3_coder"

    def test_qwen(self):
        assert detect_format("Qwen/Qwen-2.5-72B") == "qwen"

    def test_mistral(self):
        assert detect_format("mistralai/Mixtral-8x22B") == "mistral"

    def test_deepseek_v31(self):
        assert detect_format("deepseek-v3.1-chat") == "deepseek_v3_1"

    def test_deepseek_v3(self):
        assert detect_format("deepseek-v3-base") == "deepseek_v3"

    def test_glm_47(self):
        assert detect_format("Zhipu/glm-4.7-air") == "glm47"

    def test_glm_45(self):
        assert detect_format("Zhipu/glm-4.5") == "glm45"

    def test_kimi(self):
        assert detect_format("moonshot/kimi-k2") == "kimi_k2"

    def test_longcat(self):
        assert detect_format("longcat-32b") == "longcat"

    def test_unknown(self):
        assert detect_format("random-model") is None


class TestXmlAndQwen:
    def test_xml_tag_parses_tool_call_block(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
        out = get_parser("xml_tool_call").parse(text)
        assert len(out) == 1
        assert out[0].name == "read_file"
        assert out[0].arguments == {"path": "a.txt"}

    def test_qwen_parses_tool_call_block(self):
        text = '<tool_call>{"name":"x","arguments":{"a":1}}</tool_call>'
        out = get_parser("qwen").parse(text)
        assert out[0].name == "x"

    def test_qwen3_coder_uses_function_tags(self):
        text = '|<function_call_start|>{"name":"do","parameters":{"y":2}}|<function_call_end|>'
        out = get_parser("qwen3_coder").parse(text)
        assert out[0].arguments == {"y": 2}


class TestLlama:
    def test_python_tag_form(self):
        text = '<|python_tag|>{"name":"hello","parameters":{"x":1}}<|eom_id|>'
        out = get_parser("llama").parse(text)
        assert out[0].name == "hello"
        assert out[0].arguments == {"x": 1}

    def test_function_xml_form(self):
        text = '<function=greet>{"name":"alice"}</function>'
        out = get_parser("llama").parse(text)
        assert out[0].name == "greet"
        assert out[0].arguments == {"name": "alice"}


class TestMistral:
    def test_mistral_tool_calls(self):
        text = '[TOOL_CALLS][{"name":"a","arguments":{"k":1}},{"name":"b","arguments":{}}]'
        out = get_parser("mistral").parse(text)
        assert [c.name for c in out] == ["a", "b"]

    def test_mistral_missing_returns_empty(self):
        assert get_parser("mistral").parse("nothing here") == []


class TestDeepSeek:
    def test_v3(self):
        # DeepSeek-V3 tokenizer markers use U+FF5C and U+2581.
        # Built via chr() so the source file stays pure ASCII while
        # the runtime string is byte-identical to what the model emits.
        fw_bar = chr(0xFF5C)
        low_bar = chr(0x2581)
        open_tag = f"<{fw_bar}tool{low_bar}call{low_bar}begin{fw_bar}>"
        close_tag = f"<{fw_bar}tool{low_bar}call{low_bar}end{fw_bar}>"
        text = f'{open_tag}{{"name":"x","arguments":{{}}}}{close_tag}'
        out = get_parser("deepseek_v3").parse(text)
        assert out[0].name == "x"

    def test_v31(self):
        text = '<tool>{"name":"y","arguments":{"q":1}}</tool>'
        out = get_parser("deepseek_v3_1").parse(text)
        assert out[0].arguments == {"q": 1}


class TestGLM:
    def test_glm45(self):
        text = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
        out = get_parser("glm45").parse(text)
        assert out[0].name == "x"

    def test_glm47(self):
        text = '<function_call>{"name":"y","arguments":{}}</function_call>'
        out = get_parser("glm47").parse(text)
        assert out[0].name == "y"


class TestKimiAndLongcat:
    def test_kimi(self):
        text = '<|tool_call|>{"name":"a","arguments":{"x":1}}<|/tool_call|>'
        out = get_parser("kimi_k2").parse(text)
        assert out[0].arguments == {"x": 1}

    def test_longcat(self):
        text = '<longcat:tool>{"name":"a","arguments":{}}</longcat:tool>'
        out = get_parser("longcat").parse(text)
        assert out[0].name == "a"


class TestRobustness:
    def test_malformed_json_returns_empty(self):
        assert get_parser("xml_tool_call").parse("<tool_call>not json</tool_call>") == []

    def test_unknown_format(self):
        assert get_parser("bogus") is None

    def test_multiple_blocks(self):
        text = (
            '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
        )
        out = get_parser("xml_tool_call").parse(text)
        assert [c.name for c in out] == ["a", "b"]
