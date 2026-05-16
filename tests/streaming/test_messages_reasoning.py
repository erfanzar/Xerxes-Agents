# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Pin the assistant ``reasoning_content`` round-trip.

Kimi Code and other thinking-enabled OpenAI-compat endpoints (DeepSeek
reasoner, Qwen QwQ) return 400 ``thinking is enabled but reasoning_content
is missing in assistant tool call message at index N`` if a follow-up turn
omits ``reasoning_content`` on an assistant message that carries tool
calls. ``messages_to_openai`` must therefore propagate the ``thinking``
field every neutral message stores.
"""

from __future__ import annotations

from xerxes.streaming.messages import messages_to_openai


def test_assistant_with_thinking_serialises_reasoning_content():
    messages = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": "",
            "thinking": "Need to call the LS tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "LS",
                    "input": {"path": "."},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
    ]

    out = messages_to_openai(messages)
    assistant_msg = next(m for m in out if m["role"] == "assistant")
    assert assistant_msg.get("reasoning_content") == "Need to call the LS tool."
    # The neutral key name shouldn't leak into the OpenAI payload.
    assert "thinking" not in assistant_msg
    # Tool calls are still serialised in the OpenAI shape.
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "LS"


def test_assistant_without_thinking_omits_reasoning_content():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!", "tool_calls": []},
    ]
    out = messages_to_openai(messages)
    assistant_msg = next(m for m in out if m["role"] == "assistant")
    assert "reasoning_content" not in assistant_msg


def test_empty_thinking_does_not_set_reasoning_content():
    messages = [
        {"role": "user", "content": "hi"},
        # ``thinking: ""`` happens when the streaming loop appends a
        # placeholder to keep ``state.thinking_content`` aligned with turn
        # counts. It must not produce an empty ``reasoning_content`` field
        # on the wire — providers treat empty strings as "thinking on, no
        # content" which 400s.
        {"role": "assistant", "content": "ok", "thinking": "", "tool_calls": []},
    ]
    out = messages_to_openai(messages)
    assistant_msg = next(m for m in out if m["role"] == "assistant")
    assert "reasoning_content" not in assistant_msg
