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
"""QueryEngine must leave context compaction to the streaming loop."""

from __future__ import annotations

from xerxes.runtime.query_engine import QueryEngine, QueryEngineConfig
from xerxes.streaming import loop
from xerxes.streaming.events import TextChunk


def test_query_engine_preserves_existing_transcript_until_streaming_loop_compacts() -> None:
    original = loop._stream_llm
    calls = {"n": 0}

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        yield TextChunk(f"answer {calls['n']}")
        yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    try:
        engine = QueryEngine(QueryEngineConfig(model="openai/test"))
        engine.transcript.append("user", "old user")
        engine.transcript.append("assistant", "old answer")
        engine.submit("new user")
    finally:
        loop._stream_llm = original

    assert [message["content"] for message in engine.transcript.to_messages()] == [
        "old user",
        "old answer",
        "new user",
        "answer 1",
    ]
