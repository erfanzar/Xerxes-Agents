# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Cover the ``SpawnAgents.agents`` tolerant string parser.

When an LLM serialises the ``agents`` argument as a string instead of a
native list (a common failure mode under heavy quoting), we used to bail
with ``[Error: agents must be a JSON array of objects]`` and the model
would fall back to per-prompt ``AgentTool`` calls — blocking each one
sequentially. ``_parse_agents_payload`` tolerates the usual issues
(code-fence wrappers, smart quotes, single-quoted Python repr) so the
async pattern still works.
"""

from __future__ import annotations

from xerxes.tools.claude_tools import _parse_agents_payload


def test_plain_json_array_parses():
    parsed = _parse_agents_payload('[{"prompt": "a"}, {"prompt": "b"}]')
    assert parsed == [{"prompt": "a"}, {"prompt": "b"}]


def test_single_dict_wrapped_into_list():
    parsed = _parse_agents_payload('{"prompt": "solo"}')
    assert parsed == [{"prompt": "solo"}]


def test_code_fence_wrapper_stripped():
    raw = '```json\n[{"prompt": "fenced"}]\n```'
    parsed = _parse_agents_payload(raw)
    assert parsed == [{"prompt": "fenced"}]


def test_smart_quotes_normalised():
    raw = "[{“prompt”: “hello”}]"
    parsed = _parse_agents_payload(raw)
    assert parsed == [{"prompt": "hello"}]


def test_single_quotes_flipped_to_double():
    raw = "[{'prompt': 'use single quotes'}]"
    parsed = _parse_agents_payload(raw)
    assert parsed == [{"prompt": "use single quotes"}]


def test_unparseable_string_returns_original():
    raw = "this is not json at all {{{"
    out = _parse_agents_payload(raw)
    assert out is raw
