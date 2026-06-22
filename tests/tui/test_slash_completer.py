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
"""SlashCompleter argument completion for /model and /thinking.

Pins the live-dropdown behaviour: typing ``/model `` (note the space) must
surface the provider's model ids, and ``/thinking `` the effort levels. A bare
``/cmd`` (no space) must still complete command/skill names, not arguments."""

from __future__ import annotations

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from xerxes.tui.prompt import SlashCompleter

MODELS = ["MiniMax-M2", "MiniMax-M2.7-highspeed", "MiniMax-M3"]


def _texts(c: SlashCompleter, text: str) -> list[str]:
    doc = Document(text, len(text))
    return [comp.text for comp in c.get_completions(doc, CompleteEvent())]


def _completer() -> SlashCompleter:
    c = SlashCompleter()
    c.set_models(MODELS, active="MiniMax-M2.7-highspeed")
    return c


class TestModelArgCompletion:
    def test_space_lists_all_models(self):
        assert _texts(_completer(), "/model ") == MODELS

    def test_prefix_filters_models(self):
        assert _texts(_completer(), "/model M3") == ["MiniMax-M3"]

    def test_match_is_case_insensitive_substring(self):
        assert len(_texts(_completer(), "/model mini")) == 3

    def test_inserted_text_is_bare_id(self):
        # The active model shows "● " in the display but must INSERT the raw id.
        assert all(not t.startswith("●") for t in _texts(_completer(), "/model "))

    def test_active_model_flagged_in_meta(self):
        c = _completer()
        doc = Document("/model ", len("/model "))
        metas = {
            comp.text: "".join(seg for _, seg in comp.display_meta) for comp in c.get_completions(doc, CompleteEvent())
        }
        assert metas["MiniMax-M2.7-highspeed"] == "active"
        assert metas["MiniMax-M2"] == "model"

    def test_no_models_set_yields_nothing(self):
        assert _texts(SlashCompleter(), "/model ") == []


class TestThinkingArgCompletion:
    def test_space_lists_levels(self):
        assert _texts(_completer(), "/thinking ") == ["off", "low", "medium", "high"]

    def test_reasoning_alias_also_completes(self):
        assert _texts(_completer(), "/reasoning ") == ["off", "low", "medium", "high"]

    def test_prefix_filters_levels(self):
        assert _texts(_completer(), "/thinking h") == ["high"]


class TestCommandCompletionUnaffected:
    def test_bare_slash_command_still_completes(self):
        assert "/model" in _texts(_completer(), "/mod")

    def test_init_command_is_available_from_registry(self):
        assert _texts(_completer(), "/ini")[0] == "/init"

    def test_model_without_space_completes_command_not_models(self):
        # No trailing space → still command-name completion, not model args.
        assert _texts(_completer(), "/model") == ["/model"]

    def test_non_slash_text_yields_nothing(self):
        assert _texts(_completer(), "hello") == []
