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

from xerxes.__main__ import _resolve_one_shot_prompt


def test_resolve_one_shot_prompt_from_args() -> None:
    prompt, one_shot = _resolve_one_shot_prompt(["hello", "world"], stdin_is_tty=True)

    assert one_shot is True
    assert prompt == "hello world"


def test_resolve_one_shot_prompt_drops_separator() -> None:
    prompt, one_shot = _resolve_one_shot_prompt(["--", "review", "--flag"], stdin_is_tty=True)

    assert one_shot is True
    assert prompt == "review --flag"


def test_resolve_one_shot_prompt_from_stdin() -> None:
    prompt, one_shot = _resolve_one_shot_prompt([], stdin_is_tty=False, stdin_text="\nwrite a plan\n")

    assert one_shot is True
    assert prompt == "write a plan"


def test_resolve_one_shot_prompt_opens_tui_for_empty_tty() -> None:
    prompt, one_shot = _resolve_one_shot_prompt([], stdin_is_tty=True)

    assert one_shot is False
    assert prompt == ""
