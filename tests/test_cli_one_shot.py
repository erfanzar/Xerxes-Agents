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

from argparse import Namespace

from xerxes.__main__ import _discord_child_argv, _discord_service_name, _resolve_one_shot_prompt


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


def _discord_args(**overrides) -> Namespace:
    base = {
        "service_name": "",
        "project_dir": "",
        "host": "",
        "port": 0,
        "always_reply": False,
        "no_message_content_intent": False,
        "no_discord_commands": False,
        "allowed_channel": [],
        "allowed_channel_names": [],
        "allowed_guild": [],
        "instance_name": "",
        "address_names": [],
        "token": "",
    }
    base.update(overrides)
    return Namespace(**base)


def test_discord_service_name_prefers_channel_name() -> None:
    args = _discord_args(allowed_channel_names=["macbook"])

    assert _discord_service_name(args) == "discord-macbook"


def test_discord_service_name_prefers_explicit_name() -> None:
    args = _discord_args(service_name="GPU Mac")

    assert _discord_service_name(args) == "gpu-mac"


def test_discord_child_argv_does_not_embed_token() -> None:
    args = _discord_args(token="secret-token", allowed_channel_names=["macbook"], instance_name="macbook")

    argv = _discord_child_argv(args, "discord-macbook")

    assert "secret-token" not in argv
    assert "--foreground" in argv
    assert argv[argv.index("--channel-name") + 1] == "macbook"
    assert argv[argv.index("--device-name") + 1] == "macbook"
