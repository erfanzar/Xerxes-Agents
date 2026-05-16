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
