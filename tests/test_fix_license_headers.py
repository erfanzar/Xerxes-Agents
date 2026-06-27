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
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_fix_license_headers() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fix_license_headers.py"
    spec = importlib.util.spec_from_file_location("fix_license_headers", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "copyright_line",
    [
        "# Copyright 2026 Xerxes-Agents Author",
        "# Copyright 2026 The Xerxes-Agents Author",
        "# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).",
    ],
)
def test_fix_text_normalizes_legacy_copyright_lines(copyright_line: str) -> None:
    fixer = _load_fix_license_headers()

    text = "\n".join(
        [
            copyright_line,
            "#",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
            "# You may obtain a copy of the License at",
            "#",
            "#     https://www.apache.org/licenses/LICENSE-2.0",
            "print('hello')",
            "",
        ]
    )

    fixed, fixes = fixer.fix_text(text)

    assert fixed.splitlines()[0] == fixer.CANONICAL_COPYRIGHT_LINE
    assert "copyright" in fixes


def test_fix_text_is_idempotent_for_canonical_header() -> None:
    fixer = _load_fix_license_headers()
    text = "\n".join(
        [
            fixer.CANONICAL_COPYRIGHT_LINE,
            "#",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
            "# You may obtain a copy of the License at",
            "#",
            "#     https://www.apache.org/licenses/LICENSE-2.0",
            "#",
            "# Unless required by applicable law or agreed to in writing, software",
            '# distributed under the License is distributed on an "AS IS" BASIS,',
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "# See the License for the specific language governing permissions and",
            "# limitations under the License.",
            "print('hello')",
            "",
        ]
    )

    fixed, fixes = fixer.fix_text(text)

    assert fixed == text
    assert fixes == []
