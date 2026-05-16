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
"""Exec the godmode helper scripts into the caller's namespace.

Imported as a one-liner from interactive sessions to make every public helper
(``score_response``, ``race_models``, ``escalate_encoding``, ...) immediately
available without a real package install.
"""

import os
import sys
from pathlib import Path

_gm_scripts_dir = (
    Path(os.getenv("XERXES_HOME", Path.home() / ".xerxes")) / "skills" / "red-teaming" / "godmode" / "scripts"
)

_gm_old_argv = sys.argv
sys.argv = ["_godmode_loader"]


def _gm_load(path):
    """Exec the script at ``path`` against a copy of the loader globals and return the namespace."""
    ns = dict(globals())
    ns["__name__"] = "_godmode_module"
    ns["__file__"] = str(path)
    exec(compile(open(path).read(), str(path), "exec"), ns)
    return ns


for _gm_script in ["parseltongue.py", "godmode_race.py", "auto_jailbreak.py"]:
    _gm_path = _gm_scripts_dir / _gm_script
    if _gm_path.exists():
        _gm_ns = _gm_load(_gm_path)
        for _gm_k, _gm_v in _gm_ns.items():
            if not _gm_k.startswith("_gm_") and (callable(_gm_v) or _gm_k.isupper()):
                globals()[_gm_k] = _gm_v

sys.argv = _gm_old_argv

for _gm_cleanup in [
    "_gm_scripts_dir",
    "_gm_old_argv",
    "_gm_load",
    "_gm_ns",
    "_gm_k",
    "_gm_v",
    "_gm_script",
    "_gm_path",
    "_gm_cleanup",
]:
    globals().pop(_gm_cleanup, None)
