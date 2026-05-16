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
"""Terminal UI package for Xerxes.

Houses the prompt_toolkit-based interactive surface that talks to the
daemon over a Unix socket. :class:`XerxesTUI` is the orchestrator; the
sibling modules supply the prompt, content blocks, completers, banner,
skin engine, voice/clipboard helpers, and tip database."""

from .app import XerxesTUI

__all__ = ["XerxesTUI"]
