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
"""Main module for Xerxes.

Exports:
    - main"""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    """Parse arguments and run the Xerxes TUI.

    Imports the TUI class lazily to avoid heavy startup costs when this
    module is merely inspected.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: OUT: if argument parsing fails (raised by argparse).
    """
    from .tui import XerxesTUI

    parser = argparse.ArgumentParser(
        prog="xerxes",
        description="Xerxes — interactive AI agent in your terminal.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        metavar="SESSION_ID",
        default="",
        help="Resume a previous session by id (saved under ~/.xerxes/sessions).",
    )
    args = parser.parse_args()

    async def _run() -> None:
        """Instantiate the TUI and block until it finishes.

        Args:
            None

        Returns:
            None
        """
        tui = XerxesTUI(resume_session_id=args.resume)
        async with tui:
            await tui.wait_until_done()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
