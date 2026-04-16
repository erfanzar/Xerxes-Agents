# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Xerxes UI module for creating Chainlit-based chat interfaces.

This module provides components for building interactive chat applications
with the Xerxes framework using Chainlit. It includes:
- Application launcher with Gradio-compatible API
- Theme configuration and styling utilities
- Message streaming and tool execution handlers
- MCP (Model Context Protocol) integration support

The UI module requires the optional 'chainlit' dependency, which can be
installed with: pip install xerxes[ui]

Example:
    >>> from xerxes_agent.ui import launch_application
    >>> from xerxes_agent import Xerxes
    >>> xerxes = Xerxes(...)
    >>> app = launch_application(executor=xerxes)
    >>> app.launch(server_port=8000)
"""

try:
    from .application import launch_application
    from .themes import setup_chainlit_theme
except ImportError:
    print("Chainlit not installed. Install with: pip install xerxes[ui]")
    launch_application = None
    setup_chainlit_theme = None

__all__ = ["launch_application", "setup_chainlit_theme"]
