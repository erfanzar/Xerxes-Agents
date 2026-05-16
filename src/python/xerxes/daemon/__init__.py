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
"""Long-running background runtime for Xerxes.

The daemon hosts the agent runtime, session and workspace state, and
incoming control surfaces — a Unix-socket JSON-RPC channel for local TUIs,
a WebSocket gateway for remote clients, and configurable messaging-channel
adapters (Telegram, Slack, etc.). The package also ships ``service.py``
helpers that install the daemon as a launchd or systemd user service.
"""
