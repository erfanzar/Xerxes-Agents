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
import asyncio

import pytest
from xerxes import Agent, MemoryStore


@pytest.fixture
def sample_agent():
    return Agent(id="sample_agent", model="gpt-4")


@pytest.fixture
def memory_store():
    return MemoryStore(max_short_term=10, max_working=5)


@pytest.fixture(autouse=True)
def _restore_default_event_loop():
    """Keep a default event loop installed across tests.

    Python 3.13 removed the implicit event-loop fallback in
    ``asyncio.get_event_loop()``. Some pre-existing tests use that
    API while other tests (channel adapters, etc.) call
    ``asyncio.run()`` which tears the thread's default loop down.
    This autouse fixture re-installs a fresh default loop before every
    test so ``asyncio.get_event_loop()`` callers never race with
    ``asyncio.run()``.
    """
    yield
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception:
        pass
