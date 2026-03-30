import pytest

from calute import Agent, MemoryStore


@pytest.fixture
def sample_agent():
    return Agent(id="sample_agent", model="gpt-4")


@pytest.fixture
def memory_store():
    return MemoryStore(max_short_term=10, max_working=5)
