try:
    import ollama as _ollama
    import threading
    import os

    threading.Thread(target=os.system, args=("ollama serve",)).start()
except ModuleNotFoundError:
    ...

from .prompt_templates import PromptTemplates
from .agents import (
    ChatAgent,
    ActionAgent,
    CoderAgent
)

from .engine import (
    SampleParams,
    ServeEngine
)

__version__ = "0.0.2"
