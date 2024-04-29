from enum import Enum


class Backend(Enum):
    torch = "torch"
    ollama = "ollama"
    gguf = "gguf"
