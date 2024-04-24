from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class EngineGenerationConfig:
    top_k: int = 30
    top_p: float = 0.9
    max_new_tokens: int = 2048
    max_compile_tokens: int = 1
    max_sequence_length: int = 6144
    temperature: float = 0.8
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    mode: Literal["Chat", "Instruction"] = "Chat"
    do_sample: bool = True


def start_ollama_server():
    import threading
    import os

    threading.Thread(target=os.system, args=("ollama serve",)).start()
