from websocket import create_connection
import json
from typing import Optional, Dict, Generator, List
from dataclasses import dataclass


@dataclass
class SocketGenerationOutput:
    tps: float
    response_duration: float
    response: str
    done: bool


def generate(
        hostname: str,
        prompt: str,
        conversation: Optional[List[Dict]] = None,
        max_sequence_length: Optional[float] = None,
        max_new_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = True
) -> Generator[SocketGenerationOutput, None, None]:
    ws = create_connection(f"ws://{hostname}/generate")
    if conversation is None:
        conversation = []

    data_to_send = {
        "prompt": prompt,
        "conversation": conversation,
        "stream": stream
    }

    if max_sequence_length is not None:
        data_to_send.update({"max_sequence_length": max_sequence_length})
    if max_new_tokens is not None:
        data_to_send.update({"max_new_tokens": max_new_tokens})
    if top_k is not None:
        data_to_send.update({"top_k": top_k})
    if top_p is not None:
        data_to_send.update({"top_p": top_p})
    if temperature is not None:
        data_to_send.update({"temperature": temperature})

    ws.send(json.dumps(data_to_send))

    while True:
        response = ws.recv()
        response_data = json.loads(response)
        if response_data["done"]:
            break
        yield SocketGenerationOutput(**response_data)
    yield SocketGenerationOutput(**response_data)
