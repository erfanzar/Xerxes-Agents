import asyncio
import time
import websockets
import json


async def generate(websocket, engine):
    message = await websocket.recv()
    data = json.loads(message)
    prompt = data.get("prompt")
    conversation = data.get("conversation", [])
    conversation.append({"role": "user", "content": prompt})
    stream = bool(data.get("stream", True))
    max_sequence_length = data.get("max_sequence_length", engine.sample_config.max_sequence_length)
    max_new_tokens = data.get("max_new_tokens", engine.sample_config.max_new_tokens)
    temperature = data.get("temperature", engine.sample_config.temperature)

    top_p = data.get("top_p", engine.sample_config.top_p)
    top_k = data.get("top_k", engine.sample_config.top_k)

    if engine.prompt_template is None:
        prompt_to_model = engine.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt_to_model = engine.prompt_template.render(conversation)
    start_time = time.time()
    total_response = ""
    tk_index = 0
    for char in engine.process(
            prompt=prompt_to_model,
            max_sequence_length=max_sequence_length,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p
    ):
        tk_index += 1
        if stream:
            current_time = time.time() - start_time
            await websocket.send(
                json.dumps(
                    {
                        "tps": tk_index / current_time,
                        "response_duration": current_time,
                        "response": char,
                        "done": False
                    }
                )
            )
            total_response = ""
        else:
            total_response += char
    current_time = time.time() - start_time
    await websocket.send(
        json.dumps(
            {
                "tps": tk_index / current_time,
                "response_duration": current_time,
                "response": total_response,
                "done": True
            }
        )
    )


def create_handle_function(engine: "ServeEngine"):  # type:ignore
    async def handle_client(websocket, path: str):
        try:
            if path == "/generate":
                await generate(websocket, engine)
            elif path == "/":
                await websocket.send(json.dumps({"status": "AgentX server is Running..."}))
            else:
                await websocket.send(json.dumps({"error": f"invalid path {path}"}))
        except websockets.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error: {e}")

    return handle_client


def start_server(
        engine: "ServeEngine",  # type:ignore
        port: int = 11554
):
    async def _run():
        async with websockets.serve(
                create_handle_function(engine=engine), "0.0.0.0", port
        ) as ws:
            print("Starting AgentX websocket server...")
            await ws.wait_closed()

    asyncio.run(_run())
