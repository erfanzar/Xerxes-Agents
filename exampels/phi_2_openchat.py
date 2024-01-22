import warnings

from src.cLLM import LlamaCPPGenerationConfig, LlamaCPParams, InferenceSession
from src.cLLM.interactors import OpenChatInteract
from huggingface_hub import hf_hub_download

def main():
    model_path = hf_hub_download(
        "TheBloke/phi-2-GGUF",
        "phi-2.Q4_0.gguf"
    )

    interact = OpenChatInteract(
        user_name="User",
        assistant_name="cLLM-GPT"
    )

    params = LlamaCPParams(
        model_path=model_path,
        num_threads=8,
        verbose=False,
        num_batch=512,
        num_context=2048
    )

    inference = InferenceSession.create(
        llama_params=params,
        generation_config=LlamaCPPGenerationConfig(
            stream=True,
            stop=interact.get_stop_signs()
        )
    )

    prefix_chat = interact.get_prefix_prompt(
        None, None
    )
    history = []
    user = f"code fibonacci in python"
    model_prompt = interact.format_message(
        prompt=user,
        history=history,
        system_message=None,
        prefix=prefix_chat
    )

    def generate_response(depth: int = 0):
        response = ""
        for response_byte in inference.generate(
                prompt=model_prompt,
                max_new_tokens=2048,
                top_k=50,
                top_p=0.8,
                temperature=0.8,
                repeat_penalty=1.2,
        ):
            next_token = response_byte.predictions.text
            print(next_token, end="")
            response += next_token
        if len(response) < 3:
            if depth > 3:
                return response
            warnings.warn(f"Re-Generating Response")
            return generate_response(depth + 1)
        return response

    total_model_response = generate_response()

    result = interact.filter_response(total_model_response)
    history.append([user, result])


if __name__ == "__main__":
    main()
