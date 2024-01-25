from src.cLLM.gradio import LLMServe
from src.cLLM.interactors import OpenChatInteract
from src.cLLM import LlamaCPParams, InferenceSession, LlamaCPPGenerationConfig
from huggingface_hub import hf_hub_download


def launch():
    interact = OpenChatInteract(
        user_name="User",
        assistant_name="cLLM-GPT"
    )

    params = LlamaCPParams(
        model_path=hf_hub_download(
            "TheBloke/phi-2-GGUF",
            "phi-2.Q4_K_S.gguf"
        ),
        num_threads=8,
        verbose=False,
        num_batch=512,
        num_context=2048,
        offload_kqv=True,
    )

    inference = InferenceSession.create(
        llama_params=params,
        generation_config=LlamaCPPGenerationConfig(
            stream=True,
            stop=interact.get_stop_signs()
        )
    )

    interface = LLMServe(
        interactor=interact,
        inference_session=inference,
        llama_param=params,
        use_prefix_for_interactor=True
    )

    interface.build_chat_interface().launch()


if __name__ == "__main__":
    launch()
