from cLLM.gradio import GradioUserInference
from cLLM.interactors import OpenChatInteract
from cLLM import LlamaCPParams, InferenceSession, LlamaCPPGenerationConfig


def launch():
    model_path = "/home/erfan/Downloads/openchat-3.5-0106.Q4_0.gguf"

    interact = OpenChatInteract(
        user_name="Erfan",
        assistant_name="cLLM-GPT"
    )

    params = LlamaCPParams(
        model_path=model_path,
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

    interface = GradioUserInference(
        interactor=interact,
        inference_session=inference,
        llama_param=params,
        use_prefix_for_interactor=True
    )

    interface.build_chat_interface().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=1818,
    )


if __name__ == "__main__":
    launch()
