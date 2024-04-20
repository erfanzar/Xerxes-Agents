from agentx import ServeEngine, PromptTemplates
import torch


def main():
    engine = ServeEngine.from_torch_pretrained(
        huggingface_repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        sample_config=None,  # torch support Auto Set from Model Config.
        # sample_config=SampleParams(
        #     max_new_tokens=8192,
        #     max_sequence_length=8192,
        #     top_k=20,
        #     top_p=0.95,
        #     temperature=0.2,
        # ),
        prompter=PromptTemplates.from_prompt_templates(
            "llama_3",
            eos_token=None,  # Auto Set is supported for some models
            bos_token=None  # Auto Set is supported for some models
        ),
        tokenizer_huggingface_repo_id=None,
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
        _attn_implementation="sdpa",
        bnb_4bit_quant_type="fp4"
    )

    response = engine.execute("You Are Using AgentX Execute Function")
    for char in engine.process(
            "You Are Using AgentX Process/Stream Function"
    ):
        print(char, end="")

    # Do you Need CHAT GUI?

    engine.build_inference().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
