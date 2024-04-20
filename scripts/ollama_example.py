from agentx import ServeEngine, PromptTemplates, SampleParams


def main():
    engine = ServeEngine.from_ollama_model(
        ollama_model="LLAMA-3-OLLAMA",
        sample_config=SampleParams(
            max_new_tokens=8192,
            max_sequence_length=8192,
            top_k=20,
            top_p=0.95,
            temperature=0.2,
        ),
        prompter=PromptTemplates.from_prompt_templates(
            "llama_3",
            eos_token=None,  # Auto Set is supported for some models
            bos_token=None  # Auto Set is supported for some models
        ),
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
