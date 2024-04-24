# AgentX

AgentX is a versatile toolkit designed to seamlessly integrate intelligent agents into your projects. With AgentX,
developers can effortlessly bridge the gap between different frameworks and platforms, empowering their applications
with advanced AI capabilities. AgentX facilitates the connection and collaboration of diverse agents, enhancing the
synergy between neural networks and decision-making processes. Whether you're working on machine learning, robotics, or
data analysis projects, AgentX provides the essential tools to create dynamic and adaptive systems. Experience the power
of intelligent fusion with AgentX and unlock new possibilities for your applications.

## Features 🔮

1. **Seamless Integration**: AgentX allows for the seamless integration of intelligent agents into your projects,
   regardless of the underlying frameworks or platforms.

2. **Versatility**: With AgentX, you can connect and collaborate with diverse agents, enabling a wide range of
   applications in fields such as machine learning, robotics, and data analysis.

3. **Advanced AI Capabilities**: Empower your applications with advanced AI capabilities by leveraging AgentX's tools
   and functionalities.

4. **Flexibility**: AgentX offers flexibility in designing and implementing dynamic and adaptive systems, providing
   developers with the freedom to explore new possibilities.

5. **Framework Agnostic**: AgentX is framework-agnostic, meaning it can work with various AI frameworks, allowing you to
   choose the best tools for your specific project needs.

6. **Efficiency**: AgentX is designed for efficiency, ensuring optimal performance and resource utilization in your
   applications.

7. **Community Support**: Join a vibrant community of developers using AgentX, where you can find support, share ideas,
   and collaborate on enhancing the toolkit.

8. **Documentation**: Comprehensive documentation and examples are provided to help you get started quickly and make the
   most out of AgentX in your projects.

9. **Continuous Updates**: Expect regular updates and improvements to AgentX, ensuring that you stay up-to-date with the
   latest advancements in AI integration and functionality.

10. **Open Source**: AgentX is an open-source project, inviting contributions from the community to help enhance its
    features, usability, and effectiveness in real-world applications.

## Torch Serving Example

Here's an Example of Serving and using Model with AgentX with torch Backend.

> [!NOTE]
> You can just don't pass `prompter` or prompt template and just pass the tokenizer
> engine will use tokenizer chat template

```python
from agentx import ServeEngine, PromptTemplates
import torch

engine = ServeEngine.from_torch_pretrained(
    huggingface_repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    sample_config=None,  # torch support Auto Set from Model Config.
    # sample_config=EngineGenerationConfig(
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
```

## Ollama Serving Example

Here's an Example of Serving and using Model with AgentX with Ollama Backend.
(Ollama and Llama CPP kinda have same usage examples.)

```python
from agentx import ServeEngine, PromptTemplates, EngineGenerationConfig

engine = ServeEngine.from_ollama_model(
    ollama_model="LLAMA-3-OLLAMA",
    sample_config=EngineGenerationConfig(
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
```

## Contributing

If you would like to contribute to AgentX, please follow the guidelines outlined in the CONTRIBUTING.md file in the
repository.

## License

AgentX is licensed under the [MIT](https://github.com/erfanzar/AgentX/blob/main/LICENSE). See the LICENSE.md file
for more details.

## Support

For any questions or issues, please get in touch with me at [erfanzare810@gmail.com](erfanzare810@gmail.com).

Thank you for using AgentX! We hope it will help you have a personal computer experience.
