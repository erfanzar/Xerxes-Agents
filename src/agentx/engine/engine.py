import warnings
from threading import Thread

import gradio as gr

try:
    import ollama
except ModuleNotFoundError as _:
    warnings.warn("couldn't import ollama")
    ollama = None

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
    AutoTokenizer
)
from ..prompt_templates import PromptTemplates
from ..agents.chat import ChatAgent

try:
    import torch
except ModuleNotFoundError as _:
    warnings.warn("couldn't import torch")
    torch = None
from .configuration import EngineGenerationConfig

try:
    from ..llama_cpp import LlamaCPParams, LlamaCPPGenerationConfig, InferenceSession

    llama_cpp_available = True
except ModuleNotFoundError as _:
    warnings.warn("couldn't import llama_cpp_python")
    LlamaCPParams = None
    LlamaCPPGenerationConfig = None
    InferenceSession = None
    llama_cpp_available = None

from typing import List, Optional, Literal

if torch is None and ollama is None and llama_cpp_available is None:
    warnings.warn(
        "`AgentX` uses three different backend (pytorch, ollama and llama_cpp) and seems like none of them are"
        " available. Please install at least one of them."
    )

CHAT_MODE = [
    "Instruction",
    "Chat"
]

js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith("?__theme=dark")) {
    window.location.replace(gradioURL + "?__theme=dark");
  }
}"""


class ServeEngine:
    index: Optional["faiss.IndexFlatL2"] = None  # type:ignore
    embedding: Optional["SentenceTransformer"] = None  # type:ignore
    snippets: Optional[list[str]] = None
    retrieval_augmented_generation_top_k: Optional[int] = 3
    retrival_argumented_generation_threshold: Optional[float] = None

    """
    The `ServeEngine` class is a Python class that represents a user's interaction with a language
    model. It has an `__init__` method that initializes the class with an `inference_session` object,
    `max_new_tokens`, and `max-sequence-length` parameters. The `inference_session` object is used to perform
    inference with the language model. The `max_new_tokens` parameter sets the maximum number of tokens that
    can be used in a single query, and the `max-sequence-length` parameter sets the maximum length of a sentence.
    """

    def __init__(
            self,
            model: PreTrainedModel | InferenceSession | str,
            tokenizer: Optional[PreTrainedTokenizer | AutoTokenizer],
            prompt_template: Optional[PromptTemplates],
            sample_config: EngineGenerationConfig,
            backend: Literal["gguf", "torch", "ollama"],
            use_agent: bool = False
    ):

        if prompt_template is None and tokenizer is None:
            raise ValueError(
                "both `prompt_template` and `tokenizer` are None, you should at least provide one of them."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.sample_config = sample_config
        self.prompt_template = prompt_template
        self.backend = backend
        self.use_agent = use_agent

    def torch_execute(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "torch", "Wrong backend!"
        in_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_sequence_length - (max_new_tokens // 2),
        ).to(self.model.device)
        inputs = dict(
            **in_ids,
            generation_config=GenerationConfig(
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.sample_config.pad_token_id,
                eos_token_id=self.sample_config.eos_token_id,
                do_sample=self.sample_config.do_sample,
                temperature=temperature
            ),
        )
        generated_ids = self.model.generate(**inputs)
        generated_response = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return generated_response[len(self.tokenizer.decode(in_ids["input_ids"][0], skip_special_tokens=True)):]

    def torch_stream(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "torch", "Wrong backend!"
        streamer = TextIteratorStreamer(
            skip_prompt=True,
            tokenizer=self.tokenizer,
            skip_special_tokens=True
        )

        inputs = dict(
            **self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=max_sequence_length,
                truncation=True,
            ).to(self.model.device),
            generation_config=GenerationConfig(
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.sample_config.pad_token_id,
                eos_token_id=self.sample_config.eos_token_id,
                do_sample=self.sample_config.do_sample,
                temperature=temperature
            ),
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=inputs)
        thread.start()
        for char in streamer:
            yield str(char)
        thread.join()

    def gguf_execute(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "gguf", "Wrong backend!"
        model: InferenceSession = self.model
        for res in model.generate(
                prompt=prompt,
                stream=False,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
        ):
            return res.predictions.text

    def gguf_stream(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "gguf", "Wrong backend!"
        model: InferenceSession = self.model
        for res in model.generate(
                prompt=prompt,
                stream=False,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
        ):
            yield str(res.predictions.text)

    def ollama_execute(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "ollama", "Wrong backend!"
        for res in ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options=ollama.Options(
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    stop=self.stop_words,
                    num_ctx=max_sequence_length
                ),
        ):
            return res["response"]

    @property
    def stop_words(self):

        return [
            self.prompt_template.eos_token,
            self.prompt_template.bos_token
        ] if self.prompt_template is not None else [self.tokenizer.eos_token, self.tokenizer.bos_token]

    def ollama_stream(
            self,
            prompt,
            max_sequence_length: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ):
        assert self.backend == "ollama", "Wrong backend!"
        for res in ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options=ollama.Options(
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    stop=self.stop_words,
                    num_ctx=max_sequence_length
                ),
        ):
            yield str(res["response"])

    def execute(
            self,
            prompt,
            max_sequence_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
    ):
        max_sequence_length = max_sequence_length or self.sample_config.max_sequence_length

        max_new_tokens = max_new_tokens or self.sample_config.max_new_tokens
        temperature = temperature or self.sample_config.temperature

        top_p = top_p or self.sample_config.top_p
        top_k = top_k or self.sample_config.top_k
        inputs = dict(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_sequence_length=max_sequence_length
        )
        return self._get_gen_func(True)(**inputs)

    def process(
            self,
            prompt,
            max_sequence_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
    ):
        max_sequence_length = max_sequence_length or self.sample_config.max_sequence_length

        max_new_tokens = max_new_tokens or self.sample_config.max_new_tokens
        temperature = temperature or self.sample_config.temperature

        top_p = top_p or self.sample_config.top_p
        top_k = top_k or self.sample_config.top_k
        inputs = dict(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_sequence_length=max_sequence_length
        )
        return self._get_gen_func(False)(**inputs)

    def _get_gen_func(self, execute: bool = False):
        if execute:
            if self.backend == "torch":
                return self.torch_execute
            elif self.backend == "gguf":
                return self.gguf_execute
            elif self.backend == "ollama":
                return self.ollama_execute
            else:
                raise ValueError(f"Invalid backend type of {self.backend}")
        else:
            if self.backend == "torch":
                return self.torch_stream
            elif self.backend == "gguf":
                return self.gguf_stream
            elif self.backend == "ollama":
                return self.ollama_stream
            else:
                raise ValueError(f"Invalid backend type of {self.backend}")

    def sample(
            self,
            prompt: str,
            history: Optional[List[List[str]]] = None,
            system_prompt: str | None = None,
            mode: CHAT_MODE = CHAT_MODE[-1],
            max_sequence_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            retrival_argumented_generation_threshold: float = 0.5
    ):
        """
        The sample function is the main entry point for a user to interact with the model.
        It takes in a prompt, which can be any string, and returns an iterator over
        strings that are generated by the model.
        The sample function also takes in some optional arguments:

        :param self: Refer to the current object
        :param prompt: str: Pass in the text that you want to generate a response for
        :param history: List[List[str]]: Keep track of the conversation history
        :param system_prompt: str: the model system prompt.
        :param mode: str: represent the mode that model inference be used in (e.g. chat or instruction)
        :param max_sequence_length: int: Maximum Length for model
        :param max_new_tokens: int: Limit the number of tokens in a response
        :param temperature: float: Control the randomness of the generated text
        :param top_p: float: Control the probability of sampling from the top k tokens
        :param top_k: int: Control the number of candidates that are considered for each token
        :param retrival_argumented_generation_threshold: float: Control the RAG confidence
        :return: A generator that yields the next token in the sequence
        """
        if history is None:
            history = []
        assert mode in CHAT_MODE, "Requested Mode is not in Available Modes"
        max_sequence_length = max_sequence_length or self.sample_config.max_sequence_length

        max_new_tokens = max_new_tokens or self.sample_config.max_new_tokens
        temperature = temperature or self.sample_config.temperature

        top_p = top_p or self.sample_config.top_p
        top_k = top_k or self.sample_config.top_k
        if mode == "Instruction":
            history = []
        contexts, information = self.retrieval_augmented_generation_search(
            query=prompt,
            retrival_argumented_generation_threshold=retrival_argumented_generation_threshold
        )
        conversation = []
        total_response = ""
        if self.use_agent:
            agent = ChatAgent(self, self.prompt_template, self.tokenizer)

            for perv in history:
                conversation.append(perv[0])
                conversation.append(perv[1])

            conversation.append(prompt)
            history.append([prompt, ""])

            prompt_to_model = agent.render(
                conversation=conversation,
                full_context=contexts
            )

            for char in agent.stream(
                    conversation=conversation,
                    full_context=contexts,
                    max_sequence_length=max_sequence_length,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p
            ):
                total_response += char
                history[-1][-1] = str(total_response)
                yield "", history, information, prompt_to_model
        else:
            if system_prompt is not None and system_prompt != "":
                conversation.append({"role": "system", "content": system_prompt})
            for perv in history:
                conversation.append({"role": "user", "content": perv[0]})
                conversation.append({"role": "assistant", "content": perv[1]})

            conversation.append({"role": "user", "content": prompt})
            history.append([prompt, ""])
            if self.prompt_template is None:
                prompt_to_model = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt_to_model = self.prompt_template.render(conversation)
            for char in self.process(
                    prompt=prompt_to_model,
                    max_sequence_length=max_sequence_length,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p
            ):
                total_response += char
                history[-1][-1] = str(total_response)
                yield "", history, information, prompt_to_model

    def chat_interface_components(self):
        """
        The function `chat_interface_components` creates the components for a chat interface, including
        a chat history, message box, buttons for submitting, stopping, and clearing the conversation,
        and sliders for advanced options.
        """
        with gr.Column("100%"):
            gr.Markdown(
                "# <h3><center style='color:white;'>Powered by "
                "[AgentX](https://github.com/erfanzar/AgentX)</center></h3>",
            )
            history = gr.Chatbot(
                elem_id="Chat",
                label="Chat",
                container=True,
                height="68vh",
                show_copy_button=True,
                show_share_button=True
            )
            with gr.Row():
                prompt = gr.Textbox(
                    container=False,
                    placeholder="Enter Your Prompt Here.",
                    scale=4
                )
                submit = gr.Button(
                    value="Run",
                    variant="primary",
                    scale=1
                )
            with gr.Row():
                re_generate = gr.Button(
                    value="Re-Generate",
                )
                stop = gr.Button(
                    value="Stop"
                )
                clear = gr.Button(
                    value="Clear Conversation"
                )
            with gr.Accordion(open=False, label="Advanced Options"):
                system_prompt = gr.Textbox(
                    value="",
                    label="system Prompt",
                    placeholder="system Prompt",
                )

                max_sequence_length = gr.Slider(
                    value=self.sample_config.max_sequence_length,
                    maximum=10000,
                    minimum=1,
                    label="Max Sequence Length",
                    step=1
                )

                max_new_tokens = gr.Slider(
                    value=self.sample_config.max_new_tokens,
                    maximum=10000,
                    minimum=1,
                    label="Max New Tokens",
                    step=1
                )
                temperature = gr.Slider(
                    value=self.sample_config.temperature,
                    maximum=1,
                    minimum=0.1,
                    label="Temperature",
                    step=0.01
                )

                retrival_argumented_generation_threshold = gr.Slider(
                    value=0.51,
                    maximum=1,
                    minimum=0.1,
                    label="Retrieval Augmented Generation Threshold",
                    step=0.01
                )

                top_p = gr.Slider(
                    value=self.sample_config.top_p,
                    maximum=1,
                    minimum=0.1,
                    label="Top P",
                    step=0.01
                )
                top_k = gr.Slider(
                    value=self.sample_config.top_k,
                    maximum=100,
                    minimum=1,
                    label="Top K",
                    step=1
                )
                mode = gr.Dropdown(
                    choices=CHAT_MODE,
                    value=self.sample_config.mode,
                    label="Mode",
                    multiselect=False
                )
                retrieval_augmented_generation_information = gr.TextArea(
                    placeholder="Retrieval Augmented Generation Information",
                    max_lines=100,
                    label="Retrieval Augmented Generation Information"
                )
                prompt_to_model = gr.TextArea(
                    placeholder="Agent Prompt",
                    max_lines=100,
                    label="ChatAgent Prompt to Model"
                )

        inputs = [
            prompt,
            history,
            system_prompt,
            mode,
            max_sequence_length,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            retrival_argumented_generation_threshold
        ]

        clear.click(fn=lambda: [], outputs=[history])

        sub_event = submit.click(
            fn=self.sample,
            inputs=inputs,
            outputs=[
                prompt,
                history,
                retrieval_augmented_generation_information,
                prompt_to_model
            ]
        )
        re_generate_event = re_generate.click(
            fn=self._re_generate,
            inputs=inputs,
            outputs=[
                prompt,
                history,
                retrieval_augmented_generation_information,
                prompt_to_model
            ]
        )
        txt_event = prompt.submit(
            fn=self.sample,
            inputs=inputs,
            outputs=[
                prompt,
                history,
                retrieval_augmented_generation_information,
                prompt_to_model
            ]
        )

        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[
                txt_event,
                sub_event,
                re_generate_event
            ]
        )

    def _re_generate(
            self,
            prompt,
            history,
            system_prompt,
            mode,
            max_sequence_length,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            retrival_argumented_generation_threshold
    ):

        if history is None or len(history) == 0:
            gr.Warning("There's no history for this chat to re-generate response.")
        else:
            prompt = history[-1][0]
            history = history[0:-1]

            for holder, history, information, prompt_to_model in self.sample(
                    prompt=prompt,
                    history=history,
                    system_prompt=system_prompt,
                    mode=mode,
                    max_sequence_length=max_sequence_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    retrival_argumented_generation_threshold=retrival_argumented_generation_threshold
            ):
                yield holder, history, information, prompt_to_model

    def build_chat_interface(self) -> gr.Blocks:
        """
        The build function is the main entry point for your app.
        It should return a single gr.Block instance that will be displayed in the browser.

        :param self: Make the class methods work with an instance of the class
        :return: A block, which is then queued
        """
        with gr.Blocks(
                theme=gr.themes.Soft(
                    primary_hue=gr.themes.colors.orange,
                    secondary_hue=gr.themes.colors.orange,
                ),
                title="Chat",
                css="footer {visibility: hidden}"
        ) as block:
            self.chat_interface_components()
        block.queue()
        return block

    def build_inference(self) -> gr.Blocks:
        """
        The function "build_inference" returns a gr.Blocks object that contains two tabs, one for model
        interface components and one for chat interface components.
        :return: a gr.Blocks object.
        """
        with gr.Blocks(
                theme=gr.themes.Soft(
                    primary_hue=gr.themes.colors.orange,
                    secondary_hue=gr.themes.colors.orange,
                ),
                css="footer {visibility: hidden}",
                title="AgentX inference",
        ) as block:
            # with gr.Tab("Chat"):
            self.chat_interface_components()
        return block

    def add_retrieval_augmented_generation(
            self,
            index: "faiss.IndexFlatL2",  # type:ignore
            embedding: "SentenceTransformer",  # type:ignore
            snippets: list[str],
            retrieval_augmented_generation_top_k: int,
            retrival_argumented_generation_threshold: Optional[float] = None
    ):
        self.index = index
        self.embedding = embedding
        self.snippets = snippets
        self.retrieval_augmented_generation_top_k = retrieval_augmented_generation_top_k
        self.retrival_argumented_generation_threshold = retrival_argumented_generation_threshold

    @staticmethod
    def search(
            query: str,
            index: "faiss.IndexFlatL2",  # type:ignore
            embedding: "SentenceTransformer",  # type:ignore
            snippets: list,
            k: int,
    ):
        score, index = index.search(
            embedding.encode([query]),
            k=k
        )
        index = index[0].tolist()
        score = score[0].tolist()
        return [(snippets[idx], score[i]) for i, idx in enumerate(index)]

    def retrieval_augmented_generation_search(
            self,
            query: str,
            k: Optional[int] = None,
            base_question: Optional[str] = None,
            retrival_argumented_generation_threshold: Optional[float] = None,
            verbose: bool = False,
            **kwargs
    ):
        index: "faiss.IndexFlatL2" | None = self.index  # type:ignore
        embedding: "SentenceTransformer" | None = self.embedding  # type:ignore
        snippets: list[str] | None = self.snippets
        information = "Retrival Augmented Generation Search Information"
        if index is not None and embedding is not None and embedding is not None:
            contexts_and_scores = self.search(
                query=query,
                embedding=embedding,
                snippets=snippets,
                k=k or self.retrieval_augmented_generation_top_k,
                index=index
            )
            retrival_argumented_generation_threshold = (
                    retrival_argumented_generation_threshold or self.retrival_argumented_generation_threshold
            )

            contexts_and_scores = contexts_and_scores if retrival_argumented_generation_threshold is None else (
                [[snippet, score] for snippet, score in contexts_and_scores if
                 score > retrival_argumented_generation_threshold]
            )

            if not len(contexts_and_scores) > 0:
                return "", information

            contexts = [
                snippet["content"] for snippet, score in contexts_and_scores  # type:ignore
            ]

            for snippet, score in contexts_and_scores:
                for key, value in snippet.items():
                    information += f"\n{key} => {value}"
                information += f"\n~~~~~~ Confidence Score {score} ~~~~~~\n"
            if len(contexts) > 0:
                return "\n\n".join(context for context in contexts), information
        else:
            warnings.warn("Retrival Argumented Generation is disabled")
        return "", information

    @classmethod
    def from_torch_pretrained(
            cls,
            huggingface_repo_id: str,
            *,
            sample_config: Optional[EngineGenerationConfig] = None,
            prompter: Optional[PromptTemplates] = None,
            tokenizer_huggingface_repo_id: str | None = None,
            bnb_4bit_compute_dtype=torch.float16,
            device_map: str = "auto",
            _attn_implementation: str = "sdpa",
            bnb_4bit_quant_type: str = "fp4",
            use_agent: bool = False,
            trust_remote_code: bool = False
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            huggingface_repo_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type
            ),
            torch_dtype=torch.float16,
            device_map=device_map,
            _attn_implementation=_attn_implementation,
            trust_remote_code=trust_remote_code
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_huggingface_repo_id or huggingface_repo_id
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        max_position_embeddings = getattr(
            model.config, "max_position_embeddings", 4096)
        if sample_config is None:
            sample_config = EngineGenerationConfig(
                do_sample=True,
                top_k=30,
                top_p=1,
                temperature=0.2,
                mode="Chat",
                max_new_tokens=max_position_embeddings // 2,
                max_sequence_length=max_position_embeddings,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        return cls(
            model=model,
            tokenizer=tokenizer,
            sample_config=sample_config,
            prompt_template=prompter,
            backend="torch",
            use_agent=use_agent
        )

    @classmethod
    def gguf_from_hub(
            cls,
            huggingface_repo_id: str,
            filename: str,
            sample_config: Optional[EngineGenerationConfig],
            prompter: Optional[PromptTemplates] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            llama_cpp_param_kwargs: dict = None,
            use_agent: bool = False,
    ):
        if llama_cpp_param_kwargs is None:
            llama_cpp_param_kwargs = {}
        from huggingface_hub import hf_hub_download
        model = InferenceSession.create(
            LlamaCPParams(
                model_path=hf_hub_download(huggingface_repo_id, filename),
                num_context=sample_config.max_sequence_length,
                **llama_cpp_param_kwargs
            ),
            generation_config=LlamaCPPGenerationConfig(
                max_new_tokens=sample_config.max_new_tokens,
                temperature=sample_config.temperature,
                top_k=sample_config.top_k,
                top_p=sample_config.top_p,
            )
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            sample_config=sample_config,
            prompt_template=prompter,
            backend="gguf",
            use_agent=use_agent
        )

    @classmethod
    def from_ollama_model(
            cls,
            ollama_model: str,
            sample_config: Optional[EngineGenerationConfig],
            prompter: Optional[PromptTemplates] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            use_agent: bool = False,
    ):

        return cls(
            model=ollama_model,
            tokenizer=tokenizer,
            sample_config=sample_config,
            prompt_template=prompter,
            backend="ollama",
            use_agent=use_agent
        )

    def __repr__(self):
        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_src = f"\t{k} : " + \
                               v.__str__().replace("\n", "\n\t") + "\n"
                    string += repr_src if len(
                        repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                except TypeError:
                    ...
        return string + ")"

    def __str__(self):
        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
