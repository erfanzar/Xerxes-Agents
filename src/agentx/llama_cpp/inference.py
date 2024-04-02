import warnings

from ._configuration import LlamaCPP, LlamaCPPGenerationConfig, LlamaCPParams
from dataclasses import dataclass
from typing import Optional, Literal, List, Iterable, Sequence


@dataclass
class _InferencePredictions:
    text: str
    index: int
    logprobs: Optional[float]
    finish_reason: Optional[bool]


@dataclass
class InferenceGenerationOutput:
    id: str
    object: str
    created: str
    model: str
    predictions: _InferencePredictions


class InferenceSession:
    def __init__(
            self,
            model: LlamaCPP,
            generation_config: Optional[LlamaCPPGenerationConfig] = None,
    ):

        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.

        :param self: Represent the instance of the class
        :param model: LlamaCPP: Pass the model to the class
        :param generation_config: Optional[LlamaCPPGenerationConfig]: Set the generation config to none
        :return: The model and the generation config
        """
        if generation_config is None:
            warnings.warn(
                "Passing `LlamaCPPGenerationConfig` as None will "
                "initialize `LlamaCPPGenerationConfig` with default values"
            )
            generation_config = LlamaCPPGenerationConfig()
        self.model = model
        self.generation_config = generation_config

    def generate(
            self,
            prompt: Optional[str] = None,
            input_ids: Optional[Sequence[int]] = None,
            suffix: Optional[str] = None,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            typical_p: Optional[float] = None,
            logprobs: Optional[int] = None,
            echo: Optional[bool] = None,
            stop: Optional[List[str]] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            repeat_penalty: Optional[float] = None,
            top_k: Optional[int] = None,
            seed: Optional[int] = None,
            mirostat_mode: Optional[int] = None,
            mirostat_tau: Optional[float] = None,
            mirostat_eta: Optional[float] = None,
            stream: Optional[bool] = None
    ) -> Iterable[InferenceGenerationOutput]:

        """
        The generate function is used to generate text from a prompt.

        :param self: Represent the instance of the class
        :param prompt: Optional[str]: Pass the prompt to the model
        :param input_ids: Optional[Sequence[int]]: Pass the input ids to the model
        :param suffix: Optional[str]: Add a suffix to the generated text
        :param max_new_tokens: Optional[int]: Limit the number of tokens that can be generated
        :param temperature: Optional[float]: Control the randomness of the text generation
        :param top_p: Optional[float]: Set the top_p value for nucleus sampling
        :param min_p: Optional[float]: Set the minimum probability of a token to be generated
        :param typical_p: Optional[float]: Set the typical probability of a token
        :param logprobs: Optional[int]: Return the log probabilities of the generated tokens
        :param echo: Optional[bool]: Determine whether the input should be echoed in the output
        :param stop: Optional[List[str]]: Stop the generation when a specific word is generated
        :param frequency_penalty: Optional[float]: Penalize the generation of tokens that appear frequently in the training data
        :param presence_penalty: Optional[float]: Penalize the presence of tokens in the generated text
        :param repeat_penalty: Optional[float]: Penalize the model for repeating itself
        :param top_k: Optional[int]: Limit the number of tokens that are considered for each step
        :param seed: Optional[int]: Set the seed for the random number generator
        :param mirostat_mode: Optional[int]: Control the generation of text
        :param mirostat_tau: Optional[float]: Control the amount of randomness in the generation process
        :param mirostat_eta: Optional[float]: Control the amount of randomness in the generation
        :param stream: Optional[bool]: Determine whether the generation should be done in a streaming fashion or not
        :return: An iterable of `InferenceGenerationOutput` objects

        """
        stream = stream or self.generation_config.stream

        generation_kwargs = dict(
            seed=seed or self.generation_config.seed,
            max_tokens=max_new_tokens or self.generation_config.max_new_tokens,
            frequency_penalty=frequency_penalty or self.generation_config.frequency_penalty,
            presence_penalty=presence_penalty or self.generation_config.presence_penalty,
            repeat_penalty=repeat_penalty or self.generation_config.repeat_penalty,
            mirostat_mode=mirostat_mode or self.generation_config.mirostat_mode,
            mirostat_tau=mirostat_tau or self.generation_config.mirostat_tau,
            mirostat_eta=mirostat_eta or self.generation_config.mirostat_eta,
            top_k=top_k or self.generation_config.top_k,
            top_p=top_p or self.generation_config.top_p,
            suffix=suffix or self.generation_config.suffix,
            min_p=min_p or self.generation_config.min_p,
            temperature=temperature or self.generation_config.seed,
            echo=echo or self.generation_config.echo,
            stop=stop or self.generation_config.stop or ["THERES_NO_STOP_TOKEN_OR_EOS_TOKEN"],
            typical_p=typical_p or self.generation_config.typical_p,
            logprobs=logprobs or self.generation_config.logprobs
        )
        if input_ids is None and prompt is None:
            raise ValueError(
                "You can not pass `input_ids` and `prompt` both None you should pass at least one of them."
            )
        elif input_ids is not None and prompt is not None:
            raise ValueError(
                "You can only pass `input_ids` or `prompt` only one of them will be used."
            )
        elif input_ids is None and prompt is not None:
            if stream:
                for model_response in self.model(
                        prompt,
                        stream=stream,
                        **generation_kwargs
                ):
                    predictions = _InferencePredictions(
                        **model_response["choices"][0]
                    )
                    predictions.text = predictions.text.replace("<0x0A>", "\n")
                    response = InferenceGenerationOutput(
                        predictions=predictions,
                        created=model_response["created"],
                        model=model_response["model"],
                        object=model_response["object"],
                        id=model_response["id"]
                    )
                    yield response
            else:
                model_response = self.model(
                    prompt,
                    stream=stream,
                    **generation_kwargs
                )

                predictions = _InferencePredictions(
                    **model_response["choices"][0]
                )
                predictions.text = predictions.text.replace("<0x0A>", "\n")
                response = InferenceGenerationOutput(
                    predictions=predictions,
                    created=model_response["created"],
                    model=model_response["model"],
                    object=model_response["object"],
                    id=model_response["id"]
                )
                yield response
        elif input_ids is not None and prompt is None:

            for model_response in self.model.generate(
                    input_ids,
                    **generation_kwargs
            ):
                yield model_response
        else:
            raise ValueError("There's a Problem with your inputs.")

    @staticmethod
    def llama_chat_template(
            message: str,
            chat_history: Optional[List[str] | List[List[str]]] = None,
            system_prompt: str = None
    ):
        """
        The chat_template function takes in a message, chat_history and system prompt.
        It then formats the message into a template that can be used to train the model.
        The function returns a string of text formatted as follows:

        :param message: str: Pass in the user's message to be added to the chat history
        :param chat_history: Optional[List[str] | List[List[str]]]: Pass in a list of strings or a list of lists
        :param system_prompt: str: Set the prompt for the system
        :return: prompt string
        """
        if system_prompt == "":
            system_prompt = None
        if chat_history is None:
            chat_history = []
        do_strip = False
        texts = [
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        ] if system_prompt is not None else [f"<s>[INST] "]
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(
                f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
        message = message.strip() if do_strip else message
        texts.append(f"{message} [/INST]")
        return "".join(texts)

    @staticmethod
    def os_chat_template(
            message: str,
            chat_history: Optional[List[str] | List[List[str]]] = None,
            system_prompt: Optional[str] = None
    ):
        """
        The os_chat_template function takes in a message, chat history, and system prompt.
        It returns a string that is formatted to be used as the input for the OpenSubtitles dataset.
        The format of this string is:

        :param message: str: Pass in the user"s message to the assistant
        :param chat_history: Optional[List[str] | List[List[str]]]: Specify the history of the conversation
        :param system_prompt: Optional[str]: Add a system prompt to the chat history
        :return: prompt string
        """
        if chat_history is None:
            chat_history = []
        system = f"<|system|>\n{system_prompt}</s>\n" if system_prompt is not None else ""
        ua = ""
        for user_input, response in chat_history:
            ua += f"<|user|>\n{user_input}</s>\n<|assistant|>\n{response}</s>\n"
        return system + ua + f"<|user|>\n{message}</s>\n<|assistant|>\n"

    def get_chat_template(self, template_name: Literal["Llama2", "OpenChat"] = "Llama2"):
        if template_name == "Llama2":
            return self.llama_chat_template
        elif template_name == "OpenChat":
            return self.os_chat_template
        else:
            raise ValueError("UnKnown Chat Template requested")

    @classmethod
    def create(
            cls,
            llama_params: LlamaCPParams,
            generation_config: Optional[LlamaCPPGenerationConfig] = None
    ):

        """
        The create function is used to create a new instance of the LlamaCP class.
        It takes in two arguments:
            - llama_params: A LlamaCPParams object that contains all the parameters for
                creating a new model. This includes things like number of layers,
                activation functions, etc. See the documentation for more details on this
                object and its properties.

        :param cls: Create an instance of the class
        :param llama_params: LlamaCPParams: Initialize the model
        :param generation_config: Optional[LlamaCPPGenerationConfig]: Define the generation configuration
        :return: A class instance
        """
        return cls(
            model=llama_params.init_model(),
            generation_config=generation_config
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
                repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                string += repr_src if len(repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
