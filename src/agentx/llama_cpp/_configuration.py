import warnings
from typing import List, Optional, Union, Dict

from llama_cpp import (
    Llama as _Llama,
    llama_grammar_free,
    llama_free,
    llama_free_model,
    llama_batch_free,
    StoppingCriteriaList,
    LogitsProcessorList,
    LlamaGrammar,
    llama_chat_format,
    llama_cpp
)

from llama_cpp.llama_cpp import _lib as llama_cpp_lib

from ._utils import BaseCPPClass


class LlamaCPP(_Llama):
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


class LlamaCPPGenerationConfig(BaseCPPClass):
    def __init__(
            self,
            suffix: Optional[str] = None,
            max_new_tokens: Optional[int] = 2048,
            temperature: float = 0.8,
            top_p: float = 0.95,
            min_p: float = 0.05,
            typical_p: float = 1.0,
            logprobs: Optional[int] = None,
            echo: bool = False,
            stop: Optional[Union[str, List[str]]] = None,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            repeat_penalty: float = 1.1,
            top_k: int = 40,
            stream: bool = True,
            seed: Optional[int] = None,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            grammar: Optional[LlamaGrammar] = None,
            logit_bias: Optional[Dict[str, float]] = None,
    ):

        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.


        :param self: Bind the attributes with an object
        :param suffix: Optional[str]: Add a suffix to the end of the generated text
        :param max_new_tokens: Optional[int]: Limit the number of tokens that can be generated
        :param temperature: float: Control the randomness of the output
        :param top_p: float: Control the diversity of the generated text
        :param min_p: float: Set the minimum probability of a token to be generated
        :param typical_p: float: Set the typical probability of a token
        :param logprobs: Optional[int]: Set the number of log probabilities to be returned
        :param echo: bool: Determine whether to echo the input text
        :param stop: Optional[Union[str: Stop the generation process
        :param List[str]]]: Store the list of strings that are used to stop the generation process
        :param frequency_penalty: float: Penalize the frequency of a token in the training corpus
        :param presence_penalty: float: Penalize the presence of a token in the input
        :param repeat_penalty: float: Penalize the model for repeating words
        :param top_k: int: Determine the number of tokens to consider when generating the next token
        :param stream: bool: Determine whether the model should generate text in a streaming fashion
        :param seed: Optional[int]: Set the random seed for the generation process
        :param tfs_z: float: Control the z-score threshold for tfs
        :param mirostat_mode: int: Determine the mode of mirostat
        :param mirostat_tau: float: Set the temperature of the model
        :param mirostat_eta: float: Control the rate of decay in the mirostat algorithm
        :param stopping_criteria: Optional[StoppingCriteriaList]: Determine when to stop generating text
        :param logits_processor: Optional[LogitsProcessorList]: Specify a list of logits processors
        :param grammar: Optional[LlamaGrammar]: Pass in a grammar object
        :param logit_bias: Optional[Dict[str, float]] : Bias the model towards certain tokens
        :return: An instance of the class
        """
        if max_new_tokens is None:
            max_new_tokens = -1
            warnings.warn(
                "`max_new_tokens` will be set to -1 for infinity generation"
            )
        if stop is None:
            stop = []

        if seed is None:
            seed = -1  # Random seed in llama.cpp
        self.grammar = grammar
        self.stream = stream
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.typical_p = typical_p
        self.logprobs = logprobs
        self.echo = echo
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.logits_processor = logits_processor
        self.seed = seed
        self.tfs_z = tfs_z
        self.mirostat_mode = mirostat_mode
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.suffix = suffix
        self.stopping_criteria = stopping_criteria
        self.logit_bias = logit_bias

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


class LlamaCPParams(BaseCPPClass):
    def __init__(
            self,
            model_path: str,
            *,
            num_gpu_layers: int = 0,
            main_gpu: int = 0,
            tensor_split: Optional[List[float]] = None,
            vocab_only: bool = False,
            use_mmap: bool = True,
            use_mlock: bool = False,
            kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None,
            seed: int = 42,
            num_context: int = 2048,
            num_batch: int = 512,
            num_threads: Optional[int] = None,
            num_threads_batch: Optional[int] = None,
            rope_freq_base: float = 0.0,
            rope_freq_scale: float = 0.0,
            yarn_ext_factor: float = -1.0,
            yarn_attn_factor: float = 1.0,
            yarn_beta_fast: float = 32.0,
            yarn_beta_slow: float = 1.0,
            yarn_orig_ctx: int = 0,
            mul_mat_q: bool = True,
            logits_all: bool = False,
            embedding: bool = False,
            offload_kqv: bool = False,
            last_num_tokens_size: int = 64,
            lora_base: Optional[str] = None,
            lora_scale: float = 1.0,
            lora_path: Optional[str] = None,
            numa: bool = False,
            chat_format: str = "llama-2",
            chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
            verbose: bool = True,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the object with all of its attributes and other components.

        :param self: Refer to the instance of the class
        :param model_path: str: Specify the path to the model
        :param *: Pass a variable number of arguments to a function
        :param num_gpu_layers: int: Specify the number of layers that will be run on a gpu
        :param main_gpu: int: Specify the gpu to use for inference
        :param tensor_split: Optional[List[float]]: Specify how the tensors are split between gpus
        :param vocab_only: bool: Load the vocabulary only
        :param use_mmap: bool: Determine whether to use mmap or not
        :param use_mlock: bool: Lock the memory of the model in ram
        :param kv_overrides: Optional[Dict[str, Union[bool, int, float]]]: Override the default values of the parameters
         in the model
        :param seed: int: Set the random seed for the model
        :param num_context: int: Set the number of context tokens to use
        :param num_batch: int: Set the number of batches to be processed at once
        :param num_threads: Optional[int]: Set the number of threads used for inference
        :param num_threads_batch: Optional[int]: Specify the number of threads to use for batching
        :param rope_freq_base: float: Set the rope_freq_base parameter in llama
        :param rope_freq_scale: float: Scale the frequency of words in the rope
        :param yarn_ext_factor: float: Control the amount of context that is used for each token
        :param yarn_attn_factor: float: Control the amount of attention that is applied to the context
        :param yarn_beta_fast: float: Set the beta parameter for yarn
        :param yarn_beta_slow: float: Control the speed of the model
        :param yarn_orig_ctx: int: Determine whether the context is used in yarn
        :param mul_mat_q: bool: Determine whether the query matrix should be multiplied with the key and value matrices
        :param logits_all: bool: Determine whether to return all the logits or only the top one
        :param embedding: bool: Determine whether the model is used for embedding or not
        :param offload_kqv: bool: Determine whether to offload the kv embedding matrix from cpu memory to gpu memory
        :param last_num_tokens_size: int: Set the size of the last_num_tokens array
        :param lora_base: Optional[str]: Specify the base of the logarithm used in lora
        :param lora_scale: float: Scale the lora score
        :param lora_path: Optional[str]: Specify the path to a file containing
        :param numa: bool: Determine whether the numa node should be used
        :param chat_format: str: Specify the format of the input text
        :param chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler]: Define a custom chat
        completion handler
        :param verbose: bool: Print out the progress of loading the model
        """
        self.model_path = model_path
        self.num_gpu_layers = num_gpu_layers
        self.main_gpu = main_gpu
        self.tensor_split = tensor_split
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.kv_overrides = kv_overrides
        self.seed = seed
        self.num_context = num_context
        self.num_batch = num_batch
        self.num_threads = num_threads
        self.num_threads_batch = num_threads_batch
        self.rope_freq_base = rope_freq_base
        self.rope_freq_scale = rope_freq_scale
        self.yarn_ext_factor = yarn_ext_factor
        self.yarn_attn_factor = yarn_attn_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_orig_ctx = yarn_orig_ctx
        self.mul_mat_q = mul_mat_q
        self.logits_all = logits_all
        self.embedding = embedding
        self.offload_kqv = offload_kqv
        self.last_num_tokens_size = last_num_tokens_size
        self.lora_base = lora_base
        self.lora_scale = lora_scale
        self.lora_path = lora_path
        self.numa = numa
        self.chat_format = chat_format
        self.chat_handler = chat_handler
        self.verbose = verbose

    def init_model(self) -> LlamaCPP:
        """
        The init_model function is used to initialize the model.

        :param self: Refer to the instance of a class
        :return: A llamacpp object
        """
        return LlamaCPP(
            model_path=self.model_path,
            verbose=self.verbose,
            mul_mat_q=self.mul_mat_q,
            n_gpu_layers=self.num_gpu_layers,
            yarn_beta_slow=self.yarn_beta_slow,
            lora_scale=self.lora_scale,
            yarn_orig_ctx=self.yarn_orig_ctx,
            yarn_ext_factor=self.yarn_ext_factor,
            n_threads_batch=self.num_threads_batch,
            last_n_tokens_size=self.last_num_tokens_size,
            yarn_attn_factor=self.yarn_attn_factor,
            use_mmap=self.use_mmap,
            lora_base=self.lora_base,
            vocab_only=self.vocab_only,
            logits_all=self.logits_all,
            yarn_beta_fast=self.yarn_beta_fast,
            rope_freq_base=self.rope_freq_base,
            rope_freq_scale=self.rope_freq_scale,
            n_batch=self.num_batch,
            main_gpu=self.main_gpu,
            use_mlock=self.use_mlock,
            embedding=self.embedding,
            lora_path=self.lora_path,
            n_threads=self.num_threads,
            n_ctx=self.num_context,
            chat_format=self.chat_format,
            offload_kqv=self.offload_kqv,
            chat_handler=self.chat_handler,
            kv_overrides=self.kv_overrides,
            tensor_split=self.tensor_split,
            seed=self.seed,
            numa=self.numa
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
