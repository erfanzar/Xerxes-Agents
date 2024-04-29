import click
import transformers

from .utils import Backend
from .engine import EngineGenerationConfig, ServeEngine
from .prompt_templates import PromptTemplates
from .engine_websocket import start_server


@click.group()
def cli():
    ...


@cli.command()
@click.option("--top-k", type=int, default=30, help="The number of most likely tokens to keep for top-k sampling.")
@click.option("--top-p", type=float, default=0.95,
              help="The cumulative probability of token candidates that must be exceeded for nucleus sampling."
              )
@click.option("--max-new-tokens", type=int, default=2048, help="The maximum number of new tokens to generate.")
@click.option("--max-compile-tokens", type=int, default=1, help="The maximum number of tokens to compile.")
@click.option("--max-sequence-length", type=int, default=8192, help="The maximum length of the sequence.")
@click.option("--temperature", type=float, default=0.8, help="The value used to module the next token probabilities.")
@click.option("--eos-token-id", type=int, default=None, help="The end-of-sequence token ID.")
@click.option("--pad-token-id", type=int, default=None, help="The padding token ID.")
@click.option("--mode", type=click.Choice(["Chat", "Instruction"]), default="Chat", help="The mode of operation.")
@click.option("--do-sample", is_flag=True, default=True, help="Whether to use sampling instead of greedy decoding.")
@click.option("--backend", type=click.Choice([backend.value for backend in Backend]), default=Backend.ollama.value,
              help="The backend to use for generation.")
@click.option("--port", type=int, default=11555, help="The port for websocket connections.")
@click.option("--huggingface-repo-id", type=str, default=None, help="The Hugging Face repository ID.")
@click.option("--ollama-model", type=str, default=None, help="The OLLaMA model to use.")
@click.option("--filename", type=str, default=None, help="The filename to use.")
@click.option("--prompter", type=str, default=None, help="The prompter to use.")
@click.option("--tokenizer-huggingface-repo-id", type=str, default=None,
              help="The Hugging Face repository ID for the tokenizer.")
@click.option("--bnb-4bit-compute-dtype", type=str, default=None, help="The BNB 4-bit compute data type.")
@click.option("--device-map", type=str, default="auto", help="The device map to use.")
@click.option("--attn-implementation", type=str, default="sdpa", help="The attention implementation to use.")
@click.option("--bnb-4bit-quant-type", type=str, default="fp4", help="The BNB 4-bit quantization type.")
@click.option("--use-agent", is_flag=True, default=False, help="Whether to use an agent.")
@click.option("--trust-remote-code", is_flag=True, default=False, help="Whether to trust remote code.")
def serve(
        top_k,
        top_p,
        max_new_tokens,
        max_compile_tokens,
        max_sequence_length,
        temperature,
        eos_token_id,
        pad_token_id,
        mode,
        do_sample,
        backend,
        port,
        huggingface_repo_id,
        ollama_model,
        filename,
        prompter,
        tokenizer_huggingface_repo_id,
        bnb_4bit_compute_dtype,
        device_map,
        attn_implementation,
        bnb_4bit_quant_type,
        use_agent,
        trust_remote_code
):
    generation_config = EngineGenerationConfig(
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_compile_tokens=max_compile_tokens,
        max_sequence_length=max_sequence_length,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        mode=mode,
        do_sample=do_sample
    )
    tokenizer = None
    if tokenizer_huggingface_repo_id is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_huggingface_repo_id)
    elif huggingface_repo_id is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(huggingface_repo_id)
    if prompter is not None:
        prompter = PromptTemplates.from_prompt_templates(
            prompt_template=prompter
        )
    if backend == Backend.torch.value:
        assert huggingface_repo_id is not None, "`huggingface_repo_id` is required for torch backend"
        engine = ServeEngine.from_torch_pretrained(
            huggingface_repo_id=huggingface_repo_id,
            sample_config=generation_config,
            tokenizer_huggingface_repo_id=tokenizer_huggingface_repo_id,
            use_agent=use_agent,
            _attn_implementation=attn_implementation,
            device_map=device_map,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            trust_remote_code=trust_remote_code,
            prompter=prompter
        )
    elif backend == Backend.ollama.value:
        assert ollama_model is not None, "`ollama_model` is required for ollama backend"
        engine = ServeEngine.from_ollama_model(
            use_agent=use_agent,
            sample_config=generation_config,
            ollama_model=ollama_model,
            tokenizer=tokenizer,
            prompter=prompter
        )
    elif backend == Backend.gguf.value:
        assert huggingface_repo_id is not None, "`huggingface_repo_id` is required for gguf backend"
        assert filename is not None, "`filename` is required for gguf backend"
        engine = ServeEngine.gguf_from_hub(
            use_agent=use_agent,
            tokenizer=tokenizer,
            sample_config=generation_config,
            filename=filename,
            huggingface_repo_id=huggingface_repo_id,
            prompter=prompter
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    start_server(engine=engine, port=port)


if __name__ == "__main__":
    cli()
