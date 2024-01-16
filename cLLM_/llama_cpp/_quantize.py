import llama_cpp
import termcolor


def quantize_gguf(
        input_filename: str,
        output_filename: str,
        q_type: str
):
    return_code = llama_cpp.llama_model_quantize(
        input_filename.encode("utf-8"),
        output_filename.encode("utf-8"),
        q_type
    )
    if return_code != 0:
        raise SystemError(
            "Couldn't quantize the model"
        )
    return True
