# cLLM

cLLM is an Open-source library that use [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
and [llama.cpp](https://github.com/ggerganov/llama.cpp) and provide a Low and High level API and allow developer to be
more pythonic.

## Features 🔮

- **C++ Llama.cpp GGML Framework**: The program is built using the C++ language and utilizes the Llama.cpp framework for
  efficient performance.

- **EasyDeL Platform**: if you use the provided open-source models The models have been trained using the EasyDeL
  platform,
  ensuring high-quality and accurate
  assistance.

- **Customized Models**: Users can access models customized for their specific needs, such as coding assistance, grammar
  correction, and more.

- **OpenAI API**: The structure of APIs that will be provided in upcoming version will be OpenAI API like.

## Installation with Specific Hardware Acceleration (BLAS, CUDA, Metal, etc.)

> [!TIP]
> The default behavior for `llama.cpp` installation is to build for CPU only on Linux and Windows and to use
> Metal on
> macOS. However, `llama.cpp` supports various hardware acceleration backends such as OpenBLAS, cuBLAS, CLBlast,
> HIPBLAS,
> and Metal.

To install with a specific hardware acceleration backend, you can set the `CMAKE_ARGS` environment variable before
installing. Here are the instructions for different backends:

**Buildings for OpenBLAS**

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install cLLM-python
```

**Buildings for cuBLAS**

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install cLLM-python
```

**Buildings for Metal**

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install cLLM-python
```

**Buildings for CLBlast**

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install cLLM-python
```

**Buildings for hipBLAS**

```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install cLLM-python
```

You can set the `CMAKE_ARGS` environment variable accordingly based on your specific hardware acceleration requirements
before installing `llama.cpp`.

## Contributing

If you would like to contribute to cLLM, please follow the guidelines outlined in the CONTRIBUTING.md file in the
repository.

## License

cLLM is licensed under the [MIT](https://github.com/erfanzar/cLLM/blob/main/LICENSE). See the LICENSE.md file
for more details.

## Support

For any questions or issues, please get in touch with me at [erfanzare810@gmail.com](erfanzare810@gmail.com).

Thank you for using cLLM! We hope it will help you have a personal computer experience.
