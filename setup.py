from setuptools import setup, find_packages
from cLLM import __version__

setup(
    name="cLLM-python",
    version=__version__,
    author="Erfan Zare Chavoshi",
    author_email="erfanzare82@eyahoo.com",
    description=(
        "cLLM is an Open-source library that use llama-cpp-python and llama.cpp and "
        "provide a Low and High level API and allow developer to be more pythonic."
    ),
    url="https://github.com/erfanzar/cLLM",
    packages=["cLLM"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "llama-cpp-python",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    # package_dir={"": ""}
)
