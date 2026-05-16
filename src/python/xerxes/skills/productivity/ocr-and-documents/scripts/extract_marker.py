# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-fidelity PDF to markdown/JSON conversion using the ``marker-pdf`` package."""

import os
import sys


def convert(path, output_dir=None, output_format="markdown", use_llm=False):
    """Convert ``path`` to ``output_format`` using marker-pdf and print the result.

    Args:
        path: PDF file to convert.
        output_dir: Directory where embedded images are written, when present.
        output_format: ``"markdown"`` (default) or ``"json"``.
        use_llm: When True, enable marker's LLM post-processing.
    """
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config_dict = {}
    if use_llm:
        config_dict["use_llm"] = True

    config_parser = ConfigParser(config_dict)
    models = create_model_dict()
    converter = PdfConverter(config=config_parser.generate_config_dict(), artifact_dict=models)
    rendered = converter(path)

    if output_format == "json":
        import json

        print(
            json.dumps(
                {
                    "markdown": rendered.markdown,
                    "metadata": rendered.metadata if hasattr(rendered, "metadata") else {},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(rendered.markdown)

    if output_dir and hasattr(rendered, "images") and rendered.images:
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name, img_data in rendered.images.items():
            img_path = os.path.join(output_dir, name)
            with open(img_path, "wb") as f:
                f.write(img_data)
        print(f"\nSaved {len(rendered.images)} image(s) to {output_dir}/", file=sys.stderr)


def check_requirements():
    """Verify that at least 5 GB of free disk space is available for marker-pdf."""

    import shutil

    free_gb = shutil.disk_usage("/").free / (1024**3)
    if free_gb < 5:
        print(f"⚠️  Only {free_gb:.1f}GB free. marker-pdf needs ~5GB for PyTorch + models.")
        print("Use pymupdf instead (scripts/extract_pymupdf.py) or free up disk space.")
        sys.exit(1)
    print(f"✓ {free_gb:.1f}GB free — sufficient for marker-pdf")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if args[0] == "--check":
        check_requirements()
        sys.exit(0)

    path = args[0]
    output_dir = None
    output_format = "markdown"
    use_llm = False

    if "--output_dir" in args:
        idx = args.index("--output_dir")
        output_dir = args[idx + 1]
    if "--json" in args:
        output_format = "json"
    if "--use_llm" in args:
        use_llm = True

    convert(path, output_dir=output_dir, output_format=output_format, use_llm=use_llm)
