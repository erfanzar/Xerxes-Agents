---
name: ocr-and-documents
description: Extract local PDF text, tables, images, metadata, and OCR/layout output through caller-selected Bun-native conversion ports.
version: 0.3.0
tags: [pdf, documents, ocr, text-extraction]
source: bundled
subcommands: [pymupdf, marker]
---

# OCR & Documents

The historical `pymupdf` and `marker` command names are now Bun-native CLI routes. They do not run PyMuPDF, marker-pdf, Python, or a subprocess. Instead, the caller explicitly selects an adapter that supplies the native PDF/OCR engine.

## Explicit adapter

Choose a local adapter module for every operational command:

```bash
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --markdown
```

The module must export `default` or `ocrDocumentsCliAdapter`. It can provide:

- `pdfConverter`: a Bun/JavaScript/WASM `PdfTextConversionPort` for text, markdown, tables, images, and metadata.
- `markerConverter`: a Bun/JavaScript/WASM/host-service `OcrConversionPort` for high-fidelity OCR and layout extraction.
- `diskSpace`: an `OcrDocumentDiskSpacePort` for `marker --check`.
- `filesystem`: optional image-output filesystem port. Without it, output uses the bundled Bun filesystem adapter.

The adapter is executable code; select only a module you control. Xerxes does not infer a conversion engine from installed packages or environment state.

## PDF extraction

`pymupdf` is retained as the compatibility command name, but it runs the caller-provided native `pdfConverter`.

```bash
# Plain text, matching the historical page headings
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf

# Markdown, tables, embedded images, metadata, and zero-based page selection
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --markdown --pages 0-4
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --tables
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --images out/
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --metadata
```

`--images` with no directory writes under `./images/`.

## OCR and layout extraction

`marker` is retained as the compatibility command name, but it runs the caller-provided native `markerConverter`.

```bash
# The adapter controls how disk capacity is measured
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts marker --check

# Markdown output (default), structured JSON, images, and optional LLM post-processing
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts marker scanned.pdf
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts marker scanned.pdf --json
xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts marker scanned.pdf --output-dir out/ --use-llm
```

The legacy `--output_dir` and `--use_llm` spellings remain accepted for migration compatibility. OCR output is determined entirely by the injected native converter; no hidden model download or Python fallback occurs.
