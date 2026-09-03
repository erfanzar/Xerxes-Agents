---
name: pdf
description: PDF files — create, merge, split, extract text and tables, fill forms, encrypt with pypdf, pdfplumber, and reportlab.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [pdf, documents, forms, text-extraction, reportlab, pypdf, pdfplumber, productivity]
source: https://github.com/NousResearch/hermes-agent/tree/main/skills/productivity/pdf
---

# PDF

Create PDFs from structured specs, extract text/tables/metadata,
merge/split/rotate/watermark pages, fill AcroForm fields, and
encrypt/decrypt — using `pypdf`, `pdfplumber`, and `reportlab`.

For natural-language edits of an existing PDF use the bundled
`nano-pdf` skill. For scanned/image-only PDFs use the bundled
`ocr-and-documents` skill — do not report empty extracted text as
"no content".

## When to Use

- Generate a report, invoice, or multi-page document as PDF.
- Pull text, tables, metadata, or form-field values out of a PDF.
- Merge, split, rotate, extract page subsets, or watermark PDFs.
- Fill or flatten AcroForm forms; encrypt or decrypt with passwords.
- NOT for pixel-perfect HTML-to-PDF rendering (use a headless browser).

## Prerequisites

```bash
uv pip install pypdf pdfplumber reportlab   # or: pip install ...
```

## Inspect First

```python
from pypdf import PdfReader

r = PdfReader("doc.pdf")
print(len(r.pages), r.metadata)
print(r.is_encrypted)  # if true, decrypt before any other operation
```

If `page.extract_text()` returns empty strings but pages exist, the PDF
is image-only — route to `ocr-and-documents` instead of fabricating
text.

## Create

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table

styles = getSampleStyleSheet()
doc = SimpleDocTemplate("out.pdf", pagesize=A4)
story = [Paragraph("Q3 Report", styles["Title"]),
         Spacer(1, 12),
         Paragraph("Summary of findings.", styles["BodyText"]),
         Table([["Region", "Units"], ["West", "12"]])]
doc.build(story)
```

## Extract Text and Tables

```python
import pdfplumber

with pdfplumber.open("doc.pdf") as pdf:
    for i, page in enumerate(pdf.pages, 1):
        print(f"--- page {i} ---")
        print(page.extract_text() or "")
        for table in page.extract_tables():
            print(table)  # list of row lists
```

Table extraction is heuristic: borderless or merged-cell tables may
need `extract_tables(table_settings={...})` tuning or manual cleanup.

## Merge, Split, Rotate

```python
from pypdf import PdfWriter, PdfReader

# Merge
w = PdfWriter()
for path in ["a.pdf", "b.pdf"]:
    for page in PdfReader(path).pages:
        w.add_page(page)
with open("merged.pdf", "wb") as f:
    w.write(f)

# Split / rotate (rotation must be a multiple of 90)
r = PdfReader("doc.pdf")
w = PdfWriter()
for i in [0, 1, 6]:            # 0-based here; CLIs are usually 1-based
    r.pages[i].rotate(90)
    w.add_page(r.pages[i])
with open("part.pdf", "wb") as f:
    w.write(f)
```

## Fill Forms

```python
from pypdf import PdfReader, PdfWriter

r = PdfReader("form.pdf")
fields = r.get_fields()
print({name: f.get("/V") for name, f in (fields or {}).items()})

w = PdfWriter(clone_from="form.pdf")
w.update_page_form_field_values(w.pages[0], {"FieldName": "value"})
with open("filled.pdf", "wb") as f:
    w.write(f)
```

Checkbox fields accept `true`/`false`; radio/choice values must match
the field's export options. After filling, viewers only render values
if appearance streams exist — flatten if display fidelity matters, and
verify with `get_fields()`, not just visually.

## Encrypt / Decrypt

```python
w = PdfWriter(clone_from="doc.pdf")
w.encrypt(user_password="secret", algorithm="AES-256")
with open("enc.pdf", "wb") as f:
    w.write(f)
```

- **Permission flags don't enforce.** Owner-password bits (no-print,
  no-copy) are polite requests; only the user password actually gates
  content. Never present permission flags as security.
- Encrypted inputs must be decrypted before any other operation.

## Pitfalls

- **Scanned PDFs**: no text layer means route to OCR; do not fabricate.
- **Page indexing**: pypdf APIs are 0-based; most CLIs are 1-based.
  Don't double-convert.
- **Compression**: deflating content streams saves 0–20%; it is not a
  substitute for image downsampling.
- **Metadata scope**: classic DocInfo only; embedded XMP is left
  untouched and may show different values in some viewers.
- **PDF/A is out of scope** for pypdf/reportlab; archival conformance
  needs Ghostscript plus a validator like veraPDF.

## Verification

- After create/merge/split: re-open with `PdfReader`, confirm
  `len(reader.pages)` and rotation.
- After extraction: check output is non-empty and spot-check a known
  string or cell.
- After form fill: re-read fields and compare values exactly,
  including non-ASCII.
- After encrypt: opening without a password fails; after decrypt,
  extracted text matches the original.

---

Adapted from the `pdf` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Nous Research.
