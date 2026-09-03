---
name: docx
description: Create, read, edit, template, and review Word .docx files using python-docx and LibreOffice.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [word, docx, documents, office, templates, productivity]
source: https://github.com/NousResearch/hermes-agent/tree/main/skills/productivity/docx
---

# Docx

Create, read, edit, and template Microsoft Word `.docx` files with
`python-docx`. Handles text, styles, lists, tables, images,
headers/footers, and `{{token}}` templating. Does not render documents
(PDF conversion needs LibreOffice — see below) or edit legacy `.doc`.

## When to Use

- The user asks to generate a Word document (report, letter, contract).
- You need the text, outline, styles, or embedded images of a `.docx`.
- You must change an existing `.docx`: replace text, edit table cells,
  insert/delete paragraphs, apply styles.
- You have a `.docx` template with `{{placeholders}}` to fill from data.
- Not for: `.doc` (legacy), `.odt`, or pixel-perfect layout work.

## Prerequisites

```bash
# Install with uv (recommended — already available in Xerxes)
uv pip install python-docx

# Or with pip
pip install python-docx
```

Import name is `docx`; `lxml` comes with it. Images must exist locally
(PNG/JPEG). Optional: LibreOffice (`soffice`) for PDF conversion.

## Reading a Document

```python
from docx import Document

doc = Document("input.docx")
for p in doc.paragraphs:
    print(p.style.name, "|", p.text)
for table in doc.tables:
    for row in table.rows:
        print([cell.text for cell in row.cells])
for section in doc.sections:  # headers/footers
    print(section.header.paragraphs[0].text if section.header.paragraphs else "")
```

## Creating a Document

```python
from docx import Document
from docx.shared import Mm

doc = Document()
doc.add_heading("Q3 Report", level=1)
doc.add_paragraph("Intro paragraph with ")
run = doc.paragraphs[-1].add_run("bold text")
run.bold = True
doc.add_paragraph("First item", style="List Bullet")
table = doc.add_table(rows=2, cols=3)
table.style = "Table Grid"
table.cell(0, 0).text = "Header"
doc.add_picture("chart.png", width=Mm(120))
doc.save("out.docx")
```

## Editing an Existing Document

```python
# Find/replace that preserves the first run's formatting
from docx import Document

doc = Document("in.docx")
for p in doc.paragraphs:
    if "old" in p.text:
        for run in p.runs:
            run.text = run.text.replace("old", "new")
doc.save("out.docx")
```

- Table cells: `table.cell(r, c).text = "X"` (resets run formatting in
  that cell).
- Insert/delete paragraphs: build on `doc.element.body` via
  `paragraph._p` operations, or rebuild the document when structure
  changes are extensive.

## Template Fill

Put `{{name}}`-style tokens in the document, then replace them the same
way as find/replace, across paragraphs and table cells. Fail loudly and
list tokens that remain unfilled instead of silently shipping a
half-filled template.

## Converting to PDF

When LibreOffice is installed, convert headlessly:

```bash
soffice --headless --convert-to pdf --outdir outdir/ file.docx
```

Check availability first (`command -v soffice || command -v
libreoffice`). If neither exists, tell the user PDF conversion is
unavailable in this environment rather than improvising — python-docx
cannot render PDFs.

## Pitfalls

- **Tokens split across runs.** Word fragments text into several runs;
  a naive per-run replace can miss a token spanning runs. If a replace
  did not match, join paragraph text first, replace there, and rewrite
  the paragraph as a single run.
- **Style names must exist.** Applying an undefined style raises
  `KeyError`. Built-ins like `Heading 1`, `List Bullet`, `List Number`,
  `Table Grid` exist in the default template.
- **Field codes are computed by Word.** TOC/page-number fields show
  placeholder text until opened in Word/LibreOffice.
- **Never unzip-and-sed the XML.** Raw text substitution in
  `document.xml` corrupts files easily. Edit through python-docx.
- **Tracked changes and comments** live in dedicated XML parts that
  python-docx does not model. Read them by unzipping and inspecting, or
  ask the user to resolve them in Word; do not silently accept/reject.

## Verification

- After create/edit, re-read the output and confirm expected strings
  appear and old strings are gone.
- Open the file size check: a save that silently failed usually leaves
  the original untouched — compare timestamps/sizes before claiming an
  edit landed.
- For layout-sensitive output, convert to PDF with LibreOffice and
  inspect the result.

---

Adapted from the `docx` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Nous Research.
