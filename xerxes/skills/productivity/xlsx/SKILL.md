---
name: xlsx
description: Create, read, edit Excel .xlsx workbooks and convert to/from CSV using openpyxl.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [excel, spreadsheet, xlsx, csv, openpyxl, productivity]
source: https://github.com/NousResearch/hermes-agent/tree/main/skills/productivity/xlsx
---

# Xlsx

Work with Excel `.xlsx` workbooks using Python and `openpyxl`: build
styled multi-sheet workbooks with formulas, inspect or dump existing
files, edit cells and structure, and convert to/from CSV.

## When to Use

- Creating `.xlsx` reports: multiple sheets, number formats, styling,
  merged cells, freeze panes, autofilter, native tables.
- Reading a workbook: sheet inventory, dumping data as JSON or CSV,
  listing formulas vs cached values.
- Editing existing files: set cells, append rows, copy/rename sheets.
- CSV interop with type inference and non-UTF-8 encodings.
- Not for the legacy `.xls` binary format (convert first:
  `soffice --headless --convert-to xlsx old.xls`).

## Prerequisites

```bash
uv pip install openpyxl   # or: pip install openpyxl
```

Optional: LibreOffice (`soffice`) for headless recalculation or
format conversion.

## Reading a Workbook

```python
from openpyxl import load_workbook

wb = load_workbook("report.xlsx")
print(wb.sheetnames)
ws = wb["Data"]
for row in ws.iter_rows(values_only=True):
    print(row)

# Formula strings
for row in ws.iter_rows():
    for cell in row:
        if isinstance(cell.value, str) and cell.value.startswith("="):
            print(cell.coordinate, cell.value)

# Cached results (None unless a real spreadsheet app saved the file last)
wb2 = load_workbook("report.xlsx", data_only=True)
```

## Creating a Workbook

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active
ws.title = "Data"
ws.append(["Region", "Units", "Revenue"])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = PatternFill("solid", fgColor="DDDDDD")
ws.append(["West", 12, 950.0])
ws["C2"].number_format = '"$"#,##0.00'
ws.column_dimensions[get_column_letter(1)].width = 18
ws.freeze_panes = "A2"
wb.save("report.xlsx")
```

Formulas are strings starting with `=`. openpyxl itself NEVER
evaluates them. To force recalculation on open:

```python
wb.calculation.fullCalcOnLoad = True
```

Or recalculate headlessly after saving:

```bash
soffice --headless --convert-to xlsx:"Calc MS Excel 2007 XML" --outdir out/ report.xlsx
```

## Editing an Existing Workbook

```python
from openpyxl import load_workbook

wb = load_workbook("report.xlsx")
ws = wb["Data"]
ws["B2"] = 42
ws.append([1, "x", True])
wb.save("report.xlsx")   # edits in place — copy the file first if needed
```

**Never save a workbook loaded with `data_only=True`** — that silently
discards every formula (cached values replace them).

## CSV Interop

```python
import csv
from openpyxl import Workbook, load_workbook

# CSV -> xlsx with naive type inference
wb = Workbook(); ws = wb.active
with open("data.csv", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        ws.append([_typed(v) for v in row])
wb.save("out.xlsx")

# xlsx -> CSV
ws = load_workbook("report.xlsx")["Data"]
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(ws.iter_rows(values_only=True))
```

European CSVs often use `;` delimiters and decimal commas — pass
`delimiter=";"` and expect values like `"12,5"` to stay strings.

## Pitfalls

- **openpyxl does not calculate.** Formula results exist only via
  `data_only=True` and only if a real spreadsheet app saved the file
  last. Otherwise you get `None`.
- **Insert/delete does not shift references.** Raw openpyxl
  `insert_rows`/`delete_cols` leave formulas, merges, and table ranges
  pointing at stale coordinates. For formula-heavy sheets, either
  rewrite references yourself or do structural edits before adding
  formulas.
- **Loading strips charts/images.** openpyxl does not round-trip
  charts: editing a charted workbook and saving drops the charts.
  Re-add charts after editing, or avoid re-saving charted files.
- **Sheet protection is NOT security.** It signals "don't edit this"
  to well-behaved apps and nothing more; it does not encrypt anything.
- **Dates are datetimes.** Excel stores dates as serial numbers;
  openpyxl returns `datetime`/`date` objects. Emit ISO strings in
  dumps.
- Sheet names are capped at 31 chars and reject `[ ] : * ? / \`.

## Verification

- After creating: re-open and confirm sheet names, dimensions, and
  merged ranges match intent.
- After edits: re-dump the touched range; if formulas were written,
  confirm they are listed and `fullCalcOnLoad` was set.
- For a visual check: `soffice --headless --convert-to pdf out.xlsx`
  and inspect the PDF.

---

Adapted from the `xlsx` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Nous Research.
