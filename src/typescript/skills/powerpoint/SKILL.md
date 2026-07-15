---
name: powerpoint
description: Safely edit and repack caller-provided OOXML presentation and document directories with Xerxes' Bun-native Office commands.
version: 1.0.0
author: Xerxes Agents
license: Apache-2.0
tags: [powerpoint, pptx, ooxml, office, bun]
required_tools: [exec_command]
platforms: [macos, linux, windows]
---

# PowerPoint and OOXML

Use this skill when a user asks Xerxes to make a structural change to a PowerPoint presentation or another Office Open XML package. The executable behavior lives in Xerxes' native TypeScript runtime; this bundle supplies only the operating workflow.

## Preconditions

- Work on a caller-provided unpacked OOXML directory. Do not guess a source package or search for private documents.
- Preserve the original `.pptx`, `.docx`, or `.xlsx` file outside the working directory.
- Inspect the requested parts before changing them and keep all paths inside the user's workspace.
- Run `xerxes skill powerpoint --help` if the installed runtime and this document disagree. The CLI help is authoritative.

## Native commands

```text
xerxes skill powerpoint add-slide <unpacked-dir> <slideN.xml|slideLayoutN.xml>
xerxes skill powerpoint clean <unpacked-dir>
xerxes skill powerpoint pack <unpacked-dir> <output.docx|output.pptx|output.xlsx> [--original <file>] [--validate true|false]
xerxes skill powerpoint merge-runs <unpacked-docx-dir>
xerxes skill powerpoint simplify-redlines <unpacked-docx-dir>
```

The commands use Bun and Xerxes' native OOXML implementation. Do not add a Python, LibreOffice, or package-install fallback when a native command reports an unsupported structure.

## Workflow

1. Confirm the input directory and the requested output path.
2. Read the relevant OOXML parts and identify the smallest structural change.
3. Run one mutation command at a time.
4. Review the command output and inspect the changed XML before continuing.
5. Repack to a new output file. Pass `--original <file>` when an original package is available so the runtime can perform its supported baseline checks.
6. Report the output path and the validation Xerxes actually performed.

## Command boundaries

- `add-slide` creates and registers a slide from an existing slide or layout part. Follow the returned instruction if the presentation manifest needs a new entry.
- `clean` removes OOXML parts that the native relationship scan finds unreferenced. Use it only on a disposable working copy.
- `merge-runs` and `simplify-redlines` operate on unpacked WordprocessingML directories, despite this skill's PowerPoint-facing name.
- `pack` writes a deterministic Office ZIP and performs structural checks. It does not render slides, execute macros, validate every Office schema, or promise pixel-identical output.

## Safety and verification

- Never overwrite the only copy of a user's document.
- Treat macros, external relationships, encrypted packages, ZIP64 archives, and unsupported OOXML extensions as explicit unsupported cases.
- Do not claim visual correctness from structural validation alone. Ask the user to open the produced document in a compatible Office viewer when appearance matters.
- Surface malformed XML, missing relationships, ambiguous tracked-change authors, and unsupported archive features as actionable errors.
