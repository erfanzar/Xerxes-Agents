// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../../core/errors.js'
import { ToolRegistry } from '../../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../../types/toolCalls.js'
import { optionalString, requiredString } from '../inputs.js'
import { WorkspacePathResolver } from '../pathSafety.js'

export const NOTEBOOK_EDIT_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'NotebookEditTool',
    description: 'Replace the source of one existing .ipynb cell, addressed by zero-based index. Overwriting is all '
      + 'it can do: cells cannot be inserted, appended, deleted, or reordered, and an index at or past the end is '
      + 'refused with the valid range in the message. cell_type is only nominally optional — it defaults to "code" '
      + 'and is written unconditionally, so omitting it while editing a markdown cell converts that cell to code '
      + 'and leaves it without the outputs and execution_count keys nbformat expects; always pass the type the cell '
      + 'should keep. Existing outputs are preserved untouched and therefore go stale against the new source. The '
      + 'whole notebook is reserialized with one-space indentation, so even a one-line change produces a whole-file '
      + 'diff. Read the notebook first — ReadFile shows the raw JSON — to get the index right, because there is no '
      + 'dry run and no way to address a cell by its content.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        notebook_path: { type: 'string', description: 'Workspace-relative .ipynb path.' },
        cell_index: { type: 'integer', minimum: 0, description: 'Zero-based cell index.' },
        new_source: { type: 'string', description: 'New complete cell source; may be empty.' },
        cell_type: { type: 'string', enum: ['code', 'markdown'], default: 'code' },
      },
      required: ['notebook_path', 'cell_index', 'new_source'],
    },
  },
}

/** Register the notebook-only part of the old Claude file-tool surface. */
export function registerClaudeNotebookTools(
  registry: ToolRegistry,
  paths: WorkspacePathResolver,
  agentId = 'default',
): readonly ToolDefinition[] {
  registry.replace(NOTEBOOK_EDIT_TOOL_DEFINITION, inputs => editNotebookCell(inputs, paths), agentId)
  return [NOTEBOOK_EDIT_TOOL_DEFINITION]
}

/** Edit one notebook cell without bypassing workspace containment checks. */
export async function editNotebookCell(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const notebookPath = requiredString(inputs, 'notebook_path')
  const cellIndex = requiredNonnegativeInteger(inputs, 'cell_index')
  const newSource = requiredText(inputs, 'new_source')
  const cellType = optionalString(inputs, 'cell_type') ?? 'code'
  if (cellType !== 'code' && cellType !== 'markdown') {
    throw new ValidationError('cell_type', 'must be code or markdown', cellType)
  }
  const target = await paths.resolve(notebookPath)
  let parsed: unknown
  try {
    parsed = JSON.parse(await Bun.file(target).text()) as unknown
  } catch (error) {
    throw new ValidationError('notebook_path', `must contain valid notebook JSON: ${errorMessage(error)}`, notebookPath)
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.cells)) {
    throw new ValidationError('notebook_path', 'must contain a cells array', notebookPath)
  }
  if (cellIndex >= parsed.cells.length) {
    throw new ValidationError('cell_index', `is outside the notebook cell range 0-${Math.max(parsed.cells.length - 1, 0)}`, cellIndex)
  }
  const cell = parsed.cells[cellIndex]
  if (!isRecord(cell)) throw new ValidationError('cell_index', 'must refer to an object cell', cellIndex)
  cell.source = splitLines(newSource)
  cell.cell_type = cellType
  const relativePath = await paths.relative(target)
  const checked = await paths.recheck(target)
  await Bun.write(checked, `${JSON.stringify(parsed, null, 1)}\n`)
  return `Updated cell ${cellIndex} in ${relativePath} (${cellType}, ${newSource.length} chars).`
}

function requiredNonnegativeInteger(inputs: JsonObject, name: string): number {
  const value = inputs[name]
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new ValidationError(name, 'must be a non-negative integer', value)
  }
  return value
}

function requiredText(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string') throw new ValidationError(name, 'must be a string', value)
  return value
}

function splitLines(value: string): string[] {
  return value.match(/[^\n]*\n|[^\n]+$/g) ?? []
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
