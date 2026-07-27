// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AGENT_MEMORY_TOOL_DEFINITIONS } from '../src/tools/agentMemoryTools.js'
import { NOTEBOOK_EDIT_TOOL_DEFINITION } from '../src/tools/claudeTools/notebook.js'
import { LSP_TOOL_DEFINITION } from '../src/tools/claudeTools/search.js'
import { DATA_TOOL_DEFINITIONS } from '../src/tools/dataTools.js'
import { MEMORY_TOOL_DEFINITIONS } from '../src/tools/memoryTools.js'
import { EXEC_COMMAND_DEFINITION } from '../src/tools/processTools.js'
import { SYSTEM_TOOL_DEFINITIONS } from '../src/tools/systemTools.js'
import type { JsonObject, ToolDefinition } from '../src/types/toolCalls.js'
import { WEB_TOOL_DEFINITIONS } from '../src/tools/webTools.js'

const DEFINITIONS: readonly ToolDefinition[] = [
  ...SYSTEM_TOOL_DEFINITIONS,
  ...DATA_TOOL_DEFINITIONS,
  ...WEB_TOOL_DEFINITIONS,
  ...MEMORY_TOOL_DEFINITIONS,
  ...AGENT_MEMORY_TOOL_DEFINITIONS,
  EXEC_COMMAND_DEFINITION,
  LSP_TOOL_DEFINITION,
  NOTEBOOK_EDIT_TOOL_DEFINITION,
]

function describes(name: string): string {
  const found = DEFINITIONS.find(definition => definition.function.name === name)
  if (found === undefined) throw new Error('no definition named ' + name)
  return found.function.description
}

// A one-clause description is the failure this lane exists to prevent: the doctrine that
// stops a tool being misused is only paid for when the tool is loaded, so it belongs here
// and not in the shared system prompt.
test('every tool description carries more than a single restated clause', () => {
  for (const definition of DEFINITIONS) {
    expect(definition.function.description.length).toBeGreaterThan(240)
  }
})

// The web tools take `timeout` in seconds while every other bounded tool in the runtime
// takes milliseconds; the schema exposes a bare integer, so only the description can say so.
test('web tool descriptions state the timeout unit the schema cannot', () => {
  for (const definition of WEB_TOOL_DEFINITIONS) {
    expect(definition.function.description).toContain('SECONDS')
    expect(properties(definition).timeout).toMatchObject({ maximum: 300, minimum: 1 })
  }
})

// A description that quotes a default drifts silently when the constant moves.
test('exec_command quotes the same limits its schema declares', () => {
  const schema = properties(EXEC_COMMAND_DEFINITION)
  const description = EXEC_COMMAND_DEFINITION.function.description
  expect(description).toContain(String(numberField(schema.timeout_ms, 'default')))
  expect(description).toContain(String(numberField(schema.max_output_chars, 'default')))
})

test('the highest-cost footgun of each tool is named in its own description', () => {
  // Silent misuse rather than an error is what needs pre-empting, tool by tool.
  expect(describes('exec_command')).toContain('no shell')
  expect(describes('exec_command')).toContain('exitCode is a normal successful call')
  expect(describes('EnvironmentManager')).toContain('identical to a genuinely unset variable')
  expect(describes('DateTimeProcessor')).toContain('day/month/year')
  expect(describes('CSVProcessor')).toContain('every cell comes back as a string')
  expect(describes('TextProcessor')).toContain('never opens a file')
  expect(describes('JSONProcessor')).toContain('does NOT read a file')
  expect(describes('DataConverter')).toContain('not byte-safe')
  expect(describes('WebScraper')).toContain('no JavaScript ever runs')
  expect(describes('APIClient')).toContain('only for GET and HEAD')
  expect(describes('RSSReader')).toContain('no autodiscovery')
  expect(describes('URLAnalyzer')).toContain('NO network request')
  expect(describes('delete_memory')).toContain('wipes it')
  expect(describes('search_memory')).toContain('not semantic search')
  expect(describes('agent_memory_write')).toContain('never a patch')
  expect(describes('agent_memory_learn')).toContain('currently ignored')
  expect(describes('LSPTool')).toContain('zero-based')
  expect(describes('NotebookEditTool')).toContain('converts that cell to code')
})

function properties(definition: ToolDefinition): JsonObject {
  const schema = definition.function.parameters as JsonObject
  const found = schema.properties
  if (!isObject(found)) throw new Error(definition.function.name + ' has no properties')
  return found
}

function numberField(value: unknown, key: string): number {
  if (!isObject(value)) throw new Error('expected a schema object')
  const found = value[key]
  if (typeof found !== 'number') throw new Error('expected a numeric ' + key)
  return found
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
