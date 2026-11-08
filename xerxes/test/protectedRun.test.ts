// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { filterProtectedTools, parseProtectedRunConfig } from '../src/runtime/protectedRun.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

function tool(name: string): ToolDefinition {
  return { type: 'function', function: { name, description: name, parameters: { type: 'object', properties: {} } } }
}

test('parseProtectedRunConfig: flag, env, runtime, default false', () => {
  expect(parseProtectedRunConfig({}, {})).toBe(false)
  expect(parseProtectedRunConfig({ protected_run: true }, {})).toBe(true)
  expect(parseProtectedRunConfig({ protected_run: true }, { XERXES_PROTECTED_RUN: 'off' })).toBe(false)
  expect(parseProtectedRunConfig({}, { XERXES_PROTECTED_RUN: '1' })).toBe(true)
  expect(parseProtectedRunConfig({ protected_run: true }, { XERXES_PROTECTED_RUN: '0' }, true)).toBe(true)
  expect(parseProtectedRunConfig({}, {}, false)).toBe(false)
})

test('parseProtectedRunConfig rejects unreadable values', () => {
  expect(() => parseProtectedRunConfig({ protected_run: 'yes' }, {})).toThrow(ValidationError)
  expect(() => parseProtectedRunConfig({}, { XERXES_PROTECTED_RUN: 'maybe' })).toThrow('XERXES_PROTECTED_RUN')
})

test('filterProtectedTools removes shell, terminal, MCP, plugin and computer tools in order', () => {
  const names = [
    'ReadFile', 'exec_command', 'ExecuteShell', 'Bash', 'shell', 'run_command', 'pty_open', 'pty_write', 'write_stdin',
    'check_command', 'mcp__server__tool', 'mcp_other', 'plugin_demo', 'PluginManageTool', 'computer_use',
    'WriteFile', 'XlsxWriter',
  ]
  const { kept, removed } = filterProtectedTools(names.map(tool))
  expect(kept.map(entry => entry.function.name)).toEqual(['ReadFile', 'WriteFile', 'XlsxWriter'])
  expect(removed.map(entry => entry.function.name)).toEqual(names.filter(name => !['ReadFile', 'WriteFile', 'XlsxWriter'].includes(name)))
})
