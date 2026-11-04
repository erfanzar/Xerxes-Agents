// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { runResultChecks } from '../src/runtime/resultChecks.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { registerXlsxTools } from '../src/tools/xlsxTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

async function inWorkspace(run: (workspace: string, registry: ToolRegistry) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-xlsx-'))
  try {
    const registry = new ToolRegistry()
    const paths = new WorkspacePathResolver(workspace)
    registerXlsxTools(registry, { resolvePath: candidate => paths.resolve(candidate) })
    await run(workspace, registry)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

const context = { metadata: {} }

async function run(registry: ToolRegistry, name: string, input: JsonObject): Promise<JsonObject> {
  return JSON.parse(await registry.execute(call(name, input), context)) as JsonObject
}

test('XlsxWriter output round-trips through XlsxReader and keeps value types', async () => {
  await inWorkspace(async (_workspace, registry) => {
    const written = await run(registry, 'XlsxWriter', {
      path: 'out/book.xlsx',
      sheets: [
        { name: 'Data', rows: [['name', 'qty', 'ok', 'note'], ['Ada & <Bob>', 3, true, null], ['  padded ', 2.5, false, 'line\nbreak']] },
        { name: 'Empty', rows: [] },
      ],
    })
    expect(written.success).toBe(true)
    expect(written.sheets).toBe(2)

    const read = await run(registry, 'XlsxReader', { path: 'out/book.xlsx' })
    expect(read.sheet).toBe('Data')
    expect(read.sheets).toEqual(['Data', 'Empty'])
    expect(read.rows).toEqual([
      ['name', 'qty', 'ok', 'note'],
      ['Ada & <Bob>', 3, true, null],
      ['  padded ', 2.5, false, 'line\nbreak'],
    ])
    const second = await run(registry, 'XlsxReader', { path: 'out/book.xlsx', sheet: 'empty' })
    expect(second.rows).toEqual([])
  })
})

test('XlsxReader honours max_rows and reports truncation', async () => {
  await inWorkspace(async (_workspace, registry) => {
    await run(registry, 'XlsxWriter', { path: 'big.xlsx', sheets: [{ name: 'S', rows: [[1], [2], [3], [4]] }] })
    const read = await run(registry, 'XlsxReader', { path: 'big.xlsx', max_rows: 2 })
    expect(read.rows).toEqual([[1], [2]])
    expect(read.truncated).toBe(true)
  })
})

test('runResultChecks accepts a workbook written by XlsxWriter', async () => {
  await inWorkspace(async (workspace, registry) => {
    await run(registry, 'XlsxWriter', { path: 'ok.xlsx', sheets: [{ name: 'S', rows: [['a', 'b'], [1, 2]] }] })
    expect(await runResultChecks({ workspaceRoot: workspace, changedFiles: ['ok.xlsx'] })).toEqual([])
  })
})

test('XlsxWriter validates sheet names and cells and refuses silent overwrite', async () => {
  await inWorkspace(async (_workspace, registry) => {
    const write = (input: JsonObject): Promise<string> => registry.execute(call('XlsxWriter', input), context)
    for (const name of ['', 'x'.repeat(32), 'a/b', 'a:b', 'a?b', 'a*b', 'a[b', 'a]b', 'a\\b', "'quoted"]) {
      await expect(write({ path: 'a.xlsx', sheets: [{ name, rows: [] }] })).rejects.toThrow()
    }
    await expect(write({ path: 'a.xlsx', sheets: [{ name: 'S', rows: [[{ nested: 1 }]] }] })).rejects.toThrow()
    await expect(write({ path: 'a.xlsx', sheets: [{ name: 'S', rows: [['bad\u0001']] }] })).rejects.toThrow()
    await expect(write({ path: 'a.xlsx', sheets: [{ name: 'S', rows: [] }, { name: 's', rows: [] }] })).rejects.toThrow('duplicates')
    await expect(write({ path: 'a.csv', sheets: [{ name: 'S', rows: [] }] })).rejects.toThrow('.xlsx')
    await expect(write({ path: 'a.xlsx', sheets: [] })).rejects.toThrow()

    await write({ path: 'a.xlsx', sheets: [{ name: 'S', rows: [['one']] }] })
    await expect(write({ path: 'a.xlsx', sheets: [{ name: 'S', rows: [['two']] }] })).rejects.toThrow('overwrite')
    await write({ path: 'a.xlsx', overwrite: true, sheets: [{ name: 'S', rows: [['two']] }] })
  })
})

test('XlsxWriter and XlsxReader refuse paths outside the workspace and non-workbooks', async () => {
  await inWorkspace(async (workspace, registry) => {
    await expect(registry.execute(call('XlsxWriter', { path: '../escape.xlsx', sheets: [{ name: 'S', rows: [] }] }), context)).rejects.toThrow()
    await expect(registry.execute(call('XlsxReader', { path: '/etc/passwd' }), context)).rejects.toThrow()
    await expect(registry.execute(call('XlsxReader', { path: 'missing.xlsx' }), context)).rejects.toThrow('does not exist')
    await mkdir(join(workspace, 'd'), { recursive: true })
    await writeFile(join(workspace, 'd', 'fake.xlsx'), 'a,b\n1,2\n')
    await expect(registry.execute(call('XlsxReader', { path: 'd/fake.xlsx' }), context)).rejects.toThrow('not a readable')
  })
})

test('the written workbook is a valid archive for the system unzip', async () => {
  const unzip = Bun.which('unzip')
  if (unzip === null) return
  await inWorkspace(async (workspace, registry) => {
    await run(registry, 'XlsxWriter', { path: 'z.xlsx', sheets: [{ name: 'S', rows: [['a']] }] })
    const listing = Bun.spawnSync([unzip, '-l', join(workspace, 'z.xlsx')])
    const text = listing.stdout.toString()
    expect(listing.exitCode).toBe(0)
    expect(text).toContain('[Content_Types].xml')
    expect(text).toContain('xl/workbook.xml')
    expect(text).toContain('xl/worksheets/sheet1.xml')
    const test = Bun.spawnSync([unzip, '-tq', join(workspace, 'z.xlsx')])
    expect(test.exitCode).toBe(0)
  })
})
