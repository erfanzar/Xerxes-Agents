// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, utimes, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore } from '../src/bridge/profiles.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { MAX_TASK_OUTPUTS, TASK_OUTPUTS_KEY, taskOutputFiles, taskOutputsOf, withTurnOutputs } from '../src/daemon/taskOutputs.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { groupOutputs, taskOutputsView } from '../src/desktop/renderer/OutputsPanel.js'
import { looksLikeFilePath, Markdown } from '../src/desktop/renderer/markdown.js'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { MAX_LOCKED_INPUTS, RunSafety } from '../src/runtime/runSafety.js'

async function folder(prefix: string): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix))
}

test('a run lists what it wrote, by any means, and never its inputs', async () => {
  const root = await folder('xerxes-run-outputs-')
  try {
    await writeFile(join(root, 'bank.csv'), 'a\n')
    await mkdir(join(root, 'out'))
    const run = await RunSafety.start({ outputFolder: root, sandboxUnavailable: 'test' })
    expect(await run.outputs()).toEqual([])
    // A script wrote these, not a file tool: the list does not care how.
    await writeFile(join(root, 'out', 'summary.csv'), 'region,total\n')
    await writeFile(join(root, 'out', 'make.py'), 'print(1)\n')
    await writeFile(join(root, 'bank.csv'), 'changed\n')
    const listed = await run.outputs()
    expect(listed?.map(output => output.path).sort()).toEqual(['out/make.py', 'out/summary.csv'])
    expect(listed?.find(output => output.path === 'out/summary.csv')?.size).toBe(13)
    // An input someone approved changing counts as written.
    run.release('bank.csv')
    expect((await run.outputs())?.map(output => output.path)).toContain('bank.csv')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('a folder too large to know its inputs has no output list rather than a wrong one', async () => {
  const root = await folder('xerxes-run-outputs-big-')
  try {
    await Promise.all(Array.from({ length: MAX_LOCKED_INPUTS + 1 }, (_, i) => writeFile(join(root, `f${i}.txt`), '')))
    const run = await RunSafety.start({ outputFolder: root, sandboxUnavailable: 'test' })
    expect(await run.outputs()).toBeUndefined()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('the saved list keeps the first turn a file appeared in, refuses paths outside the task and is bounded', async () => {
  const first = withTurnOutputs([], [{ path: 'out/a.csv', size: 1, modifiedAt: 1 }], 1)
  const second = withTurnOutputs(first, [
    { path: 'out/a.csv', size: 2, modifiedAt: 2 }, { path: 'out/b.csv', size: 1, modifiedAt: 2 }, { path: '../escape.csv', size: 1, modifiedAt: 2 },
    // Housekeeping is not a result: Xerxes's own files, hidden files, bytecode caches.
    { path: '.xerxes/memory.db', size: 1, modifiedAt: 2 }, { path: 'out/.DS_Store', size: 1, modifiedAt: 2 }, { path: 'out/__pycache__/m.pyc', size: 1, modifiedAt: 2 },
  ], 3)
  expect(second).toEqual([{ path: 'out/a.csv', turn: 1 }, { path: 'out/b.csv', turn: 3 }])
  expect(taskOutputsOf({ [TASK_OUTPUTS_KEY]: [...second, { path: '/etc/passwd', turn: 1 }, { path: 'x', turn: -1 }, 'junk'] })).toEqual(second)
  expect(taskOutputsOf({})).toEqual([])
  const many = withTurnOutputs([], Array.from({ length: MAX_TASK_OUTPUTS + 20 }, (_, i) => ({ path: `out/${i}.csv`, size: 1, modifiedAt: i })), 1)
  expect(many).toHaveLength(MAX_TASK_OUTPUTS)

  const root = await folder('xerxes-task-outputs-')
  try {
    await mkdir(join(root, 'out'))
    await writeFile(join(root, 'out', 'a.csv'), 'abc')
    await utimes(join(root, 'out', 'a.csv'), new Date(1_000_000), new Date(1_000_000))
    await writeFile(join(root, 'out', 'b.csv'), 'b')
    // b.csv is newest; a file deleted since drops out of the list.
    const files = await taskOutputFiles(root, [...second, { path: 'out/gone.csv', turn: 2 }])
    expect(files.map(file => [file.path, file.size, file.turn])).toEqual([['out/b.csv', 1, 3], ['out/a.csv', 3, 1]])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

/** A model whose turn leaves a result file behind, as a script it ran would. */
class WritesOnce implements LlmClient {
  constructor(private readonly root: string) {}
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    await writeFile(join(this.root, 'out', 'result.csv'), 'region,total\nNorth,2\n')
    yield { content: 'Wrote out/result.csv.' }
  }
}

test('each turn records the files it wrote with the task, and outputs.list reads them back', async () => {
  const root = await folder('xerxes-turn-outputs-')
  try {
    await writeFile(join(root, 'input.csv'), 'x\n')
    await mkdir(join(root, 'out'))
    const runner = new AgentTurnRunner({
      llm: new WritesOnce(root), model: 'fixture',
      runSafetyForSession: async (_session, outputFolder) => RunSafety.start({ outputFolder, sandboxUnavailable: 'test' }),
    })
    const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: root, sessionDirectory: join(root, '.sessions') })
    const socketPath = join(root, 'd.sock')
    const server = new DaemonServer({ profileStore: new ProfileStore(join(root, 'profiles.json')), runtime, socketPath })
    await server.start()
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
    socket.setEncoding('utf8')
    let buffer = '', id = 0
    const pending = new Map<number, (value: Record<string, unknown>) => void>()
    socket.on('data', chunk => {
      buffer += String(chunk)
      for (let newline = buffer.indexOf('\n'); newline >= 0; newline = buffer.indexOf('\n')) {
        const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown> }
        buffer = buffer.slice(newline + 1)
        if (typeof frame.id === 'number') pending.get(frame.id)?.(frame.result ?? {})
      }
    })
    const call = (method: string, params: Record<string, unknown> = {}) => new Promise<Record<string, unknown>>(resolve => {
      id += 1; pending.set(id, resolve); socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    })
    try {
      await call('initialize', { session_key: 'task', project_dir: root })
      expect(await call('outputs.list')).toMatchObject({ ok: true, outputs: [] })
      await runtime.submitTurn('task', 'write the result', () => {})
      const listed = await call('outputs.list')
      expect(listed).toMatchObject({ ok: true, outputs: [{ path: 'out/result.csv', turn: 1, size: 21 }] })
      expect(JSON.stringify(listed)).not.toContain('input.csv')
    } finally {
      socket.destroy()
      await server.stop()
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('the panel reads the list strictly, puts results before scripts, and replies link only real file names', () => {
  const view = taskOutputsView({ cwd: '/w', outputs: [
    { path: 'out/make.py', turn: 1, size: 10, modified_at: 3 },
    { path: 'out/summary.csv', turn: 1, size: 10, modified_at: 2 },
    { path: 'out/_debug.txt', turn: 2, size: 1, modified_at: 4 },
    { path: 'bad' },
  ] })
  expect(view?.outputs).toHaveLength(3)
  const grouped = groupOutputs(view!.outputs)
  expect(grouped.results.map(output => output.path)).toEqual(['out/summary.csv'])
  expect(grouped.scripts.map(output => output.path)).toEqual(['out/_debug.txt', 'out/make.py'])
  expect(taskOutputsView({ ok: true })).toBeNull()

  for (const path of ['out/region_summary.csv', 'notes.md', './report.xlsx', 'src/a.ts', '~/Documents/x.pdf']) expect(looksLikeFilePath(path)).toBe(true)
  for (const text of ['snap.runMode', 'os.path', 'e.g.', '1.2.3', 'npm run build', 'https://x.com/a.csv', 'out/']) expect(looksLikeFilePath(text)).toBe(false)
})

test('a file named in code in a reply is a link, also inside bold', () => {
  const html = renderToStaticMarkup(createElement(Markdown, { text: 'Wrote **`out/units.csv`** from `sales.csv`; ran `python3 -I x.py`.' }))
  expect(html.match(/md__link--code/g)).toHaveLength(2)
  expect(html).not.toContain('`')
  expect(html).toContain('<code>python3 -I x.py</code>')
})
