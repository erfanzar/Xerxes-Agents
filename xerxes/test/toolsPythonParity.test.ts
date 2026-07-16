// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { MarkdownAgentWorkspace } from '../src/channels/workspace.js'
import { SpawnedAgentManager } from '../src/operators/subagents.js'
import { ClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import { readFile as codingReadFile } from '../src/tools/codingTools.js'
import { readFile as standaloneReadFile } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import {
  WorkspaceFileNotFoundError,
  workspaceAppend,
  workspaceDiff,
  workspaceList,
  workspaceRead,
  workspaceWrite,
} from '../src/tools/workspaceTools.js'

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-tools-python-parity-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

test('ReadFile and read_file treat JSON null chunk arguments as omitted defaults', async () => {
  await inTemporaryDirectory(async directory => {
    const paths = new WorkspacePathResolver(directory)
    const payload = Array.from({ length: 1_000 }, (_value, index) => `payload-${index}\n`).join('')
    await Bun.write(join(directory, 'big.txt'), payload)

    const standalone = await standaloneReadFile({ file_path: 'big.txt', offset: null, limit: null }, paths)
    expect(standalone).toContain('payload-0')
    expect(standalone).toContain('payload-399')
    expect(standalone).not.toContain('payload-400')
    expect(standalone).toContain('Continue with offset=400, limit=400')
    expect(await standaloneReadFile({ file_path: 'big.txt', limit: -1 }, paths)).toContain('payload-999')

    const coding = await codingReadFile({ file_path: 'big.txt', start_line: null, end_line: null }, paths)
    expect(coding).toContain('payload-0')
    expect(coding).toContain('payload-399')
    expect(coding).not.toContain('payload-400')
    expect(coding).toContain('Continue with start_line=401')
  })
})

test('SpawnAgents accepts safe common JSON-adjacent agent payloads without evaluating text', async () => {
  const manager = new SpawnedAgentManager({
    runner: async request => ({ content: `done:${request.input}` }),
  })
  const tools = new ClaudeAgentTools({ manager })

  const payloads = [
    '```json\n[{"name":"fenced","prompt":"fenced task","title":"Fenced task"}]\n```',
    '{"name":"solo","prompt":"solo task","title":"Solo task"}',
    '[{“name”: “smart”, “prompt”: “smart task”, “title”: “Smart task”}]',
    "[{'name': 'single', 'prompt': 'single task', 'title': 'Single task'}]",
  ]
  for (const agents of payloads) {
    const result = await tools.execute(
      'SpawnAgents',
      { agents, timeout: 1, wait: true },
      { metadata: {} },
    ) as Array<{ readonly last_output: string; readonly status: string }>
    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({ status: 'completed', last_output: expect.stringContaining('task') })
  }
  await expect(tools.execute(
    'SpawnAgents',
    { agents: 'this is not JSON', wait: false },
    { metadata: {} },
  )).rejects.toThrow('agents')
  for (const handle of manager.listHandles()) manager.close(handle.id)
})

test('markdown workspace list/read/write/append/diff preserve the Python workspace-tools contract', async () => {
  await inTemporaryDirectory(async directory => {
    const workspace = new MarkdownAgentWorkspace(join(directory, 'agent'))
    const seeded = await workspaceList(workspace)
    expect(seeded.map(entry => entry.path)).toEqual(expect.arrayContaining([
      'AGENTS.md', 'SOUL.md', 'USER.md', 'MEMORY.md', 'TOOLS.md',
    ]))
    expect(await workspaceRead('SOUL.md', workspace)).toContain('Xerxes')
    await expect(workspaceRead('missing.md', workspace)).rejects.toBeInstanceOf(WorkspaceFileNotFoundError)
    await expect(workspaceRead('../../outside.md', workspace)).rejects.toThrow('workspace root')

    expect(await workspaceWrite('notes/entry.md', 'first line', workspace)).toEqual({
      path: 'notes/entry.md', bytes: 10, created: true,
    })
    expect(await workspaceWrite('notes/entry.md', 'second', workspace)).toEqual({
      path: 'notes/entry.md', bytes: 6, created: false,
    })
    expect(await workspaceAppend('notes/entry.md', 'third', workspace)).toEqual({
      path: 'notes/entry.md', appendedBytes: 6, created: false,
    })
    expect(await workspaceRead('notes/entry.md', workspace)).toBe('second\nthird')

    await workspaceWrite('diff.md', 'hello\nworld\n', workspace)
    const diff = await workspaceDiff('diff.md', 'hello\nuniverse\n', workspace)
    expect(diff).toContain('-world')
    expect(diff).toContain('+universe')
    expect(await workspaceDiff('diff.md', 'hello\nworld\n', workspace)).toBe('')
    expect(await workspaceDiff('new.md', 'fresh\n', workspace)).toContain('+fresh')

    await workspaceWrite('large.md', 'x'.repeat(1_000), workspace)
    expect(await workspaceRead('large.md', workspace, { maxBytes: 100 })).toContain('truncated')
  })
})
