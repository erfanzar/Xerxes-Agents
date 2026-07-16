// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  DAEMON_PROTOCOL_VERSION,
  DaemonBuildFingerprint,
  captureDaemonBuildFingerprint,
  computeDaemonBuildId,
  type DaemonSourceReader,
} from '../src/daemon/fingerprint.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { daemonBuildIdForEntry, sourceDaemonBuildId } from '../src/daemon/sourceBuild.js'
import {
  BunDaemonLogFiles,
  DaemonLogger,
  type DaemonLogFiles,
  type DaemonLogOutput,
} from '../src/daemon/log.js'

test('daemon build fingerprint hashes ordered source bytes and explicit missing-file markers', async () => {
  const source = new Map<string, Uint8Array>([
    ['daemon/a.ts', bytes('first')],
    ['daemon/b.ts', bytes('second')],
  ])
  const reader = new MemorySourceReader(source)
  const options = {
    sourceRoot: '/workspace/src',
    sourceReader: reader,
    files: ['daemon/a.ts', 'daemon/missing.ts', 'daemon/b.ts'],
  }

  const buildId = await computeDaemonBuildId(options)

  expect(buildId).toBe('366b8df5dd3a3d01')
  expect(reader.calls).toEqual([
    ['/workspace/src', 'daemon/a.ts'],
    ['/workspace/src', 'daemon/missing.ts'],
    ['/workspace/src', 'daemon/b.ts'],
  ])
  expect(await computeDaemonBuildId(options)).toBe(buildId)

  source.set('daemon/b.ts', bytes('changed'))
  expect(await computeDaemonBuildId(options)).not.toBe(buildId)
})

test('captured fingerprint retains v35 identity after the caller source view changes', async () => {
  const source = new Map<string, Uint8Array>([['daemon/a.ts', bytes('before')]])
  const captured = await captureDaemonBuildFingerprint({
    sourceRoot: '/source',
    sourceReader: new MemorySourceReader(source),
    files: ['daemon/a.ts'],
  })
  source.set('daemon/a.ts', bytes('after'))

  expect(captured).toBeInstanceOf(DaemonBuildFingerprint)
  expect(captured.toRecord()).toEqual({
    daemon_protocol: DAEMON_PROTOCOL_VERSION,
    daemon_build_id: captured.daemonBuildId(),
  })
  expect(captured.daemonBuildId()).toMatch(/^[0-9a-f]{16}$/)
})

test('captured v35 identity composes through the existing runtime buildId option without server changes', async () => {
  const captured = await captureDaemonBuildFingerprint({
    sourceRoot: '/source',
    sourceReader: new MemorySourceReader(new Map([['daemon/a.ts', bytes('runtime')]])),
    files: ['daemon/a.ts'],
  })
  const runtime = new InMemoryDaemonRuntime(undefined, {
    buildId: captured.daemonBuildId(),
    currentProjectDirectory: '/project',
    sessionDirectory: '/sessions',
    workspaceRoot: '/workspace',
  })

  expect(runtime.status()).toMatchObject(captured.toRecord())
})

test('source checkout identity covers nested runtime files and skips built distributions', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-source-build-'))
  const built = await mkdtemp(join(tmpdir(), 'xerxes-built-runtime-'))
  try {
    await mkdir(join(root, 'daemon'), { recursive: true })
    await mkdir(join(root, 'tools'), { recursive: true })
    await writeFile(join(root, 'cli.ts'), 'export const cli = true\n')
    await writeFile(join(root, 'daemon', 'server.ts'), 'export const server = true\n')
    await writeFile(join(root, 'tools', 'agent.ts'), 'export const tool = "before"\n')
    await writeFile(join(root, 'agent.yaml'), 'tools: [AgentTool]\n')

    const before = await sourceDaemonBuildId(root)
    await writeFile(join(root, 'agent.yaml'), 'tools: [AgentTool, SpawnAgents]\n')
    const afterAgentDefinition = await sourceDaemonBuildId(root)
    await writeFile(join(root, 'tools', 'agent.ts'), 'export const tool = "after"\n')
    const afterTool = await sourceDaemonBuildId(root)

    expect(before).toMatch(/^[0-9a-f]{16}$/)
    expect(afterAgentDefinition).toMatch(/^[0-9a-f]{16}$/)
    expect(afterTool).toMatch(/^[0-9a-f]{16}$/)
    expect(afterAgentDefinition).not.toBe(before)
    expect(afterTool).not.toBe(afterAgentDefinition)
    expect(await sourceDaemonBuildId(built)).toBeUndefined()

    const builtEntry = join(built, 'cli.js')
    await writeFile(builtEntry, 'console.log("before")\n')
    const builtBefore = await daemonBuildIdForEntry(built, builtEntry)
    await writeFile(builtEntry, 'console.log("after")\n')
    const builtAfter = await daemonBuildIdForEntry(built, builtEntry)
    expect(builtBefore).toMatch(/^[0-9a-f]{16}$/)
    expect(builtAfter).toMatch(/^[0-9a-f]{16}$/)
    expect(builtAfter).not.toBe(builtBefore)
    expect(await daemonBuildIdForEntry(built, join(built, 'missing.js'))).toBeUndefined()
  } finally {
    await rm(root, { recursive: true, force: true })
    await rm(built, { recursive: true, force: true })
  }
})

test('runtime status reports live host tool and skill inventory', () => {
  let activeSubagents = 3
  let tools = 17
  let skills = 4
  const runtime = new InMemoryDaemonRuntime(undefined, {
    model: 'gpt-4o',
    statusInventory: () => ({ activeSubagents, skills, tools }),
  })

  expect(runtime.status()).toMatchObject({ active_subagents: 3, skills: 4, tools: 17 })
  activeSubagents = -1
  tools = 21
  skills = -1
  expect(runtime.status()).toMatchObject({ active_subagents: 0, skills: 0, tools: 21 })

  const runtimeWithoutSubagentInventory = new InMemoryDaemonRuntime(undefined, {
    model: 'gpt-4o',
    statusInventory: () => ({ skills: 1, tools: 2 }),
  })
  expect(runtimeWithoutSubagentInventory.status()).not.toHaveProperty('active_subagents')
})

test('fingerprint does not invent source contents when the injected reader fails', async () => {
  const reader: DaemonSourceReader = {
    async readFile() {
      throw new Error('source root is unavailable')
    },
  }
  await expect(computeDaemonBuildId({
    sourceRoot: '/source',
    sourceReader: reader,
    files: ['daemon/a.ts'],
  })).rejects.toThrow('source root is unavailable')
})

test('daemon logger writes atomic daily JSONL records and rotates using the injected UTC clock', async () => {
  const files = new MemoryLogFiles()
  const output = new MemoryOutput()
  const timestamps = [
    new Date('2026-07-13T23:59:58.000Z'),
    new Date('2026-07-14T00:00:01.000Z'),
  ]
  const logger = new DaemonLogger({
    directory: '/logs',
    files,
    output,
    clock: () => timestamps.shift() ?? new Date('2026-07-14T00:00:01.000Z'),
    idFactory: sequence('one', 'two'),
  })

  const first = await logger.info('daemon_started', { worker: 1 })
  const second = await logger.error('turn_failed', { reason: 'timeout' })

  expect(first).toEqual({ ts: '2026-07-13T23:59:58.000Z', level: 'info', event: 'daemon_started', worker: 1 })
  expect(second).toEqual({ ts: '2026-07-14T00:00:01.000Z', level: 'error', event: 'turn_failed', reason: 'timeout' })
  expect(files.directWrites).toEqual([
    '/logs/.daemon-2026-07-13.one.tmp',
    '/logs/.daemon-2026-07-14.two.tmp',
  ])
  expect(files.renames).toEqual([
    ['/logs/.daemon-2026-07-13.one.tmp', '/logs/daemon-2026-07-13.jsonl'],
    ['/logs/.daemon-2026-07-14.two.tmp', '/logs/daemon-2026-07-14.jsonl'],
  ])
  expect(files.text('/logs/daemon-2026-07-13.jsonl')).toBe(
    '{"ts":"2026-07-13T23:59:58.000Z","level":"info","event":"daemon_started","worker":1}\n',
  )
  expect(files.text('/logs/daemon-2026-07-14.jsonl')).toBe(
    '{"ts":"2026-07-14T00:00:01.000Z","level":"error","event":"turn_failed","reason":"timeout"}\n',
  )
  expect(logger.currentPath).toBe('/logs/daemon-2026-07-14.jsonl')
  expect(output.chunks).toEqual(['[info] daemon_started\n', '[error] turn_failed\n'])
})

test('daemon logger serializes concurrent appends and closes without retaining caller field objects', async () => {
  const files = new MemoryLogFiles()
  const output = new MemoryOutput()
  let timestamp = 0
  const fields = { nested: { count: 1 } }
  const logger = new DaemonLogger({
    directory: '/logs',
    files,
    output,
    clock: () => new Date(`2026-07-13T12:00:0${timestamp++}.000Z`),
    idFactory: sequence('first', 'second'),
  })

  const [first, second] = await Promise.all([
    logger.info('first', fields),
    logger.info('second', { value: 2 }),
  ])
  fields.nested.count = 99

  expect(first).toEqual({ ts: '2026-07-13T12:00:00.000Z', level: 'info', event: 'first', nested: { count: 1 } })
  expect(second).toEqual({ ts: '2026-07-13T12:00:01.000Z', level: 'info', event: 'second', value: 2 })
  expect(files.text('/logs/daemon-2026-07-13.jsonl')).toBe([
    '{"ts":"2026-07-13T12:00:00.000Z","level":"info","event":"first","nested":{"count":1}}',
    '{"ts":"2026-07-13T12:00:01.000Z","level":"info","event":"second","value":2}',
    '',
  ].join('\n'))

  await logger.close()
  await expect(logger.info('after_close')).rejects.toThrow('daemon logger is closed')
})

test('Bun daemon log files persist a complete daily JSONL replacement through the real filesystem port', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-log-'))
  const temporary = join(directory, '.daemon-2026-07-13.atomic.tmp')
  const output = new MemoryOutput()
  const logger = new DaemonLogger({
    directory,
    files: new BunDaemonLogFiles(),
    output,
    clock: () => new Date('2026-07-13T12:00:00.000Z'),
    idFactory: () => 'atomic',
  })
  try {
    await logger.info('persisted', { source: 'bun' })
    expect(await Bun.file(join(directory, 'daemon-2026-07-13.jsonl')).text()).toBe(
      '{"ts":"2026-07-13T12:00:00.000Z","level":"info","event":"persisted","source":"bun"}\n',
    )
    expect(await Bun.file(temporary).exists()).toBeFalse()
  } finally {
    await logger.close()
    await rm(directory, { recursive: true, force: true })
  }
})

class MemorySourceReader implements DaemonSourceReader {
  readonly calls: Array<readonly [string, string]> = []
  private readonly source: ReadonlyMap<string, Uint8Array>

  constructor(source: ReadonlyMap<string, Uint8Array>) {
    this.source = source
  }

  async readFile(sourceRoot: string, relativePath: string): Promise<Uint8Array | undefined> {
    this.calls.push([sourceRoot, relativePath])
    return this.source.get(relativePath)
  }
}

class MemoryLogFiles implements DaemonLogFiles {
  readonly directWrites: string[] = []
  readonly directories: string[] = []
  readonly renames: Array<readonly [string, string]> = []
  private readonly content = new Map<string, string>()

  async ensureDirectory(directory: string): Promise<void> {
    this.directories.push(directory)
  }

  async readText(path: string): Promise<string | undefined> {
    return this.content.get(path)
  }

  async remove(path: string): Promise<void> {
    this.content.delete(path)
  }

  async rename(from: string, to: string): Promise<void> {
    const content = this.content.get(from)
    if (content === undefined) throw new Error('temporary log file was not written')
    this.content.set(to, content)
    this.content.delete(from)
    this.renames.push([from, to])
  }

  async writeText(path: string, content: string): Promise<void> {
    this.content.set(path, content)
    this.directWrites.push(path)
  }

  text(path: string): string | undefined {
    return this.content.get(path)
  }
}

class MemoryOutput implements DaemonLogOutput {
  readonly chunks: string[] = []

  write(chunk: string): void {
    this.chunks.push(chunk)
  }
}

function bytes(value: string): Uint8Array {
  return new TextEncoder().encode(value)
}

function sequence(...values: string[]): () => string {
  let index = 0
  return () => values[index++] ?? `overflow-${index}`
}
