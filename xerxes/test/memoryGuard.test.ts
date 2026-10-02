// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, setSystemTime, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  defaultMemoryLimitBytes,
  hostMemoryGuardPorts,
  linuxFootprint,
  MemoryGuard,
  memoryGuardNote,
  memoryGuardSupported,
  readMemoryGuardSettings,
  resolveMemoryLimitBytes,
  saveMemoryGuardSettings,
  type MemoryGuardPorts,
  type ProcessRow,
} from '../src/runtime/memoryGuard.js'

const GB = 1024 ** 3

function fakePorts(rows: readonly ProcessRow[], sizes: Readonly<Record<number, number>>) {
  const signals: Array<[number, string]> = []
  const ports: MemoryGuardPorts = {
    listProcesses: async () => rows,
    footprintBytes: pid => sizes[pid],
    kill: (pid, signal) => { signals.push([pid, signal]) },
  }
  return { ports, signals }
}

// daemon 100 → sh 200 → python 201 (leaking) and 202; a small 300; a person's shell 400 → 401.
const tree: readonly ProcessRow[] = [
  { pid: 200, ppid: 100, command: '/bin/sh -c pytest -q' },
  { pid: 201, ppid: 200, command: '.venv/bin/python3 -u -m pytest -q' },
  { pid: 202, ppid: 200, command: 'cat' },
  { pid: 300, ppid: 100, command: 'git status' },
  { pid: 400, ppid: 100, command: '/bin/zsh -l' },
  { pid: 401, ppid: 400, command: 'big-local-build' },
  { pid: 500, ppid: 1, command: 'unrelated app' },
]

test('a child tree over the limit is stopped deepest first, and its tools are told why', async () => {
  const { ports, signals } = fakePorts(tree, { 200: 1 * 1024 ** 2, 201: 9 * GB, 202: 2 * 1024 ** 2, 300: 40 * 1024 ** 2, 400: 1024 ** 2, 401: 20 * GB, 500: 30 * GB })
  const stops: unknown[] = []
  const guard = new MemoryGuard({ rootPid: 100, limitBytes: () => 8 * GB, ports, graceMs: 20, exclude: () => new Set([400]), onStop: stop => stops.push(stop) })
  const stopped = await guard.tick()
  expect(stopped).toHaveLength(1)
  expect(stopped[0]).toMatchObject({ rootPid: 200, pids: [200, 201, 202], command: '.venv/bin/python3 -u -m pytest -q', limitBytes: 8 * GB })
  expect(stopped[0]!.bytes).toBeGreaterThan(9 * GB)
  expect(stops).toHaveLength(1)
  // Deepest first; the git command, the person's own shell and unrelated apps are untouched.
  expect(signals).toEqual([[202, 'SIGTERM'], [201, 'SIGTERM'], [200, 'SIGTERM']])
  for (const pid of [200, 201, 202]) expect(memoryGuardNote(pid, 0)).toContain('over the 8.0 GB limit')
  expect(memoryGuardNote(300, 0)).toBeUndefined()
  // A tree being stopped is not stopped again on the next pass.
  expect(await guard.tick()).toHaveLength(0)
  await Bun.sleep(40)
  expect(signals.filter(([, signal]) => signal === 'SIGKILL').map(([pid]) => pid)).toEqual([202, 201, 200])
})

test('a forked worker pool is measured by what it holds, not once per worker for the pages it shares', async () => {
  // daemon 100 → python 200 (6 GB) → 8 forked workers sharing its pages copy-on-write.
  const workers = [201, 202, 203, 204, 205, 206, 207, 208]
  const rows: ProcessRow[] = [{ pid: 200, ppid: 100, command: 'python train.py' }, ...workers.map(pid => ({ pid, ppid: 200, command: 'python train.py' }))]
  const kb = (bytes: number) => Math.round(bytes / 1024)
  // VmRSS bills every worker for the 6 GB it shares; PSS splits each shared page among the 9 mappers.
  const proc: Record<string, string> = {}
  for (const pid of [200, ...workers]) {
    const own = pid === 200 ? 0.5 * GB : 0.1 * GB
    proc[`/proc/${pid}/status`] = `Name:\tpython\nVmRSS:\t${kb(6 * GB + own)} kB\nRssAnon:\t${kb(6 * GB + own)} kB\n`
    proc[`/proc/${pid}/smaps_rollup`] = `Rss:            ${kb(6 * GB + own)} kB\nPss:            ${kb(6 * GB / 9 + own)} kB\n`
  }
  const read = (path: string) => {
    if (!(path in proc)) throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
    return proc[path]!
  }
  const signals: Array<[number, string]> = []
  const ports: MemoryGuardPorts = { listProcesses: async () => rows, footprintBytes: pid => linuxFootprint(pid, read), kill: (pid, signal) => { signals.push([pid, signal]) } }
  // Half a 64 GB host: the job really holds about 7 GB.
  expect(await new MemoryGuard({ rootPid: 100, limitBytes: () => 32 * GB, ports, graceMs: 10 }).tick()).toHaveLength(0)
  expect(signals).toEqual([])
  expect(linuxFootprint(200, read)).toBe(kb(6 * GB / 9 + 0.5 * GB) * 1024)
  // An old kernel without smaps_rollup still does not count mapped files.
  expect(linuxFootprint(1, () => { throw new Error('ENOENT') })).toBeUndefined()
  expect(linuxFootprint(9, path => path.endsWith('status') ? 'VmRSS:\t900 kB\nRssAnon:\t100 kB\n' : (() => { throw new Error('ENOENT') })())).toBe(100 * 1024)
})

test('a stop note is not handed to a later process that reuses the pid', async () => {
  const realNow = Date.now()
  setSystemTime(new Date(realNow - 60_000))
  try {
    const { ports } = fakePorts([{ pid: 777_001, ppid: 100, command: 'pytest -n 40' }], { 777_001: 20 * GB })
    expect(await new MemoryGuard({ rootPid: 100, limitBytes: () => 8 * GB, ports, graceMs: 10 }).tick()).toHaveLength(1)
  } finally {
    setSystemTime()
  }
  // The process the guard stopped started before the stop; a new one with that pid started after it.
  expect(memoryGuardNote(777_001, realNow - 120_000)).toContain('[memory guard] Stopped')
  expect(memoryGuardNote(777_001, realNow)).toBeUndefined()
})

test('a background command that exits normally on a pid the guard once stopped is not reported as stopped', async () => {
  const { BackgroundCommandManager } = await import('../src/tools/backgroundCommands.js')
  const manager = new BackgroundCommandManager()
  const started = manager.start({ command: process.execPath, args: ['-e', 'console.error("fine")'], cwd: tmpdir() })
  // An earlier stop recorded for this pid, before this process existed.
  setSystemTime(new Date(Date.now() - 60_000))
  try {
    const { ports } = fakePorts([{ pid: started.pid, ppid: 100, command: 'old job' }], { [started.pid]: 20 * GB })
    expect(await new MemoryGuard({ rootPid: 100, limitBytes: () => 8 * GB, ports, graceMs: 10 }).tick()).toHaveLength(1)
  } finally {
    setSystemTime()
  }
  try {
    const result = await manager.check(started.procId, 10_000, 10_000)
    expect(result).toMatchObject({ running: false, exitCode: 0 })
    expect(result.stderr).not.toContain('[memory guard]')
  } finally {
    await manager.disposeAll()
  }
})

test('an off guard, or trees under the limit, stop nothing', async () => {
  const { ports, signals } = fakePorts(tree, { 201: 9 * GB })
  expect(await new MemoryGuard({ rootPid: 100, limitBytes: () => 0, ports }).tick()).toHaveLength(0)
  expect(await new MemoryGuard({ rootPid: 100, limitBytes: () => 16 * GB, ports }).tick()).toHaveLength(0)
  expect(signals).toEqual([])
})

test('the limit is the environment override, else the saved setting, else half the computer', async () => {
  expect(defaultMemoryLimitBytes(16 * GB)).toBe(8 * GB)
  expect(resolveMemoryLimitBytes({}, {}, 16 * GB)).toBe(8 * GB)
  expect(resolveMemoryLimitBytes({ limitMb: 4096 }, {}, 16 * GB)).toBe(4 * GB)
  expect(resolveMemoryLimitBytes({ limitMb: 0 }, {}, 16 * GB)).toBe(0)
  expect(resolveMemoryLimitBytes({ limitMb: 4096 }, { XERXES_COMMAND_MEMORY_LIMIT_MB: '2048' }, 16 * GB)).toBe(2 * GB)
  expect(resolveMemoryLimitBytes({}, { XERXES_COMMAND_MEMORY_LIMIT_MB: 'lots' }, 16 * GB)).toBe(8 * GB)

  const directory = await mkdtemp(join(tmpdir(), 'xerxes-memory-guard-'))
  try {
    const file = join(directory, 'memory-guard.json')
    expect(await readMemoryGuardSettings(file)).toEqual({})
    await saveMemoryGuardSettings(file, 6144)
    expect(await readMemoryGuardSettings(file)).toEqual({ limitMb: 6144 })
    await saveMemoryGuardSettings(file, undefined)
    expect(await readMemoryGuardSettings(file)).toEqual({})
    await expect(saveMemoryGuardSettings(file, -1)).rejects.toThrow('whole number')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test.skipIf(!memoryGuardSupported())('a real command that outgrows the limit is stopped and noted', async () => {
  // A child that holds ~300 MB until it is killed.
  const spawnedAt = Date.now()
  const child = Bun.spawn([process.execPath, '-e', 'const keep = Buffer.alloc(300 * 1024 * 1024, 1); setInterval(() => keep[0]++, 50)'], { stdout: 'ignore', stderr: 'ignore' })
  try {
    const ports = hostMemoryGuardPorts()
    // Wait for the allocation to show up in the footprint.
    for (let attempt = 0; attempt < 50 && (ports.footprintBytes(child.pid) ?? 0) < 200 * 1024 ** 2; attempt++) await Bun.sleep(100)
    expect(ports.footprintBytes(child.pid) ?? 0).toBeGreaterThan(200 * 1024 ** 2)
    const guard = new MemoryGuard({ rootPid: process.pid, limitBytes: () => 150 * 1024 ** 2, ports, graceMs: 500,
      // Only the child under test: this test runner may have other children.
      exclude: () => new Set((Bun.spawnSync(['ps', '-A', '-o', 'pid=,ppid=']).stdout.toString().split('\n')
        .map(line => line.trim().split(/\s+/).map(Number)).filter(([pid, ppid]) => ppid === process.pid && pid !== child.pid).map(([pid]) => pid!))) })
    const stopped = await guard.tick()
    expect(stopped.map(stop => stop.rootPid)).toContain(child.pid)
    expect(await child.exited).not.toBe(0)
    expect(memoryGuardNote(child.pid, spawnedAt)).toContain('[memory guard] Stopped')
  } finally {
    child.kill('SIGKILL')
  }
})

test.skipIf(!memoryGuardSupported())('the command tool tells the agent its command was stopped by the guard', async () => {
  const { executeCommand } = await import('../src/tools/processTools.js')
  const { WorkspacePathResolver } = await import('../src/tools/pathSafety.js')
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-memory-guard-tool-'))
  const ports = hostMemoryGuardPorts()
  const before = new Set((await ports.listProcesses()).filter(row => row.ppid === process.pid).map(row => row.pid))
  // Watch only children that appear after this point: the command under test.
  const guard = new MemoryGuard({ rootPid: process.pid, limitBytes: () => 150 * 1024 ** 2, ports, graceMs: 500, exclude: () => before })
  const stopWatching = guard.start(200)
  try {
    const result = await executeCommand({
      cmd: process.execPath,
      args: ['-e', 'const keep = Buffer.alloc(300 * 1024 * 1024, 1); setInterval(() => keep[0]++, 50)'],
      timeout_ms: 20_000,
    }, new WorkspacePathResolver(directory)) as { exitCode: number | null; stderr: string }
    expect(result.exitCode).not.toBe(0)
    expect(result.stderr).toContain('[memory guard] Stopped')
    expect(result.stderr).toContain('run less at once')
  } finally {
    stopWatching()
    await rm(directory, { recursive: true, force: true })
  }
}, 30_000)
