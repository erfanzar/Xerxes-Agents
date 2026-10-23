// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { ensureHubDirectories, hubPaths } from '../src/hub/paths.js'
import { getHubStatus, HubLifecycleError, runHub, startHub, stopHub, type HubHandle, type HubStatus } from '../src/hub/lifecycle.js'
import { runHubCli } from '../src/hub/cli.js'

const temporary: string[] = []
const handles: HubHandle[] = []
afterEach(async () => {
  await Promise.all(handles.splice(0).map(handle => handle.stop()))
  await Promise.all(temporary.splice(0).map(path => rm(path, { recursive: true, force: true })))
})
async function fixture() {
  const home = await mkdtemp(join(tmpdir(), 'xh-life-'))
  temporary.push(home)
  return { home, paths: hubPaths({ XERXES_HOME: home }) }
}

describe('Hub lifecycle', () => {
  test('real foreground control socket reports readiness, private modes, workspace health, and runs shutdown hooks', async () => {
    const { paths } = await fixture()
    const steps: string[] = []
    const handle = await runHub({ paths, installSignalHandlers: false, shutdownHooks: [() => { steps.push('phones') }, () => { steps.push('runtime-pool') }], workspaceHealth: () => [{ id: 'w1', health: 'unavailable' }] })
    handles.push(handle)
    const status = await getHubStatus({ paths })
    expect(status).toMatchObject({ service: 'xerxes-hub', pid: process.pid, listeners: { control: true, http: false }, workspaces: [{ id: 'w1', health: 'unavailable' }] })
    expect((await stat(paths.pidPath)).mode & 0o777).toBe(0o600)
    expect((await stat(paths.controlSocketPath)).mode & 0o777).toBe(0o600)
    expect(JSON.parse(await readFile(paths.pidPath, 'utf8')).hostname).toBeString()
    expect(await stopHub({ paths })).toEqual({ stopped: true })
    await handle.closed
    expect(steps).toEqual(['phones', 'runtime-pool'])
    expect(handle.admitting).toBe(false)
    expect(await getHubStatus({ paths })).toBeNull()
    expect(await Bun.file(paths.pidPath).exists()).toBe(false)
  })

  test('concurrent starts spawn only once and the second reports the verified live Hub', async () => {
    const { paths } = await fixture()
    let spawns = 0
    const spawn = async () => {
      spawns++
      const handle = await runHub({ paths, installSignalHandlers: false })
      handles.push(handle)
      return { pid: process.pid }
    }
    const outcomes = await Promise.all([startHub({ paths, spawn }), startHub({ paths, spawn })])
    expect(spawns).toBe(1)
    expect(outcomes.map(outcome => outcome.alreadyRunning).sort()).toEqual([false, true])
    await expect(runHub({ paths, installSignalHandlers: false })).rejects.toBeInstanceOf(HubLifecycleError)
  })

  test('a stale record naming a live non-Hub pid is reclaimed without ever signalling it or a daemon', async () => {
    const { paths } = await fixture()
    await ensureHubDirectories(paths)
    await writeFile(paths.pidPath, JSON.stringify({ pid: process.pid, started_at: '2026-01-01T00:00:00.000Z', version: '0', build_id: 'not-hub', instance_id: 'stale-instance', hostname: 'test', control_socket: paths.controlSocketPath }))
    const signals: number[] = []
    expect(await stopHub({ paths, processProbe: () => true, signal: pid => { signals.push(pid) } })).toEqual({ stopped: false })
    const handle = await runHub({ paths, installSignalHandlers: false })
    handles.push(handle)
    expect((await getHubStatus({ paths }))?.pid).toBe(process.pid)
    expect(signals).toEqual([])
  })

  test('start timeout reports the log tail, not success', async () => {
    const { paths } = await fixture()
    await expect(startHub({ paths, timeoutMs: 20, spawn: async request => {
      await writeFile(request.logPath, 'startup failed: bind denied\n', { mode: 0o600 })
      return { pid: 99999999 }
    } })).rejects.toThrow('startup failed: bind denied')
  })

  test('shutdown continues through hook failures, closes admission first and cleans ownership', async () => {
    const { paths } = await fixture()
    let handle: HubHandle
    const steps: string[] = []
    handle = await runHub({ paths, installSignalHandlers: false, shutdownHooks: [() => { expect(handle.admitting).toBe(false); steps.push('first'); throw new Error('hook failed') }, () => { steps.push('second') }] })
    handles.push(handle)
    await expect(handle.stop()).rejects.toThrow('hook failed')
    await expect(handle.closed).rejects.toThrow('hook failed')
    expect(steps).toEqual(['first', 'second'])
    expect(await getHubStatus({ paths })).toBeNull()
    expect(await Bun.file(paths.pidPath).exists()).toBe(false)
  })

  test('SIGTERM fallback revalidates the complete identity and never signals a replaced process', async () => {
    const { paths } = await fixture()
    await ensureHubDirectories(paths)
    const status: HubStatus = { service: 'xerxes-hub', pid: 12345, started_at: '2026-01-01T00:00:00.000Z', version: 'test', build_id: 'test', instance_id: 'original', hostname: 'test', listeners: { control: true, http: false }, workspaces: [] }
    const { service: _service, listeners: _listeners, workspaces: _workspaces, ...identity } = status
    await writeFile(paths.pidPath, JSON.stringify({ ...identity, control_socket: paths.controlSocketPath }))
    let now = 0, probes = 0, live = true
    const signals: number[] = [], methods: string[] = []
    const clock = { now: () => now, sleep: async (ms: number) => { now += ms } }
    const options = { paths, clock, timeoutMs: 30, processProbe: () => live, signal: (pid: number) => { signals.push(pid); live = false } }
    await expect(stopHub({ ...options, controlRequest: async (_path, method) => {
      methods.push(method)
      if (method === 'stop') return { ok: true }
      probes++
      return probes > 1 ? { ...status, instance_id: 'reused' } : status
    } })).rejects.toMatchObject({ code: 'stop_failed' })
    expect(signals).toEqual([])
    expect(methods).toEqual(['status', 'stop', 'status'])
    expect(await stopHub({ ...options, controlRequest: async (_path, method) => method === 'status' ? status : { ok: true } })).toEqual({ stopped: true })
    expect(signals).toEqual([12345])
  })

  test('real detached CLI child starts, answers status, and stops without a daemon', async () => {
    const { home, paths } = await fixture()
    // Port 0: a real Hub already running on this machine owns the default port.
    const options = { paths, environment: { ...process.env, XERXES_HOME: home, XERXES_HUB_PORT: '0' }, scriptPath: resolve(import.meta.dir, '../src/cli.ts'), timeoutMs: 10000 }
    try {
      const started = await startHub(options)
      expect(started.status.pid).not.toBe(process.pid)
      expect(started.alreadyRunning).toBe(false)
      expect((await stat(started.logPath!)).mode & 0o777).toBe(0o600)
      expect((await getHubStatus({ paths }))?.instance_id).toBe(started.status.instance_id)
      expect(await stopHub({ paths })).toEqual({ stopped: true })
      expect(await getHubStatus({ paths })).toBeNull()
    } finally { await stopHub({ paths }) }
  }, 20000)

  test('CLI status supports JSON and unknown subcommands fail with usage', async () => {
    const { paths } = await fixture()
    const output: string[] = [], errors: string[] = []
    const io = { stdout: (line: string) => { output.push(line) }, stderr: (line: string) => { errors.push(line) } }
    expect(await runHubCli(['status', '--json'], { paths, ...io })).toBe(1)
    expect(JSON.parse(output[0]!)).toEqual({ running: false })
    expect(await runHubCli(['not-a-command'], { paths, ...io })).toBe(1)
    expect(errors.join('\n')).toContain('Usage: xerxes hub')
  })
})
