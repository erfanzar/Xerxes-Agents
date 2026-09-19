// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { spawn } from 'node:child_process'
import { expect, it, vi } from 'vitest'
import { startSshSocketTunnel } from '../lib/sshSocketTunnel.js'

function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'xr-tunnel-'))
  const processes: ReturnType<typeof processFixture>[] = []
  function processFixture() { return Object.assign(new EventEmitter(), { stderr: new EventEmitter(), kill: vi.fn() }) }
  const launch = vi.fn(() => { const child = processFixture(); processes.push(child); return child })
  const failure = vi.fn()
  const options = { ssh: ['-F', '/private/config'], target: 'saved-alias', controlPath: join(dir, 'control.sock'),
    localSocket: join(dir, 'rpc.sock'), remoteSocket: '/remote/daemon.sock', onFailure: failure,
    spawnProcess: launch as unknown as typeof spawn, startupTimeoutMs: 100 }
  return { dir, processes, launch, failure, options }
}

it('adds only the daemon socket through the owned master and keeps that master live', async () => {
  const f = fixture(); const tunnel = startSshSocketTunnel(f.options)
  try {
    expect(f.launch.mock.calls).toHaveLength(1)
    const masterArgs = (f.launch.mock.calls[0] as unknown as [string, string[]])[1]
    expect(masterArgs).toEqual(expect.arrayContaining(['-M', '-S', f.options.controlPath, 'ClearAllForwardings=yes', 'ControlPersist=no']))
    expect(masterArgs).not.toContain('-L')
    writeFileSync(f.options.controlPath, '')
    await vi.waitFor(() => expect(f.launch).toHaveBeenCalledTimes(2), { interval: 5 })
    const controlArgs = (f.launch.mock.calls[1] as unknown as [string, string[]])[1]
    expect(controlArgs).toEqual(['-F', '/dev/null', '-S', f.options.controlPath, '-o', 'ControlMaster=no', '-O', 'forward', '-L', `${f.options.localSocket}:${f.options.remoteSocket}`, '--', 'saved-alias'])
    writeFileSync(f.options.localSocket, '')
    f.processes[1]!.emit('close', 0)
    expect(f.failure).not.toHaveBeenCalled()
    expect(f.processes[0]!.kill).not.toHaveBeenCalled()
    f.processes[0]!.stderr.emit('data', 'Permission denied sensitive-sentinel')
    f.processes[0]!.emit('close', 255)
    expect(f.failure).toHaveBeenCalledOnce()
    expect(f.failure.mock.calls[0]![0].message).toContain('authentication failed')
    expect(f.failure.mock.calls[0]![0].message).not.toContain('sensitive-sentinel')
  } finally { tunnel.close(); rmSync(f.dir, { recursive: true, force: true }) }
})

it('cancellation before master readiness prevents a late control request', async () => {
  const f = fixture(); const tunnel = startSshSocketTunnel(f.options)
  try {
    tunnel.close(); writeFileSync(f.options.controlPath, '')
    await new Promise(resolve => setTimeout(resolve, 40))
    expect(f.launch).toHaveBeenCalledOnce()
    expect(f.processes[0]!.kill).toHaveBeenCalledWith('SIGTERM')
    expect(f.failure).not.toHaveBeenCalled()
  } finally { tunnel.close(); rmSync(f.dir, { recursive: true, force: true }) }
})

it('missing master readiness times out and releases the owned process', async () => {
  const f = fixture(); const tunnel = startSshSocketTunnel({ ...f.options, startupTimeoutMs: 15 })
  try {
    await vi.waitFor(() => expect(f.failure).toHaveBeenCalledOnce(), { interval: 5 })
    expect(f.failure.mock.calls[0]![0].message).toContain('timed out')
    expect(f.processes[0]!.kill).toHaveBeenCalledWith('SIGTERM')
  } finally { tunnel.close(); rmSync(f.dir, { recursive: true, force: true }) }
})

it('forward refusal closes the master without echoing diagnostics or opening another connection', async () => {
  const f = fixture(); const tunnel = startSshSocketTunnel(f.options)
  try {
    writeFileSync(f.options.controlPath, '')
    await vi.waitFor(() => expect(f.launch).toHaveBeenCalledTimes(2), { interval: 5 })
    f.processes[1]!.stderr.emit('data', 'sensitive-forward-diagnostic')
    f.processes[1]!.emit('close', 255)
    expect(f.failure).toHaveBeenCalledOnce()
    expect(f.failure.mock.calls[0]![0].message).not.toContain('sensitive-forward-diagnostic')
    expect(f.processes[0]!.kill).toHaveBeenCalledWith('SIGTERM')
    expect(f.launch).toHaveBeenCalledTimes(2)
  } finally { tunnel.close(); rmSync(f.dir, { recursive: true, force: true }) }
})
