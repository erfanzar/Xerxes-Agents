// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import {
  desktopCall,
  desktopError,
  diffSections,
  specialistsOf,
} from '../src/desktop/renderer/desktopRpc.js'
import { remoteTarget, remoteAddress, runCaptured } from '../src/desktop/main/remote.js'
import type { XerxesBridge } from '../src/desktop/renderer/types.js'

test('RPC panels preserve session scope and surface daemon failures', async () => {
  const calls: unknown[] = []
  const bridge: Pick<XerxesBridge, 'call'> = {
    call: async <T>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })
      return { ok: false, error: 'revision changed' } as T
    },
  }
  await expect(
    desktopCall(bridge, 's1', 'agentPreset.projectWrite', { id: 'reviewer', revision: 'r1' }),
  ).rejects.toThrow('revision changed')
  expect(calls).toEqual([
    {
      method: 'agentPreset.projectWrite',
      params: { id: 'reviewer', revision: 'r1', session_key: 's1' },
    },
  ])
})
test('untracked and tracked previews are indexed by daemon file headers', () => {
  const sections = diffSections({
    files: 2,
    untracked: ['new file.ts'],
    lines: [
      { kind: 'file', text: 'tracked.ts' },
      { kind: 'del', text: 'old' },
      { kind: 'file', text: 'new file.ts' },
      { kind: 'add', text: 'new' },
    ],
  })
  expect(sections).toEqual([
    { path: 'tracked.ts', start: 0, end: 2, untracked: false },
    { path: 'new file.ts', start: 2, end: 4, untracked: true },
  ])
  expect(diffSections({})).toEqual([])
  expect(() => diffSections({ lines: {} })).toThrow('Invalid daemon list response')
})
test('specialist discovery does not turn broken records into an empty catalog', () => {
  expect(
    specialistsOf([{ id: 'reviewer', description: 'Reviews', error: 'invalid yaml' }]),
  ).toEqual([{ id: 'reviewer', description: 'Reviews', error: 'invalid yaml' }])
  expect(() => specialistsOf([{ name: 'reviewer' }])).toThrow('Invalid specialist record')
})
test('remote boundary rejects option injection, controls and invalid socket forwarding', () => {
  const machine = { alias: 'gpu', target: 'me@host', workspacePath: '/home/me/project with spaces' }
  expect(remoteTarget(machine)).toEqual(machine)
  for (const target of ['-oProxyCommand=evil', 'host;echo bad', 'me@host\nother', 'host:22'])
    expect(() => remoteTarget({ ...machine, target })).toThrow()
  expect(() => remoteTarget({ ...machine, workspacePath: '/tmp/x\n' })).toThrow()
  expect(
    remoteAddress(
      'setup logs\nXERXES_REMOTE_READY {"socketPath":"/tmp/daemon.sock","projectDir":"/repo"}\n',
    ),
  ).toEqual({ socketPath: '/tmp/daemon.sock', projectDir: '/repo' })
  expect(() =>
    remoteAddress('XERXES_REMOTE_READY {"socketPath":"/tmp/x:22","projectDir":"/repo"}'),
  ).toThrow()
  expect(() => remoteAddress('setup failed')).toThrow()
})
test('remote subprocess calls report exit failure, cancellation, and deadline', async () => {
  const controller = new AbortController()
  await expect(
    runCaptured(
      process.execPath,
      ['-e', 'console.error("test failure"); process.exit(2)'],
      controller.signal,
      3000,
    ),
  ).rejects.toThrow('test failure')
  const pending = runCaptured(
    process.execPath,
    ['-e', 'setTimeout(()=>{},10000)'],
    controller.signal,
    3000,
  )
  controller.abort()
  await expect(pending).rejects.toThrow('cancelled')
  await expect(
    runCaptured(
      process.execPath,
      ['-e', 'setTimeout(()=>{},10000)'],
      new AbortController().signal,
      20,
    ),
  ).rejects.toThrow('timed out')
})

test('dictation uses explicit credentials and forwards audio only through its port', async () => {
  const { dictationPort, transcribeDictation } = await import('../src/desktop/main/voice.js')
  expect(() => dictationPort({})).toThrow('XERXES_DICTATION_BASE_URL')
  const seen: unknown[] = []
  const port = {
    providerName: 'test',
    async transcribe(request: unknown, signal?: AbortSignal) {
      seen.push(request, signal)
      return { backend: 'test', model: 'test', text: 'Review this change.' }
    },
  }
  const signal = new AbortController().signal
  expect(
    await transcribeDictation(
      { bytes: new Uint8Array([1, 2]), mediaType: 'audio/webm;codecs=opus' },
      port,
      signal,
    ),
  ).toBe('Review this change.')
  expect(seen[1]).toBe(signal)
  await expect(
    transcribeDictation({ bytes: new Uint8Array(), mediaType: 'audio/webm' }, port, signal),
  ).rejects.toThrow('Recording must contain')
  await expect(
    transcribeDictation({ bytes: new Uint8Array([1]), mediaType: 'text/html' }, port, signal),
  ).rejects.toThrow('Unsupported')
})

test('native failures retain actionable messages without IPC internals', () => {
  expect(
    desktopError(
      new Error(
        "Error invoking remote method 'desktop:voice': Error: Configure a transcription provider",
      ),
    ),
  ).toBe('Configure a transcription provider')
  expect(desktopError(new Error('Permission denied'))).toBe('Permission denied')
  expect(desktopError(new Error('rpc -32000: expected 5-field cron expression'))).toBe('expected 5-field cron expression')
})
