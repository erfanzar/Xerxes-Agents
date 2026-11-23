// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import { createHubController } from '../src/client/hubController.js'
import { createHubStore } from '../src/client/hubStore.js'
import type { HubConnection } from '../src/client/hubConnection.js'
import { createHubCockpit } from '../src/desktop/main/hubCockpit.js'
import { runHubIdentityCli } from '../src/hub/identity/cli.js'
import { cleanupAll, tempPaths } from './hubIdentityHarness.js'

afterEach(cleanupAll)

test('xerxes hub devices list/revoke support --json output cleanly', async () => {
  const { paths } = tempPaths()
  const run = async (args: string[]) => {
    const stdout: string[] = []
    const stderr: string[] = []
    const code = await runHubIdentityCli(args, {
      paths,
      environment: { XERXES_HUB_PUBLIC_ORIGINS: 'https://hub.example.ts.net' },
      out: line => stdout.push(line),
      err: line => stderr.push(line),
    })
    return { code, stdout: stdout.join('\n'), stderr: stderr.join('\n') }
  }

  expect((await run(['owner', 'init', '--name', 'Erfan'])).code).toBe(0)
  const listedEmpty = await run(['devices', 'list', '--json'])
  expect(listedEmpty.code, listedEmpty.stderr).toBe(0)
  expect(JSON.parse(listedEmpty.stdout)).toEqual({ devices: [] })
})

test('hubCockpit lists and revokes paired devices via validated remote CLI calls', async () => {
  const ran: string[][] = []
  let saved = {
    target: 'xerxes@203.0.113.10',
    origin: 'https://hub.example.ts.net',
    fingerprint: 'SHA256:abc',
    hubVersion: '0.8.52',
    linkedMachineId: 'm1',
  }
  const cockpit = createHubCockpit({
    runner: {
      run: async args => {
        ran.push([...args])
        if (args.includes('list')) {
          return {
            exitCode: 0,
            stdout: JSON.stringify({
              devices: [{
                id: 'dev_phone1',
                user_id: 'usr_1',
                display_name: 'Erfan',
                label: 'iPhone',
                status: 'active',
                created_at: '2026-10-20T00:00:00Z',
                last_seen_at: '2026-10-21T10:00:00Z',
                revoked_at: null,
              }],
            }),
            stderr: '',
            timedOut: false,
          }
        }
        return { exitCode: 0, stdout: JSON.stringify({ ok: true, device_id: 'dev_phone1' }), stderr: '', timedOut: false }
      },
    },
    http: async () => ({ status: 200, body: '{}', setCookies: [] }),
    connect: () => { throw new Error('no socket') },
    storage: {
      loadSettings: () => saved,
      saveSettings: next => { saved = next as typeof saved },
      loadSession: async () => undefined,
      saveSession: async () => {},
    },
    emit: () => {},
  })

  const listed = await cockpit.handle('devices', {}, 1)
  expect(listed).toMatchObject({ ok: true, devices: [{ id: 'dev_phone1', label: 'iPhone', status: 'active' }] })
  expect(ran[0]).toEqual(['remote', 'xerxes@203.0.113.10', 'devices', 'list', '--json'])

  const revoked = await cockpit.handle('revokeDevice', { deviceId: 'dev_phone1' }, 1)
  expect(revoked).toEqual({ ok: true, deviceId: 'dev_phone1' })
  expect(ran[1]).toEqual(['remote', 'xerxes@203.0.113.10', 'devices', 'revoke', 'dev_phone1', '--json'])

  const badId = await cockpit.handle('revokeDevice', { deviceId: '../bad' }, 1)
  expect(badId.ok).toBe(false)
})

test('HubController createTask opens a session, submits optional prompt, and manages paired devices', async () => {
  const store = createHubStore()
  store.dispatch({ type: 'connection', status: { state: 'live' } })
  store.dispatch({ type: 'workspaces', list: [{ id: 'w1', label: 'Mac · app', health: 'online', models: [], efforts: [] }] })

  const commands: unknown[] = []
  const requests: [string, unknown][] = []
  const conn = {
    status: () => ({ state: 'live' as const }),
    helloInfo: () => ({ userId: 'usr_1' }),
    connect() {},
    close() {},
    reconnectNow() {},
    onStatus: () => () => {},
    onNotice: () => () => {},
    subscribe: () => ({ dispose() {}, cursor: () => undefined }),
    watermark: async () => 'c0',
    request: async (method: string, params?: unknown) => {
      requests.push([method, params])
      if (method === 'hub.devices.list') {
        return {
          can_revoke_all: true,
          devices: [{ id: 'dev_a', user_id: 'usr_1', display_name: 'Erfan', label: 'Phone', status: 'active', current: true }],
        }
      }
      if (method === 'hub.devices.revoke') return { ok: true, device_id: 'dev_b' }
      if (method === 'hub.workspaces.list') return { workspaces: [{ id: 'w1', label: 'Mac · app', health: 'online' }] }
      if (method === 'hub.snapshot') return { workspaces: [] }
      return {}
    },
    command: async (input: { action: string; sessionId?: string; args?: Record<string, unknown> }) => {
      commands.push(input)
      if (input.action === 'session.open') {
        return { ok: true, command_id: 'cmd-1', state: 'accepted', result: { session_id: 'sess-99' } }
      }
      return { ok: true, command_id: 'cmd-2', state: 'accepted', result: {} }
    },
  } as unknown as HubConnection

  const controller = createHubController({ connection: conn, store })
  const created = await controller.createTask!('w1', 'Fix the build')
  expect(created).toEqual({ ok: true, workspaceId: 'w1', sessionId: 'sess-99' })
  expect(commands).toHaveLength(2)
  expect((commands[0] as { action: string }).action).toBe('session.open')
  expect((commands[1] as { action: string; sessionId: string; args: { text: string } })).toMatchObject({
    action: 'turn.submit',
    sessionId: 'sess-99',
    args: { text: 'Fix the build' },
  })

  await controller.loadDevices!()
  expect(store.getSnapshot().devices).toMatchObject({
    phase: 'ready',
    canRevokeAll: true,
    list: [{ id: 'dev_a', label: 'Phone', current: true }],
  })
  expect(await controller.revokeDevice!('dev_b')).toBe(true)
  expect(requests.some(entry => entry[0] === 'hub.devices.revoke')).toBe(true)
})
