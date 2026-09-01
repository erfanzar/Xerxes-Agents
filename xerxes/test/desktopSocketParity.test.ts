// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { daemonAddress } from '../src/desktop/main/spawn.js'
import { daemonPaths as gatewayPaths } from '../src/ui/gatewayClient.js'

// Byte-for-byte parity between the desktop's address derivation and the TUI
// gateway client's. A one-character drift makes the app find no daemon and
// launch a second one on a socket nobody else will ever use.

const PROJECT = '/fixture/some-project'
const DIGEST = createHash('sha256').update(PROJECT, 'utf8').digest('hex').slice(0, 16)

let savedSocket: string | undefined
let savedHome: string | undefined

beforeEach(() => {
  savedSocket = process.env.XERXES_DAEMON_SOCKET
  savedHome = process.env.XERXES_HOME
  delete process.env.XERXES_DAEMON_SOCKET
  delete process.env.XERXES_HOME
})

afterEach(() => {
  if (savedSocket === undefined) delete process.env.XERXES_DAEMON_SOCKET
  else process.env.XERXES_DAEMON_SOCKET = savedSocket
  if (savedHome === undefined) delete process.env.XERXES_HOME
  else process.env.XERXES_HOME = savedHome
})

test('posix default lands beside the pid file under ~/.xerxes/daemon/projects', () => {
  const address = daemonAddress(PROJECT)
  expect(address.socketPath).toBe(join(homedir(), '.xerxes', 'daemon', 'projects', `${DIGEST}.sock`))
  expect(address.pidPath).toBe(join(homedir(), '.xerxes', 'daemon', 'projects', `${DIGEST}.pid`))
})

test('desktop derivation equals the TUI gateway client on every platform', () => {
  for (const platform of ['darwin', 'linux', 'win32'] as const) {
    const desktop = daemonAddress(PROJECT, undefined, platform)
    const gateway = gatewayPaths(PROJECT, platform)
    expect({ platform, ...desktop }).toEqual({ platform, ...gateway })
  }
})

test('windows resolves a named pipe; the pid file stays on disk', () => {
  const address = daemonAddress(PROJECT, undefined, 'win32')
  expect(address.socketPath).toBe(`\\\\.\\pipe\\xerxes-${DIGEST}`)
  expect(address.pidPath.endsWith(`${DIGEST}.pid`)).toBe(true)
})

test('XERXES_HOME and XERXES_DAEMON_SOCKET behave identically on both sides', () => {
  process.env.XERXES_HOME = '/tmp/alt-home'
  process.env.XERXES_DAEMON_SOCKET = '  /tmp/override.sock  '
  for (const platform of ['darwin', 'linux', 'win32'] as const) {
    expect({ platform, ...daemonAddress(PROJECT, undefined, platform) }).toEqual({
      platform,
      ...gatewayPaths(PROJECT, platform),
    })
  }
  expect(daemonAddress(PROJECT).socketPath).toBe('/tmp/override.sock')
  expect(daemonAddress(PROJECT).pidPath).toBe(join('/tmp/alt-home', 'daemon', 'projects', `${DIGEST}.pid`))
})
