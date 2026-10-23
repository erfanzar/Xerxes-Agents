// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'

import { runHubCli } from '../src/hub/cli.js'
import type { CommandResult, CommandRunner } from '../src/hub/commandRunner.js'
import { ensureLauncher, launcherScript } from '../src/hub/hubSetup.js'
import { remoteXerxesCommand, runHubRemote } from '../src/hub/hubLink.js'
import { hubPaths } from '../src/hub/paths.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'
import { disposeAll, tempDir } from './hubHarness.js'

afterEach(disposeAll)

async function cli(args: string[], home: string, environment: Record<string, string | undefined> = {}) {
  const paths = hubPaths({ XERXES_HOME: home })
  const out: string[] = []
  const err: string[] = []
  const code = await runHubCli(args, { paths, lifecycle: { paths, environment: { XERXES_HOME: home, ...environment } }, stdout: line => out.push(line), stderr: line => err.push(line) })
  return { code, out, err, paths }
}
/** Stdout of a `--json` command must be exactly one JSON object. */
function oneObject(out: string[]): Record<string, unknown> {
  expect(out).toHaveLength(1)
  const value: unknown = JSON.parse(out[0]!)
  expect(value !== null && typeof value === 'object' && !Array.isArray(value)).toBe(true)
  return value as Record<string, unknown>
}

describe('hub CLI --json', () => {
  test('info prints exactly one object with the Hub key and origin', async () => {
    const home = await tempDir('xh-cli-')
    const first = await cli(['info', '--json'], home, { XERXES_HUB_PUBLIC_ORIGINS: 'https://hub.tail1.ts.net' })
    expect(first.code).toBe(0)
    const info = oneObject(first.out)
    expect(Object.keys(info).sort()).toEqual(['fingerprint', 'hub_id', 'public_key_pem', 'public_origin', 'version'])
    expect(info.public_origin).toBe('https://hub.tail1.ts.net')
    expect(String(info.public_key_pem)).toContain('BEGIN PUBLIC KEY')
    expect(String(info.hub_id)).toBe(`hub-${String(info.fingerprint).slice(0, 16)}`)
    // No origin configured: null, never a guess. Same key on every call.
    const second = oneObject((await cli(['info', '--json'], home)).out)
    expect(second.public_origin).toBeNull()
    expect(second.hub_id).toBe(info.hub_id)
    expect(JSON.stringify(info)).not.toContain('PRIVATE')
  })

  test('machines add-local is idempotent by socket, replaces per-project entries and lists machines', async () => {
    const home = await tempDir('xh-cli-')
    const paths = hubPaths({ XERXES_HOME: home })
    const registry = new WorkspaceRegistry(paths.workspacesPath)
    let revision = (await registry.load()).revision
    for (const dir of ['/srv/a', '/srv/b']) {
      revision = (await registry.add({ label: dir, projectDir: dir, endpoint: { kind: 'local', socketPath: '/tmp/tunnel.sock' }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, revision)).revision
    }
    revision = (await registry.add({ label: 'other', projectDir: '/srv/c', endpoint: { kind: 'local', socketPath: '/tmp/other.sock' }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, revision)).revision

    const added = oneObject((await cli(['machines', 'add-local', 'Mac', '--socket', '/tmp/tunnel.sock', '--json'], home)).out)
    expect(added).toMatchObject({ created: true })
    expect((added.removed_workspace_ids as string[])).toHaveLength(2)
    const again = oneObject((await cli(['machines', 'add-local', 'Mac renamed', '--socket', '/tmp/tunnel.sock', '--json'], home)).out)
    expect(again).toMatchObject({ created: false, machine_id: added.machine_id, removed_workspace_ids: [] })

    const rows = (await registry.load()).workspaces
    expect(rows).toHaveLength(2) // the machine plus the entry on another socket
    const machine = rows.find(row => row.id === added.machine_id)!
    expect(machine).toMatchObject({ label: 'Mac renamed', projectDir: '/', connection: { autoConnect: true, discover: true } })

    const listed = oneObject((await cli(['machines', 'list', '--json'], home)).out)
    const machines = listed.machines as Array<Record<string, unknown>>
    expect(machines).toHaveLength(2)
    expect(machines.find(row => row.id === added.machine_id)).toMatchObject({ label: 'Mac renamed', endpoint_kind: 'local', discover: true, projects: [] })
    // The Hub is not running: nothing is claimed to be online.
    const legacy = machines.find(row => row.discover === false)!
    expect((legacy.projects as Array<Record<string, unknown>>)[0]).toMatchObject({ project_dir: '/srv/c', health: 'offline' })

    const removed = oneObject((await cli(['machines', 'remove', String(added.machine_id), '--json'], home)).out)
    expect(removed).toEqual({ removed: added.machine_id })
    expect((await registry.load()).workspaces.map(row => row.id)).not.toContain(added.machine_id)
    expect((await cli(['machines', 'add-local', 'NoSocket', '--json'], home)).code).toBe(1)
  })

  test('machines add-route / remove-route keep fallback routes on one machine, idempotently, one JSON object each', async () => {
    const home = await tempDir('xh-cli-')
    const added = oneObject((await cli(['machines', 'add-local', 'TPU', '--socket', '/tmp/tpu-tunnel.sock', '--json'], home)).out)
    const id = String(added.machine_id)
    const hub = ['--alias', 'tpu', '--remote-socket', '/home/erfan/.xerxes/daemon/global.sock']
    const first = oneObject((await cli(['machines', 'add-route', id, ...hub, '--json'], home)).out)
    expect(first).toMatchObject({ machine_id: id, added: true, routes: [{ via: 'hub', endpoint: { kind: 'ssh', sshAlias: 'tpu', remoteSocketPath: '/home/erfan/.xerxes/daemon/global.sock' } }] })
    expect(oneObject((await cli(['machines', 'add-route', id, ...hub, '--json'], home)).out)).toMatchObject({ added: false })
    const relayed = oneObject((await cli(['machines', 'add-route', id, '--via', 'mac', '--socket', '/tmp/mac-relay.sock', '--json'], home)).out)
    expect((relayed.routes as Array<{ via: string }>).map(route => route.via)).toEqual(['hub', 'mac'])
    // Unsafe aliases and a route on an unknown machine are refused.
    expect((await cli(['machines', 'add-route', id, '--alias', '-oProxyCommand=x', '--remote-socket', '/r.sock', '--json'], home)).code).toBe(1)
    expect((await cli(['machines', 'add-route', 'nope', ...hub, '--json'], home)).code).toBe(1)
    expect((await cli(['machines', 'add-route', id, '--alias', 'tpu', '--json'], home)).code).toBe(1)
    const removed = oneObject((await cli(['machines', 'remove-route', id, ...hub, '--json'], home)).out)
    expect((removed.routes as Array<{ via: string }>).map(route => route.via)).toEqual(['mac'])
  })

  test('trust needs a target and a socket that a daemon listens on', async () => {
    const home = await tempDir('xh-cli-')
    expect((await cli(['trust', 'xerxes@hub'], home)).code).toBe(1)
    const missing = await cli(['trust', 'xerxes@hub', '--socket', join(home, 'none.sock'), '--json'], home)
    expect(missing.code).toBe(1)
    expect(missing.out).toEqual([])
    expect(missing.err.join('\n')).toContain('No Xerxes daemon is listening')
  })

  test('pair --json carries the pairing, comparison code and QR; owner-less Hubs say so on stderr', async () => {
    const home = await tempDir('xh-cli-')
    const env = { XERXES_HUB_PUBLIC_ORIGINS: 'https://hub.test' }
    const none = await cli(['pair', '--json'], home, env)
    expect(none.code).toBe(1)
    expect(none.out).toEqual([])
    expect((await cli(['owner', 'init', '--name', 'Erfan'], home, env)).code).toBe(0)
    const paired = await cli(['pair', '--label', 'Desktop', '--no-confirm', '--json'], home, env)
    expect(paired.code).toBe(0)
    const body = oneObject(paired.out)
    expect(Object.keys(body).sort()).toEqual(['comparison_code', 'confirm_required', 'expires_at', 'pairing_id', 'qr_svg', 'url'])
    expect(body.confirm_required).toBe(false)
    expect(String(body.url)).toContain('https://hub.test/pair#')
    expect(String(body.comparison_code)).toMatch(/^\d{6}$/)
    expect(String(body.qr_svg)).toContain('<svg')
    expect(Number.isFinite(Date.parse(String(body.expires_at)))).toBe(true)
    const confirmable = oneObject((await cli(['pair', '--json'], home, env)).out)
    expect(confirmable.confirm_required).toBe(true)
  })
})

describe('hub remote and the launcher', () => {
  const fake = (handler: (argv: readonly string[]) => Partial<CommandResult>) => {
    const calls: string[][] = []
    const run: CommandRunner = async argv => { calls.push([...argv]); return { code: 0, stdout: '', stderr: '', timedOut: false, ...handler(argv) } }
    return { calls, run }
  }

  test('runs "xerxes hub <args>" over a non-interactive, never multiplexed ssh with every argument quoted', async () => {
    const runner = fake(() => ({ stdout: '{"ok":true}\n', stderr: 'note', code: 3 }))
    const result = await runHubRemote({ ssh: '/usr/bin/ssh', target: 'xerxes@hub', args: ['info', '--json', "it's"] }, runner.run)
    expect(result).toMatchObject({ code: 3, stdout: '{"ok":true}\n', stderr: 'note' })
    const argv = runner.calls[0]!
    for (const option of ['BatchMode=yes', 'ControlMaster=no', 'ControlPath=none']) expect(argv).toContain(option)
    expect(argv[argv.indexOf('--') + 1]).toBe('xerxes@hub')
    const command = argv.at(-1)!
    expect(command).toContain('"$HOME/.xerxes/bin/xerxes" \'hub\' \'info\' \'--json\' \'it\'\\\'\'s\'')
    expect(command).toContain('exec xerxes ')
    await expect(runHubRemote({ ssh: '/usr/bin/ssh', target: '-oProxyCommand=x', args: ['info'] }, runner.run)).rejects.toThrow('Invalid SSH target')
    await expect(runHubRemote({ ssh: '/usr/bin/ssh', target: 'xerxes@hub', args: [] }, runner.run)).rejects.toThrow('usage')
    expect(remoteXerxesCommand(['hub', 'info'], '~/.bun/bin/bun ~/x/cli.js')).toBe("~/.bun/bin/bun ~/x/cli.js 'hub' 'info'")
  })

  test('the hub remote command relays stdout, stderr and the exit code of the remote command', async () => {
    const home = await tempDir('xh-cli-')
    const none = await cli(['remote'], home)
    expect(none.code).toBe(1)
    expect((await cli(['remote', '-oProxyCommand=x', 'info'], home)).code).toBe(1)
  })

  test('hub setup writes the launcher 0755 with the absolute bun and cli, and rewrites it only when it differs', async () => {
    const home = await tempDir('xh-launch-')
    const path = await ensureLauncher(home, '/opt/bun', "/opt/x y/cli.js")
    expect(path).toBe(join(home, '.xerxes', 'bin', 'xerxes'))
    const text = await readFile(path, 'utf8')
    expect(text).toBe(launcherScript('/opt/bun', '/opt/x y/cli.js'))
    expect(text.startsWith('#!/bin/sh\n')).toBe(true)
    expect(text).toContain("exec '/opt/bun' '/opt/x y/cli.js' \"$@\"")
    expect((await stat(path)).mode & 0o777).toBe(0o755)
    const before = (await stat(path)).mtimeMs
    await ensureLauncher(home, '/opt/bun', '/opt/x y/cli.js')
    expect((await stat(path)).mtimeMs).toBe(before)
    await ensureLauncher(home, '/opt/bun2', '/opt/cli.js')
    expect(await readFile(path, 'utf8')).toContain("exec '/opt/bun2' '/opt/cli.js'")
    expect((await stat(path)).mode & 0o777).toBe(0o755)
  })
})
