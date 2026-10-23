// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { shareRemoteWorkspace, shareSlug, shareTunnelArgs, type HubShareDeps, type ShareProcess } from '../src/desktop/main/hubShare.js'
import type { CliOutcome, CliRunner } from '../src/desktop/main/hubCli.js'

interface Fake {
  readonly deps: HubShareDeps
  readonly ssh: string[][]
  readonly spawned: string[][]
  readonly cli: string[][]
  readonly logs: string[]
  readonly kills: number[]
}

function fake(options: {
  target?: string | undefined
  attach?: CliOutcome
  trust?: CliOutcome
  socketAppears?: boolean
} = {}): Fake {
  const ssh: string[][] = [], spawned: string[][] = [], cli: string[][] = [], logs: string[] = [], kills: number[] = []
  let forwarding = false
  const runner: CliRunner = {
    async run(args) {
      cli.push([...args])
      if (args.includes('attach')) return options.attach ?? { exitCode: 0, stdout: JSON.stringify({ machine_id: 'm-tpu', runtime_id: 'rt', attached: 'route', added: true }), stderr: '', timedOut: false }
      return options.trust ?? { exitCode: 0, stdout: JSON.stringify({ hub_id: 'hub-1', enrolled: true }), stderr: '', timedOut: false }
    },
  }
  const deps: HubShareDeps = {
    target: () => ('target' in options ? options.target : 'xerxes@hub'),
    async ssh(args) {
      ssh.push([...args])
      const command = args.at(-1) ?? ''
      if (command.startsWith('mkdir')) return { code: 0, stdout: '/home/xerxes\n', stderr: '' }
      if (command.startsWith('test -S')) return { code: forwarding && options.socketAppears !== false ? 0 : 1, stdout: '', stderr: '' }
      return { code: 0, stdout: '', stderr: '' }
    },
    spawnSsh(args) {
      spawned.push([...args]); forwarding = true
      let finish!: (code: number | null) => void
      const process: ShareProcess = { exited: new Promise(resolve => { finish = resolve }), kill: () => { kills.push(1); forwarding = false; finish(0) } }
      return process
    },
    cli: runner,
    log: message => { logs.push(message) },
    sleep: () => Promise.resolve(),
  }
  return { deps, ssh, spawned, cli, logs, kills }
}

test('a connected SSH workspace is relayed to the Hub, attached by runtime identity, and its daemon trusts the Hub key', async () => {
  const f = fake()
  const handle = shareRemoteWorkspace(f.deps, { label: 'TPU-workspace', localSocket: '/tmp/xd-abc/rpc.sock' })
  expect(await handle.ready).toEqual({ state: 'shared', machineId: 'm-tpu', attached: 'route', trusted: true })
  const socket = '/home/xerxes/.xerxes/tunnels/mac-relay-tpu-workspace.sock'
  expect(f.spawned[0]).toEqual(shareTunnelArgs('xerxes@hub', socket, '/tmp/xd-abc/rpc.sock'))
  expect(f.spawned[0]).toContain('ExitOnForwardFailure=yes')
  expect(f.spawned[0]).toContain('ForwardAgent=no')
  expect(f.cli[0]).toEqual(['remote', 'xerxes@hub', 'machines', 'attach', '--socket', socket, '--label', 'TPU-workspace', '--json'])
  expect(f.cli[1]).toEqual(['trust', 'xerxes@hub', '--socket', '/tmp/xd-abc/rpc.sock', '--json'])
  // A stale socket file is removed before the forward binds; closing stops it and removes it again.
  expect(f.ssh.some(args => args.at(-1) === `rm -f '${socket}'`)).toBe(true)
  const removals = f.ssh.filter(args => args.at(-1) === `rm -f '${socket}'`).length
  await handle.close()
  expect(f.kills.length).toBe(1)
  expect(f.ssh.filter(args => args.at(-1) === `rm -f '${socket}'`).length).toBe(removals + 1)
  // Nothing that looks like a credential crosses to the Hub.
  expect(JSON.stringify([f.ssh, f.spawned, f.cli])).not.toMatch(/BEGIN|password|api[_-]?key|token/i)
})

test('without a saved Hub nothing is attempted', async () => {
  const f = fake({ target: undefined })
  const handle = shareRemoteWorkspace(f.deps, { label: 'TPU', localSocket: '/tmp/x/rpc.sock' })
  expect((await handle.ready).state).toBe('skipped')
  expect(f.ssh).toEqual([])
  expect(f.spawned).toEqual([])
  await handle.close()
})

test('an unsafe Hub target or local socket is refused before any process starts', async () => {
  for (const [target, socket] of [['-oProxyCommand=x', '/tmp/x/rpc.sock'], ['xerxes@hub', 'relative.sock'], ['xerxes@hub', '/tmp/a:b.sock']] as const) {
    const f = fake({ target })
    const outcome = await shareRemoteWorkspace(f.deps, { label: 'TPU', localSocket: socket }).ready
    expect(outcome.state === 'skipped' || outcome.state === 'failed').toBe(true)
    expect(f.spawned).toEqual([])
  }
})

test('a refused attach is a logged failure, and a daemon that cannot trust the Hub is reported as read only', async () => {
  const refused = fake({ attach: { exitCode: 1, stdout: '', stderr: 'unknown command attach\n', timedOut: false } })
  const failed = await shareRemoteWorkspace(refused.deps, { label: 'TPU', localSocket: '/tmp/x/rpc.sock' }).ready
  expect(failed.state).toBe('failed')
  expect(refused.logs.join('\n')).toContain('unknown command attach')
  const untrusted = fake({ trust: { exitCode: 1, stdout: '', stderr: 'old daemon\n', timedOut: false } })
  const shared = await shareRemoteWorkspace(untrusted.deps, { label: 'TPU', localSocket: '/tmp/x/rpc.sock' }).ready
  expect(shared).toMatchObject({ state: 'shared', trusted: false })
  expect(untrusted.logs.join('\n')).toContain('can only read it')
})

test('a forward that never appears on the Hub host fails without attaching', async () => {
  const f = fake({ socketAppears: false })
  const handle = shareRemoteWorkspace(f.deps, { label: 'TPU', localSocket: '/tmp/x/rpc.sock' })
  expect((await handle.ready).state).toBe('failed')
  expect(f.cli).toEqual([])
  await handle.close()
})

test('labels become safe socket names', () => {
  expect(shareSlug('TPU workspace / EasyDeL')).toBe('tpu-workspace-easydel')
  expect(shareSlug('***')).toBe('remote')
})
