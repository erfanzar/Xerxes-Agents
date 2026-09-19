// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Opt-in real-process acceptance. Requires bun run build:runtime first.
import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { DaemonRpc } from '../../../src/desktop/main/daemon.js'
import { daemonAddress } from '../../../src/desktop/main/spawn.js'
import { requestDaemonControl } from '../../../src/daemon/controlClient.js'
import { GatewayClient } from '../../../src/ui/gatewayClient.js'
import { prepareManagedRuntime } from '../../../src/ui/lib/managedRuntime.js'

const root = await mkdtemp('/tmp/x-runtime-update-')
const oldBuild = 'aaaaaaaaaaaaaaaa', newBuild = 'bbbbbbbbbbbbbbbb'
const cli = await Bun.file(new URL('../../../dist/cli.js', import.meta.url)).arrayBuffer()
for (const [release, build] of [['old', oldBuild], ['new', newBuild]]) {
  await mkdir(join(root, release!))
  await Bun.write(join(root, release!, 'cli.js'), cli)
  await Bun.write(join(root, release!, 'build-id'), build!)
}
try {
  for (const mode of ['local', 'managed-remote']) {
    const home = join(root, mode), workspace = join(home, 'workspace')
    await mkdir(workspace, { recursive: true })
    process.env.XERXES_HOME = home
    delete process.env.XERXES_DAEMON_SOCKET
    delete process.env.XERXES_EXPECTED_DAEMON_BUILD_ID
    const env = { ...process.env, XERXES_BUN_DAEMON: join(root, 'new/cli.js'), XERXES_TUI_BUN_DAEMON: join(root, 'new/cli.js'), XERXES_BUN: process.execPath }
    const address = daemonAddress(workspace, env)
    const child = Bun.spawn([process.execPath, join(root, 'old/cli.js'), 'daemon', '--project-dir', workspace, '--socket', address.socketPath, '--pid-file', address.pidPath], { env, stdout: 'ignore', stderr: 'ignore' })
    const rpc = new DaemonRpc({ projectDir: workspace, env })
    try {
      const deadline = Date.now() + 15000
      for (;;) {
        try { await requestDaemonControl(address.socketPath, 'runtime.status', {}, { timeoutMs: 500 }); break }
        catch (error) { if (Date.now() >= deadline) throw error; await Bun.sleep(50) }
      }
      const before = await rpc.call<Record<string, unknown>>('runtime.status')
      if (before.daemon_build_id !== oldBuild) throw new Error('Old build did not start')
      const opened = await rpc.call<{ session: { id: string } }>('initialize', { session_key: 'preserved', history_limit: 100 })
      if (mode === 'local') {
        const updated = await rpc.restartRuntime()
        if (updated.ok !== true) throw new Error(JSON.stringify(updated))
      } else {
        rpc.dispose()
        const updated = await prepareManagedRuntime(verify => new GatewayClient({ projectDir: workspace, bunBinary: process.execPath,
          bunDaemonPath: join(root, 'new/cli.js'), ...(verify ? { expectedDaemonBuildId: newBuild } : {}),
        }), newBuild, async () => { await child.exited })
        if (updated.busy) throw new Error('Unexpected busy runtime')
      }
      const check = new DaemonRpc({ projectDir: workspace, env })
      try {
        const after = await check.call<Record<string, unknown>>('runtime.status')
        const resumed = await check.call<{ session: { id: string } }>('initialize', { resume_session_id: opened.session.id, history_limit: 100 })
        if (after.daemon_build_id !== newBuild || after.pid === before.pid || resumed.session.id !== opened.session.id) throw new Error('Build replacement or session preservation failed')
        console.log(`PASS ${mode}: ${oldBuild} -> ${newBuild}; new PID; original session preserved`)
      } finally { check.dispose() }
    } finally {
      rpc.dispose()
      await requestDaemonControl(address.socketPath, 'shutdown', {}).catch(() => {})
      if (child.exitCode === null) child.kill()
      await Bun.sleep(250)
    }
  }
} finally { await rm(root, { recursive: true, force: true }) }
