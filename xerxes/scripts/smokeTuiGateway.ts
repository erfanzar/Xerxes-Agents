// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Integration smoke test: spawn/attach the Bun Xerxes daemon over the Unix
// socket, prove request/response + event streaming work. No provider/API key
// needed — exercises runtime.status + initialize. Run with:
//   XERXES_TUI_BUN=/path/to/bun bun scripts/smokeTuiGateway.ts
// Set XERXES_TUI_BUN_DAEMON when the UI is not colocated with this runtime.
import { resolve } from 'node:path'

import { daemonBuildIdForEntry } from '../src/daemon/sourceBuild.js'
import { GatewayClient } from '../src/ui/gatewayClient.js'
import type { AnyEvent } from '../src/ui/gatewayTypes.js'

const projectDir = process.env.SMOKE_PROJECT_DIR ?? process.cwd()
const runtimeSource = resolve(import.meta.dir, '../src')
const runtimeEntry = resolve(runtimeSource, 'cli.ts')
const expectedDaemonBuildId = await daemonBuildIdForEntry(runtimeSource, runtimeEntry)
if (!expectedDaemonBuildId) {
  throw new Error(`could not fingerprint the Bun daemon entry at ${runtimeEntry}`)
}
const client = new GatewayClient({
  projectDir,
  expectedDaemonBuildId,
  bunDaemonPath: runtimeEntry
})

const seen: string[] = []
client.on('event', (e: AnyEvent) => {
  seen.push(e.type)
  if (e.type === 'gateway.stderr') {
    return
  }
  console.log(`  event: ${e.type}`, JSON.stringify(e.payload).slice(0, 160))
})

function fail(msg: string): never {
  console.error(`SMOKE FAIL: ${msg}`)
  console.error(client.stderrSnapshot().split('\n').slice(-15).join('\n'))
  shutdown()
  process.exit(1)
}

function shutdown(): void {
  if (client.didSpawnDaemon) {
    client.kill('smoke-gateway')
  } else {
    client.close()
  }
}

try {
  console.log(`session key: ${client.sessionKey}`)
  console.log('connecting (spawning daemon if needed)…')
  await client.start()
  console.log(`connected. spawned daemon = ${client.didSpawnDaemon}`)

  const status = await client.request<Record<string, unknown>>('runtime.status')
  console.log('runtime.status:', JSON.stringify(status).slice(0, 400))
  if (!status || status.ok !== true || status.daemon_protocol !== 35 || status.runtime !== 'bun-typescript') {
    fail('runtime.status did not identify the Bun v35 daemon')
  }

  const sessionInfo = new Promise<void>(res => client.once('session.info', () => res()))
  await client.request('initialize', { session_key: client.sessionKey })
  await Promise.race([sessionInfo, new Promise<void>(res => setTimeout(res, 4000))])

  console.log(`\nevents seen: ${seen.filter(t => t !== 'gateway.stderr').join(', ') || '(none)'}`)
  console.log('SMOKE PASS')
  shutdown()
  process.exit(0)
} catch (err) {
  fail(String((err as Error)?.message ?? err))
}
