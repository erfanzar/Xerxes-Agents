// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Live smoke for the daemon `complete` RPC. Run:
//   XERXES_PYTHON=../.venv/bin/python SMOKE_PROJECT_DIR=.. npx tsx scripts/smoke-complete.ts
import { GatewayClient } from '../src/gatewayClient.js'

const client = new GatewayClient({ projectDir: process.env.SMOKE_PROJECT_DIR })
await client.start()
const slash = (await client.request('complete', { text: '/prov' })) as { kind: string; completions: { label: string }[] }
console.log('slash /prov →', slash.kind, slash.completions.slice(0, 4).map(c => c.label))
const path = (await client.request('complete', { text: 'open ./src/ap' })) as { kind: string; completions: { label: string }[] }
console.log('path ./src/ap →', path.kind, path.completions.slice(0, 6).map(c => c.label))
console.log(slash.completions.length && path.kind === 'path' ? 'SMOKE PASS' : 'SMOKE FAIL')
if (client.didSpawnDaemon) {
  client.kill('smoke-complete')
} else {
  client.close()
}
process.exit(0)
