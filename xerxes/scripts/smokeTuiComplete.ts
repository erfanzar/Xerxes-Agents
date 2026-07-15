// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Live smoke for the Bun daemon completion RPC. No provider/API key is needed.
// Run with: SMOKE_PROJECT_DIR=.. bun scripts/smokeTuiComplete.ts
import { GatewayClient } from '../src/ui/gatewayClient.js'

const client = new GatewayClient({ projectDir: process.env.SMOKE_PROJECT_DIR })

try {
  await client.start()
  const slash = (await client.request('complete', { text: '/prov' })) as {
    kind: string
    completions: { label: string }[]
  }
  console.log(
    'slash /prov →',
    slash.kind,
    slash.completions.slice(0, 4).map(c => c.label)
  )
  const path = (await client.request('complete', { text: 'open ./src/ap' })) as {
    kind: string
    completions: { label: string }[]
  }
  console.log(
    'path ./src/ap →',
    path.kind,
    path.completions.slice(0, 6).map(c => c.label)
  )
  if (!slash.completions.length || path.kind !== 'path') {
    throw new Error('completion response did not contain slash and path candidates')
  }
  console.log('SMOKE PASS')
} catch (error) {
  console.error(`SMOKE FAIL: ${error instanceof Error ? error.message : String(error)}`)
  process.exitCode = 1
} finally {
  if (client.didSpawnDaemon) {
    client.kill('smoke-complete')
  } else {
    client.close()
  }
}
