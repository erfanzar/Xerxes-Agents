// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { probeRemote, quote, runCaptured, sshOptions, type RemoteRunner, type RemoteTarget } from './remote.js'
import { sshDestination } from '../../daemon/sshTarget.js'

export interface ProviderCliStatus { readonly claude: boolean; readonly codex: boolean }
const PROBE = "console.log('XERXES_PROVIDER_CLIS '+JSON.stringify({claude:typeof Bun.which('claude')==='string',codex:typeof Bun.which('codex')==='string'}))"
const PACKAGES = { claude: '@anthropic-ai/claude-code', codex: '@openai/codex' } as const

export function parseProviderCliProbe(output: string): ProviderCliStatus {
  const line = output.split('\n').find(value => value.startsWith('XERXES_PROVIDER_CLIS '))
  if (!line) throw new Error('Provider CLI probe returned no status')
  const value = JSON.parse(line.slice('XERXES_PROVIDER_CLIS '.length)) as Record<string, unknown>
  return { claude: value.claude === true, codex: value.codex === true }
}

/** Install only missing provider CLIs, then verify both are executable. */
export async function ensureRemoteProviderClis(machine: RemoteTarget, signal: AbortSignal, run: RemoteRunner = runCaptured): Promise<ProviderCliStatus> {
  const initial = parseProviderCliProbe(await probeRemote(machine, PROBE, signal, run))
  for (const name of ['claude', 'codex'] as const) {
    if (initial[name]) continue
    const command = `PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"; export PATH; bun install -g ${PACKAGES[name]}`
    await run('ssh', [...sshOptions, '-T', '--', sshDestination(machine.target), 'exec sh -c ' + quote(command)], signal, 300000)
  }
  const verified = parseProviderCliProbe(await probeRemote(machine, PROBE, signal, run))
  if (!verified.claude || !verified.codex) throw new Error('Provider CLI installation did not make both claude and codex executable')
  return verified
}
