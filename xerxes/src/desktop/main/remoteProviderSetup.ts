// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { probeRemote, quote, runCaptured, sshOptions, type RemoteRunner, type RemoteTarget } from './remote.js'
import { sshDestination } from '../../daemon/sshTarget.js'

export interface ProviderCliStatus { readonly claude: boolean; readonly codex: boolean; readonly agy: boolean }
export const AGY_REMOTE_INSTALL_URL = 'https://antigravity.google/cli/install.sh'
const PROBE = "console.log('XERXES_PROVIDER_CLIS '+JSON.stringify({claude:typeof Bun.which('claude')==='string',codex:typeof Bun.which('codex')==='string',agy:typeof Bun.which('agy')==='string'}))"
const PACKAGES = { claude: '@anthropic-ai/claude-code', codex: '@openai/codex' } as const

/** Safe POSIX-sh install command for `agy`: downloads to a temp file first, never `curl | bash`. */
export function remoteAgyInstallCommand(): string {
  return [
    'PATH="$HOME/.bun/bin:$HOME/.local/bin:$HOME/.claude/local:$HOME/.gemini/antigravity/bin:$PATH"; export PATH',
    'tmp="$(mktemp "${TMPDIR:-/tmp}/xerxes-agy-install.XXXXXX")"',
    'trap \'rm -f "$tmp"\' EXIT INT TERM',
    `curl -fsSL ${AGY_REMOTE_INSTALL_URL} -o "$tmp"`,
    'bash "$tmp"',
  ].join('; ')
}

export function parseProviderCliProbe(output: string): ProviderCliStatus {
  const line = output.split('\n').find(value => value.startsWith('XERXES_PROVIDER_CLIS '))
  if (!line) throw new Error('Provider CLI probe returned no status')
  const value = JSON.parse(line.slice('XERXES_PROVIDER_CLIS '.length)) as Record<string, unknown>
  return { claude: value.claude === true, codex: value.codex === true, agy: value.agy === true }
}

/** Install only missing provider CLIs (`claude`, `codex`, `agy`), then verify all three are executable. */
export async function ensureRemoteProviderClis(machine: RemoteTarget, signal: AbortSignal, run: RemoteRunner = runCaptured): Promise<ProviderCliStatus> {
  const initial = parseProviderCliProbe(await probeRemote(machine, PROBE, signal, run))
  if (initial.claude && initial.codex && initial.agy) return initial
  const failures: string[] = []
  for (const name of ['claude', 'codex', 'agy'] as const) {
    if (initial[name]) continue
    const command = name === 'agy'
      ? remoteAgyInstallCommand()
      : `PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"; export PATH; bun install -g ${PACKAGES[name]}`
    try {
      await run('ssh', [...sshOptions, '-T', '--', sshDestination(machine.target), 'exec sh -c ' + quote(command)], signal, 300000)
    } catch (error) {
      failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const verified = parseProviderCliProbe(await probeRemote(machine, PROBE, signal, run))
  const missing = (['claude', 'codex', 'agy'] as const).filter(name => !verified[name])
  if (missing.length > 0) {
    const detail = failures.length > 0 ? ` (${failures.join('; ')})` : ''
    throw new Error(`Provider CLI installation did not make ${missing.join(', ')} executable${detail}`)
  }
  return verified
}
