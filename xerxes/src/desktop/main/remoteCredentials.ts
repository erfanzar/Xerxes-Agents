// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn, type ChildProcess } from 'node:child_process'

import { sshDestination } from '../../daemon/sshTarget.js'
import { quote, sshOptions, type RemoteTarget } from './remote.js'

export interface RemoteProfileInput {
  readonly name: string
  readonly base_url: string
  readonly api_key: string
  readonly model: string
  readonly provider?: string
  readonly sampling?: Record<string, unknown>
  readonly oauth_account_id?: string
  readonly isolated_auth?: boolean
}
export interface RemoteOAuthSyncInput {
  readonly kind: 'claude' | 'codex' | 'claude-code'
  readonly accountId: string
  readonly label: string
  readonly profile: RemoteProfileInput
  readonly credential: unknown
}
export interface RemoteApiKeySyncInput { readonly kind: 'api_key'; readonly profile: RemoteProfileInput }
export type RemoteProviderSyncInput = RemoteProfileInput | RemoteOAuthSyncInput | RemoteApiKeySyncInput

/** Receiver source is static: secrets exist only in the stdin JSON payload. */
export function remoteProfileReceiverSource(): string {
  return String.raw`
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
const input = await new Response(Bun.stdin.stream()).text()
if (input.length > 262144) throw new Error('profile payload too large')
const value = JSON.parse(input)
if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('invalid profile')
const oauth = ['claude', 'codex', 'claude-code'].includes(value.kind)
const profile = value.kind === 'api_key' || oauth ? value.profile : value
if (!profile || typeof profile !== 'object' || Array.isArray(profile)) throw new Error('invalid linked profile')
if (typeof profile.name !== 'string' || !profile.name.trim() || profile.name.length > 128) throw new Error('invalid profile name')
if (typeof profile.base_url !== 'string' || !profile.base_url.trim() || profile.base_url.length > 4096) throw new Error('invalid endpoint')
if (typeof profile.api_key !== 'string' || (!oauth && !profile.api_key.trim()) || profile.api_key.length > 32768) throw new Error('invalid credential')
if (typeof profile.model !== 'string' || !profile.model.trim() || profile.model.length > 512) throw new Error('invalid model')
if (profile.provider !== undefined && (typeof profile.provider !== 'string' || profile.provider.length > 128)) throw new Error('invalid provider')
if (profile.sampling !== undefined && (typeof profile.sampling !== 'object' || profile.sampling === null || Array.isArray(profile.sampling))) throw new Error('invalid sampling')
if (profile.isolated_auth !== undefined && typeof profile.isolated_auth !== 'boolean') throw new Error('invalid authentication isolation')
if (oauth) {
  if (typeof value.accountId !== 'string' || !/^[a-f0-9-]{36}$/.test(value.accountId)) throw new Error('invalid account id')
  if (!value.credential || typeof value.credential !== 'object' || Array.isArray(value.credential)) throw new Error('invalid OAuth credential')
  const directory = join(process.env.HOME, '.xerxes', 'provider-accounts', value.accountId)
  if (typeof value.label !== 'string' || !value.label.trim() || value.label.length > 80 || /[\0\r\n]/.test(value.label)) throw new Error('invalid account label')
  if (profile.oauth_account_id !== value.accountId) throw new Error('linked profile account does not match')
  const credentialDirectory = value.kind === 'codex' ? join(directory, '.codex') : join(directory, 'auth')
  const credentialFile = value.kind === 'codex' ? join(credentialDirectory, 'auth.json') : join(credentialDirectory, 'anthropic-oauth.json')
  await mkdir(credentialDirectory, { recursive: true, mode: 0o700 })
  await writeFile(credentialFile, JSON.stringify(value.kind === 'codex' ? { tokens: value.credential } : value.credential, null, 2) + '\n', { mode: 0o600 })
}
const home = join(process.env.HOME, '.xerxes')
await mkdir(home, { recursive: true, mode: 0o700 })
const file = join(home, 'profiles.json'), temp = join(home, '.profiles-credential-' + process.pid + '.tmp')
let document = { active: null, profiles: {} }
try {
  const current = JSON.parse(await readFile(file, 'utf8'))
  if (current && typeof current === 'object' && !Array.isArray(current)) {
    document = { ...current, active: typeof current.active === 'string' ? current.active : null, profiles: current.profiles && typeof current.profiles === 'object' && !Array.isArray(current.profiles) ? current.profiles : {} }
  }
} catch (error) { if (error.code !== 'ENOENT') throw error }
const existing = document.profiles[profile.name] && typeof document.profiles[profile.name] === 'object' ? document.profiles[profile.name] : {}
document.profiles[profile.name] = {
  ...existing, ...profile, name: profile.name, base_url: profile.base_url, api_key: profile.api_key, model: profile.model,
  ...(profile.provider === undefined ? {} : { provider: profile.provider }),
  ...(profile.sampling === undefined ? {} : { sampling: profile.sampling }),
}
try {
  await writeFile(temp, JSON.stringify(document, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
  await rename(temp, file)
  if (oauth) {
    const root = join(home, 'provider-accounts'), registryFile = join(root, 'accounts.json'), registryTemp = join(root, '.accounts-sync-' + process.pid + '.tmp')
    let accounts = []
    try {
      const registry = JSON.parse(await readFile(registryFile, 'utf8'))
      if (!registry || !Array.isArray(registry.accounts)) throw new Error('invalid account registry')
      accounts = registry.accounts
    } catch (error) { if (error.code !== 'ENOENT') throw error }
    const existingAccount = accounts.find(account => account.id === value.accountId)
    const account = { id: value.accountId, kind: value.kind, label: value.label, profileName: profile.name, createdAt: existingAccount?.createdAt ?? new Date().toISOString() }
    try {
      await writeFile(registryTemp, JSON.stringify({ accounts: [...accounts.filter(row => row.id !== value.accountId), account] }, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
      await rename(registryTemp, registryFile)
    } catch (error) {
      await rm(registryTemp, { force: true }).catch(() => undefined)
      throw error
    }
  }
  console.log(JSON.stringify({ ok: true, profile: profile.name }))
} catch (error) {
  await rm(temp, { force: true }).catch(() => undefined)
  throw error
}
`
}

export type SecretProcessSpawner = (binary: string, args: readonly string[]) => ChildProcess
export function spawnSecretWriter(target: RemoteTarget, payload: RemoteProviderSyncInput, signal: AbortSignal, timeoutMs = 30000, spawnImpl: SecretProcessSpawner = (binary, args) => spawn(binary, [...args], { stdio: ['pipe', 'pipe', 'pipe'] })): Promise<string> {
  if (signal.aborted) return Promise.reject(new Error('Remote credential transfer cancelled'))
  const args = [...sshOptions, '-T', '--', sshDestination(target.target), 'PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"; export PATH; exec bun -e ' + quote(remoteProfileReceiverSource())]
  return new Promise((resolve, reject) => {
    const child = spawnImpl('ssh', args)
    let output = '', settled = false
    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      child.stdin?.end()
      if (error) { child.kill('SIGKILL'); reject(error) } else resolve(output)
    }
    const abort = () => finish(new Error('Remote credential transfer cancelled'))
    const timer = setTimeout(() => finish(new Error('Remote credential transfer timed out')), timeoutMs)
    signal.addEventListener('abort', abort, { once: true })
    child.stdout?.on('data', chunk => { output = (output + String(chunk)).slice(-16384) })
    child.once('error', () => finish(new Error('Remote credential transfer failed')))
    child.once('close', code => finish(code === 0 ? undefined : new Error('Remote credential transfer failed')))
    child.stdin?.on('error', () => finish(new Error('Remote credential transfer failed')))
    child.stdin?.end(JSON.stringify(payload))
    if (signal.aborted) abort()
  })
}

/** Parse only the nonsecret receiver receipt; never return remote stdout/errors. */
export function remoteProfileReceipt(output: string): string {
  const value = JSON.parse(output) as Record<string, unknown>
  if (value.ok !== true || typeof value.profile !== 'string' || !value.profile) throw new Error('Remote credential transfer was not acknowledged')
  return value.profile
}
