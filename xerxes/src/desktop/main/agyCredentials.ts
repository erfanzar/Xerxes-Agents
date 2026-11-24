// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access, constants, readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { loadAgyRuntimeEnv } from '../../auth/agyRuntimeEnv.js'
import type { ProviderAccountRegistry } from '../../auth/providerAccountRegistry.js'
import { sshDestination } from '../../daemon/sshTarget.js'
import { cliSearchDirectories } from './localProviderSetup.js'
import { quote, sshOptions, type RemoteTarget } from './remote.js'
import type { SecretProcessSpawner } from './remoteCredentials.js'

/**
 * Antigravity CLI (`agy`) local inspection and SSH credential transfer.
 *
 * Mirrors Xerxes's `remoteCredentials.ts` security model:
 * - No Electron imports (pure main-process module, directly unit-tested).
 * - Renderer receives ONLY secret-free metadata (`AgyLocalStatus`).
 * - Secrets travel exclusively over SSH stdin to a static Bun receiver, never
 *   in argv, shell strings, logs, stdout, stderr, or IPC replies.
 * - Remote files are written atomically (temp + rename) with mode 0600,
 *   preserving unrelated keys in `~/.gemini/config/config.json` and
 *   `~/.gemini/antigravity-cli/settings.json`.
 *
 * Only credentials the user owns independently of Antigravity (Google Cloud
 * ADC, a Gemini API key) are transferable. Antigravity's own Google sign-in
 * (`keyring`) is reported for status but never read, exported, or planted on
 * another host: each host signs in through `agy` itself (`agy acp --login`).
 */

export type AgyCredentialMode = 'adc' | 'gemini_api_key' | 'keyring'
/** Credential sources Xerxes may copy to an SSH host; Antigravity's own sign-in is excluded. */
export type AgyTransferMode = Exclude<AgyCredentialMode, 'keyring'>

export interface AgyCredentialSourceStatus {
  readonly mode: AgyCredentialMode
  readonly label: string
  readonly available: boolean
  readonly detail: string
  readonly headlessSafe: boolean
  /** False for Antigravity's own sign-in, which stays on the host where `agy` created it. */
  readonly transferable: boolean
  /** Saved Gemini account id if `gemini_api_key` comes from the Xerxes account registry. */
  readonly accountId?: string
}

export interface AgyLocalStatus {
  readonly agyInstalled: boolean
  readonly sources: readonly AgyCredentialSourceStatus[]
  readonly recommendedMode: AgyTransferMode | null
  readonly canSync: boolean
  readonly destination: string | null
  readonly syncedDestination: string | null
  readonly syncedMode: AgyTransferMode | null
}

export interface AgyAdcSyncPayload {
  readonly kind: 'agy'
  readonly mode: 'adc'
  readonly adc: Record<string, unknown>
  readonly quotaProject?: string
  readonly location?: string
}

export interface AgyGeminiApiKeySyncPayload {
  readonly kind: 'agy'
  readonly mode: 'gemini_api_key'
  readonly apiKey: string
  readonly baseUrl?: string
}

export type AgyRemoteSyncPayload =
  | AgyAdcSyncPayload
  | AgyGeminiApiKeySyncPayload

export interface AgyInspectOptions {
  readonly home?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly registry?: Pick<ProviderAccountRegistry, 'list' | 'syncPayload'>
  /** Optional seam for testing OS keyring existence without touching the host keychain. */
  readonly hasKeyringEntry?: () => Promise<boolean>
}

interface AgySyncReview {
  readonly id: string
  readonly mode: AgyTransferMode
  readonly accountId?: string
  readonly target: string
  readonly expiresAt: number
}

const MAX_ADC_BYTES = 128_000
const MAX_TOOL_OUTPUT_BYTES = 16_384

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function cleanShortString(value: unknown, max = 256): string | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (!trimmed || trimmed.length > max || /[\0\r\n]/.test(trimmed)) return undefined
  return trimmed
}

async function isExecutable(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

async function findAgyBinary(env: Readonly<Record<string, string | undefined>>, home: string): Promise<boolean> {
  const dirs = [...(env.PATH ?? '').split(':').filter(Boolean), ...cliSearchDirectories(home)]
  for (const dir of dirs) {
    if (await isExecutable(join(dir, 'agy'))) return true
  }
  return false
}

export function adcCredentialsPath(home = homedir()): string {
  return join(home, '.config', 'gcloud', 'application_default_credentials.json')
}

export function agyOauthTokenPath(home = homedir()): string {
  return join(home, '.gemini', 'antigravity-cli', 'antigravity-oauth-token')
}

/** Whether `agy` keeps a file-based sign-in here; only checks existence, never reads it. */
async function hasFileOauthToken(home: string): Promise<boolean> {
  try {
    await access(agyOauthTokenPath(home), constants.R_OK)
    return true
  } catch {
    return false
  }
}

async function readJsonFile(path: string): Promise<Record<string, unknown> | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, 'utf8'))
    return isPlainRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

async function runTool(binary: string, args: readonly string[], timeoutMs = 4000): Promise<{ code: number; stdout: string }> {
  return new Promise(resolve => {
    let stdout = ''
    let settled = false
    const child = spawn(binary, [...args], { stdio: ['ignore', 'pipe', 'ignore'] })
    const finish = (code: number) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code, stdout })
    }
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      finish(-1)
    }, timeoutMs)
    child.stdout?.on('data', chunk => {
      if (stdout.length < MAX_TOOL_OUTPUT_BYTES) stdout += String(chunk)
    })
    child.once('error', () => finish(-1))
    child.once('close', code => finish(code ?? -1))
  })
}

/** Check existence of an `agy` keyring login without reading the secret value. */
async function defaultHasKeyringEntry(): Promise<boolean> {
  if (process.platform === 'darwin') {
    const res = await runTool('/usr/bin/security', ['find-generic-password', '-s', 'gemini', '-a', 'antigravity'])
    return res.code === 0
  }
  if (process.platform === 'linux') {
    if (!process.env.DBUS_SESSION_BUS_ADDRESS) return false
    // Query only D-Bus Secret Service object paths (never `GetSecret` or `secret-tool lookup/search`, which print secrets).
    const res = await runTool('gdbus', [
      'call',
      '--session',
      '--dest',
      'org.freedesktop.secrets',
      '--object-path',
      '/org/freedesktop/secrets',
      '--method',
      'org.freedesktop.Secret.Service.SearchItems',
      "{'service': 'gemini', 'account': 'antigravity'}",
    ])
    return res.code === 0 && res.stdout.includes('/org/freedesktop/secrets/')
  }
  return false
}

/**
 * Inspect local `agy` credential sources on this computer.
 * Returns strictly secret-free metadata suitable for IPC / renderer display.
 */
export async function inspectLocalAgyCredentials(
  options: AgyInspectOptions = {},
  context: { readonly destination?: string | null; readonly syncedDestination?: string | null; readonly syncedMode?: AgyTransferMode | null } = {},
): Promise<AgyLocalStatus> {
  const home = options.home ?? homedir()
  const env = options.env ?? process.env

  const [agyInstalled, adcDoc, runtimeEnv, accounts, keyringInOs, fileOauthPresent] = await Promise.all([
    findAgyBinary(env, home),
    readJsonFile(adcCredentialsPath(home)),
    loadAgyRuntimeEnv(home),
    options.registry ? options.registry.list().catch(() => []) : Promise.resolve([]),
    (options.hasKeyringEntry ?? defaultHasKeyringEntry)().catch(() => false),
    hasFileOauthToken(home).catch(() => false),
  ])
  const subscriptionPresent = keyringInOs || fileOauthPresent

  const adcType = adcDoc ? cleanShortString(adcDoc.type, 64) : undefined
  const adcProject = adcDoc ? cleanShortString(adcDoc.quota_project_id, 128) : undefined
  const adcAvailable = Boolean(adcDoc && adcType)

  const geminiAccount = accounts.find(account => account.provider === 'gemini')
  const envGeminiKeyPresent = Boolean(cleanShortString(env.GEMINI_API_KEY, 4096) || cleanShortString(runtimeEnv.GEMINI_API_KEY, 4096))
  const geminiAvailable = Boolean(geminiAccount || envGeminiKeyPresent)
  const geminiDetail = geminiAccount
    ? `Uses saved Gemini account "${geminiAccount.label}" for headless agy sessions (modelProvider: gemini).`
    : envGeminiKeyPresent
      ? 'Uses local GEMINI_API_KEY for headless agy sessions (modelProvider: gemini).'
      : 'Add a Gemini API-key account above or set GEMINI_API_KEY to transfer key-based agy auth.'

  const sources: AgyCredentialSourceStatus[] = [
    {
      mode: 'keyring',
      label: 'Gemini Subscription (Antigravity)',
      available: subscriptionPresent,
      headlessSafe: false,
      transferable: false,
      detail: subscriptionPresent
        ? 'Signed in to Antigravity on this computer. This sign-in stays here; on an SSH host, run `agy acp --login` there, or transfer Google Cloud ADC / a Gemini API key (recommended for headless and third-party harness use).'
        : 'Not signed in locally yet. Run `agy acp --login` in a terminal, or configure Google Cloud ADC / a Gemini API key (recommended for headless and third-party harness use).',
    },
    {
      mode: 'adc',
      label: 'Google Cloud ADC',
      available: adcAvailable,
      headlessSafe: true,
      transferable: true,
      detail: adcAvailable
        ? `Local Application Default Credentials (${adcType}${adcProject ? ` · ${adcProject}` : ''}). Works on headless Linux SSH hosts.`
        : 'Not found at ~/.config/gcloud/application_default_credentials.json. Run `gcloud auth application-default login` to enable headless ADC.',
    },
    {
      mode: 'gemini_api_key',
      label: 'Gemini API key',
      available: geminiAvailable,
      headlessSafe: true,
      transferable: true,
      detail: geminiDetail,
      ...(geminiAccount ? { accountId: geminiAccount.id } : {}),
    },
  ]

  const recommendedMode: AgyTransferMode | null = adcAvailable
    ? 'adc'
    : geminiAvailable
      ? 'gemini_api_key'
      : null

  const destination = context.destination ?? null
  return {
    agyInstalled,
    sources,
    recommendedMode,
    canSync: Boolean(destination),
    destination,
    syncedDestination: context.syncedDestination ?? null,
    syncedMode: context.syncedMode ?? null,
  }
}

/**
 * Read the actual local `agy` credential bundle for a confirmed transfer.
 * Must stay exclusively inside the main process.
 */
export async function readLocalAgyCredentialPayload(
  mode: AgyTransferMode,
  options: AgyInspectOptions & { readonly accountId?: string } = {},
): Promise<AgyRemoteSyncPayload> {
  const home = options.home ?? homedir()
  const env = options.env ?? process.env

  if (mode === 'adc') {
    const raw = await readFile(adcCredentialsPath(home), 'utf8')
    if (Buffer.byteLength(raw, 'utf8') > MAX_ADC_BYTES) {
      throw new Error('Local ADC credential file exceeds size limit')
    }
    const parsed: unknown = JSON.parse(raw)
    if (!isPlainRecord(parsed) || typeof parsed.type !== 'string' || !parsed.type.trim()) {
      throw new Error('Local ADC credential file is invalid')
    }
    const runtimeEnv = await loadAgyRuntimeEnv(home)
    const quotaProject = cleanShortString(env.GOOGLE_CLOUD_QUOTA_PROJECT ?? runtimeEnv.GOOGLE_CLOUD_QUOTA_PROJECT ?? parsed.quota_project_id, 128)
    const location = cleanShortString(env.GOOGLE_CLOUD_LOCATION ?? runtimeEnv.GOOGLE_CLOUD_LOCATION, 64)
    return {
      kind: 'agy',
      mode: 'adc',
      adc: parsed,
      ...(quotaProject ? { quotaProject } : {}),
      ...(location ? { location } : {}),
    }
  }

  if (mode === 'gemini_api_key') {
    const runtimeEnv = await loadAgyRuntimeEnv(home)
    let apiKey = ''
    let baseUrl = cleanShortString(env.GOOGLE_GEMINI_BASE_URL ?? runtimeEnv.GOOGLE_GEMINI_BASE_URL, 2048)

    if (options.registry) {
      const accounts = await options.registry.list()
      const picked = options.accountId
        ? accounts.find(row => row.id === options.accountId && row.provider === 'gemini')
        : accounts.find(row => row.provider === 'gemini')
      if (picked) {
        const payload = await options.registry.syncPayload(picked.id)
        if (payload.profile.api_key.trim()) {
          apiKey = payload.profile.api_key.trim()
        }
      }
    }
    if (!apiKey) {
      apiKey = cleanShortString(env.GEMINI_API_KEY ?? runtimeEnv.GEMINI_API_KEY, 4096) ?? ''
    }
    if (!apiKey) {
      throw new Error('No local Gemini API key is available to transfer for agy')
    }
    return {
      kind: 'agy',
      mode: 'gemini_api_key',
      apiKey,
      ...(baseUrl ? { baseUrl } : {}),
    }
  }

  throw new Error(`Antigravity credential mode "${String(mode)}" cannot be transferred`)
}

/**
 * Static remote receiver script run via `bun -e` over SSH.
 * Secrets arrive ONLY on stdin JSON, are written atomically with mode 0600,
 * and are never printed to stdout or stderr.
 */
export function remoteAgyReceiverSource(): string {
  return String.raw`
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

const input = await new Response(Bun.stdin.stream()).text()
if (input.length > 262144) throw new Error('agy payload too large')
const value = JSON.parse(input)
if (!value || typeof value !== 'object' || Array.isArray(value) || value.kind !== 'agy') {
  throw new Error('invalid agy payload')
}
const home = process.env.HOME
if (!home) throw new Error('missing HOME')

async function atomicJson(file, data) {
  const dir = dirname(file)
  await mkdir(dir, { recursive: true, mode: 0o700 })
  const temp = join(dir, '.xerxes-agy-' + process.pid + '-' + Math.random().toString(16).slice(2) + '.tmp')
  try {
    await writeFile(temp, JSON.stringify(data, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
    await rename(temp, file)
  } catch (error) {
    await rm(temp, { force: true }).catch(() => undefined)
    throw error
  }
}

async function readObj(file) {
  try {
    const parsed = JSON.parse(await readFile(file, 'utf8'))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch (error) {
    if (error && error.code === 'ENOENT') return {}
    throw error
  }
}

const envFile = join(home, '.xerxes', 'provider-accounts', 'agy', 'env.json')

if (value.mode === 'adc') {
  if (!value.adc || typeof value.adc !== 'object' || Array.isArray(value.adc) || typeof value.adc.type !== 'string' || !value.adc.type.trim()) {
    throw new Error('invalid adc credential')
  }
  await atomicJson(join(home, '.config', 'gcloud', 'application_default_credentials.json'), value.adc)
  const geminiConfigFile = join(home, '.gemini', 'config', 'config.json')
  const geminiConfig = await readObj(geminiConfigFile)
  const userSettings = geminiConfig.userSettings && typeof geminiConfig.userSettings === 'object' && !Array.isArray(geminiConfig.userSettings)
    ? geminiConfig.userSettings : {}
  await atomicJson(geminiConfigFile, { ...geminiConfig, userSettings: { ...userSettings, enableAdc: true } })
  const currentEnv = await readObj(envFile)
  const nextEnv = { ...currentEnv, AGY_ADC_AUTH: 'true' }
  if (typeof value.quotaProject === 'string' && value.quotaProject.trim()) nextEnv.GOOGLE_CLOUD_QUOTA_PROJECT = value.quotaProject.trim()
  if (typeof value.location === 'string' && value.location.trim()) nextEnv.GOOGLE_CLOUD_LOCATION = value.location.trim()
  await atomicJson(envFile, nextEnv)
  console.log(JSON.stringify({ ok: true, provider: 'agy', mode: 'adc' }))
  process.exit(0)
}

if (value.mode === 'gemini_api_key') {
  if (typeof value.apiKey !== 'string' || !value.apiKey.trim() || value.apiKey.length > 16384 || /[\0\r\n]/.test(value.apiKey)) {
    throw new Error('invalid gemini api key')
  }
  const settingsFile = join(home, '.gemini', 'antigravity-cli', 'settings.json')
  const settings = await readObj(settingsFile)
  await atomicJson(settingsFile, { ...settings, modelProvider: 'gemini' })
  const currentEnv = await readObj(envFile)
  const nextEnv = { ...currentEnv, GEMINI_API_KEY: value.apiKey.trim() }
  if (typeof value.baseUrl === 'string' && value.baseUrl.trim() && value.baseUrl.length <= 2048 && !/[\0\r\n]/.test(value.baseUrl)) {
    nextEnv.GOOGLE_GEMINI_BASE_URL = value.baseUrl.trim()
  }
  await atomicJson(envFile, nextEnv)
  console.log(JSON.stringify({ ok: true, provider: 'agy', mode: 'gemini_api_key' }))
  process.exit(0)
}

throw new Error('unsupported agy credential mode')
`
}

export function spawnAgySecretWriter(
  target: RemoteTarget,
  payload: AgyRemoteSyncPayload,
  signal: AbortSignal,
  timeoutMs = 30_000,
  spawnImpl: SecretProcessSpawner = (binary, args) => spawn(binary, [...args], { stdio: ['pipe', 'pipe', 'pipe'] }),
): Promise<string> {
  if (signal.aborted) return Promise.reject(new Error('Remote agy credential transfer cancelled'))
  const args = [
    ...sshOptions, '-T', '--', sshDestination(target.target),
    'PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"; export PATH; exec bun -e ' + quote(remoteAgyReceiverSource()),
  ]
  return new Promise((resolve, reject) => {
    const child = spawnImpl('ssh', args)
    let output = ''
    let settled = false
    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      child.stdin?.end()
      if (error) { child.kill('SIGKILL'); reject(error) } else resolve(output)
    }
    const abort = () => finish(new Error('Remote agy credential transfer cancelled'))
    const timer = setTimeout(() => finish(new Error('Remote agy credential transfer timed out')), timeoutMs)
    signal.addEventListener('abort', abort, { once: true })
    child.stdout?.on('data', chunk => { output = (output + String(chunk)).slice(-16384) })
    child.once('error', () => finish(new Error('Remote agy credential transfer failed')))
    child.once('close', code => {
      if (code === 0) { finish(); return }
      finish(new Error('Remote agy credential transfer failed'))
    })
    child.stdin?.on('error', () => finish(new Error('Remote agy credential transfer failed')))
    child.stdin?.end(JSON.stringify(payload))
    if (signal.aborted) abort()
  })
}

/** Parse only the non-secret receiver acknowledgement. */
export function remoteAgyReceipt(output: string): AgyTransferMode {
  const lines = output.split('\n')
  for (let index = lines.length - 1; index >= 0; index--) {
    let value: unknown
    try {
      value = JSON.parse(lines[index]!.trim())
    } catch {
      continue
    }
    if (!isPlainRecord(value)) continue
    if (value.ok === true && value.provider === 'agy' && isAgyTransferMode(value.mode)) {
      return value.mode
    }
  }
  throw new Error('Remote agy credential transfer was not acknowledged')
}

export function isAgyTransferMode(value: unknown): value is AgyTransferMode {
  return value === 'adc' || value === 'gemini_api_key'
}

/**
 * Coordinates one-use human review & stdin-only SSH transfer for `agy` credentials.
 * Bound to a single desktop workspace center (local or remote SSH).
 */
export class AgyCredentialTransfer {
  private review: AgySyncReview | undefined
  private syncedDestination: string | null = null
  private syncedMode: AgyTransferMode | null = null

  constructor(
    private readonly remote: RemoteTarget | undefined,
    private readonly inspectOptions: AgyInspectOptions = {},
    private readonly now: () => number = Date.now,
    private readonly writer?: (payload: AgyRemoteSyncPayload) => Promise<string>,
  ) {}

  async status(): Promise<AgyLocalStatus> {
    return inspectLocalAgyCredentials(this.inspectOptions, {
      destination: this.remote?.target ?? null,
      syncedDestination: this.syncedDestination,
      syncedMode: this.syncedMode,
    })
  }

  async reviewSync(requestedMode?: unknown, accountId?: string): Promise<Record<string, unknown>> {
    if (!this.remote) throw new Error('Open an SSH workspace to transfer Antigravity (agy) credentials')
    const current = await this.status()
    let mode: AgyTransferMode
    if (requestedMode !== undefined && requestedMode !== '') {
      if (requestedMode === 'keyring') {
        throw new Error('The Antigravity sign-in stays on this computer. Run `agy acp --login` on the SSH host, or transfer Google Cloud ADC or a Gemini API key.')
      }
      if (!isAgyTransferMode(requestedMode)) throw new Error('Choose Google Cloud ADC or a Gemini API key for agy transfer')
      mode = requestedMode
    } else if (current.recommendedMode) {
      mode = current.recommendedMode
    } else {
      throw new Error('No local Antigravity (agy) credentials were found on this computer')
    }
    const source = current.sources.find(item => item.mode === mode)
    if (!source || !source.available) {
      throw new Error(`Antigravity (${mode}) credentials are not available on this computer`)
    }
    const id = randomUUID()
    const resolvedAccountId = accountId?.trim() || source.accountId
    this.review = {
      id,
      mode,
      ...(resolvedAccountId ? { accountId: resolvedAccountId } : {}),
      target: this.remote.target,
      expiresAt: this.now() + 300_000,
    }
    return {
      ok: true,
      review: id,
      mode,
      label: source.label,
      headless_safe: source.headlessSafe,
      destination: this.remote.target,
      expires_at: this.review.expiresAt,
    }
  }

  async confirmSync(reviewId: string, confirm: boolean): Promise<Record<string, unknown>> {
    const review = this.review
    const remote = this.remote
    if (!remote || !confirm || !review || review.id !== reviewId || review.target !== remote.target || review.expiresAt < this.now()) {
      throw new Error('Antigravity credential transfer review expired; review it again')
    }
    this.review = undefined
    let stage: 'source' | 'transfer' | 'receipt' | 'identity' = 'source'
    try {
      const payload = await readLocalAgyCredentialPayload(review.mode, {
        ...this.inspectOptions,
        ...(review.accountId ? { accountId: review.accountId } : {}),
      })
      stage = 'transfer'
      const output = this.writer
        ? await this.writer(payload)
        : await spawnAgySecretWriter(remote, payload, new AbortController().signal)
      stage = 'receipt'
      const acknowledged = remoteAgyReceipt(output)
      stage = 'identity'
      if (acknowledged !== review.mode) throw new Error('mode mismatch')
      this.syncedDestination = remote.target
      this.syncedMode = acknowledged
      return { ok: true, provider: 'agy', mode: acknowledged, destination: remote.target }
    } catch (error) {
      const hint = stage === 'source'
        ? 'source: the local agy credential could not be read'
        : stage === 'transfer'
          ? 'transfer: the SSH credential transfer did not complete'
          : stage === 'receipt'
            ? 'receipt: the remote host replied without a valid receipt'
            : 'identity: the remote host acknowledged a different mode'
      throw new Error(`Antigravity (agy) credential transfer was not acknowledged (${hint}); review the transfer again before retrying`)
    }
  }
}
