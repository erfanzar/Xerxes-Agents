// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access, constants, readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { buildAgySubprocessEnv, loadAgyRuntimeEnv } from '../../auth/agyRuntimeEnv.js'
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
 */

export type AgyCredentialMode = 'adc' | 'gemini_api_key' | 'keyring'

export interface AgyCredentialSourceStatus {
  readonly mode: AgyCredentialMode
  readonly label: string
  readonly available: boolean
  readonly detail: string
  readonly headlessSafe: boolean
  /** Saved Gemini account id if `gemini_api_key` comes from the Xerxes account registry. */
  readonly accountId?: string
}

export interface AgyLocalStatus {
  readonly agyInstalled: boolean
  readonly sources: readonly AgyCredentialSourceStatus[]
  readonly recommendedMode: AgyCredentialMode | null
  readonly canSync: boolean
  readonly destination: string | null
  readonly syncedDestination: string | null
  readonly syncedMode: AgyCredentialMode | null
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

export interface AgyKeyringSyncPayload {
  readonly kind: 'agy'
  readonly mode: 'keyring'
  readonly secret: string
  /** Decoded JSON OAuth token written to `~/.gemini/antigravity-cli/antigravity-oauth-token` (0600) for headless Linux/SSH hosts. */
  readonly oauthToken?: Record<string, unknown>
  readonly settings?: Record<string, unknown>
}

export type AgyRemoteSyncPayload =
  | AgyAdcSyncPayload
  | AgyGeminiApiKeySyncPayload
  | AgyKeyringSyncPayload

export interface AgyInspectOptions {
  readonly home?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly registry?: Pick<ProviderAccountRegistry, 'list' | 'syncPayload'>
  /** Optional seam for testing OS keyring existence without touching the host keychain. */
  readonly hasKeyringEntry?: () => Promise<boolean>
  /** Optional seam for reading OS keyring secret in tests. */
  readonly readKeyringSecret?: () => Promise<string | undefined>
}

interface AgySyncReview {
  readonly id: string
  readonly mode: AgyCredentialMode
  readonly accountId?: string
  readonly target: string
  readonly expiresAt: number
}

const MAX_ADC_BYTES = 128_000
const MAX_KEYRING_SECRET_BYTES = 128_000

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

export function agySettingsPath(home = homedir()): string {
  return join(home, '.gemini', 'antigravity-cli', 'settings.json')
}

export function agyOauthTokenPath(home = homedir()): string {
  return join(home, '.gemini', 'antigravity-cli', 'antigravity-oauth-token')
}

const GO_KEYRING_B64_PREFIX = 'go-keyring-base64:'

/**
 * Decode an `agy` keyring or file-based OAuth token payload into its plain JSON object.
 * Supports `go-keyring-base64:<base64>` (macOS Keychain / go-keyring), hex-encoded
 * `security -w` output, and raw JSON (`antigravity-oauth-token`).
 */
export function decodeAgyOauthToken(raw: string): Record<string, unknown> | undefined {
  let candidate = raw.trim()
  if (!candidate) return undefined
  if (/^[0-9a-fA-F]+$/.test(candidate) && candidate.length % 2 === 0 && candidate.length >= 16) {
    try {
      const hexDecoded = Buffer.from(candidate, 'hex').toString('utf8')
      if (hexDecoded.startsWith(GO_KEYRING_B64_PREFIX) || hexDecoded.trimStart().startsWith('{')) {
        candidate = hexDecoded.trim()
      }
    } catch {
      // Keep original candidate.
    }
  }
  if (candidate.startsWith(GO_KEYRING_B64_PREFIX)) {
    try {
      candidate = Buffer.from(candidate.slice(GO_KEYRING_B64_PREFIX.length), 'base64').toString('utf8').trim()
    } catch {
      return undefined
    }
  }
  try {
    const parsed: unknown = JSON.parse(candidate)
    if (!isPlainRecord(parsed)) return undefined
    if (!('token' in parsed) || parsed.token === null || parsed.token === undefined) return undefined
    return parsed
  } catch {
    return undefined
  }
}

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
      if (stdout.length < MAX_KEYRING_SECRET_BYTES) stdout += String(chunk)
    })
    child.once('error', () => finish(-1))
    child.once('close', code => finish(code ?? -1))
  })
}

/** Check existence of an `agy` keyring login without reading the secret value on macOS. */
async function defaultHasKeyringEntry(): Promise<boolean> {
  if (process.platform === 'darwin') {
    const res = await runTool('/usr/bin/security', ['find-generic-password', '-s', 'gemini', '-a', 'antigravity'])
    return res.code === 0
  }
  if (process.platform === 'linux') {
    if (!process.env.DBUS_SESSION_BUS_ADDRESS) return false
    const res = await runTool('secret-tool', ['lookup', 'service', 'gemini', 'account', 'antigravity'])
    return res.code === 0 && res.stdout.trim().length > 0
  }
  return false
}

/** Read the local `agy` OS keyring secret strictly in the main process when confirmed by the user. */
async function defaultReadKeyringSecret(): Promise<string | undefined> {
  if (process.platform === 'darwin') {
    const res = await runTool('/usr/bin/security', ['find-generic-password', '-s', 'gemini', '-a', 'antigravity', '-w'])
    const value = res.code === 0 ? res.stdout.replace(/\r?\n$/, '') : ''
    return value.length > 0 ? value : undefined
  }
  if (process.platform === 'linux') {
    if (!process.env.DBUS_SESSION_BUS_ADDRESS) return undefined
    const res = await runTool('secret-tool', ['lookup', 'service', 'gemini', 'account', 'antigravity'])
    const value = res.code === 0 ? res.stdout.replace(/\r?\n$/, '') : ''
    return value.length > 0 ? value : undefined
  }
  return undefined
}

/**
 * Inspect local `agy` credential sources on this computer.
 * Returns strictly secret-free metadata suitable for IPC / renderer display.
 */
export async function inspectLocalAgyCredentials(
  options: AgyInspectOptions = {},
  context: { readonly destination?: string | null; readonly syncedDestination?: string | null; readonly syncedMode?: AgyCredentialMode | null } = {},
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
      headlessSafe: true,
      detail: subscriptionPresent
        ? 'Local Gemini Subscription / Antigravity Google sign-in found on this computer. Transfers securely to headless Linux/TPU SSH hosts (mode 0600).'
        : 'Not signed in locally yet. Choose Gemini Subscription above to sign in with Google, or run `agy acp --login`.',
    },
    {
      mode: 'adc',
      label: 'Google Cloud ADC',
      available: adcAvailable,
      headlessSafe: true,
      detail: adcAvailable
        ? `Local Application Default Credentials (${adcType}${adcProject ? ` · ${adcProject}` : ''}). Works on headless Linux SSH hosts.`
        : 'Not found at ~/.config/gcloud/application_default_credentials.json. Run `gcloud auth application-default login` to enable headless ADC.',
    },
    {
      mode: 'gemini_api_key',
      label: 'Gemini API key',
      available: geminiAvailable,
      headlessSafe: true,
      detail: geminiDetail,
      ...(geminiAccount ? { accountId: geminiAccount.id } : {}),
    },
  ]

  // Prefer the user's Gemini Subscription (Antigravity OAuth) first — it works on
  // both headless Linux/TPU SSH hosts (file token storage) and macOS hosts.
  const recommendedMode: AgyCredentialMode | null = subscriptionPresent
    ? 'keyring'
    : adcAvailable
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
  mode: AgyCredentialMode,
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

  let secret = await (options.readKeyringSecret ?? defaultReadKeyringSecret)()
  if (!secret) {
    try {
      const fromFile = (await readFile(agyOauthTokenPath(home), 'utf8')).trim()
      if (fromFile.length > 0) secret = fromFile
    } catch {
      // No local file-based OAuth token either.
    }
  }
  if (!secret || Buffer.byteLength(secret, 'utf8') > MAX_KEYRING_SECRET_BYTES) {
    throw new Error('No local Gemini Subscription (Antigravity) sign-in is available on this computer')
  }
  const oauthToken = decodeAgyOauthToken(secret)
  const rawSettings = await readJsonFile(agySettingsPath(home))
  const settings = rawSettings
    ? Object.fromEntries(Object.entries(rawSettings).filter(([key, val]) => !(key === 'modelProvider' && val === 'gemini')))
    : undefined
  return {
    kind: 'agy',
    mode: 'keyring',
    secret,
    ...(oauthToken ? { oauthToken } : {}),
    ...(settings && Object.keys(settings).length > 0 ? { settings } : {}),
  }
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

if (value.mode === 'keyring') {
  if (typeof value.secret !== 'string' || !value.secret.length || value.secret.length > 131072 || /[\0\r\n]/.test(value.secret)) {
    throw new Error('invalid keyring credential')
  }
  let stored = false
  if (value.oauthToken && typeof value.oauthToken === 'object' && !Array.isArray(value.oauthToken) && 'token' in value.oauthToken) {
    // Headless-safe file token storage read natively by agy on SSH / headless Linux hosts.
    await atomicJson(join(home, '.gemini', 'antigravity-cli', 'antigravity-oauth-token'), value.oauthToken)
    stored = true
  }
  const allowKeyring = process.env.XERXES_SKIP_OS_KEYRING !== '1'
  if (allowKeyring && process.platform === 'darwin') {
    const escaped = value.secret.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
    const proc = Bun.spawn(['/usr/bin/security', '-i'], {
      stdin: new TextEncoder().encode('add-generic-password -U -s gemini -a antigravity -w "' + escaped + '"\n'),
      stdout: 'ignore',
      stderr: 'ignore',
      timeout: 3000,
    })
    if ((await proc.exited) === 0) stored = true
  } else if (allowKeyring && process.platform === 'linux' && process.env.DBUS_SESSION_BUS_ADDRESS && Bun.which('secret-tool')) {
    const proc = Bun.spawn(['secret-tool', 'store', '--label=Antigravity CLI', 'service', 'gemini', 'account', 'antigravity'], {
      stdin: new TextEncoder().encode(value.secret),
      stdout: 'ignore',
      stderr: 'ignore',
      timeout: 3000,
    })
    if ((await proc.exited) === 0) stored = true
  }
  if (!stored) {
    console.log(JSON.stringify({ ok: false, code: 'keyring_unavailable' }))
    process.exit(1)
  }
  const settingsFile = join(home, '.gemini', 'antigravity-cli', 'settings.json')
  const current = await readObj(settingsFile)
  const merged = value.settings && typeof value.settings === 'object' && !Array.isArray(value.settings)
    ? { ...current, ...value.settings }
    : { ...current }
  if (merged.modelProvider === 'gemini') delete merged.modelProvider
  if (Object.keys(merged).length > 0 || current.modelProvider === 'gemini') {
    await atomicJson(settingsFile, merged)
  }
  console.log(JSON.stringify({ ok: true, provider: 'agy', mode: 'keyring' }))
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
      // Check for the non-secret structured keyring_unavailable notice before collapsing to generic error.
      if (output.includes('"keyring_unavailable"')) {
        finish(new Error('keyring_unavailable'))
        return
      }
      finish(new Error('Remote agy credential transfer failed'))
    })
    child.stdin?.on('error', () => finish(new Error('Remote agy credential transfer failed')))
    child.stdin?.end(JSON.stringify(payload))
    if (signal.aborted) abort()
  })
}

/** Parse only the non-secret receiver acknowledgement. */
export function remoteAgyReceipt(output: string): AgyCredentialMode {
  const lines = output.split('\n')
  for (let index = lines.length - 1; index >= 0; index--) {
    let value: unknown
    try {
      value = JSON.parse(lines[index]!.trim())
    } catch {
      continue
    }
    if (!isPlainRecord(value)) continue
    if (value.ok === true && value.provider === 'agy' && (value.mode === 'adc' || value.mode === 'gemini_api_key' || value.mode === 'keyring')) {
      return value.mode
    }
  }
  throw new Error('Remote agy credential transfer was not acknowledged')
}

export function isAgyCredentialMode(value: unknown): value is AgyCredentialMode {
  return value === 'adc' || value === 'gemini_api_key' || value === 'keyring'
}

/**
 * Coordinates one-use human review & stdin-only SSH transfer for `agy` credentials.
 * Bound to a single desktop workspace center (local or remote SSH).
 */
export class AgyCredentialTransfer {
  private review: AgySyncReview | undefined
  private syncedDestination: string | null = null
  private syncedMode: AgyCredentialMode | null = null

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
    let mode: AgyCredentialMode
    if (requestedMode !== undefined && requestedMode !== '') {
      if (!isAgyCredentialMode(requestedMode)) throw new Error('Choose Google Cloud ADC, Gemini API key, or OS Keyring for agy transfer')
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
      if (error instanceof Error && error.message === 'keyring_unavailable') {
        throw new Error('The SSH host could not store the Antigravity OAuth token. Transfer Google Cloud ADC or a Gemini API key instead.')
      }
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

interface PendingAgyLogin {
  readonly id: string
  readonly child: ChildProcess
  readonly done: Promise<void>
  readonly timer: ReturnType<typeof setTimeout>
}

export interface AgyLoginManagerOptions {
  readonly home?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly findAgyPath?: () => Promise<string | undefined>
  readonly spawnLogin?: (binary: string, args: readonly string[], env: NodeJS.ProcessEnv) => ChildProcess
}

async function resolveAgyBinaryPath(env: Readonly<Record<string, string | undefined>>, home: string): Promise<string | undefined> {
  const dirs = [...(env.PATH ?? '').split(':').filter(Boolean), ...cliSearchDirectories(home)]
  for (const dir of dirs) {
    const candidate = join(dir, 'agy')
    if (await isExecutable(candidate)) return candidate
  }
  return undefined
}

/**
 * Coordinates interactive `agy acp --login` Google OAuth sign-in in the main
 * process: spawns `agy acp --login`, extracts the Google OAuth URL for the
 * browser, and submits the pasted authorization code/URL to stdin on completion.
 */
export class AgyLoginManager {
  private active: PendingAgyLogin | undefined

  constructor(private readonly options: AgyLoginManagerOptions = {}) {}

  async start(): Promise<{ readonly loginId: string; readonly url: string }> {
    this.cancel()
    const home = this.options.home ?? homedir()
    const env = this.options.env ?? process.env
    const binary = await (this.options.findAgyPath ?? (() => resolveAgyBinaryPath(env, home)))()
    if (!binary) {
      throw new Error('Antigravity CLI (agy) is not installed yet on this computer.')
    }
    const cleanEnv = buildAgySubprocessEnv(env, { stripSshSession: process.platform === 'darwin' })
    const childEnv: NodeJS.ProcessEnv = {}
    for (const [key, value] of Object.entries(cleanEnv)) {
      if (typeof value === 'string') childEnv[key] = value
    }
    childEnv.PATH = [...cliSearchDirectories(home), childEnv.PATH ?? ''].filter(Boolean).join(':')

    const child = (this.options.spawnLogin ?? ((bin, args, spawnEnv) =>
      spawn(bin, [...args], { env: spawnEnv, stdio: ['pipe', 'pipe', 'pipe'] })
    ))(binary, ['acp', '--login'], childEnv)

    const id = randomUUID()
    let output = ''
    let resolveUrl: ((url: string) => void) | undefined
    let rejectUrl: ((err: Error) => void) | undefined
    const urlPromise = new Promise<string>((resolve, reject) => {
      resolveUrl = resolve
      rejectUrl = reject
    })

    let resolveDone: (() => void) | undefined
    let rejectDone: ((err: Error) => void) | undefined
    const done = new Promise<void>((resolve, reject) => {
      resolveDone = resolve
      rejectDone = reject
    })
    // Prevent unhandled rejections if the process exits before complete() is called.
    done.catch(() => undefined)

    const timer = setTimeout(() => {
      this.cancel(id)
    }, 10 * 60_000)
    timer.unref?.()

    const urlTimer = setTimeout(() => {
      rejectUrl?.(new Error('Timed out waiting for Antigravity sign-in URL.'))
      this.cancel(id)
    }, 15_000)
    urlTimer.unref?.()

    const scanOutput = (chunk: Buffer | string) => {
      output = (output + String(chunk)).slice(-8192)
      const match = output.match(/https:\/\/accounts\.google\.com\/[^\s]+/)
      if (match?.[0]) {
        try {
          const parsed = new URL(match[0])
          if (parsed.protocol === 'https:') {
            clearTimeout(urlTimer)
            resolveUrl?.(parsed.toString())
            resolveUrl = undefined
            rejectUrl = undefined
          }
        } catch {
          // Wait for full URL chunk.
        }
      }
    }
    child.stdout?.on('data', scanOutput)
    child.stderr?.on('data', scanOutput)
    child.once('error', error => {
      clearTimeout(urlTimer)
      clearTimeout(timer)
      if (this.active?.id === id) this.active = undefined
      rejectUrl?.(error)
      rejectDone?.(error)
    })
    child.once('close', code => {
      clearTimeout(urlTimer)
      clearTimeout(timer)
      if (this.active?.id === id) this.active = undefined
      if (code === 0) {
        resolveDone?.()
      } else {
        const detail = output.trim().split('\n').filter(line => !line.includes('https://accounts.google.com/')).slice(-2).join(' ').trim()
        const err = new Error(detail || `Antigravity sign-in exited (${code ?? 'unknown'})`)
        rejectUrl?.(err)
        rejectDone?.(err)
      }
    })

    this.active = { id, child, done, timer }
    const url = await urlPromise
    return { loginId: id, url }
  }

  async complete(loginId: string, codeOrUrl: string): Promise<void> {
    const active = this.active
    const cleaned = codeOrUrl.trim()
    if (!active || active.id !== loginId) {
      throw new Error('Sign-in session expired. Start Google sign-in again.')
    }
    if (!cleaned || /[\0\r\n]/.test(cleaned) || cleaned.length > 4096) {
      throw new Error('Paste the authorization code or callback URL from your browser.')
    }
    active.child.stdin?.write(`${cleaned}\n`)
    active.child.stdin?.end()
    await active.done
  }

  cancel(loginId?: string): void {
    const active = this.active
    if (!active) return
    if (loginId && active.id !== loginId) return
    this.active = undefined
    clearTimeout(active.timer)
    try {
      active.child.kill('SIGTERM')
    } catch {
      // Already exited.
    }
  }
}
