// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { ProfileStore, type ProviderProfile } from '../bridge/profiles.js'
import { xerxesHome } from '../daemon/paths.js'
import { AnthropicOAuthSession } from './anthropicOAuth.js'
import { CodexSession, codexAuthHeaders, importCodexCliTokens } from './codexAuth.js'
import { loginWithChatGpt, openInBrowser } from './codexLogin.js'
import { CredentialStorage } from './storage.js'
import { fetchClaudeUsage, fetchCodexUsage, type ProviderUsageReport, type UsageRequestOptions } from './usage.js'
import { ConfigurationError } from '../core/errors.js'
import { claudeCodeLogin } from './claudeCodeLogin.js'
import { profileQuota, type ProfileQuota } from './profileUsage.js'

export type OAuthAccountKind = 'claude' | 'codex' | 'claude-code'
export type ProviderAccountKind = OAuthAccountKind | 'api_key'
export interface ProviderAccountMetadata {
  readonly id: string
  readonly kind: OAuthAccountKind
  readonly label: string
  readonly profileName: string
  readonly createdAt: string
}
export interface ProviderAccountSummary extends Omit<ProviderAccountMetadata, 'kind'> {
  readonly kind: ProviderAccountKind
  readonly provider: string
  readonly model: string
}
export interface OAuthSyncPayload {
  readonly kind: OAuthAccountKind
  readonly accountId: string
  readonly label: string
  readonly profile: ProviderProfile
  readonly credential: unknown
}
export interface ApiKeySyncPayload {
  readonly kind: 'api_key'
  readonly profile: ProviderProfile
}
export type ProviderSyncPayload = OAuthSyncPayload | ApiKeySyncPayload

interface RegistryDocument { readonly accounts: readonly ProviderAccountMetadata[] }

function defaultBase(kind: OAuthAccountKind): { provider: string; baseUrl: string; model: string } {
  if (kind === 'codex') return { provider: 'openai-codex', baseUrl: 'https://chatgpt.com/backend-api/codex', model: 'gpt-5.6-codex' }
  if (kind === 'claude-code') return { provider: 'claude-code', baseUrl: 'https://api.anthropic.com', model: 'claude-sonnet-4-5' }
  return { provider: 'anthropic', baseUrl: 'https://api.anthropic.com', model: 'claude-sonnet-4-5' }
}

/** Isolated OAuth account directories plus the existing named API-key profiles. */
export class ProviderAccountRegistry {
  constructor(
    private readonly root = join(xerxesHome(), 'provider-accounts'),
    private readonly profilesFile = join(xerxesHome(), 'profiles.json'),
    private readonly environment: Readonly<Record<string, string | undefined>> = process.env,
  ) {}

  private directory(id: string): string {
    if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('Invalid provider account id')
    return join(this.root, id)
  }

  private async document(): Promise<RegistryDocument> {
    try {
      const value = JSON.parse(await readFile(join(this.root, 'accounts.json'), 'utf8')) as RegistryDocument
      return { accounts: Array.isArray(value.accounts) ? value.accounts : [] }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { accounts: [] }
      throw error
    }
  }

  private async saveDocument(document: RegistryDocument): Promise<void> {
    await mkdir(this.root, { recursive: true, mode: 0o700 })
    const temporary = join(this.root, `.accounts-${randomUUID()}.tmp`)
    await writeFile(temporary, JSON.stringify(document, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
    await rename(temporary, join(this.root, 'accounts.json'))
  }

  private async metadata(id: string): Promise<ProviderAccountMetadata> {
    const account = (await this.document()).accounts.find(row => row.id === id)
    if (!account) throw new Error('Provider account not found')
    return account
  }

  async list(): Promise<readonly ProviderAccountSummary[]> {
    const oauth = (await this.document()).accounts
    const profiles = new ProfileStore(this.profilesFile).list()
    const result: ProviderAccountSummary[] = []
    for (const account of oauth) {
      const profile = profiles.find(row => row.name === account.profileName)
      if (profile) result.push({ ...account, provider: profile.provider, model: profile.model })
    }
    for (const profile of profiles.filter(row => !row.oauth_account_id && row.api_key.trim())) {
      result.push({ id: `api:${profile.name}`, kind: 'api_key', label: profile.name, profileName: profile.name, createdAt: '', provider: profile.provider, model: profile.model })
    }
    return result
  }

  private pendingPath(id: string): string {
    this.directory(id)
    return join(this.root, 'pending', `${id}.json`)
  }

  private async newAccount(kind: OAuthAccountKind, label: string): Promise<ProviderAccountMetadata> {
    const clean = label.trim()
    if (!clean || clean.length > 80 || /[\0\r\n]/.test(clean)) throw new Error('Account label is required')
    const id = randomUUID()
    const profileStore = new ProfileStore(this.profilesFile)
    const base = `${kind}-${clean}`
    let profileName = base, suffix = 2
    while (profileStore.get(profileName)) profileName = `${base}-${suffix++}`
    return { id, kind, label: clean, profileName, createdAt: new Date().toISOString() }
  }

  private async register(account: ProviderAccountMetadata): Promise<ProviderAccountMetadata> {
    const document = await this.document(), profileStore = new ProfileStore(this.profilesFile)
    let registered = account
    if (profileStore.get(account.profileName)) {
      const base = `${account.kind}-${account.label}`
      let profileName = base, suffix = 2
      while (profileStore.get(profileName)) profileName = `${base}-${suffix++}`
      registered = { ...account, profileName }
    }
    const defaults = defaultBase(registered.kind)
    profileStore.save({ name: registered.profileName, apiKey: '', model: defaults.model, provider: defaults.provider, baseUrl: defaults.baseUrl, setActive: false, oauthAccountId: registered.id })
    try {
      await this.saveDocument({ accounts: [...document.accounts.filter(row => row.id !== registered.id), registered] })
    } catch (error) {
      new ProfileStore(this.profilesFile).delete(registered.profileName)
      throw error
    }
    await mkdir(this.directory(registered.id), { recursive: true, mode: 0o700 })
    return registered
  }

  private async pending(id: string): Promise<ProviderAccountMetadata | undefined> {
    try {
      return JSON.parse(await readFile(this.pendingPath(id), 'utf8')) as ProviderAccountMetadata
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    }
  }

  /** Temporary login state only: no account or provider profile is visible yet. */
  async prepare(kind: OAuthAccountKind, label: string): Promise<ProviderAccountMetadata> {
    const account = await this.newAccount(kind, label)
    await mkdir(this.directory(account.id), { recursive: true, mode: 0o700 })
    try {
      await mkdir(join(this.root, 'pending'), { recursive: true, mode: 0o700 })
      await writeFile(this.pendingPath(account.id), JSON.stringify(account, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
    } catch (error) {
      await rm(this.directory(account.id), { recursive: true, force: true })
      throw error
    }
    return account
  }

  /** Commit only after the provider credential is actually readable. */
  async finishLogin(id: string): Promise<ProviderAccountMetadata> {
    const account = await this.pending(id)
    if (!account) return this.metadata(id)
    const directory = this.directory(id)
    const credential = account.kind === 'codex'
      ? (await importCodexCliTokens({ CODEX_HOME: join(directory, '.codex') }, directory))?.toRecord()
      : await new AnthropicOAuthSession({ xerxesHome: directory }).stored()
    if (!credential) throw new Error('Provider account has no saved credential')
    const registered = await this.register(account)
    await rm(this.pendingPath(id), { force: true })
    return registered
  }

  /** Remove only an unfinished login; a concurrently completed account is kept. */
  async discard(id: string): Promise<void> {
    if (!await this.pending(id)) return
    await rm(this.directory(id), { recursive: true, force: true })
    await rm(this.pendingPath(id), { force: true })
  }

  async create(kind: OAuthAccountKind, label: string): Promise<ProviderAccountMetadata> {
    return this.register(await this.newAccount(kind, label))
  }

  async login(id: string, openUrl: (url: string) => void = openInBrowser, onAuthorizationUrl: (url: string) => void = () => {}): Promise<ProviderAccountMetadata> {
    const pending = await this.pending(id)
    const account = pending ?? await this.metadata(id), directory = this.directory(id)
    if (account.kind === 'codex') {
      const { token } = await loginWithChatGpt({ openUrl, onAuthorizationUrl })
      const storage = new CredentialStorage(join(directory, 'credentials'))
      await new CodexSession({ storage, environment: { CODEX_HOME: join(directory, '.codex') }, homeDirectory: directory }).store(token)
      // The sync receiver uses the CLI-compatible plaintext form, mode 0600.
      await mkdir(join(directory, '.codex'), { recursive: true, mode: 0o700 })
      await writeFile(join(directory, '.codex', 'auth.json'), JSON.stringify({ tokens: token.toRecord() }, null, 2) + '\n', { mode: 0o600 })
    } else {
      await new AnthropicOAuthSession({ xerxesHome: directory }).login({ openUrl, onAuthorizationUrl })
    }
    return pending ? this.finishLogin(id) : account
  }

  sessionOptions(id: string): { codexSession: CodexSession; anthropicOAuthSession: AnthropicOAuthSession; claudeEnvironment: Record<string, string>; isolatedAuth: true } {
    const directory = this.directory(id)
    const claudeEnvironment: Record<string, string> = {}
    for (const [key, value] of Object.entries(this.environment)) {
      if (value !== undefined && !key.startsWith('ANTHROPIC_') && !['CLAUDE_CODE_OAUTH_TOKEN', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX'].includes(key)) claudeEnvironment[key] = value
    }
    claudeEnvironment.CLAUDE_CONFIG_DIR = directory
    // metadata is checked by callers that enumerate accounts; this path remains
    // valid only for registry-created ids.
    return {
      codexSession: new CodexSession({ storage: new CredentialStorage(join(directory, 'credentials')), environment: { CODEX_HOME: join(directory, '.codex') }, homeDirectory: directory }),
      anthropicOAuthSession: new AnthropicOAuthSession({ xerxesHome: directory, environment: {} }),
      claudeEnvironment,
      isolatedAuth: true,
    }
  }

  async usage(id: string): Promise<ProfileQuota | ProviderUsageReport> {
    if (id.startsWith('api:')) {
      const profile = new ProfileStore(this.profilesFile).get(id.slice(4))
      if (!profile) throw new Error('Provider profile not found')
      return profileQuota(profile)
    }
    return this.subscriptionUsage(id)
  }

  /** A named account never borrows a global login, keychain entry, or token. */
  async subscriptionUsage(id: string, options: UsageRequestOptions = {}): Promise<ProviderUsageReport> {
    options.signal?.throwIfAborted()
    const account = await this.metadata(id), sessions = this.sessionOptions(id)
    const request: UsageRequestOptions = { ...options, environment: {} }
    const signedOut = (error: unknown): never => {
      if (error instanceof ConfigurationError) throw new ConfigurationError('provider_account', `${account.label} has no saved sign-in. Sign in to this account again from Account Center.`)
      throw error
    }
    if (account.kind === 'codex') {
      const credential = await sessions.codexSession.credential(options.signal).catch(signedOut)
      return fetchCodexUsage(codexAuthHeaders(credential), request)
    }
    if (account.kind === 'claude-code' && !await sessions.anthropicOAuthSession.stored()) {
      const login = await claudeCodeLogin({ environment: { CLAUDE_CONFIG_DIR: this.directory(id) }, readKeychain: async () => undefined }).catch(signedOut)
      return fetchClaudeUsage(login.accessToken, request)
    }
    const credential = await sessions.anthropicOAuthSession.credential(options.signal).catch(signedOut)
    return fetchClaudeUsage(credential.access, request)
  }

  async syncPayload(id: string): Promise<ProviderSyncPayload> {
    if (id.startsWith('api:')) {
      const profile = new ProfileStore(this.profilesFile).get(id.slice(4))
      if (!profile) throw new Error('Provider profile not found')
      return { kind: 'api_key', profile }
    }
    const account = await this.metadata(id), directory = this.directory(id)
    const profile = new ProfileStore(this.profilesFile).get(account.profileName)
    if (!profile) throw new Error('Linked provider profile not found')
    const credential = account.kind === 'codex'
      ? (await importCodexCliTokens({ CODEX_HOME: join(directory, '.codex') }, directory))?.toRecord()
      : await new AnthropicOAuthSession({ xerxesHome: directory }).stored()
    if (!credential) throw new Error('Provider account has no saved credential')
    return { kind: account.kind, accountId: id, label: account.label, profile, credential }
  }

  async remove(id: string): Promise<void> {
    const account = await this.metadata(id)
    new ProfileStore(this.profilesFile).delete(account.profileName)
    await rm(this.directory(id), { recursive: true, force: true })
    await this.saveDocument({ accounts: (await this.document()).accounts.filter(row => row.id !== id) })
  }
}
