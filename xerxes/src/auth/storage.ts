// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'node:crypto'
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { OAuthToken } from '../mcp/oauth.js'

const CREDENTIAL_FILE_MODE = 0o600
const CREDENTIAL_DIRECTORY_MODE = 0o700
const ENCRYPTION_ALGORITHM = 'aes-256-gcm'
const ENCRYPTION_VERSION = 1
const KEY_BYTES = 32
const IV_BYTES = 12
const PROVIDER_NAME = /^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,127})$/

export interface CredentialStorageOptions {
  /** Explicit encryption key material. It is hashed into a 256-bit AES key. */
  readonly credentialKey?: string | Uint8Array
  /** Optional key-file location. Defaults beside the credentials directory. */
  readonly keyPath?: string
}

interface EncryptedCredential {
  readonly algorithm: typeof ENCRYPTION_ALGORITHM
  readonly ciphertext: string
  readonly iv: string
  readonly tag: string
  readonly version: typeof ENCRYPTION_VERSION
}

/**
 * Encrypted filesystem-backed OAuth token store.
 *
 * Each credential is encrypted with AES-256-GCM and atomically replaced from a
 * same-directory, `0600` temporary file. Provider identifiers are constrained
 * to filenames, so callers cannot escape the configured credential directory.
 */
export class CredentialStorage {
  readonly baseDirectory: string

  private readonly configuredKey: string | Uint8Array | undefined
  private keyPromise: Promise<Buffer> | undefined
  private readonly keyPath: string

  constructor(baseDirectory = join(xerxesHome(), 'credentials'), options: CredentialStorageOptions = {}) {
    this.baseDirectory = resolve(baseDirectory)
    this.keyPath = resolve(options.keyPath ?? join(dirname(this.baseDirectory), '.credential_key'))
    this.configuredKey = options.credentialKey ?? nonEmptyEnvironmentKey()
  }

  /** Return a default store rooted at `$XERXES_HOME/credentials`. */
  static default(options: CredentialStorageOptions = {}): CredentialStorage {
    return new CredentialStorage(undefined, options)
  }

  /** Encrypt and atomically save a token. Returns the credential path. */
  async save(provider: string, token: OAuthToken): Promise<string> {
    const normalizedProvider = validatedProvider(provider)
    const key = await this.encryptionKey()
    const content = Buffer.from(JSON.stringify(token.toRecord()), 'utf8')
    const envelope = encrypt(normalizedProvider, content, key)
    const path = this.pathFor(normalizedProvider)
    await writeAtomic(path, `${JSON.stringify(envelope)}\n`)
    return path
  }

  /** Load an encrypted token, or return undefined for a missing/corrupt credential. */
  async load(provider: string): Promise<OAuthToken | undefined> {
    const normalizedProvider = validatedProvider(provider)
    const path = this.pathFor(normalizedProvider)
    let raw: string
    try {
      raw = await readFile(path, 'utf8')
    } catch (error) {
      // Only a missing file means "not logged in". Other filesystem failures
      // (EACCES, EISDIR, ...) must surface instead of masquerading as absence.
      if (isMissingFile(error)) {
        return undefined
      }
      throw error
    }

    try {
      const parsed = JSON.parse(raw) as unknown
      if (!isEncryptedCredential(parsed)) {
        // Read legacy plaintext records so users can upgrade without losing tokens.
        return OAuthToken.fromRecord(parsed)
      }
      const key = await this.encryptionKey()
      const plain = decrypt(normalizedProvider, parsed, key)
      return OAuthToken.fromRecord(JSON.parse(plain.toString('utf8')) as unknown)
    } catch {
      return undefined
    }
  }

  /** Remove a provider credential. Returns true only when a file was removed. */
  async remove(provider: string): Promise<boolean> {
    const path = this.pathFor(validatedProvider(provider))
    try {
      await unlink(path)
      return true
    } catch (error) {
      if (isMissingFile(error)) {
        return false
      }
      throw error
    }
  }

  /** Return stored provider names in lexical order. */
  async listProviders(): Promise<string[]> {
    try {
      const entries = await readdir(this.baseDirectory, { encoding: 'utf8', withFileTypes: true })
      return entries
        .filter(entry => entry.isFile() && entry.name.endsWith('.json'))
        .map(entry => entry.name.slice(0, -'.json'.length))
        .filter(name => PROVIDER_NAME.test(name))
        .sort()
    } catch (error) {
      if (isMissingFile(error)) {
        return []
      }
      throw error
    }
  }

  private pathFor(provider: string): string {
    return join(this.baseDirectory, `${provider}.json`)
  }

  private encryptionKey(): Promise<Buffer> {
    this.keyPromise ??= this.createEncryptionKey()
    return this.keyPromise
  }

  private async createEncryptionKey(): Promise<Buffer> {
    if (this.configuredKey !== undefined) {
      return createHash('sha256').update(this.configuredKey).digest()
    }
    return readOrCreateKey(this.keyPath)
  }
}

let defaultStorage: CredentialStorage | undefined

export function defaultCredentialStorage(): CredentialStorage {
  defaultStorage ??= CredentialStorage.default()
  return defaultStorage
}

export async function save(provider: string, token: OAuthToken): Promise<string> {
  return defaultCredentialStorage().save(provider, token)
}

export async function load(provider: string): Promise<OAuthToken | undefined> {
  return defaultCredentialStorage().load(provider)
}

export async function remove(provider: string): Promise<boolean> {
  return defaultCredentialStorage().remove(provider)
}

export async function listProviders(): Promise<string[]> {
  return defaultCredentialStorage().listProviders()
}

async function writeAtomic(path: string, data: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: CREDENTIAL_DIRECTORY_MODE })
  const temporaryPath = join(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomBytes(12).toString('hex')}.tmp`,
  )
  let handle: Awaited<ReturnType<typeof open>> | undefined
  try {
    handle = await open(temporaryPath, 'wx', CREDENTIAL_FILE_MODE)
    await handle.writeFile(data, 'utf8')
    await handle.sync()
    await handle.close()
    handle = undefined
    await rename(temporaryPath, path)
  } catch (error) {
    await handle?.close().catch(() => undefined)
    await unlink(temporaryPath).catch(() => undefined)
    throw error
  }
}

async function readOrCreateKey(path: string): Promise<Buffer> {
  try {
    return validateKey(await readFile(path))
  } catch (error) {
    if (!isMissingFile(error)) {
      throw error
    }
  }

  await mkdir(dirname(path), { recursive: true, mode: CREDENTIAL_DIRECTORY_MODE })
  const fresh = randomBytes(KEY_BYTES)
  let handle: Awaited<ReturnType<typeof open>> | undefined
  try {
    handle = await open(path, 'wx', CREDENTIAL_FILE_MODE)
    await handle.writeFile(fresh)
    await handle.sync()
    await handle.close()
    return fresh
  } catch (error) {
    await handle?.close().catch(() => undefined)
    if (!isAlreadyExists(error)) {
      throw error
    }
    return validateKey(await readFile(path))
  }
}

function validateKey(value: Buffer): Buffer {
  if (value.length !== KEY_BYTES) {
    throw new Error(`Credential key at rest must be exactly ${KEY_BYTES} bytes`)
  }
  return value
}

function encrypt(provider: string, plain: Buffer, key: Buffer): EncryptedCredential {
  const iv = randomBytes(IV_BYTES)
  const cipher = createCipheriv(ENCRYPTION_ALGORITHM, key, iv)
  cipher.setAAD(Buffer.from(aad(provider), 'utf8'))
  const ciphertext = Buffer.concat([cipher.update(plain), cipher.final()])
  return {
    version: ENCRYPTION_VERSION,
    algorithm: ENCRYPTION_ALGORITHM,
    iv: base64Url(iv),
    tag: base64Url(cipher.getAuthTag()),
    ciphertext: base64Url(ciphertext),
  }
}

function decrypt(provider: string, envelope: EncryptedCredential, key: Buffer): Buffer {
  const decipher = createDecipheriv(ENCRYPTION_ALGORITHM, key, fromBase64Url(envelope.iv))
  decipher.setAAD(Buffer.from(aad(provider), 'utf8'))
  decipher.setAuthTag(fromBase64Url(envelope.tag))
  return Buffer.concat([decipher.update(fromBase64Url(envelope.ciphertext)), decipher.final()])
}

function aad(provider: string): string {
  return `xerxes/oauth/v${ENCRYPTION_VERSION}/${provider}`
}

function isEncryptedCredential(value: unknown): value is EncryptedCredential {
  if (!isRecord(value)) {
    return false
  }
  return value.version === ENCRYPTION_VERSION
    && value.algorithm === ENCRYPTION_ALGORITHM
    && typeof value.iv === 'string'
    && typeof value.tag === 'string'
    && typeof value.ciphertext === 'string'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function validatedProvider(provider: string): string {
  if (!PROVIDER_NAME.test(provider)) {
    throw new Error('Credential provider must be a simple non-empty filename')
  }
  return provider
}

function nonEmptyEnvironmentKey(): string | undefined {
  const value = process.env.XERXES_CREDENTIAL_KEY
  return value ? value : undefined
}

function isMissingFile(error: unknown): boolean {
  return isNodeError(error, 'ENOENT')
}

function isAlreadyExists(error: unknown): boolean {
  return isNodeError(error, 'EEXIST')
}

function isNodeError(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code
}

function base64Url(value: Uint8Array): string {
  return Buffer.from(value).toString('base64').replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
}

function fromBase64Url(value: string): Buffer {
  const padding = '='.repeat((4 - value.length % 4) % 4)
  return Buffer.from(`${value.replaceAll('-', '+').replaceAll('_', '/')}${padding}`, 'base64')
}
