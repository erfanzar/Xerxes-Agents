// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { chmod, mkdir, open, readFile, rename, unlink } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'

import { credentialEncryptionKey } from '../auth/storage.js'
import { PLUGIN_SECRET_NAME } from './plugins.js'

const ALGORITHM = 'aes-256-gcm'
const PLUGIN_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/
const MAX_SECRET_BYTES = 16_384

interface SealedValue { readonly iv: string; readonly tag: string; readonly data: string }
type SecretDocument = Record<string, Record<string, SealedValue>>

/** Where secret values come from; injected so tests never touch the user's key. */
export interface PluginSecretStoreOptions {
  /** Encryption key material (hashed to 256 bits). Defaults to the credential key. */
  readonly key?: () => Promise<Buffer>
}

/**
 * Per-plugin secret values, encrypted at rest with the same key as provider
 * credentials and stored in a `0600` file. Values are only ever returned to the
 * plugin call that declared them; list() returns names and whether they are set.
 */
export class PluginSecretStore {
  private queue: Promise<unknown> = Promise.resolve()

  constructor(private readonly path: string, private readonly options: PluginSecretStoreOptions = {}) {}

  /** Names set for a plugin, sorted. Never values. */
  async names(plugin: string): Promise<string[]> {
    return Object.keys((await this.read())[validPlugin(plugin)] ?? {}).sort()
  }

  async set(plugin: string, name: string, value: string): Promise<void> {
    validPlugin(plugin); validName(name)
    if (typeof value !== 'string' || !value.length) throw new TypeError('Secret value must be a non-empty string')
    if (Buffer.byteLength(value) > MAX_SECRET_BYTES) throw new TypeError(`Secret value must be at most ${MAX_SECRET_BYTES} bytes`)
    const key = await this.key()
    await this.mutate(document => {
      document[plugin] = { ...(document[plugin] ?? {}), [name]: seal(key, plugin, name, value) }
    })
  }

  /** Returns whether a value was removed. */
  async unset(plugin: string, name: string): Promise<boolean> {
    validPlugin(plugin); validName(name)
    let removed = false
    await this.mutate(document => {
      const entries = document[plugin]
      if (!entries || !(name in entries)) return
      removed = true
      delete entries[name]
      if (!Object.keys(entries).length) delete document[plugin]
    })
    return removed
  }

  /** Decrypted values of exactly the requested names that are set. */
  async values(plugin: string, names: readonly string[]): Promise<Record<string, string>> {
    if (!names.length) return {}
    const entries = (await this.read())[validPlugin(plugin)] ?? {}
    const wanted = names.filter(name => name in entries)
    if (!wanted.length) return {}
    const key = await this.key()
    return Object.fromEntries(wanted.map(name => [name, open_(key, plugin, name, entries[name]!)]))
  }

  private key(): Promise<Buffer> {
    return this.options.key ? this.options.key() : credentialEncryptionKey()
  }

  private async read(): Promise<SecretDocument> {
    let raw: string
    try { raw = await readFile(this.path, 'utf8') } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}
      throw error
    }
    const parsed: unknown = JSON.parse(raw)
    if (!isRecord(parsed)) throw new Error('Plugin secret store is malformed')
    const document: SecretDocument = {}
    for (const [plugin, entries] of Object.entries(parsed)) {
      if (!PLUGIN_NAME.test(plugin) || !isRecord(entries)) throw new Error('Plugin secret store is malformed')
      document[plugin] = {}
      for (const [name, sealed] of Object.entries(entries)) {
        if (!PLUGIN_SECRET_NAME.test(name) || !isRecord(sealed) || typeof sealed.iv !== 'string' || typeof sealed.tag !== 'string' || typeof sealed.data !== 'string') {
          throw new Error('Plugin secret store is malformed')
        }
        document[plugin]![name] = { iv: sealed.iv, tag: sealed.tag, data: sealed.data }
      }
    }
    return document
  }

  private mutate(change: (document: SecretDocument) => void): Promise<void> {
    const operation = this.queue.then(async () => {
      const document = await this.read()
      change(document)
      await writePrivate(this.path, `${JSON.stringify(document, null, 2)}\n`)
    })
    this.queue = operation.catch(() => undefined)
    return operation
  }
}

function seal(key: Buffer, plugin: string, name: string, value: string): SealedValue {
  const iv = randomBytes(12)
  const cipher = createCipheriv(ALGORITHM, key, iv)
  cipher.setAAD(Buffer.from(`xerxes/plugin-secret/v1/${plugin}/${name}`))
  const data = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()])
  return { iv: iv.toString('base64'), tag: cipher.getAuthTag().toString('base64'), data: data.toString('base64') }
}

function open_(key: Buffer, plugin: string, name: string, sealed: SealedValue): string {
  try {
    const decipher = createDecipheriv(ALGORITHM, key, Buffer.from(sealed.iv, 'base64'))
    decipher.setAAD(Buffer.from(`xerxes/plugin-secret/v1/${plugin}/${name}`))
    decipher.setAuthTag(Buffer.from(sealed.tag, 'base64'))
    return Buffer.concat([decipher.update(Buffer.from(sealed.data, 'base64')), decipher.final()]).toString('utf8')
  } catch {
    // Never include the ciphertext or value in the error.
    throw new Error(`Secret ${name} of plugin ${plugin} could not be decrypted; the key changed or the store was modified. Set it again.`)
  }
}

async function writePrivate(path: string, data: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`)
  let handle: Awaited<ReturnType<typeof open>> | undefined
  try {
    handle = await open(temporary, 'wx', 0o600)
    await handle.writeFile(data, 'utf8')
    await handle.sync()
    await handle.close()
    handle = undefined
    await rename(temporary, path)
    await chmod(path, 0o600)
  } catch (error) {
    await handle?.close().catch(() => undefined)
    await unlink(temporary).catch(() => undefined)
    throw error
  }
}

function validPlugin(plugin: string): string {
  if (typeof plugin !== 'string' || !PLUGIN_NAME.test(plugin)) throw new TypeError('Plugin name must be a simple name (letters, digits, _ . -)')
  return plugin
}

function validName(name: string): string {
  if (typeof name !== 'string' || !PLUGIN_SECRET_NAME.test(name)) throw new TypeError('Secret names are UPPER_SNAKE_CASE (A-Z, 0-9, _; at most 64 characters)')
  return name
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
