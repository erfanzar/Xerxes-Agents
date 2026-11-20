// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { constants, type Stats } from 'node:fs'
import { lstat, mkdir, open, rename, unlink } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { SlackTransportError, type SlackIdentity } from './types.js'
import { identity, record, userToken } from './wire.js'

const MAX_RECORD_BYTES = 2048

function missing(error: unknown): boolean {
  return record(error)?.code === 'ENOENT'
}

function owned(info: Stats): boolean {
  return process.geteuid !== undefined && info.uid === process.geteuid()
}

function privateFile(info: Stats): boolean {
  return info.isFile() && owned(info) && (info.mode & 0o777) === 0o600 && info.nlink === 1
}

/** Host-only token files; not a Keychain and not encrypted. Never use a workspace path. */
export class SlackCredentialStore {
  readonly #stateDir: string
  readonly #directory: string

  constructor(stateDir: string) {
    if (!stateDir || stateDir.includes('\0')) throw new SlackTransportError('invalid_request')
    this.#stateDir = resolve(stateDir)
    this.#directory = join(this.#stateDir, 'slack-credentials')
  }

  async #path(account: SlackIdentity, create: boolean): Promise<string | undefined> {
    if (create) await mkdir(this.#stateDir, { recursive: true, mode: 0o700 })
    try {
      const state = await lstat(this.#stateDir)
      if (!state.isDirectory() || !owned(state) || (state.mode & 0o022) !== 0) throw new SlackTransportError('storage_error')
      if (create) await mkdir(this.#directory, { recursive: true, mode: 0o700 })
      const dir = await lstat(this.#directory)
      if (!dir.isDirectory() || !owned(dir) || (dir.mode & 0o777) !== 0o700) throw new SlackTransportError('storage_error')
      return join(this.#directory, `${account.teamId}-${account.userId}.json`)
    } catch (error) {
      if (!create && missing(error)) return undefined
      throw error
    }
  }

  async load(input: SlackIdentity): Promise<string | undefined> {
    const account = identity(input)
    try {
      const path = await this.#path(account, false)
      if (!path) return undefined
      let file
      try { file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK) }
      catch (error) { if (missing(error)) return undefined; throw error }
      try {
        const info = await file.stat()
        if (!privateFile(info) || info.size < 1 || info.size > MAX_RECORD_BYTES) throw new SlackTransportError('storage_error')
        const bytes = Buffer.alloc(info.size + 1)
        const { bytesRead } = await file.read(bytes, 0, bytes.length, 0)
        if (bytesRead !== info.size) throw new SlackTransportError('storage_error')
        const data = record(JSON.parse(bytes.toString('utf8', 0, bytesRead)))
        if (data?.version !== 1 || data.teamId !== account.teamId || data.userId !== account.userId || !userToken(data.token)) {
          throw new SlackTransportError('storage_error')
        }
        return data.token
      } finally { await file.close() }
    } catch { throw new SlackTransportError('storage_error') }
  }

  async save(input: SlackIdentity, token: string, signal?: AbortSignal): Promise<void> {
    const account = identity(input)
    if (!userToken(token)) throw new SlackTransportError('invalid_request')
    let temporary: string | undefined
    try {
      const path = await this.#path(account, true)
      if (!path) throw new SlackTransportError('storage_error')
      try {
        if (!privateFile(await lstat(path))) throw new SlackTransportError('storage_error')
      } catch (error) { if (!missing(error)) throw error }
      const candidate = join(this.#directory, `.${randomUUID()}.tmp`)
      const file = await open(candidate, 'wx', 0o600)
      temporary = candidate
      try {
        await file.chmod(0o600)
        await file.writeFile(`${JSON.stringify({ version: 1, ...account, token })}\n`)
        await file.sync()
      } finally { await file.close() }
      if (!signal?.aborted) {
        await rename(temporary, path)
        temporary = undefined
        return
      }
    } catch { throw new SlackTransportError('storage_error') }
    finally {
      if (temporary) {
        try { await unlink(temporary) } catch (error) { if (!missing(error)) throw new SlackTransportError('storage_error') }
      }
    }
    throw new SlackTransportError('aborted')
  }

  async remove(input: SlackIdentity): Promise<void> {
    const account = identity(input)
    try {
      const path = await this.#path(account, false)
      if (path) await unlink(path)
    } catch (error) { if (!missing(error)) throw new SlackTransportError('storage_error') }
  }
}
