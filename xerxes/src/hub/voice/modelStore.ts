// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { chmod, lstat, mkdir, open, rename, unlink } from 'node:fs/promises'
import type { BunFile } from 'bun'
import { dirname, join } from 'node:path'
import {
  VOICE_MODELS, VOICE_MODEL_TIERS, resolveVoiceModelFile,
  type ResolvedVoiceModelFile, type VoiceModel, type VoiceModelId, type VoiceModelTier,
} from '../../client/voiceModelManifest.js'

export type VoiceModelFetch = (url: string, init: RequestInit) => Promise<Response>
export interface HubVoiceModelStoreOptions {
  readonly stateDir: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetch?: VoiceModelFetch
  /** Offline fixtures can replace the manifest and its resolver together. */
  readonly models?: Readonly<Record<VoiceModelId, VoiceModel>>
  readonly resolveFile?: (relative: string) => ResolvedVoiceModelFile | undefined
  readonly timeoutMs?: number
  /** Receives only sanitized failures, never the original network/filesystem error. */
  readonly onError?: (error: VoiceModelsUnavailable, context: string) => void
}
export interface VoiceModelReadiness { readonly bytes: number; readonly ready: boolean }
export interface HubVoiceModelStatus {
  readonly enabled: boolean
  readonly tiers: Readonly<Record<VoiceModelTier, VoiceModelReadiness>>
  readonly models: Readonly<Record<VoiceModelId, VoiceModelReadiness>>
}

/** Only fixed, public-safe reasons cross the HTTP/CLI boundary. */
export class VoiceModelsUnavailable extends Error {
  constructor(readonly reason: string = 'Model storage or download failed.', cause = 'storage or network failure') {
    super(reason, { cause }); this.name = 'VoiceModelsUnavailable'
  }
}

function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const aborted = (): void => {
      signal.removeEventListener('abort', aborted)
      reject(new VoiceModelsUnavailable('Model download was interrupted or timed out.', 'cancelled or deadline exceeded'))
    }
    signal.addEventListener('abort', aborted, { once: true })
    // Always observe the original promise, even when the signal was already aborted.
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', aborted))
    if (signal.aborted) aborted()
  })
}

/** Disk-backed, pinned allowlist. No provider credentials, URL input or arbitrary paths are accepted. */
export class HubVoiceModelStore {
  readonly enabled: boolean
  private readonly models: Readonly<Record<VoiceModelId, VoiceModel>>
  private readonly fetchPort: VoiceModelFetch
  private readonly baseUrl: string
  private readonly timeoutMs: number
  private readonly pending = new Map<string, Promise<string>>()
  private readonly verified = new Map<string, string>()
  private readonly shutdown = new AbortController()
  private readonly waiters = new Set<() => void>()
  private active = 0

  constructor(private readonly options: HubVoiceModelStoreOptions) {
    const env = options.environment ?? process.env
    this.enabled = env.XERXES_HUB_VOICE_MODELS !== 'off'
    this.baseUrl = env.XERXES_HUB_VOICE_MODELS_URL ?? 'https://huggingface.co'
    this.models = options.models ?? VOICE_MODELS
    this.fetchPort = options.fetch ?? ((url, init) => fetch(url, init))
    this.timeoutMs = options.timeoutMs ?? 5 * 60_000
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1 || this.timeoutMs > 5 * 60_000) throw new RangeError('Invalid model download timeout')
  }

  resolve(relative: string): ResolvedVoiceModelFile | undefined {
    if (relative.length > 300 || /[\0\\]|(^|\/)\.\.?(\/|$)/u.test(relative)) return undefined
    return (this.options.resolveFile ?? resolveVoiceModelFile)(relative)
  }

  files(tier: VoiceModelTier | 'all'): readonly ResolvedVoiceModelFile[] {
    const ids = tier === 'all' ? Object.keys(this.models) as VoiceModelId[] : VOICE_MODEL_TIERS[tier]
    return ids.flatMap(id => this.models[id].files.map(file => ({ id, model: this.models[id], file })))
  }

  private filePath(resolved: ResolvedVoiceModelFile): string {
    const model = this.models[resolved.id]
    const file = model?.files.find(entry => entry.path === resolved.file.path && entry.sha256 === resolved.file.sha256 && entry.size === resolved.file.size)
    if (!model || !file || model.repo !== resolved.model.repo || model.revision !== resolved.model.revision
      || !/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/u.test(model.repo) || !/^[0-9a-f]{40}$/u.test(model.revision)
      || !/^[A-Za-z0-9._/-]+$/u.test(file.path) || file.path.startsWith('/') || /(^|\/)\.\.?(\/|$)/u.test(`${model.repo}/${file.path}`)
      || !Number.isSafeInteger(file.size) || file.size < 0 || !/^[0-9a-f]{64}$/u.test(file.sha256)) {
      throw new VoiceModelsUnavailable('Model is not in the pinned manifest.')
    }
    return join(this.options.stateDir, 'voice-models', model.repo, model.revision, file.path)
  }

  async status(): Promise<HubVoiceModelStatus> {
    const entries = await Promise.all((Object.keys(this.models) as VoiceModelId[]).map(async id => {
      const files = this.files('all').filter(entry => entry.id === id)
      const ready = await Promise.all(files.map(async entry => {
        const stat = await lstat(this.filePath(entry)).catch((error: unknown) => {
          if (isMissing(error)) return undefined
          throw new VoiceModelsUnavailable()
        })
        return stat?.isFile() === true && stat.size === entry.file.size
      }))
      return [id, { bytes: files.reduce((sum, entry) => sum + entry.file.size, 0), ready: ready.every(Boolean) }] as const
    }))
    const models = Object.fromEntries(entries) as Record<VoiceModelId, VoiceModelReadiness>
    const tier = (name: VoiceModelTier): VoiceModelReadiness => ({
      bytes: VOICE_MODEL_TIERS[name].reduce((sum, id) => sum + models[id].bytes, 0),
      ready: VOICE_MODEL_TIERS[name].every(id => models[id].ready),
    })
    return { enabled: this.enabled, models, tiers: { standard: tier('standard'), light: tier('light') } }
  }

  get(resolved: ResolvedVoiceModelFile): Promise<string> {
    if (!this.enabled) return Promise.reject(new VoiceModelsUnavailable('Model downloads are disabled.'))
    if (this.shutdown.signal.aborted) return Promise.reject(new VoiceModelsUnavailable('Model store is closed.'))
    const path = this.filePath(resolved)
    const existing = this.pending.get(path)
    if (existing) return existing
    // Start the deadline before admission: a queued file cannot wait forever.
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)
    const signal = AbortSignal.any([this.shutdown.signal, controller.signal])
    const job = this.withSlot(signal, () => this.prepare(resolved, path, signal)).catch((error: unknown) => {
      const sanitized = error instanceof VoiceModelsUnavailable ? error : new VoiceModelsUnavailable()
      this.options.onError?.(sanitized, 'voice model download')
      throw sanitized
    }).finally(() => { clearTimeout(timer); this.pending.delete(path) })
    this.pending.set(path, job)
    return job
  }

  /** Prepare Bun's native file response inside admission, retrying a disappearing cache file once. */
  async openFile(resolved: ResolvedVoiceModelFile): Promise<BunFile> {
    if (!this.enabled) throw new VoiceModelsUnavailable('Model downloads are disabled.')
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)
    const signal = AbortSignal.any([this.shutdown.signal, controller.signal])
    try {
      // Share the initial download with every other caller before entering the
      // short file/stat admission section. The outer timer includes both waits.
      await abortable(this.get(resolved), signal)
      return await this.withSlot(signal, async () => {
        const path = this.filePath(resolved)
        for (let attempt = 0; attempt < 2; attempt += 1) {
          await this.prepare(resolved, path, signal)
          const file = Bun.file(path)
          try {
            const stat = await file.stat()
            if (!stat.isFile() || stat.size !== resolved.file.size) throw new VoiceModelsUnavailable('Model storage or download failed.', 'cached size changed')
            return file
          } catch (error) {
            if (isMissing(error) && attempt === 0) continue
            throw error
          }
        }
        throw new VoiceModelsUnavailable()
      })
    } catch (error) {
      throw error instanceof VoiceModelsUnavailable ? error : new VoiceModelsUnavailable()
    } finally { clearTimeout(timer) }
  }

  async close(): Promise<void> {
    this.shutdown.abort()
    await Promise.allSettled([...this.pending.values()])
  }

  private async withSlot<T>(signal: AbortSignal, run: () => Promise<T>): Promise<T> {
    // A released slot belongs to the next waiter, not a newly arriving request.
    if (this.active >= 2) await new Promise<void>((resolve, reject) => {
      const acquired = (): void => { cleanup(); resolve() }
      const aborted = (): void => {
        this.waiters.delete(acquired); cleanup()
        reject(new VoiceModelsUnavailable('Model download was interrupted or timed out.', 'queue deadline exceeded or cancelled'))
      }
      const cleanup = (): void => signal.removeEventListener('abort', aborted)
      this.waiters.add(acquired)
      signal.addEventListener('abort', aborted, { once: true })
      if (signal.aborted) aborted()
    })
    else this.active += 1
    try {
      if (signal.aborted) throw new VoiceModelsUnavailable('Model download was interrupted or timed out.', 'cancelled or deadline exceeded')
      return await run()
    } finally {
      const next = this.waiters.values().next().value
      if (next) { this.waiters.delete(next); next() } else this.active -= 1
    }
  }

  private async prepare(resolved: ResolvedVoiceModelFile, path: string, signal: AbortSignal): Promise<string> {
    const stat = await lstat(path).catch((error: unknown) => { if (isMissing(error)) return undefined; throw error })
    if (stat !== undefined) {
      const fingerprint = `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`
      try {
        if (stat.isFile() && stat.size === resolved.file.size
          && (this.verified.get(path) === fingerprint || await this.hashFile(path, signal) === resolved.file.sha256)) {
          this.verified.set(path, fingerprint)
          return path
        }
      } catch (error) { if (!isMissing(error)) throw error }
      this.verified.delete(path)
      await unlink(path).catch((error: unknown) => { if (!isMissing(error)) throw error })
    }
    await this.download(resolved, path, signal)
    return path
  }

  private async hashFile(path: string, signal: AbortSignal): Promise<string> {
    const reader = Bun.file(path).stream().getReader()
    const hash = createHash('sha256')
    try {
      for (;;) {
        const chunk = await abortable(reader.read(), signal)
        if (chunk.done) return hash.digest('hex')
        hash.update(chunk.value)
      }
    } finally { releaseReader(reader) }
  }

  private async download(resolved: ResolvedVoiceModelFile, path: string, signal: AbortSignal): Promise<void> {
    // mkdir's recursive mode only applies to newly created dirs: tighten each cache parent as well.
    const root = join(this.options.stateDir, 'voice-models')
    let parent = root
    for (const segment of ['', ...`${resolved.model.repo}/${resolved.model.revision}/${dirname(resolved.file.path)}`.split('/').filter(part => part !== '.')]) {
      parent = segment ? join(parent, segment) : parent
      await mkdir(parent, { recursive: true, mode: 0o700 })
      await chmod(parent, 0o700)
    }
    const temporary = `${path}.${randomUUID()}.tmp`
    let handle: Awaited<ReturnType<typeof open>> | undefined
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
    try {
      let url: URL
      try { url = new URL(`${this.baseUrl.replace(/\/+$/u, '')}/${resolved.model.repo}/resolve/${resolved.model.revision}/${resolved.file.path}`) }
      catch { throw new VoiceModelsUnavailable('Model mirror configuration is invalid.') }
      if (url.protocol !== 'https:' || url.username || url.password) throw new VoiceModelsUnavailable('Model mirror must use HTTPS without credentials.')
      const response = await this.fetchPinned(url, signal)
      if (!response.ok || !response.body) {
        void response.body?.cancel().catch(() => undefined)
        throw new VoiceModelsUnavailable('Model download was refused by the upstream server.', `upstream status ${response.status}`)
      }
      // Hash/count the decoded stream, not Content-Length (which can describe compressed bytes).
      reader = response.body.getReader()
      handle = await open(temporary, 'wx', 0o600)
      const hash = createHash('sha256')
      let size = 0
      for (;;) {
        const chunk = await abortable(reader.read(), signal)
        if (chunk.done) break
        size += chunk.value.byteLength
        if (size > resolved.file.size) throw new VoiceModelsUnavailable('Model size exceeded the pinned manifest.', 'size exceeded')
        hash.update(chunk.value)
        let offset = 0
        while (offset < chunk.value.byteLength) {
          const { bytesWritten } = await handle.write(chunk.value, offset, chunk.value.byteLength - offset)
          if (bytesWritten < 1) throw new VoiceModelsUnavailable('Model storage or download failed.', 'file write made no progress')
          offset += bytesWritten
        }
      }
      if (size !== resolved.file.size) throw new VoiceModelsUnavailable('Model verification did not match the pinned manifest.', 'size mismatch')
      if (hash.digest('hex') !== resolved.file.sha256) throw new VoiceModelsUnavailable('Model verification did not match the pinned manifest.', 'hash mismatch')
      signal.throwIfAborted()
      await handle.close(); handle = undefined
      await rename(temporary, path)
      // The just-downloaded bytes have already been verified, so do not hash them twice.
      const stat = await lstat(path)
      this.verified.set(path, `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`)
    } finally {
      if (reader) releaseReader(reader)
      try { await handle?.close() } finally {
        await unlink(temporary).catch((error: unknown) => { if (!isMissing(error)) throw error })
      }
    }
  }

  private async fetchPinned(initial: URL, signal: AbortSignal): Promise<Response> {
    let url = initial
    for (let redirects = 0; redirects <= 5; redirects += 1) {
      const response = await abortable(this.fetchPort(url.href, { redirect: 'manual', signal, credentials: 'omit', referrerPolicy: 'no-referrer' }), signal)
      if (![301, 302, 303, 307, 308].includes(response.status)) return response
      void response.body?.cancel().catch(() => undefined)
      const location = response.headers.get('location')
      if (!location) throw new VoiceModelsUnavailable('Model download returned an invalid redirect.', 'invalid redirect')
      try { url = new URL(location, url) } catch { throw new VoiceModelsUnavailable('Model download returned an invalid redirect.', 'invalid redirect') }
      if (url.protocol !== 'https:' || url.username || url.password) throw new VoiceModelsUnavailable('Model redirect was not secure.', 'insecure redirect')
    }
    throw new VoiceModelsUnavailable('Model download exceeded the redirect limit.')
  }
}

function releaseReader(reader: ReadableStreamDefaultReader<Uint8Array>): void {
  try { void reader.cancel().catch(() => undefined) } catch { /* Already detached; continue closing the file. */ }
  try { reader.releaseLock() } catch { /* Cleanup must never prevent fd close or temporary unlink. */ }
}

function isMissing(error: unknown): boolean {
  return error !== null && typeof error === 'object' && 'code' in error && error.code === 'ENOENT'
}
