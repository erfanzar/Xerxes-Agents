// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { mkdir, open, readFile, rename, rm, stat } from 'node:fs/promises'
import { basename, dirname, resolve } from 'node:path'

/**
 * Session metadata that grows with the work a session does rather than with
 * its conversation: every agent it ever ran, every goal change, every workflow
 * script. A long session held 17.8 MB of this against 1.4 MB of messages, and
 * the whole lot was re-serialized, rewritten and fsynced on every save, even
 * when nothing in it had changed.
 *
 * These keys stay complete in memory. On disk each element is stored once, by
 * content hash, in an append-only sidecar beside the transcript; the transcript
 * keeps only the ordered hashes. A save appends the elements that are new and
 * nothing else, so its cost follows what changed instead of what accumulated.
 */
export const SIDECAR_METADATA_KEYS: readonly string[] = Object.freeze([
  'xerxes_subagent_snapshots_v1',
  'xerxes_subagent_deliveries_v1',
  'xerxes_workflow_runs_v1',
  'goal_changes',
])

/** Top-level transcript field holding the hashes. Written after `messages`, so head reads never see it. */
export const METADATA_SIDECAR_FIELD = 'metadata_sidecar'

/** A sidecar this small is never worth rewriting. */
const COMPACT_MIN_BYTES = 1024 * 1024
const HASH = /^[0-9a-f]{32}$/
const LINE_PREFIX = /^\{"h":"([0-9a-f]{32})","v":/

export type MetadataSidecarRef =
  | { readonly array: readonly string[] }
  | { readonly object: readonly (readonly [string, string])[] }

export interface SplitMetadata {
  /** Metadata with the sidecar keys removed. */
  readonly metadata: Record<string, unknown>
  /** How to rebuild each removed key from the sidecar. */
  readonly refs: Readonly<Record<string, MetadataSidecarRef>>
  /** Every element the refs name, as hash → serialized value. */
  readonly entries: ReadonlyMap<string, string>
}

/** `<id>.json` → `<id>.metadata.jsonl`. Not `<id>.jsonl`, which is the message journal. */
export function sidecarPathFor(transcriptPath: string): string {
  return `${transcriptPath.replace(/\.json$/u, '')}.metadata.jsonl`
}

function entryHash(json: string): string {
  return createHash('sha256').update(json).digest('hex').slice(0, 32)
}

/**
 * Serialized form of each element object, by identity. A live session keeps
 * its agent rows and goal changes as the same objects from save to save (the
 * writers replace a changed row with a new object and never mutate one in
 * place), so an unchanged element is serialized and hashed once, not on every
 * save. Without this, splitting re-encoded all 17.8 MB of a long session's
 * metadata each time and a save cost more than the inline write it replaced.
 * Frozen so an in-place edit throws instead of being saved under a stale hash.
 */
const encoded = new WeakMap<object, { readonly hash: string; readonly json: string }>()

function encode(value: unknown): { readonly hash: string; readonly json: string } {
  const cacheable = typeof value === 'object' && value !== null
  if (cacheable) {
    const known = encoded.get(value)
    if (known) return known
  }
  const json = JSON.stringify(value) ?? 'null'
  const result = { hash: entryHash(json), json }
  if (cacheable) {
    encoded.set(value, result)
    deepFreeze(value)
  }
  return result
}

function deepFreeze(value: object): void {
  if (Object.isFrozen(value)) return
  Object.freeze(value)
  for (const child of Object.values(value)) if (typeof child === 'object' && child !== null) deepFreeze(child)
}

/** Move the growing keys out of `metadata`, element by element. Never touches disk. */
export function splitMetadata(metadata: Readonly<Record<string, unknown>>): SplitMetadata {
  const kept: Record<string, unknown> = { ...metadata }
  const refs: Record<string, MetadataSidecarRef> = {}
  const entries = new Map<string, string>()
  const add = (value: unknown): string => {
    const { hash, json } = encode(value)
    entries.set(hash, json)
    return hash
  }
  for (const key of SIDECAR_METADATA_KEYS) {
    const value = metadata[key]
    if (Array.isArray(value)) {
      refs[key] = { array: value.map(add) }
    } else if (isRecord(value)) {
      refs[key] = {
        object: Object.entries(value)
          .filter(([, item]) => item !== undefined)
          .map(([name, item]) => [name, add(item)] as const),
      }
    } else {
      continue
    }
    delete kept[key]
  }
  return { metadata: kept, refs, entries }
}

/** The transcript record as written: sidecar keys replaced by refs placed after everything else. */
export function recordWithSidecar(record: Readonly<Record<string, unknown>>, split: SplitMetadata): Record<string, unknown> {
  const written: Record<string, unknown> = { ...record, metadata: split.metadata }
  delete written[METADATA_SIDECAR_FIELD]
  return Object.keys(split.refs).length ? { ...written, [METADATA_SIDECAR_FIELD]: split.refs } : written
}

/**
 * Put sidecar values back into `metadata` in place. A key already present
 * inline wins: older transcripts and hand repairs store the value there.
 * Returns how many referenced elements could not be found.
 */
export function joinMetadata(
  metadata: Record<string, unknown>,
  refs: unknown,
  values: ReadonlyMap<string, unknown>,
): number {
  if (!isRecord(refs)) return 0
  let missing = 0
  const found = (hash: string): boolean => {
    if (values.has(hash)) return true
    missing += 1
    return false
  }
  for (const [key, raw] of Object.entries(refs)) {
    if (Object.hasOwn(metadata, key)) continue
    const ref = decodeRef(raw)
    if (!ref) { missing += 1; continue }
    metadata[key] = 'array' in ref
      ? ref.array.filter(found).map(hash => values.get(hash))
      : Object.fromEntries(ref.object.filter(([, hash]) => found(hash)).map(([name, hash]) => [name, values.get(hash)]))
  }
  return missing
}

/** Keys of `refs` whose values are not already inline, so the sidecar needs reading. */
export function unresolvedSidecarKeys(metadata: Readonly<Record<string, unknown>>, refs: unknown): string[] {
  return isRecord(refs) ? Object.keys(refs).filter(key => !Object.hasOwn(metadata, key)) : []
}

function decodeRef(value: unknown): MetadataSidecarRef | undefined {
  if (!isRecord(value)) return undefined
  if (Array.isArray(value.array) && value.array.every(isHash)) return { array: value.array as string[] }
  if (Array.isArray(value.object) && value.object.every(pair => Array.isArray(pair) && pair.length === 2 && typeof pair[0] === 'string' && isHash(pair[1]))) {
    return { object: value.object as [string, string][] }
  }
  return undefined
}

interface SidecarIndex {
  /** Hash → bytes of its line, for complete and well-formed lines only. */
  readonly sizes: ReadonlyMap<string, number>
  readonly size: number
  readonly mtimeMs: number
  readonly ino: number
  readonly endsWithNewline: boolean
}

interface SidecarScan {
  readonly index: SidecarIndex
  readonly values: Map<string, unknown>
  readonly lines: Map<string, string>
}

const EMPTY_INDEX: SidecarIndex = { sizes: new Map(), size: 0, mtimeMs: 0, ino: 0, endsWithNewline: true }

/**
 * Append-only, content-addressed storage for split metadata. Writers must hold
 * the transcript's lock. The hash index is cached per file and revalidated by
 * `stat`, so another process's write is never mistaken for our last view.
 */
export class MetadataSidecarStore {
  private readonly indexes = new Map<string, SidecarIndex>()

  /** Every stored element, hash → value. A missing file is an empty store. */
  async read(path: string): Promise<ReadonlyMap<string, unknown>> {
    return (await this.scan(path, true)).values
  }

  /**
   * Durably append the elements not stored yet. Must complete before the
   * transcript that names them is written: a transcript may only reference
   * what is already on disk.
   */
  async append(path: string, entries: ReadonlyMap<string, string>): Promise<number> {
    if (entries.size === 0) return 0
    const index = await this.index(path)
    const sizes = new Map(index.sizes)
    const fresh: string[] = []
    for (const [hash, json] of entries) {
      if (sizes.has(hash)) continue
      const line = `{"h":"${hash}","v":${json}}\n`
      fresh.push(line)
      sizes.set(hash, Buffer.byteLength(line))
    }
    if (fresh.length === 0) return 0
    // A crash mid-append leaves a torn final line; start ours on a fresh one.
    const text = (index.size > 0 && !index.endsWithNewline ? '\n' : '') + fresh.join('')
    await mkdir(dirname(path), { recursive: true })
    const handle = await open(path, 'a', 0o600)
    try {
      await handle.writeFile(text, 'utf8')
      await handle.sync()
    } finally {
      await handle.close()
    }
    const info = await stat(path)
    this.indexes.set(path, { sizes, size: info.size, mtimeMs: info.mtimeMs, ino: info.ino, endsWithNewline: true })
    return fresh.length
  }

  /**
   * Rewrite the sidecar with only `live` elements once superseded versions
   * (an agent's earlier progress, say) outweigh the live ones. Call only after
   * the transcript naming exactly `live` has been written. Returns whether it
   * rewrote. The replacement is atomic, so a crash keeps the old superset.
   */
  async compact(path: string, live: ReadonlySet<string>): Promise<boolean> {
    const index = await this.index(path)
    let liveBytes = 0
    for (const hash of live) liveBytes += index.sizes.get(hash) ?? 0
    if (index.size < COMPACT_MIN_BYTES || index.size <= liveBytes * 2) return false
    const scan = await this.scan(path, false)
    const kept = [...live].flatMap(hash => {
      const line = scan.lines.get(hash)
      return line === undefined ? [] : [line]
    })
    if (kept.length < [...live].filter(hash => scan.index.sizes.has(hash)).length) return false
    const temporary = resolve(dirname(path), `.${basename(path)}.${crypto.randomUUID()}.tmp`)
    try {
      const handle = await open(temporary, 'w', 0o600)
      try {
        await handle.writeFile(kept.map(line => `${line}\n`).join(''), 'utf8')
        await handle.sync()
      } finally {
        await handle.close()
      }
      await rename(temporary, path)
      if (process.platform !== 'win32') {
        const directory = await open(dirname(path), 'r')
        try { await directory.sync() } finally { await directory.close() }
      }
    } catch (error) {
      await rm(temporary, { force: true })
      throw error
    }
    this.indexes.delete(path)
    return true
  }

  async remove(path: string): Promise<void> {
    this.indexes.delete(path)
    await rm(path, { force: true })
  }

  private async index(path: string): Promise<SidecarIndex> {
    let info: Awaited<ReturnType<typeof stat>>
    try {
      info = await stat(path)
    } catch (error) {
      if (isMissing(error)) { this.indexes.delete(path); return EMPTY_INDEX }
      throw error
    }
    const cached = this.indexes.get(path)
    if (cached && cached.size === info.size && cached.mtimeMs === info.mtimeMs && cached.ino === info.ino) return cached
    return (await this.scan(path, false)).index
  }

  /**
   * Parse every line in full. Only complete, well-formed lines count as
   * stored: a torn line whose hash were trusted would never be re-appended,
   * and the transcript would reference an element that cannot be read.
   */
  private async scan(path: string, keepValues: boolean): Promise<SidecarScan> {
    let bytes: Buffer
    try {
      bytes = await readFile(path)
    } catch (error) {
      if (isMissing(error)) { this.indexes.delete(path); return { index: EMPTY_INDEX, values: new Map(), lines: new Map() } }
      throw error
    }
    const info = await stat(path)
    const sizes = new Map<string, number>()
    const values = new Map<string, unknown>()
    const lines = new Map<string, string>()
    const text = bytes.toString('utf8')
    let start = 0
    while (start < text.length) {
      const end = text.indexOf('\n', start)
      const complete = end >= 0
      const line = text.slice(start, complete ? end : text.length)
      start = complete ? end + 1 : text.length
      if (!complete || !LINE_PREFIX.test(line)) continue
      let parsed: unknown
      try { parsed = JSON.parse(line) } catch { continue }
      if (!isRecord(parsed) || !isHash(parsed.h) || !Object.hasOwn(parsed, 'v')) continue
      if (sizes.has(parsed.h)) continue
      sizes.set(parsed.h, Buffer.byteLength(line) + 1)
      lines.set(parsed.h, line)
      if (keepValues) values.set(parsed.h, parsed.v)
    }
    const index: SidecarIndex = {
      sizes,
      size: info.size,
      mtimeMs: info.mtimeMs,
      ino: info.ino,
      endsWithNewline: bytes.length === 0 || bytes[bytes.length - 1] === 0x0a,
    }
    this.indexes.set(path, index)
    return { index, values, lines }
  }
}

/**
 * Resolve a parsed transcript record's sidecar refs in place, removing the
 * refs field. Returns how many referenced elements were missing; 0 also when
 * the record has no sidecar.
 */
export async function joinMetadataSidecar(
  record: Record<string, unknown>,
  path: string,
  store: MetadataSidecarStore = new MetadataSidecarStore(),
): Promise<number> {
  if (!Object.hasOwn(record, METADATA_SIDECAR_FIELD)) return 0
  const refs = record[METADATA_SIDECAR_FIELD]
  delete record[METADATA_SIDECAR_FIELD]
  const metadata = isRecord(record.metadata) ? record.metadata : (record.metadata = {}) as Record<string, unknown>
  if (unresolvedSidecarKeys(metadata, refs).length === 0) return 0
  return joinMetadata(metadata, refs, await store.read(path))
}

function isHash(value: unknown): value is string {
  return typeof value === 'string' && HASH.test(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissing(error: unknown): boolean {
  return (error as NodeJS.ErrnoException | undefined)?.code === 'ENOENT'
}
