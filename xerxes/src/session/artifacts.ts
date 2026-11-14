// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Artifacts: titled documents an agent writes for the user (reports, plans,
 * specs, long write-ups) that live with the session rather than in the
 * repository.
 *
 * Layout: `<root>/<sessionId>/<id>.json`, one JSON record per artifact, where
 * `root` defaults to `<xerxes home>/artifacts`. Every write goes to a private
 * temporary file and is renamed into place, so a reader never sees half a
 * record. Everything read back from disk is validated; a corrupt or foreign
 * file is skipped, never repaired or deleted.
 */
import { mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../core/paths.js'

export const ARTIFACT_KINDS = ['markdown', 'html', 'code', 'text'] as const
export type ArtifactKind = typeof ARTIFACT_KINDS[number]

export const MAX_ARTIFACT_CONTENT_BYTES = 2 * 1024 * 1024
export const MAX_ARTIFACT_TITLE_CHARS = 200
export const MAX_ARTIFACTS_PER_SESSION = 200
export const MAX_ARTIFACT_LANGUAGE_CHARS = 40
/** JSON escaping can grow content up to six-fold (`\u0001`); anything larger is not ours. */
const MAX_ARTIFACT_FILE_BYTES = MAX_ARTIFACT_CONTENT_BYTES * 6 + 64 * 1024
const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/
const ARTIFACT_ID_PATTERN = /^[a-z0-9]{6,32}$/
const ARTIFACT_ID_LENGTH = 10

export interface Artifact {
  readonly id: string
  readonly title: string
  readonly kind: ArtifactKind
  readonly language?: string
  readonly content: string
  /** ISO 8601. */
  readonly created_at: string
  /** ISO 8601; changes on every content revision, not on pinning. */
  readonly updated_at: string
  readonly pinned: boolean
  /** 1 on create, +1 on every update. */
  readonly version: number
}

/** An artifact without its content, with the content's UTF-8 size in bytes. */
export type ArtifactSummary = Omit<Artifact, 'content'> & { readonly size: number }

export interface ArtifactDraft {
  readonly title: string
  readonly kind?: ArtifactKind
  readonly language?: string
  readonly content: string
}

export interface ArtifactRevision {
  readonly title?: string
  readonly kind?: ArtifactKind
  readonly language?: string
  readonly content: string
}

export interface ArtifactStoreOptions {
  /** Defaults to `<xerxes home>/artifacts`. */
  readonly root?: string
  readonly now?: () => Date
}

export function isArtifactKind(value: unknown): value is ArtifactKind {
  return typeof value === 'string' && (ARTIFACT_KINDS as readonly string[]).includes(value)
}

/** Per-session artifact storage under the Xerxes home. */
export class ArtifactStore {
  readonly root: string
  private readonly now: () => Date
  private readonly locks = new Map<string, Promise<void>>()

  constructor(options: ArtifactStoreOptions = {}) {
    this.root = options.root ?? join(xerxesHome(), 'artifacts')
    this.now = options.now ?? (() => new Date())
  }

  async create(sessionId: string, draft: ArtifactDraft): Promise<Artifact> {
    const title = normalizedTitle(draft.title)
    const kind = draft.kind ?? 'markdown'
    if (!isArtifactKind(kind)) throw new ValidationError('kind', `must be one of ${ARTIFACT_KINDS.join(', ')}`, kind)
    const language = normalizedLanguage(draft.language)
    const content = checkedContent(draft.content)
    return this.locked(sessionId, async directory => {
      const { artifacts, names } = await loadAll(directory)
      if (artifacts.length >= MAX_ARTIFACTS_PER_SESSION) {
        throw new ValidationError(
          'artifacts',
          `this session already holds the maximum of ${MAX_ARTIFACTS_PER_SESSION} artifacts; update an existing one instead`,
        )
      }
      const stamp = this.now().toISOString()
      const artifact: Artifact = {
        id: freshId(names),
        title,
        kind,
        ...(language === undefined ? {} : { language }),
        content,
        created_at: stamp,
        updated_at: stamp,
        pinned: false,
        version: 1,
      }
      await writeArtifact(directory, artifact)
      return artifact
    })
  }

  /** Replace the content (and any given title, kind or language) and bump the version. */
  async update(sessionId: string, id: string, revision: ArtifactRevision): Promise<Artifact> {
    const title = revision.title === undefined ? undefined : normalizedTitle(revision.title)
    if (revision.kind !== undefined && !isArtifactKind(revision.kind)) {
      throw new ValidationError('kind', `must be one of ${ARTIFACT_KINDS.join(', ')}`, revision.kind)
    }
    const language = normalizedLanguage(revision.language)
    const content = checkedContent(revision.content)
    return this.locked(sessionId, async directory => {
      const current = await readArtifact(directory, id)
      if (!current) throw new ValidationError('id', `no artifact ${id} in this session`, id)
      const next: Artifact = {
        ...current,
        ...(title === undefined ? {} : { title }),
        ...(revision.kind === undefined ? {} : { kind: revision.kind }),
        ...(language === undefined ? {} : { language }),
        content,
        updated_at: this.now().toISOString(),
        version: current.version + 1,
      }
      await writeArtifact(directory, next)
      return next
    })
  }

  /** Pin or unpin; neither the version nor `updated_at` changes. Undefined for an unknown id. */
  async pin(sessionId: string, id: string, pinned: boolean): Promise<ArtifactSummary | undefined> {
    return this.locked(sessionId, async directory => {
      const current = await readArtifact(directory, id)
      if (!current) return undefined
      const next: Artifact = { ...current, pinned }
      if (current.pinned !== pinned) await writeArtifact(directory, next)
      return summaryOf(next)
    })
  }

  async read(sessionId: string, id: string): Promise<Artifact | undefined> {
    return readArtifact(this.sessionDirectory(sessionId), id)
  }

  /** Every valid artifact of the session without content: pinned first, then newest revision first. */
  async list(sessionId: string): Promise<readonly ArtifactSummary[]> {
    const { artifacts } = await loadAll(this.sessionDirectory(sessionId))
    return artifacts.map(summaryOf).sort(compareSummaries)
  }

  private sessionDirectory(sessionId: string): string {
    if (typeof sessionId !== 'string' || !SESSION_ID_PATTERN.test(sessionId)) {
      throw new ValidationError('session_id', 'is not a valid session id', sessionId)
    }
    return join(this.root, sessionId)
  }

  /** Serialize mutations per session so a count check and its write cannot interleave. */
  private async locked<T>(sessionId: string, work: (directory: string) => Promise<T>): Promise<T> {
    const directory = this.sessionDirectory(sessionId)
    const previous = this.locks.get(directory) ?? Promise.resolve()
    const run = previous.then(() => work(directory))
    const tail = run.then(() => undefined, () => undefined)
    this.locks.set(directory, tail)
    try {
      return await run
    } finally {
      if (this.locks.get(directory) === tail) this.locks.delete(directory)
    }
  }
}

export function summaryOf(artifact: Artifact): ArtifactSummary {
  const { content, ...rest } = artifact
  return { ...rest, size: Buffer.byteLength(content, 'utf8') }
}

function compareSummaries(left: ArtifactSummary, right: ArtifactSummary): number {
  if (left.pinned !== right.pinned) return left.pinned ? -1 : 1
  if (left.updated_at !== right.updated_at) return left.updated_at < right.updated_at ? 1 : -1
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0
}

function normalizedTitle(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) throw new ValidationError('title', 'must be a non-empty string', value)
  const title = value.trim()
  if (title.length > MAX_ARTIFACT_TITLE_CHARS) {
    throw new ValidationError('title', `must be at most ${MAX_ARTIFACT_TITLE_CHARS} characters`, title.length)
  }
  return title
}

function normalizedLanguage(value: unknown): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string') throw new ValidationError('language', 'must be a string', value)
  const language = value.trim()
  if (!language) return undefined
  if (language.length > MAX_ARTIFACT_LANGUAGE_CHARS) {
    throw new ValidationError('language', `must be at most ${MAX_ARTIFACT_LANGUAGE_CHARS} characters`, language.length)
  }
  return language
}

function checkedContent(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) throw new ValidationError('content', 'must be a non-empty string', typeof value)
  const bytes = Buffer.byteLength(value, 'utf8')
  if (bytes > MAX_ARTIFACT_CONTENT_BYTES) {
    throw new ValidationError('content', `is ${bytes} bytes; the limit is ${MAX_ARTIFACT_CONTENT_BYTES} bytes (2 MiB)`, bytes)
  }
  return value
}

function freshId(taken: ReadonlySet<string>): string {
  for (;;) {
    const id = crypto.randomUUID().replaceAll('-', '').slice(0, ARTIFACT_ID_LENGTH)
    if (!taken.has(id)) return id
  }
}

async function loadAll(directory: string): Promise<{ artifacts: Artifact[]; names: Set<string> }> {
  let entries: string[]
  try {
    entries = await readdir(directory)
  } catch (error) {
    if (isMissing(error)) return { artifacts: [], names: new Set() }
    throw error
  }
  const ids = entries
    .filter(name => name.endsWith('.json'))
    .map(name => name.slice(0, -'.json'.length))
    .filter(id => ARTIFACT_ID_PATTERN.test(id))
  const loaded = await Promise.all(ids.map(id => readArtifact(directory, id)))
  return {
    artifacts: loaded.filter((artifact): artifact is Artifact => artifact !== undefined),
    names: new Set(ids),
  }
}

/** The validated record, or undefined when it is missing, oversized, unreadable or malformed. */
async function readArtifact(directory: string, id: string): Promise<Artifact | undefined> {
  if (typeof id !== 'string' || !ARTIFACT_ID_PATTERN.test(id)) return undefined
  const path = join(directory, `${id}.json`)
  let text: string
  try {
    const info = await stat(path)
    if (!info.isFile() || info.size > MAX_ARTIFACT_FILE_BYTES) return undefined
    text = await readFile(path, 'utf8')
  } catch (error) {
    if (isMissing(error)) return undefined
    throw error
  }
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return undefined
  }
  return parseArtifact(raw, id)
}

function parseArtifact(raw: unknown, id: string): Artifact | undefined {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return undefined
  const record = raw as Record<string, unknown>
  const { title, kind, language, content, created_at, updated_at, pinned, version } = record
  if (record.id !== id) return undefined
  if (typeof title !== 'string' || !title.trim() || title.length > MAX_ARTIFACT_TITLE_CHARS) return undefined
  if (!isArtifactKind(kind)) return undefined
  if (language !== undefined && (typeof language !== 'string' || language.length > MAX_ARTIFACT_LANGUAGE_CHARS)) return undefined
  if (typeof content !== 'string' || Buffer.byteLength(content, 'utf8') > MAX_ARTIFACT_CONTENT_BYTES) return undefined
  if (!isTimestamp(created_at) || !isTimestamp(updated_at)) return undefined
  if (typeof pinned !== 'boolean') return undefined
  if (typeof version !== 'number' || !Number.isSafeInteger(version) || version < 1) return undefined
  return {
    id,
    title,
    kind,
    ...(typeof language === 'string' && language ? { language } : {}),
    content,
    created_at,
    updated_at,
    pinned,
    version,
  }
}

function isTimestamp(value: unknown): value is string {
  return typeof value === 'string' && value.length <= 64 && !Number.isNaN(Date.parse(value))
}

async function writeArtifact(directory: string, artifact: Artifact): Promise<void> {
  await mkdir(directory, { recursive: true, mode: 0o700 })
  const path = join(directory, `${artifact.id}.json`)
  const temporary = join(directory, `.${artifact.id}.${crypto.randomUUID()}.tmp`)
  try {
    await writeFile(temporary, JSON.stringify(artifact, null, 2) + '\n', { mode: 0o600 })
    await rename(temporary, path)
  } catch (error) {
    await rm(temporary, { force: true })
    throw error
  }
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { code?: unknown }).code === 'ENOENT'
}
