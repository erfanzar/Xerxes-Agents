// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute, posix, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'

/** Direction of a host-to-sandbox file transfer. */
export type FileSyncDirection = 'pull' | 'push'

/** Side of the transfer on which a path is resolved or inspected. */
export type FileSyncLocation = 'local' | 'remote'

/** A file requested for transfer between a local root and a sandbox root. */
export interface FileSyncSpec {
  /** Path relative to, or already contained by, the configured local root. */
  readonly localPath: string
  /** POSIX path relative to, or already contained by, the configured remote root. */
  readonly remotePath: string
  /** Caller-owned contextual data returned with the transfer report. */
  readonly metadata?: Readonly<Record<string, unknown>>
}

/** Size information returned by an injected stat operation. */
export interface FileSyncStat {
  readonly size: number
}

/** Input supplied to an injected stat operation. */
export interface FileSyncStatRequest {
  readonly direction: FileSyncDirection
  readonly location: FileSyncLocation
  readonly path: string
}

/** Input supplied to an injected copy operation. */
export interface FileSyncCopyRequest {
  readonly destination: string
  readonly destinationLocation: FileSyncLocation
  readonly direction: FileSyncDirection
  readonly source: string
  readonly sourceLocation: FileSyncLocation
}

/** Transport-specific file operations; this module does not assume Docker or a cloud SDK. */
export interface FileSyncPorts {
  readonly copy: (request: FileSyncCopyRequest) => Promise<void> | void
  readonly stat: (request: FileSyncStatRequest) => Promise<FileSyncStat | undefined> | FileSyncStat | undefined
}

/** Explicit containment and transfer-size policy for a sync batch. */
export interface FileSyncOptions {
  /** Absolute host path that contains all local paths. */
  readonly localRoot: string
  /** Absolute POSIX sandbox path that contains all remote paths. */
  readonly remoteRoot: string
  /** Maximum permitted source-file size in bytes. Omit for no size cap. */
  readonly maxBytes?: number
}

/** Reasons for a transfer that was intentionally not attempted. */
export type FileSyncSkipReason = 'max_bytes_exceeded' | 'missing'

/** A successfully copied file. */
export interface FileSyncCopiedResult {
  readonly bytes: number
  readonly direction: FileSyncDirection
  readonly localPath: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly remotePath: string
  readonly status: 'copied'
}

/** A file omitted without invoking the copy port. */
export interface FileSyncSkippedResult {
  readonly bytes: number
  readonly direction: FileSyncDirection
  readonly localPath: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly reason: FileSyncSkipReason
  readonly remotePath: string
  readonly status: 'skipped'
}

/** A per-file failure that leaves the rest of the batch eligible to continue. */
export interface FileSyncFailedResult {
  readonly direction: FileSyncDirection
  readonly error: string
  readonly localPath: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly remotePath: string
  readonly status: 'failed'
}

/** Outcome of one requested file transfer. */
export type FileSyncResult = FileSyncCopiedResult | FileSyncFailedResult | FileSyncSkippedResult

/** Raised when a configured root or requested file path escapes its allowed location. */
export class FileSyncPathError extends ValidationError {
  constructor(field: string, value: string, root: string, reason: string) {
    super(field, reason, value, { root })
  }
}

/** Copy local files into the remote sandbox root, reporting every input spec. */
export async function syncPush(
  specs: readonly FileSyncSpec[],
  ports: FileSyncPorts,
  options: FileSyncOptions,
): Promise<FileSyncResult[]> {
  return sync('push', specs, ports, options)
}

/** Copy remote sandbox files into the local root, reporting every input spec. */
export async function syncPull(
  specs: readonly FileSyncSpec[],
  ports: FileSyncPorts,
  options: FileSyncOptions,
): Promise<FileSyncResult[]> {
  return sync('pull', specs, ports, options)
}

interface PreparedFileSyncSpec {
  readonly localPath: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly remotePath: string
}

interface NormalizedFileSyncOptions {
  readonly localRoot: string
  readonly maxBytes: number | undefined
  readonly remoteRoot: string
}

const EMPTY_METADATA: Readonly<Record<string, unknown>> = Object.freeze({})

async function sync(
  direction: FileSyncDirection,
  specs: readonly FileSyncSpec[],
  ports: FileSyncPorts,
  options: FileSyncOptions,
): Promise<FileSyncResult[]> {
  if (!Array.isArray(specs)) {
    throw new ValidationError('fileSyncSpecs', 'must be an array of file specifications', specs)
  }
  validatePorts(ports)
  const normalizedOptions = normalizeOptions(options)
  const results: FileSyncResult[] = []
  for (const spec of specs) {
    results.push(await syncOne(direction, spec, ports, normalizedOptions))
  }
  return results
}

async function syncOne(
  direction: FileSyncDirection,
  spec: FileSyncSpec,
  ports: FileSyncPorts,
  options: NormalizedFileSyncOptions,
): Promise<FileSyncResult> {
  let prepared: PreparedFileSyncSpec
  try {
    prepared = prepareSpec(spec, options)
  } catch (error) {
    return failedResult(
      displayPaths(spec),
      EMPTY_METADATA,
      `path validation failed: ${errorMessage(error)}`,
      direction,
    )
  }

  const sourceLocation: FileSyncLocation = direction === 'push' ? 'local' : 'remote'
  const sourcePath = direction === 'push' ? prepared.localPath : prepared.remotePath
  let stat: FileSyncStat | undefined
  try {
    stat = await ports.stat({ direction, location: sourceLocation, path: sourcePath })
  } catch (error) {
    return failedResult(prepared, prepared.metadata, `stat failed: ${errorMessage(error)}`, direction)
  }
  if (stat === undefined) {
    return skippedResult(prepared, prepared.metadata, 0, 'missing', direction)
  }

  let bytes: number
  try {
    bytes = validatedSize(stat)
  } catch (error) {
    return failedResult(prepared, prepared.metadata, `invalid stat result: ${errorMessage(error)}`, direction)
  }
  if (options.maxBytes !== undefined && bytes > options.maxBytes) {
    return skippedResult(prepared, prepared.metadata, bytes, 'max_bytes_exceeded', direction)
  }

  const copyRequest = copyRequestFor(direction, prepared)
  try {
    await ports.copy(copyRequest)
  } catch (error) {
    return failedResult(prepared, prepared.metadata, `copy failed: ${errorMessage(error)}`, direction)
  }
  return copiedResult(prepared, prepared.metadata, bytes, direction)
}

function prepareSpec(spec: FileSyncSpec, options: NormalizedFileSyncOptions): PreparedFileSyncSpec {
  if (spec === null || typeof spec !== 'object' || Array.isArray(spec)) {
    throw new ValidationError('fileSyncSpec', 'must be an object containing localPath and remotePath', spec)
  }
  return {
    localPath: resolveLocalPath(options.localRoot, spec.localPath),
    metadata: normalizeMetadata(spec.metadata),
    remotePath: resolveRemotePath(options.remoteRoot, spec.remotePath),
  }
}

function normalizeOptions(options: FileSyncOptions): NormalizedFileSyncOptions {
  if (options === null || typeof options !== 'object') {
    throw new ValidationError('fileSyncOptions', 'must explicitly define localRoot and remoteRoot', options)
  }
  const localRoot = normalizeLocalRoot(options.localRoot)
  const remoteRoot = normalizeRemoteRoot(options.remoteRoot)
  const maxBytes = options.maxBytes
  if (maxBytes !== undefined && (!Number.isSafeInteger(maxBytes) || maxBytes < 0)) {
    throw new ValidationError('maxBytes', 'must be a non-negative safe integer when provided', maxBytes)
  }
  return { localRoot, maxBytes, remoteRoot }
}

function validatePorts(ports: FileSyncPorts): void {
  const hasCopyPort = ports !== null && typeof ports === 'object' && typeof ports.copy === 'function'
  const hasStatPort = ports !== null && typeof ports === 'object' && typeof ports.stat === 'function'
  if (!hasCopyPort || !hasStatPort) {
    throw new ValidationError('fileSyncPorts', 'must provide asynchronous or synchronous copy and stat functions', ports)
  }
}

function normalizeLocalRoot(root: string): string {
  validatePathValue(root, 'localRoot')
  if (!isAbsolute(root)) {
    throw new FileSyncPathError('localRoot', root, root, 'localRoot must be an absolute host path')
  }
  return resolve(root)
}

function normalizeRemoteRoot(root: string): string {
  validatePathValue(root, 'remoteRoot')
  if (!posix.isAbsolute(root)) {
    throw new FileSyncPathError('remoteRoot', root, root, 'remoteRoot must be an absolute POSIX sandbox path')
  }
  return posix.resolve(root)
}

function resolveLocalPath(root: string, candidate: string): string {
  validatePathValue(candidate, 'localPath')
  const resolvedPath = isAbsolute(candidate) ? resolve(candidate) : resolve(root, candidate)
  assertContained(root, resolvedPath, 'localPath', candidate, relative)
  return resolvedPath
}

function resolveRemotePath(root: string, candidate: string): string {
  validatePathValue(candidate, 'remotePath')
  const resolvedPath = posix.isAbsolute(candidate) ? posix.resolve(candidate) : posix.resolve(root, candidate)
  assertContained(root, resolvedPath, 'remotePath', candidate, posix.relative)
  return resolvedPath
}

function assertContained(
  root: string,
  candidate: string,
  field: string,
  value: string,
  relativePath: (from: string, to: string) => string,
): void {
  const pathFromRoot = relativePath(root, candidate)
  if (
    pathFromRoot === '..'
    || pathFromRoot.startsWith(`..${sep}`)
    || pathFromRoot.startsWith('../')
    || isAbsolute(pathFromRoot)
    || posix.isAbsolute(pathFromRoot)
  ) {
    throw new FileSyncPathError(field, value, root, `${field} must remain inside its configured root`)
  }
}

function validatePathValue(value: string, field: string): void {
  if (typeof value !== 'string' || value.trim() === '' || value.includes('\0')) {
    throw new ValidationError(field, 'must be a non-empty path without NUL bytes', value)
  }
}

function normalizeMetadata(metadata: FileSyncSpec['metadata']): Readonly<Record<string, unknown>> {
  if (metadata === undefined) {
    return EMPTY_METADATA
  }
  if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)) {
    throw new ValidationError('fileSyncMetadata', 'must be a record when provided', metadata)
  }
  return Object.freeze({ ...metadata })
}

function validatedSize(stat: FileSyncStat): number {
  if (stat === null || typeof stat !== 'object' || !Number.isSafeInteger(stat.size) || stat.size < 0) {
    throw new ValidationError('fileSyncStat', 'size must be a non-negative safe integer', stat)
  }
  return stat.size
}

function copyRequestFor(direction: FileSyncDirection, spec: PreparedFileSyncSpec): FileSyncCopyRequest {
  if (direction === 'push') {
    return {
      destination: spec.remotePath,
      destinationLocation: 'remote',
      direction,
      source: spec.localPath,
      sourceLocation: 'local',
    }
  }
  return {
    destination: spec.localPath,
    destinationLocation: 'local',
    direction,
    source: spec.remotePath,
    sourceLocation: 'remote',
  }
}

function copiedResult(
  spec: PreparedFileSyncSpec,
  metadata: Readonly<Record<string, unknown>>,
  bytes: number,
  direction: FileSyncDirection,
): FileSyncCopiedResult {
  return {
    bytes,
    direction,
    localPath: spec.localPath,
    metadata,
    remotePath: spec.remotePath,
    status: 'copied',
  }
}

function skippedResult(
  spec: PreparedFileSyncSpec,
  metadata: Readonly<Record<string, unknown>>,
  bytes: number,
  reason: FileSyncSkipReason,
  direction: FileSyncDirection,
): FileSyncSkippedResult {
  return {
    bytes,
    direction,
    localPath: spec.localPath,
    metadata,
    reason,
    remotePath: spec.remotePath,
    status: 'skipped',
  }
}

function failedResult(
  spec: Pick<PreparedFileSyncSpec, 'localPath' | 'remotePath'>,
  metadata: Readonly<Record<string, unknown>>,
  error: string,
  direction: FileSyncDirection,
): FileSyncFailedResult {
  return {
    direction,
    error,
    localPath: spec.localPath,
    metadata,
    remotePath: spec.remotePath,
    status: 'failed',
  }
}

function displayPaths(spec: FileSyncSpec): Pick<PreparedFileSyncSpec, 'localPath' | 'remotePath'> {
  return {
    localPath: typeof spec?.localPath === 'string' ? spec.localPath : '<invalid local path>',
    remotePath: typeof spec?.remotePath === 'string' ? spec.remotePath : '<invalid remote path>',
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message !== '') {
    return error.message
  }
  return 'unknown error'
}
