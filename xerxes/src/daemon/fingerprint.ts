// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** The wire-level daemon protocol version shared with the Python daemon. */
export const DAEMON_PROTOCOL_VERSION = 35

const MISSING_SOURCE_MARKER = new TextEncoder().encode('<missing>')
const NULL_SEPARATOR = new Uint8Array([0])
const TEXT_ENCODER = new TextEncoder()

/**
 * Ordered source files contributing to the native daemon build identity.
 *
 * The list mirrors the Python fingerprint's daemon/runtime/command/streaming
 * coverage with the corresponding Bun modules. It is intentionally explicit:
 * callers provide both the root and a source reader rather than relying on the
 * process working directory or an import-time filesystem read.
 */
export const DAEMON_FINGERPRINT_FILES = Object.freeze([
  'daemon/fingerprint.ts',
  'daemon/server.ts',
  'daemon/runtime.ts',
  'bridge/commands.ts',
  'streaming/loop.ts',
  'context/windowUsage.ts',
] as const)

/** Read source bytes relative to one explicit source root. Return undefined only for a missing file. */
export interface DaemonSourceReader {
  readFile(sourceRoot: string, relativePath: string): Promise<Uint8Array | undefined>
}

export interface DaemonBuildIdOptions {
  readonly files?: readonly string[]
  readonly sourceReader: DaemonSourceReader
  readonly sourceRoot: string
}

/** v35 identity shape that can be passed to a daemon runtime's buildId option. */
export interface DaemonBuildIdentityRecord {
  readonly daemon_build_id: string
  readonly daemon_protocol: typeof DAEMON_PROTOCOL_VERSION
}

/**
 * Captured daemon source identity.
 *
 * Unlike Python's import-time module global, this object is explicitly created
 * by the host once it has selected the real source root and reader. The value
 * then stays stable for the lifetime of the object.
 */
export class DaemonBuildFingerprint {
  readonly buildId: string

  constructor(buildId: string) {
    if (!/^[0-9a-f]{16}$/.test(buildId)) {
      throw new TypeError('daemon build id must be a 16-character lowercase SHA-256 prefix')
    }
    this.buildId = buildId
    Object.freeze(this)
  }

  /** Python-compatible accessor for the captured build ID. */
  daemonBuildId(): string {
    return this.buildId
  }

  /** Return the protocol/build fields used by v35 initialize and status payloads. */
  toRecord(): DaemonBuildIdentityRecord {
    return Object.freeze({
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.buildId,
    })
  }
}

/** Compute a deterministic 16-character SHA-256 source fingerprint. */
export async function computeDaemonBuildId(options: DaemonBuildIdOptions): Promise<string> {
  const sourceRoot = requiredSourceRoot(options.sourceRoot)
  const files = normalizedFingerprintFiles(options.files ?? DAEMON_FINGERPRINT_FILES)
  const digest = new Bun.CryptoHasher('sha256')
  for (const relativePath of files) {
    digest.update(TEXT_ENCODER.encode(relativePath))
    digest.update(NULL_SEPARATOR)
    const source = await options.sourceReader.readFile(sourceRoot, relativePath)
    digest.update(source ?? MISSING_SOURCE_MARKER)
    digest.update(NULL_SEPARATOR)
  }
  return digest.digest('hex').slice(0, 16)
}

/** Capture a stable v35 daemon identity from one explicit source snapshot. */
export async function captureDaemonBuildFingerprint(options: DaemonBuildIdOptions): Promise<DaemonBuildFingerprint> {
  return new DaemonBuildFingerprint(await computeDaemonBuildId(options))
}

/** Python-style helper for hosts that need only the deterministic build-ID string. */
export const daemonBuildId = computeDaemonBuildId

function requiredSourceRoot(value: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError('daemon fingerprint sourceRoot must be a non-empty string')
  }
  return value
}

function normalizedFingerprintFiles(files: readonly string[]): readonly string[] {
  if (!files.length) {
    throw new TypeError('daemon fingerprint must include at least one source file')
  }
  const normalized = files.map(relativePath => {
    if (!isRelativeSourcePath(relativePath)) {
      throw new TypeError('daemon fingerprint file paths must be non-empty relative paths without traversal')
    }
    return relativePath
  })
  return Object.freeze(normalized)
}

function isRelativeSourcePath(value: string): boolean {
  if (typeof value !== 'string' || !value || value.startsWith('/') || value.startsWith('\\')) return false
  return value.split('/').every(segment => segment !== '' && segment !== '.' && segment !== '..')
}
