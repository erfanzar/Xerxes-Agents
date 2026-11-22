// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir, realpath, stat } from 'node:fs/promises'
import { extname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { parseLspConfig, type LspServerConfig } from './config.js'

export interface WorkspaceLspDiscoveryOptions {
  /**
   * Explicit user-configured servers. Never mutated; discovered candidates yield
   * to any name or enabled extension already claimed here.
   */
  readonly existingServers?: readonly LspServerConfig[]
  /** Inspect workspace-contained local binaries (`node_modules/.bin`, `.venv/bin`, `venv/bin`). Default: `true`. */
  readonly allowLocalBinaries?: boolean
  /** Inspect global `PATH` binaries via `which`. Default: `true`. */
  readonly allowSystemPath?: boolean
  /** Injectable binary resolver (defaults to `Bun.which`). */
  readonly which?: (binary: string) => string | null | undefined
  /** Optional request timeout in ms (1–120000) attached to discovered servers. */
  readonly timeoutMs?: number
}

interface CandidateSpec {
  readonly name: string
  readonly binary: string
  readonly args?: readonly string[]
  readonly languageId: string
  readonly extensions: readonly string[]
  readonly markerFiles: readonly string[]
  readonly markerExtensions: readonly string[]
  readonly localRelativeBins: readonly string[]
}

const BIN_EXT = process.platform === 'win32' ? '.cmd' : ''
const VENV_DIR = process.platform === 'win32' ? 'Scripts' : 'bin'
const VENV_EXT = process.platform === 'win32' ? '.exe' : ''

const CANDIDATE_SPECS: readonly CandidateSpec[] = [
  {
    name: 'typescript-language-server',
    binary: 'typescript-language-server',
    args: ['--stdio'],
    languageId: 'typescript',
    extensions: ['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs'],
    markerFiles: ['tsconfig.json', 'jsconfig.json', 'package.json'],
    markerExtensions: ['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs'],
    localRelativeBins: [join('node_modules', '.bin', `typescript-language-server${BIN_EXT}`)],
  },
  {
    name: 'pyright-langserver',
    binary: 'pyright-langserver',
    args: ['--stdio'],
    languageId: 'python',
    extensions: ['.py', '.pyi'],
    markerFiles: ['pyproject.toml', 'pyrightconfig.json', 'requirements.txt', 'setup.py', 'setup.cfg', 'Pipfile'],
    markerExtensions: ['.py', '.pyi'],
    localRelativeBins: [
      join('node_modules', '.bin', `pyright-langserver${BIN_EXT}`),
      join('.venv', VENV_DIR, `pyright-langserver${VENV_EXT}`),
      join('venv', VENV_DIR, `pyright-langserver${VENV_EXT}`),
    ],
  },
  {
    name: 'gopls',
    binary: 'gopls',
    args: ['serve'],
    languageId: 'go',
    extensions: ['.go'],
    markerFiles: ['go.mod', 'go.work'],
    markerExtensions: ['.go'],
    localRelativeBins: [join('bin', `gopls${VENV_EXT}`)],
  },
  {
    name: 'rust-analyzer',
    binary: 'rust-analyzer',
    languageId: 'rust',
    extensions: ['.rs'],
    markerFiles: ['Cargo.toml', 'rust-project.json'],
    markerExtensions: ['.rs'],
    localRelativeBins: [],
  },
]

const SKIP_DIRS = new Set(['.git', '.hg', '.svn', 'node_modules', 'dist', 'build', 'target', '.venv', 'venv', '__pycache__'])

/**
 * Opt-in, non-blocking workspace LSP server auto-discovery.
 *
 * Never reads untrusted `.lsp.json` files from the workspace and never mutates
 * `existingServers` or persisted user configuration.
 */
export async function discoverWorkspaceLspServers(
  workspaceRoot: string,
  options: WorkspaceLspDiscoveryOptions = {},
): Promise<readonly LspServerConfig[]> {
  if (!workspaceRoot.trim() || workspaceRoot.includes('\0')) {
    return []
  }

  let canonicalRoot: string
  try {
    canonicalRoot = await realpath(resolve(workspaceRoot))
    if (!(await stat(canonicalRoot)).isDirectory()) return []
  } catch {
    return []
  }

  const allowLocalBinaries = options.allowLocalBinaries ?? true
  const allowSystemPath = options.allowSystemPath ?? true
  const whichFn = options.which ?? ((binary: string) => Bun.which(binary))
  const existing = options.existingServers ?? []

  const existingNames = new Set(existing.map(server => server.name))
  const claimedExtensions = new Set(
    existing
      .filter(server => server.enabled)
      .flatMap(server => server.extensions.map(ext => ext.toLowerCase())),
  )

  const snapshot = await inspectShallowWorkspace(canonicalRoot)
  const discoveredRaw: Array<Record<string, unknown>> = []

  for (const spec of CANDIDATE_SPECS) {
    if (existingNames.has(spec.name)) continue

    const availableExtensions = spec.extensions.filter(ext => !claimedExtensions.has(ext.toLowerCase()))
    if (availableExtensions.length === 0) continue

    const localCommand = allowLocalBinaries
      ? await resolveContainedLocalBinary(canonicalRoot, snapshot.subdirectories, spec.localRelativeBins)
      : undefined

    const hasWorkspaceSignal = localCommand !== undefined
      || spec.markerFiles.some(marker => snapshot.fileNames.has(marker))
      || spec.markerExtensions.some(ext => snapshot.extensions.has(ext))

    if (!hasWorkspaceSignal) continue

    const rawCommand = localCommand ?? (allowSystemPath ? whichFn(spec.binary) ?? undefined : undefined)
    const command = rawCommand?.trim()
    if (!command || command.includes('\0')) continue

    for (const ext of availableExtensions) {
      claimedExtensions.add(ext.toLowerCase())
    }

    discoveredRaw.push({
      name: spec.name,
      command,
      ...(spec.args !== undefined && spec.args.length > 0 ? { args: [...spec.args] } : {}),
      languageId: spec.languageId,
      extensions: availableExtensions,
      enabled: true,
      ...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
    })
  }

  if (discoveredRaw.length === 0) return []
  return parseLspConfig({ servers: discoveredRaw })
}

/**
 * Pure non-mutating merge of explicit user LSP configs with auto-discovered servers.
 * Explicit user configs always take priority over discovered names and file suffixes.
 */
export function mergeDiscoveredLspServers(
  existingServers: readonly LspServerConfig[],
  discoveredServers: readonly LspServerConfig[],
): readonly LspServerConfig[] {
  const names = new Set(existingServers.map(server => server.name))
  const claimedExtensions = new Set(
    existingServers
      .filter(server => server.enabled)
      .flatMap(server => server.extensions.map(ext => ext.toLowerCase())),
  )

  const merged: Array<Record<string, unknown>> = existingServers.map(cloneServerConfig)
  for (const candidate of discoveredServers) {
    if (names.has(candidate.name)) continue
    const unclaimed = candidate.extensions.filter(ext => {
      if (!candidate.enabled) return true
      return !claimedExtensions.has(ext.toLowerCase())
    })
    if (unclaimed.length === 0) continue
    names.add(candidate.name)
    if (candidate.enabled) {
      for (const ext of unclaimed) claimedExtensions.add(ext.toLowerCase())
    }
    merged.push({
      ...cloneServerConfig(candidate),
      extensions: unclaimed,
    })
  }

  return parseLspConfig({ servers: merged })
}

function cloneServerConfig(server: LspServerConfig): Record<string, unknown> {
  return {
    name: server.name,
    command: server.command,
    languageId: server.languageId,
    extensions: [...server.extensions],
    enabled: server.enabled,
    ...(server.args !== undefined ? { args: [...server.args] } : {}),
    ...(server.env !== undefined ? { env: { ...server.env } } : {}),
    ...(server.timeoutMs !== undefined ? { timeoutMs: server.timeoutMs } : {}),
  }
}

async function inspectShallowWorkspace(canonicalRoot: string): Promise<{
  readonly fileNames: ReadonlySet<string>
  readonly extensions: ReadonlySet<string>
  readonly subdirectories: readonly string[]
}> {
  const fileNames = new Set<string>()
  const extensions = new Set<string>()
  const subdirectories: string[] = []

  const scanOneDir = async (dir: string, collectSubdirs: boolean): Promise<void> => {
    let entries
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue
      if (entry.isFile()) {
        fileNames.add(entry.name)
        const ext = extname(entry.name.toLowerCase())
        if (ext) extensions.add(ext)
      } else if (collectSubdirs && entry.isDirectory() && !entry.name.startsWith('.') && !SKIP_DIRS.has(entry.name)) {
        subdirectories.push(join(dir, entry.name))
      }
    }
  }

  await scanOneDir(canonicalRoot, true)
  // Scan up to 24 direct subdirectories non-blockingly so monorepo layouts (`src/`, `pkg/`, `crates/`, `xerxes/`) are recognized.
  for (const subdir of subdirectories.slice(0, 24)) {
    await scanOneDir(subdir, false)
  }

  return { fileNames, extensions, subdirectories: subdirectories.slice(0, 24) }
}

async function resolveContainedLocalBinary(
  canonicalRoot: string,
  subdirectories: readonly string[],
  relativeCandidates: readonly string[],
): Promise<string | undefined> {
  if (relativeCandidates.length === 0) return undefined
  const bases = [canonicalRoot, ...subdirectories]
  for (const base of bases) {
    for (const relBin of relativeCandidates) {
      const candidate = join(base, relBin)
      try {
        const resolved = await realpath(candidate)
        if (!isWithinWorkspace(canonicalRoot, resolved)) continue
        const info = await stat(resolved)
        if (info.isFile()) {
          return candidate
        }
      } catch {
        continue
      }
    }
  }
  return undefined
}

function isWithinWorkspace(canonicalRoot: string, candidate: string): boolean {
  const rel = relative(canonicalRoot, candidate)
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}
