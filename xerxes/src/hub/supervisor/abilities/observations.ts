// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, readdir, realpath, statfs } from 'node:fs/promises'
import { freemem, loadavg, totalmem, uptime } from 'node:os'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

const MAX_BYTES = 64 * 1024
const MAX_FILE = 1024 * 1024
const MAX_SCAN_FILES = 1000
const MAX_SCAN_BYTES = 8 * 1024 * 1024
const WIDTHS: ReadonlySet<number> = new Set([390, 768, 1280, 1440])
const PRIVATE = /(^|[/\\])(?:\.git|node_modules|\.env(?:\.[^/\\]*)?|\.ssh|credentials(?:\.[^/\\]*)?|id_rsa|id_ed25519)(?:$|[/\\])/i

export interface ObservationHosts {
  readonly github?: (view: 'issues' | 'prs' | 'ci', signal?: AbortSignal) => Promise<Readonly<Record<string, unknown>>>
  readonly screenshot?: (width: number, signal?: AbortSignal) => Promise<Readonly<Record<string, unknown>>>
  readonly accelerators?: (signal?: AbortSignal) => Promise<Readonly<Record<string, unknown>>>
}

/** Fixed read-only argv only. No shell, pager, repository hooks, external diff or textconv. */
async function readCommand(argv: readonly string[], cwd: string, signal?: AbortSignal) {
  signal?.throwIfAborted()
  const deadline = AbortSignal.timeout(5000)
  const bounded = signal === undefined ? deadline : AbortSignal.any([signal, deadline])
  const process = Bun.spawn([...argv], { cwd, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe',
    env: { PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin', GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null', GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0', LC_ALL: 'C' } })
  const abort = () => { process.kill('SIGKILL') }
  bounded.addEventListener('abort', abort, { once: true })
  if (bounded.aborted) abort()
  let used = 0
  let truncated = false
  const consume = async (stream: ReadableStream<Uint8Array>): Promise<string> => {
    const chunks: Uint8Array[] = []
    const reader = stream.getReader()
    try {
      while (true) {
        const { done, value: chunk } = await reader.read()
        if (done) break
        const remaining = Math.max(0, MAX_BYTES - used)
        chunks.push(chunk.subarray(0, remaining)); used += Math.min(remaining, chunk.length)
        if (chunk.length > remaining) { truncated = true; abort(); await reader.cancel(); break }
      }
    } finally { reader.releaseLock() }
    return Buffer.concat(chunks).toString('utf8')
  }
  try {
    const [stdout, stderr, exit] = await Promise.all([consume(process.stdout), consume(process.stderr), process.exited])
    bounded.throwIfAborted()
    if (exit !== 0 && !truncated) throw new Error(`Read command failed (${exit}): ${stderr.slice(0, 500)}`)
    return { text: stdout, truncated }
  } finally {
    bounded.removeEventListener('abort', abort)
    if (process.exitCode === null) process.kill('SIGKILL')
  }
}

function integer(value: number, min: number, max: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${label} must be an integer from ${min} to ${max}`)
}

/** Workspace-local SEE implementation. Observations never grant mutation authority. */
export class WorkspaceObservations {
  constructor(private readonly root: string, private readonly hosts: ObservationHosts = {}) {}

  private async path(path: string): Promise<string> {
    if (path === '' || path.length > 4096 || /[\0\r\n]/.test(path) || isAbsolute(path) || PRIVATE.test(path)) throw new Error('Refused workspace path (secrets require owner confirmation)')
    const root = await realpath(this.root)
    const candidate = resolve(root, path)
    const relativePath = relative(root, candidate)
    if (relativePath === '..' || relativePath.startsWith(`..${sep}`)) throw new Error('Path escapes workspace')
    // Refuse symlinks entirely: both directory traversal and leaf escapes are data-boundary violations.
    let current = root
    for (const part of relativePath.split(sep).filter(Boolean)) {
      current = join(current, part)
      if ((await lstat(current)).isSymbolicLink()) throw new Error('Symlink workspace paths are refused')
    }
    const canonical = await realpath(candidate)
    if (canonical !== root && !canonical.startsWith(`${root}${sep}`)) throw new Error('Path escapes workspace')
    return canonical
  }

  async read(path: string, offset = 0, maxBytes = MAX_BYTES) {
    integer(offset, 0, Number.MAX_SAFE_INTEGER, 'offset'); integer(maxBytes, 1, MAX_BYTES, 'max_bytes')
    const filePath = await this.path(path)
    const info = await lstat(filePath)
    if (!info.isFile()) throw new Error('Only regular files can be read')
    if (info.size > MAX_FILE) throw new Error('File exceeds the 1 MiB observation limit')
    const file = Bun.file(filePath)
    const bytes = new Uint8Array(await file.slice(offset, offset + maxBytes).arrayBuffer())
    if (bytes.includes(0)) throw new Error('Refused binary file')
    return { text: new TextDecoder().decode(bytes), offset, next_offset: offset + bytes.length < info.size ? offset + bytes.length : null,
      total_bytes: info.size, untrusted: true }
  }

  async list(path = '.', offset = 0, limit = 200) {
    integer(offset, 0, Number.MAX_SAFE_INTEGER, 'offset'); integer(limit, 1, 200, 'limit')
    const dir = await this.path(path)
    const entries = (await readdir(dir, { withFileTypes: true })).filter(entry => !PRIVATE.test(entry.name))
      .sort((a, b) => a.name.localeCompare(b.name))
    return { entries: entries.slice(offset, offset + limit).map(entry => ({ name: entry.name,
      kind: entry.isSymbolicLink() ? 'symlink' : entry.isDirectory() ? 'directory' : entry.isFile() ? 'file' : 'other' })),
      next_offset: offset + limit < entries.length ? offset + limit : null, untrusted: true }
  }

  async search(query: string, offset = 0, limit = 100, signal?: AbortSignal) {
    if (query.trim() === '' || query.length > 256) throw new Error('query must contain 1..256 characters')
    integer(offset, 0, 10000, 'offset'); integer(limit, 1, 100, 'limit')
    const hits: { path: string; line: number; text: string }[] = []
    let files = 0; let scannedBytes = 0; let matched = 0; let truncated = false
    const visit = async (path: string, depth: number): Promise<void> => {
      signal?.throwIfAborted()
      if (depth > 20 || files >= MAX_SCAN_FILES || scannedBytes >= MAX_SCAN_BYTES || hits.length > limit) { truncated = true; return }
      const dir = await this.path(path)
      for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
        signal?.throwIfAborted()
        const child = path === '.' ? entry.name : `${path}/${entry.name}`
        if (PRIVATE.test(child) || entry.isSymbolicLink()) continue
        if (entry.isDirectory()) { await visit(child, depth + 1); continue }
        if (!entry.isFile()) continue
        if (files >= MAX_SCAN_FILES || scannedBytes >= MAX_SCAN_BYTES || hits.length > limit) { truncated = true; return }
        const filePath = await this.path(child)
        const info = await lstat(filePath); files++
        if (info.size > MAX_BYTES || scannedBytes + info.size > MAX_SCAN_BYTES) { truncated = true; continue }
        const bytes = new Uint8Array(await Bun.file(filePath).arrayBuffer()); scannedBytes += bytes.length
        if (bytes.includes(0)) continue
        const lines = new TextDecoder().decode(bytes).split('\n')
        for (let line = 0; line < lines.length; line++) {
          if (!lines[line]!.includes(query)) continue
          if (matched++ >= offset) hits.push({ path: child, line: line + 1, text: lines[line]!.slice(0, 500) })
          if (hits.length > limit) { truncated = true; return }
        }
      }
    }
    await visit('.', 0)
    const more = hits.length > limit
    return { hits: hits.slice(0, limit), next_offset: more ? offset + limit : null, truncated, scanned_files: files, untrusted: true }
  }

  async git(view: 'status' | 'log' | 'diff', signal?: AbortSignal) {
    const args = view === 'status' ? ['status', '--short', '--untracked-files=no']
      : view === 'log' ? ['log', '-30', '--format=%h %s']
        : view === 'diff' ? ['diff', '--no-ext-diff', '--no-textconv', '--', '.', ':!.env*', ':!**/.env*'] : undefined
    if (args === undefined) throw new Error('Unknown git observation')
    const result = await readCommand(['git', '--no-pager', '-c', 'core.fsmonitor=false', ...args], await realpath(this.root), signal)
    return { ...result, untrusted: true }
  }

  async health(signal?: AbortSignal) {
    signal?.throwIfAborted()
    const disk = await statfs(await realpath(this.root))
    const processes = await readCommand(['/bin/ps', '-eo', 'pid,pcpu,pmem,comm'], this.root, signal)
    return { disk: { available_bytes: disk.bavail * disk.bsize, total_bytes: disk.blocks * disk.bsize },
      memory: { available_bytes: freemem(), total_bytes: totalmem() }, load: loadavg(), uptime_seconds: uptime(), processes,
      accelerators: this.hosts.accelerators === undefined ? { available: false, reason: 'No GPU/TPU observation host configured' } : await this.hosts.accelerators(signal), untrusted: true }
  }

  async github(view: 'issues' | 'prs' | 'ci', signal?: AbortSignal) {
    if (!['issues', 'prs', 'ci'].includes(view)) throw new Error('Unknown GitHub observation')
    if (this.hosts.github === undefined) throw new Error('GitHub read host is not configured for this workspace')
    return await this.hosts.github(view, signal)
  }

  async screenshot(width: number, signal?: AbortSignal) {
    if (!WIDTHS.has(width)) throw new Error('Unsupported Xerxes screenshot width; use 390, 768, 1280 or 1440')
    if (this.hosts.screenshot === undefined) throw new Error('Xerxes UI screenshot host is not configured for this workspace')
    return await this.hosts.screenshot(width, signal)
  }
}
