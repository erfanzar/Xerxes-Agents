// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, mkdir, realpath, rm, statfs, symlink } from 'node:fs/promises'
import { freemem, loadavg, tmpdir, totalmem, uptime } from 'node:os'
import { isAbsolute, join, relative } from 'node:path'
import { AnchoredWorkspaceFiles, OversizedObservationFile } from './anchoredFiles.js'

const MAX_BYTES = 64 * 1024
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
async function readCommand(argv: readonly string[], cwd: string, signal?: AbortSignal, allowNoMatch = false) {
  signal?.throwIfAborted()
  const deadline = AbortSignal.timeout(5000)
  const bounded = signal === undefined ? deadline : AbortSignal.any([signal, deadline])
  const process = Bun.spawn([...argv], { cwd, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe',
    env: { PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin', GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null', GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0',
      GIT_NO_LAZY_FETCH: '1', GIT_ALLOW_PROTOCOL: '', GIT_SSH_COMMAND: '/usr/bin/false', GIT_ASKPASS: '/usr/bin/false', LC_ALL: 'C' } })
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
    const [stdout, , exit] = await Promise.all([consume(process.stdout), consume(process.stderr), process.exited])
    bounded.throwIfAborted()
    if (exit !== 0 && !(allowNoMatch && exit === 1) && !truncated) throw new Error(`Read command failed (${exit}); repository or host diagnostics were withheld to avoid exposing credentials`)
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
  private readonly files: AnchoredWorkspaceFiles
  constructor(private readonly root: string, private readonly hosts: ObservationHosts = {}) {
    this.files = new AnchoredWorkspaceFiles(root)
  }

  private path(path: string): string {
    if (path === '' || path.length > 4096 || /[\0\r\n]/.test(path) || isAbsolute(path) || PRIVATE.test(path)) throw new Error('Refused workspace path (secrets require owner confirmation)')
    return path
  }

  async read(path: string, offset = 0, maxBytes = MAX_BYTES) {
    integer(offset, 0, Number.MAX_SAFE_INTEGER, 'offset'); integer(maxBytes, 1, MAX_BYTES, 'max_bytes')
    const { bytes, totalBytes } = await this.files.read(this.path(path), offset, maxBytes)
    if (bytes.includes(0)) throw new Error('Refused binary file')
    return { text: new TextDecoder().decode(bytes), offset, next_offset: offset + bytes.length < totalBytes ? offset + bytes.length : null,
      total_bytes: totalBytes, untrusted: true }
  }

  async list(path = '.', offset = 0, limit = 200) {
    integer(offset, 0, Number.MAX_SAFE_INTEGER, 'offset'); integer(limit, 1, 200, 'limit')
    const entries = (await this.files.list(this.path(path))).filter(entry => !PRIVATE.test(entry.name))
      .sort((a, b) => a.name.localeCompare(b.name))
    return { entries: entries.slice(offset, offset + limit),
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
      for (const entry of (await this.files.list(this.path(path))).sort((a, b) => a.name.localeCompare(b.name))) {
        signal?.throwIfAborted()
        const child = path === '.' ? entry.name : `${path}/${entry.name}`
        if (PRIVATE.test(child) || entry.kind === 'symlink') continue
        if (entry.kind === 'directory') { await visit(child, depth + 1); continue }
        if (entry.kind !== 'file') continue
        if (files >= MAX_SCAN_FILES || scannedBytes >= MAX_SCAN_BYTES || hits.length > limit) { truncated = true; return }
        files++
        let opened: Awaited<ReturnType<AnchoredWorkspaceFiles['read']>>
        try { opened = await this.files.read(this.path(child), 0, Math.min(MAX_BYTES, MAX_SCAN_BYTES - scannedBytes)) }
        catch (error) { if (error instanceof OversizedObservationFile) { truncated = true; continue } throw error }
        const { bytes, totalBytes } = opened
        scannedBytes += bytes.length
        if (totalBytes > bytes.length) { truncated = true; continue }
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
    const exclusions = ['.env*', 'credentials*', 'id_rsa*', 'id_ed25519*', '.ssh/**', '.git/**', 'node_modules/**']
      .map(pattern => `:(exclude,glob,icase)**/${pattern}`)
    const args = view === 'status' ? ['status', '--short', '--untracked-files=no', '--', '.']
      : view === 'log' ? ['log', '-30', '--format=%h %s', '--', '.']
        : view === 'diff' ? ['diff', '--no-ext-diff', '--no-textconv', '--', '.', ...exclusions] : undefined
    if (args === undefined) throw new Error('Unknown git observation')
    const root = await realpath(this.root)
    const safeGit = ['git', '--no-pager', '-c', 'core.fsmonitor=false', '-c', 'protocol.allow=never', '-c', 'core.sshCommand=/usr/bin/false', '-c', 'core.askPass=/usr/bin/false', '-c', 'credential.helper=']
    // Config listing does not evaluate clean filters. Do not echo values: even
    // read-only status/diff can execute repository-defined clean/process filters.
    const config = await readCommand([...safeGit, 'config', '--no-includes', '--null', '--name-only', '--get-regexp', '^(filter\\..*|include\\..*|includeif\\..*)$'], root, signal, true)
    if (config.truncated || config.text !== '') throw new Error('Git observation refused unsafe repository configuration (filters/includes)')
    const metadata = await readCommand([...safeGit, 'rev-parse', '--path-format=absolute', '--show-toplevel', '--absolute-git-dir', '--git-common-dir', '--show-object-format'], root, signal)
    const parts = metadata.text.trim().split('\n')
    const [top, gitDirectory, commonDirectory, format] = parts
    if (metadata.truncated || parts.length !== 4 || !top || !gitDirectory || !commonDirectory || (format !== 'sha1' && format !== 'sha256')) throw new Error('Git repository identity is unavailable')
    const subtree = relative(top, root)
    if (isAbsolute(subtree) || subtree === '..' || subtree.startsWith('../')) throw new Error('Workspace is outside its Git repository')
    const head = await readCommand([...safeGit, 'rev-parse', '--verify', '--quiet', 'HEAD'], root, signal, true)
    const revision = head.text.trim()
    if (revision && !/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(revision)) throw new Error('Invalid Git HEAD identity')
    // Config refusal is defense in depth, not the isolation boundary: a writer
    // could change it after inspection. Run reads against a private metadata
    // view with no repository config, hooks, info/attributes or helper commands.
    const temporary = await mkdtemp(join(tmpdir(), 'xerxes-git-observation-'))
    try {
      await mkdir(join(temporary, 'refs'))
      await symlink(join(commonDirectory, 'objects'), join(temporary, 'objects'), 'dir')
      await Bun.write(join(temporary, 'HEAD'), revision ? `${revision}\n` : 'ref: refs/heads/unborn\n')
      await Bun.write(join(temporary, 'config'), `[core]\nrepositoryformatversion = ${format === 'sha256' ? 1 : 0}\nbare = false\n${format === 'sha256' ? '[extensions]\nobjectformat = sha256\n' : ''}`)
      const index = Bun.file(join(gitDirectory, 'index'))
      if (await index.exists()) {
        if (index.size > 64 * 1024 * 1024) throw new Error('Git index exceeds observation budget')
        const writer = Bun.file(join(temporary, 'index')).writer()
        const reader = index.stream().getReader()
        let bytes = 0
        try {
          while (true) {
            signal?.throwIfAborted()
            const chunk = await reader.read()
            if (chunk.done) break
            bytes += chunk.value.byteLength
            if (bytes > 64 * 1024 * 1024) throw new Error('Git index exceeds observation budget')
            writer.write(chunk.value)
          }
        } finally {
          try { await reader.cancel() } finally { reader.releaseLock(); await writer.end() }
        }
      }
      const cwd = subtree ? join(top, subtree) : top
      const result = await readCommand([...safeGit, `--git-dir=${temporary}`, `--work-tree=${top}`, ...args], cwd, signal)
      return { ...result, untrusted: true }
    } finally { await rm(temporary, { recursive: true, force: true }) }
  }

  async health(signal?: AbortSignal, view: 'all' | 'disk' = 'all') {
    signal?.throwIfAborted()
    const disk = await statfs(await realpath(this.root))
    const diskResult = { available_bytes: disk.bavail * disk.bsize, total_bytes: disk.blocks * disk.bsize }
    if (view === 'disk') return { disk: diskResult, untrusted: true }
    const processes = await readCommand(['/bin/ps', '-eo', 'pid,pcpu,pmem,comm'], this.root, signal)
    return { disk: diskResult,
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
