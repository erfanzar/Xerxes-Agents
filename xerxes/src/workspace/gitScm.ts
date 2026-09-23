// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Source-control operations behind the desktop Git panel.
 *
 * `gitDiff.ts` answers "what changed" as one combined diff; a Source Control
 * view needs the index distinction (staged vs. not), per-file actions, commit,
 * branches, history and remote sync. Every call runs git through direct argv
 * (never a shell) from the repository top level — porcelain paths are
 * root-relative — with `--literal-pathspecs` so a file named `*.ts` is that
 * file, not a glob. Mutations are serialized per repository so two clicks can
 * never race on `index.lock`. Remote operations never prompt: a credential
 * request fails fast instead of hanging the daemon.
 */

import { copyFile, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { parseUnifiedDiff, type DiffLine } from './gitDiff.js'

export type ScmFileStatus = 'M' | 'A' | 'D' | 'R' | 'C' | 'T' | 'U' | '?'

export interface ScmFile {
  readonly path: string
  /** Source path of a rename or copy. */
  readonly origPath?: string
  readonly status: ScmFileStatus
}

export interface ScmStatus {
  readonly root: string
  readonly branch: string | null
  readonly detached: boolean
  readonly hasHead: boolean
  readonly upstream: string | null
  readonly ahead: number
  readonly behind: number
  readonly staged: readonly ScmFile[]
  readonly unstaged: readonly ScmFile[]
  readonly untracked: readonly ScmFile[]
  readonly conflicts: readonly ScmFile[]
  /** More entries existed than the panel budget; counts are still exact. */
  readonly truncated: boolean
  readonly counts: { readonly staged: number; readonly unstaged: number; readonly untracked: number; readonly conflicts: number }
}

export interface ScmBranch {
  readonly name: string
  readonly current: boolean
  readonly upstream: string | null
  readonly updated: string
}

export interface ScmCommit {
  readonly hash: string
  readonly short: string
  readonly subject: string
  readonly author: string
  readonly when: string
}

export interface ScmCommitDetail extends ScmCommit {
  readonly body: string
  readonly date: string
  readonly parents: number
}

export interface ScmRunOutput {
  readonly code: number
  readonly stdout: string
  readonly stderr: string
}

export interface ScmRunOptions {
  readonly timeoutMs?: number
  /** Extra environment, e.g. GIT_INDEX_FILE for a scratch index. */
  readonly env?: Readonly<Record<string, string>>
}

/** Injectable process runner: direct argv, resolves with the captured output. */
export type ScmRunner = (args: readonly string[], cwd: string, options?: ScmRunOptions) => Promise<ScmRunOutput>

const LOCAL_TIMEOUT_MS = 20_000
const REMOTE_TIMEOUT_MS = 120_000
const MAX_OUTPUT_BYTES = 8 * 1024 * 1024
/** Rows per group the panel renders; the counts stay exact beyond it. */
const MAX_FILES_PER_GROUP = 2000

export const defaultScmRunner: ScmRunner = async (args, cwd, options = {}) => {
  const proc = Bun.spawn(['git', ...args], {
    cwd,
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      ...process.env,
      // A credential or host-key prompt would block forever with no TTY.
      GIT_TERMINAL_PROMPT: '0',
      GIT_ASKPASS: process.env.GIT_ASKPASS ?? '',
      SSH_ASKPASS: process.env.SSH_ASKPASS ?? '',
      GCM_INTERACTIVE: 'never',
      GIT_SSH_COMMAND: process.env.GIT_SSH_COMMAND ?? 'ssh -o BatchMode=yes',
      LC_ALL: 'C',
      ...options.env,
    },
  })
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; proc.kill() }, options.timeoutMs ?? LOCAL_TIMEOUT_MS)
  const read = async (stream: ReadableStream<Uint8Array>): Promise<string> => {
    const reader = stream.getReader()
    const chunks: Uint8Array[] = []
    let bytes = 0
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        if (bytes < MAX_OUTPUT_BYTES) { chunks.push(value); bytes += value.byteLength }
      }
    } finally { reader.releaseLock() }
    return Buffer.concat(chunks).toString('utf8')
  }
  try {
    const [stdout, stderr, code] = await Promise.all([read(proc.stdout), read(proc.stderr), proc.exited])
    if (timedOut) return { code: 124, stdout, stderr: `git ${args[0] ?? ''} timed out` }
    return { code, stdout, stderr }
  } finally { clearTimeout(timer) }
}

/** Why a git call failed, in git's own words when it gave any. */
export class ScmError extends Error {}

function failure(output: ScmRunOutput, fallback: string): ScmError {
  const detail = (output.stderr.trim() || output.stdout.trim()).split('\n').filter(Boolean).slice(-4).join('\n')
  return new ScmError(detail || fallback)
}

/** A commit id as git prints it — never a ref name, range or option. */
export function assertScmHash(hash: unknown): string {
  if (typeof hash !== 'string' || !/^[0-9a-f]{7,64}$/i.test(hash)) throw new ScmError('Choose a commit by its hash')
  return hash
}

/** Parse `git diff-tree -z --name-status`: status, then one path (two for renames and copies). */
export function parseNameStatus(stdout: string): ScmFile[] {
  const parts = stdout.split('\0')
  const files: ScmFile[] = []
  for (let index = 0; index < parts.length; index += 1) {
    const code = parts[index]
    if (!code) continue
    const kind = code[0]!
    if (kind === 'R' || kind === 'C') {
      const origPath = parts[index + 1] ?? ''
      const path = parts[index + 2] ?? ''
      index += 2
      if (path) files.push({ path, origPath, status: kind })
      continue
    }
    const path = parts[index + 1] ?? ''
    index += 1
    if (path) files.push({ path, status: (XY_STATUS.has(kind) ? kind : 'M') as ScmFileStatus })
  }
  return files
}

/** Workspace-relative, no traversal, no control characters — same rule as `workspace.diff`. */
export function assertScmPath(path: unknown): string {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path)
    || path.split(/[\\/]/).some(part => part === '..') || /[\0\r\n]/.test(path)) {
    throw new ScmError('Choose a repository-relative file path without parent traversal')
  }
  return path
}

export function assertScmPaths(paths: unknown): string[] {
  if (!Array.isArray(paths) || paths.length === 0) throw new ScmError('Choose at least one file')
  if (paths.length > 5000) throw new ScmError('Too many files in one request')
  return paths.map(assertScmPath)
}

const XY_STATUS = new Set<string>(['M', 'A', 'D', 'R', 'C', 'T', 'U'])

/**
 * Parse `git status --porcelain=v2 -z --branch`. Entries are NUL-separated;
 * a rename (`2`) carries its source path as the following entry. Paths may
 * contain spaces, so the fixed fields are split off the front and the rest
 * of the entry is the path verbatim.
 */
export function parseScmStatus(stdout: string, root: string, maxPerGroup = MAX_FILES_PER_GROUP): ScmStatus {
  const entries = stdout.split('\0')
  let branch: string | null = null
  let detached = false
  let hasHead = true
  let upstream: string | null = null
  let ahead = 0
  let behind = 0
  const staged: ScmFile[] = []
  const unstaged: ScmFile[] = []
  const untracked: ScmFile[] = []
  const conflicts: ScmFile[] = []
  const counts = { staged: 0, unstaged: 0, untracked: 0, conflicts: 0 }
  const add = (group: ScmFile[], key: keyof typeof counts, file: ScmFile): void => {
    counts[key] += 1
    if (group.length < maxPerGroup) group.push(file)
  }
  const fields = (entry: string, count: number): { head: string[]; rest: string } => {
    const head: string[] = []
    let rest = entry
    for (let index = 0; index < count; index += 1) {
      const space = rest.indexOf(' ')
      if (space < 0) return { head: [...head, rest], rest: '' }
      head.push(rest.slice(0, space))
      rest = rest.slice(space + 1)
    }
    return { head, rest }
  }
  const status = (code: string): ScmFileStatus => (XY_STATUS.has(code) ? code : 'M') as ScmFileStatus
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index]!
    if (!entry) continue
    if (entry.startsWith('# ')) {
      const [, key, ...value] = entry.split(' ')
      const text = value.join(' ')
      if (key === 'branch.oid') hasHead = text !== '(initial)'
      else if (key === 'branch.head') { detached = text === '(detached)'; branch = detached ? null : text }
      else if (key === 'branch.upstream') upstream = text
      else if (key === 'branch.ab') {
        const match = /^\+(\d+) -(\d+)$/.exec(text)
        if (match) { ahead = Number(match[1]); behind = Number(match[2]) }
      }
      continue
    }
    const type = entry[0]
    if (type === '?') { add(untracked, 'untracked', { path: entry.slice(2), status: '?' }); continue }
    if (type === '!') continue
    if (type === 'u') {
      const { rest } = fields(entry, 10)
      add(conflicts, 'conflicts', { path: rest, status: 'U' })
      continue
    }
    if (type === '1' || type === '2') {
      const { head, rest } = fields(entry, type === '1' ? 8 : 9)
      const xy = head[1] ?? '..'
      const x = xy[0] ?? '.'
      const y = xy[1] ?? '.'
      let origPath: string | undefined
      if (type === '2') { origPath = entries[index + 1]; index += 1 }
      if (x !== '.') add(staged, 'staged', { path: rest, status: status(x), ...(origPath && (x === 'R' || x === 'C') ? { origPath } : {}) })
      if (y !== '.') add(unstaged, 'unstaged', { path: rest, status: status(y), ...(origPath && (y === 'R' || y === 'C') ? { origPath } : {}) })
    }
  }
  const truncated = counts.staged > staged.length || counts.unstaged > unstaged.length
    || counts.untracked > untracked.length || counts.conflicts > conflicts.length
  return { root, branch, detached, hasHead, upstream, ahead, behind, staged, unstaged, untracked, conflicts, truncated, counts }
}

const locks = new Map<string, Promise<unknown>>()

/** One git mutation at a time per repository: `index.lock` is not shareable. */
function serialized<T>(root: string, work: () => Promise<T>): Promise<T> {
  const previous = locks.get(root) ?? Promise.resolve()
  const next = previous.catch(() => {}).then(work)
  const settled = next.catch(() => {})
  locks.set(root, settled)
  void settled.then(() => { if (locks.get(root) === settled) locks.delete(root) })
  return next
}

export interface CommitContext {
  readonly stat: string
  /** `git diff --name-status` for the whole change. */
  readonly files: string
  readonly diff: string
  readonly truncated: boolean
  readonly recentSubjects: readonly string[]
}

/** Source Control for one repository, resolved from any directory inside it. */
export class GitScm {
  private constructor(readonly root: string, private readonly run: ScmRunner) {}

  /** `null` when `cwd` is not inside a git worktree. */
  static async open(cwd: string, run: ScmRunner = defaultScmRunner): Promise<GitScm | null> {
    let top: ScmRunOutput
    try { top = await run(['rev-parse', '--show-toplevel'], cwd) }
    catch (error) { throw new ScmError(`Could not run git: ${error instanceof Error ? error.message : String(error)}`) }
    if (top.code !== 0) return null
    const root = top.stdout.trim()
    return root ? new GitScm(root, run) : null
  }

  private async git(args: readonly string[], fallback: string, options?: ScmRunOptions): Promise<string> {
    const output = await this.run(args, this.root, options)
    if (output.code !== 0) throw failure(output, fallback)
    return output.stdout
  }

  private mutate<T>(work: () => Promise<T>): Promise<T> {
    return serialized(this.root, work)
  }

  async status(): Promise<ScmStatus> {
    const stdout = await this.git(
      ['--no-optional-locks', 'status', '--porcelain=v2', '-z', '--branch', '--untracked-files=all'],
      'git status failed',
    )
    return parseScmStatus(stdout, this.root)
  }

  /** One file's diff: the index against HEAD when `staged`, else the worktree against the index. */
  async diff(path: string, options: { staged?: boolean; untracked?: boolean } = {}): Promise<{ lines: readonly DiffLine[]; truncated: boolean }> {
    assertScmPath(path)
    const output = options.untracked
      ? await this.run(['diff', '--no-index', '--no-color', '--no-ext-diff', '--', '/dev/null', path], this.root)
      : await this.run(['--literal-pathspecs', 'diff', '--no-color', '--no-ext-diff', ...(options.staged ? ['--cached'] : []), '--', path], this.root)
    // `--no-index` exits 1 when the files differ, which is the normal case.
    if (output.code !== 0 && !(options.untracked && output.code === 1)) throw failure(output, 'git diff failed')
    const parsed = parseUnifiedDiff(output.stdout)
    return { lines: parsed.lines, truncated: parsed.truncated }
  }

  /** One commit: its message and the files it changed against its first parent. */
  async showCommit(hash: string): Promise<{ commit: ScmCommitDetail; files: ScmFile[] }> {
    assertScmHash(hash)
    const meta = await this.git(['log', '-1', '--format=%H%x1f%h%x1f%s%x1f%an%x1f%cr%x1f%aD%x1f%P%x1f%b', hash, '--'], 'Could not read the commit')
    const [full = '', short = '', subject = '', author = '', when = '', date = '', parents = '', body = ''] = meta.replace(/\n$/, '').split('\x1f')
    // --root lists a first commit's files; -m --first-parent gives a merge its changes against the mainline.
    const tree = await this.git(['diff-tree', '-r', '-z', '--root', '-M', '--no-commit-id', '--name-status', '-m', '--first-parent', hash], 'Could not list the commit\'s files')
    return {
      commit: { hash: full, short, subject, author, when, date, body: body.trim(), parents: parents.split(' ').filter(Boolean).length },
      files: parseNameStatus(tree),
    }
  }

  /** One file as that commit changed it (a rename passes both names). */
  async commitDiff(hash: string, path: string, origPath?: string): Promise<{ lines: readonly DiffLine[]; truncated: boolean }> {
    assertScmHash(hash)
    assertScmPath(path)
    if (origPath !== undefined) assertScmPath(origPath)
    const output = await this.run(['--literal-pathspecs', 'show', '--no-color', '--no-ext-diff', '--format=', '-M', '-m', '--first-parent', hash, '--', ...(origPath ? [origPath] : []), path], this.root)
    if (output.code !== 0) throw failure(output, 'git show failed')
    const parsed = parseUnifiedDiff(output.stdout)
    return { lines: parsed.lines, truncated: parsed.truncated }
  }

  stage(paths: readonly string[]): Promise<void> {
    return this.mutate(async () => { await this.git(['--literal-pathspecs', 'add', '-A', '--', ...paths], 'Could not stage') })
  }

  stageAll(): Promise<void> {
    return this.mutate(async () => { await this.git(['add', '-A'], 'Could not stage all changes') })
  }

  async unstage(paths: readonly string[]): Promise<void> {
    const hasHead = await this.hasHead()
    return this.mutate(async () => {
      if (hasHead) await this.git(['--literal-pathspecs', 'restore', '--staged', '--', ...paths], 'Could not unstage')
      else await this.git(['--literal-pathspecs', 'rm', '--cached', '-r', '-q', '--', ...paths], 'Could not unstage')
    })
  }

  async unstageAll(): Promise<void> {
    const hasHead = await this.hasHead()
    return this.mutate(async () => {
      if (hasHead) await this.git(['reset', '-q'], 'Could not unstage all changes')
      else await this.git(['rm', '--cached', '-r', '-q', '.'], 'Could not unstage all changes')
    })
  }

  /**
   * Throw away unstaged work: tracked files go back to their staged (index)
   * content, untracked files are deleted. Staged changes are never touched —
   * that is what makes "discard" safe to offer on the Changes group.
   */
  async discard(paths: readonly string[]): Promise<void> {
    const current = await this.status()
    const untracked = new Set(current.untracked.map(file => file.path))
    const tracked = paths.filter(path => !untracked.has(path))
    const fresh = paths.filter(path => untracked.has(path))
    return this.mutate(async () => {
      if (tracked.length) await this.git(['--literal-pathspecs', 'restore', '--worktree', '--', ...tracked], 'Could not discard changes')
      if (fresh.length) await this.git(['--literal-pathspecs', 'clean', '-f', '-q', '--', ...fresh], 'Could not delete untracked files')
    })
  }

  /**
   * Commit the index — or, with `all`, every change first: `add -A` stages
   * modifications, deletions and new files alike while honouring .gitignore.
   */
  async commit(message: string, options: { amend?: boolean; all?: boolean } = {}): Promise<ScmCommit> {
    const text = message.replace(/\r\n/g, '\n').trim()
    if (!text && !options.amend) throw new ScmError('Write a commit message first')
    return this.mutate(async () => {
      if (options.all) await this.git(['add', '-A'], 'Could not stage all changes')
      await this.git(['commit', '-q', ...(options.amend ? ['--amend'] : []), ...(text ? ['-m', text] : ['--no-edit'])], 'Commit failed')
      const [commit] = await this.logUnlocked(1)
      if (!commit) throw new ScmError('Committed, but the new commit could not be read back')
      return commit
    })
  }

  async branches(): Promise<ScmBranch[]> {
    const stdout = await this.git(
      ['for-each-ref', '--sort=-committerdate', '--format=%(refname:short)%1f%(HEAD)%1f%(upstream:short)%1f%(committerdate:relative)', 'refs/heads'],
      'Could not list branches',
    )
    return stdout.split('\n').filter(Boolean).map(line => {
      const [name = '', head = '', upstream = '', updated = ''] = line.split('\x1f')
      return { name, current: head === '*', upstream: upstream || null, updated }
    })
  }

  async switchBranch(name: string, options: { create?: boolean } = {}): Promise<void> {
    const check = await this.run(['check-ref-format', '--branch', name], this.root)
    if (check.code !== 0 || name.startsWith('-')) throw new ScmError(`“${name}” is not a valid branch name`)
    return this.mutate(async () => { await this.git(['switch', ...(options.create ? ['-c'] : []), name], options.create ? 'Could not create the branch' : 'Could not switch branches') })
  }

  log(limit = 20): Promise<ScmCommit[]> {
    return this.logUnlocked(limit)
  }

  private async logUnlocked(limit: number): Promise<ScmCommit[]> {
    if (!(await this.hasHead())) return []
    const count = Math.max(1, Math.min(200, Math.trunc(limit)))
    const stdout = await this.git(['log', `-n${count}`, '--format=%H%x1f%h%x1f%s%x1f%an%x1f%cr'], 'Could not read history')
    return stdout.split('\n').filter(Boolean).map(line => {
      const [hash = '', short = '', subject = '', author = '', when = ''] = line.split('\x1f')
      return { hash, short, subject, author, when }
    })
  }

  fetch(): Promise<void> {
    return this.mutate(async () => { await this.git(['fetch', '--prune'], 'Fetch failed', { timeoutMs: REMOTE_TIMEOUT_MS }) })
  }

  /** Fast-forward only: a pull never creates a merge commit behind your back. */
  pull(): Promise<void> {
    return this.mutate(async () => { await this.git(['pull', '--ff-only'], 'Pull failed', { timeoutMs: REMOTE_TIMEOUT_MS }) })
  }

  /** Push the current branch; the first push of a new branch sets its upstream. */
  async push(): Promise<void> {
    const current = await this.status()
    if (current.detached || !current.branch) throw new ScmError('Check out a branch before pushing')
    const branch = current.branch
    const upstream = current.upstream
    let remote: string | null = null
    if (!upstream) {
      const remotes = (await this.git(['remote'], 'Could not list remotes')).split('\n').filter(Boolean)
      if (!remotes.length) throw new ScmError('This repository has no remote to push to')
      remote = remotes.includes('origin') ? 'origin' : remotes[0]!
    }
    return this.mutate(async () => {
      await this.git(remote ? ['push', '-u', remote, branch] : ['push'], 'Push failed', { timeoutMs: REMOTE_TIMEOUT_MS })
    })
  }

  /** What a commit-message writer needs: the staged diff (or all changes), its stat, and the house style. */
  /**
   * What a commit-message writer needs about *everything* a "commit all"
   * would record — staged, unstaged and new files, .gitignore respected.
   * It is computed on a scratch copy of the index (GIT_INDEX_FILE), so the
   * user's own staging is never touched.
   */
  async commitContext(maxDiffBytes = 80_000): Promise<CommitContext> {
    const indexPath = resolve(this.root, (await this.git(['rev-parse', '--git-path', 'index'], 'Could not locate the index')).trim())
    const scratch = await mkdtemp(join(tmpdir(), 'xerxes-scm-'))
    const env = { GIT_INDEX_FILE: join(scratch, 'index') }
    try {
      await copyFile(indexPath, env.GIT_INDEX_FILE).catch(() => {}) // an unborn repo may have no index yet
      await this.git(['add', '-A'], 'Could not read the changes', { env })
      // `--cached` compares the scratch index with HEAD (or the empty tree before the first commit).
      const stat = (await this.git(['diff', '--cached', '--no-color', '--stat=100'], 'git diff failed', { env })).trim()
      // Every changed file by name, so a diff cut short still leaves nothing unmentioned.
      const files = (await this.git(['diff', '--cached', '--no-color', '--name-status', '-M'], 'git diff failed', { env })).trim()
      // -D drops the body of deleted files (the name says enough) and -U2 trims
      // context, so the budget goes to what actually changed.
      const full = await this.git(['diff', '--cached', '--no-color', '--no-ext-diff', '-M', '-D', '-U2'], 'git diff failed', { env })
      const truncated = Buffer.byteLength(full) > maxDiffBytes
      const diff = truncated ? Buffer.from(full).subarray(0, maxDiffBytes).toString('utf8') : full
      const recentSubjects = (await this.logUnlocked(12)).map(commit => commit.subject)
      return { stat, files, diff, truncated, recentSubjects }
    } finally {
      await rm(scratch, { recursive: true, force: true })
    }
  }

  private async hasHead(): Promise<boolean> {
    return (await this.run(['rev-parse', '--verify', '-q', 'HEAD'], this.root)).code === 0
  }
}

/**
 * Prompt for a commit message. The recent subjects carry the repository's
 * own convention (e.g. `fix(scope): …`), which beats any style we could
 * prescribe here.
 */
export function commitMessagePrompt(context: CommitContext): string {
  const style = context.recentSubjects.length
    ? `Recent commit subjects in this repository, newest first. Match their convention for the subject line (prefix, scope, tense, casing):\n${context.recentSubjects.map(subject => `- ${subject}`).join('\n')}`
    : 'Use a concise imperative subject line (e.g. "Fix token refresh race").'
  return [
    'Write a detailed git commit message for the change below. A reviewer reading only this message, months from now, should understand what changed and why without opening the diff.',
    '',
    style,
    '',
    'Format:',
    '- Line 1: the subject, at most 72 characters, no trailing period. Name the actual change, not the activity ("remove retired TUI state modules", not "update files").',
    '- Line 2: blank.',
    '- Then an opening paragraph of 1-3 sentences: what the change does as a whole and why, as far as the diff shows it.',
    '- Then a blank line and a bulleted list ("- ") of the distinct changes, grouped by area or module. Each bullet names the concrete thing that changed — the function, command, type, config key, file or behavior — and what happened to it. Group related files into one bullet instead of listing files mechanically.',
    '- Call out anything a reviewer must know: behavior changes, removed or renamed APIs, new dependencies, schema or config changes, and tests added, removed or changed.',
    '- Scale to the change: a small fix gets 1-3 bullets, a large change 5-12. Only a truly trivial one-line change may omit the body.',
    '- Wrap body lines at 72 characters.',
    '',
    'Rules:',
    '- Describe only what the diff and file list show. Do not invent motivation, tickets, benchmarks or co-authors; if the reason is not visible, describe the effect instead.',
    '- If the diff below is truncated, use the full file list and stat to cover the files it does not show.',
    '- Reply with the commit message only: no quotes, no code fences, no preamble.',
    '',
    'Changed files (name-status; everything this commit will include):',
    context.files || '(none listed)',
    '',
    'Stat:',
    context.stat || '(no stat)',
    '',
    `Diff${context.truncated ? ' (truncated — see the file list above for the rest)' : ''}:`,
    context.diff,
  ].join('\n')
}

/** Strip the wrappers models add despite being told not to. */
export function cleanCommitMessage(text: string): string {
  let body = text.replace(/\r\n/g, '\n').trim()
  const fenced = /^```[\w-]*\n([\s\S]*?)\n```$/.exec(body)
  if (fenced) body = fenced[1]!.trim()
  body = body.replace(/^(?:commit message|message)\s*:\s*/i, '')
  if (/^(["'`]).*\1$/s.test(body) && !body.slice(1, -1).includes(body[0]!)) body = body.slice(1, -1).trim()
  return body.replace(/\n{3,}/g, '\n\n')
}
