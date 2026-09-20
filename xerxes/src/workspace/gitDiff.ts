// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Bounded `git diff` collection for the TUI diff viewer (F7 / /diff).
 *
 * Runs git through direct argv (never a shell), reads the worktree diff
 * against HEAD (staged + unstaged) plus the untracked-file list, and parses
 * the unified diff into classified rows for rendering. Output is capped by
 * lines and bytes with an explicit truncation marker — a monster diff must
 * never wedge the terminal. The process runner is injectable so tests never
 * touch a real repository.
 */

export type DiffLineKind = 'add' | 'context' | 'del' | 'file' | 'hunk' | 'meta'

export interface DiffLine {
  readonly kind: DiffLineKind
  readonly text: string
  readonly oldLine?: number | undefined
  readonly newLine?: number | undefined
}

export interface GitDiffSummary {
  readonly deletions: number
  readonly files: number
  readonly insertions: number
  readonly lines: readonly DiffLine[]
  readonly truncated: boolean
  readonly untracked: readonly string[]
  readonly untrackedTruncated: boolean
}

export type GitDiffResult =
  | { readonly kind: 'clean' }
  | { readonly kind: 'error'; readonly message: string }
  | { readonly kind: 'ok'; readonly diff: GitDiffSummary }

export interface GitDiffRunOutput {
  readonly code: number
  readonly stderr: string
  readonly stdout: string
}

/** Injectable process runner: direct argv, resolves with the captured output. */
export type GitDiffRunner = (args: readonly string[], cwd: string) => Promise<GitDiffRunOutput>

export interface CollectGitDiffOptions {
  readonly path?: string
  readonly includeUntracked?: boolean
  readonly cwd: string
  readonly maxBytes?: number
  readonly maxLines?: number
  readonly maxUntracked?: number
  readonly run?: GitDiffRunner
}

const DEFAULT_MAX_LINES = 4000
const DEFAULT_MAX_BYTES = 256 * 1024
const DEFAULT_MAX_UNTRACKED = 50
/** Keep one minified/generated row from consuming the whole panel budget. */
const MAX_RENDERED_ROW_CHARS = 512
const TRUNCATED_ROW_SUFFIX = ' … [line truncated]'
/** Git's well-known empty tree, used when the repo has no HEAD commit yet. */
const EMPTY_TREE = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'

const defaultRunner: GitDiffRunner = async (args, cwd) => {
  const proc = Bun.spawn(['git', ...args], {
    cwd,
    stderr: 'pipe',
    stdin: 'ignore',
    stdout: 'pipe'
  })
  let failure = ''
  const timer = setTimeout(() => { failure = 'Git preview timed out'; proc.kill() }, 15000)
  const read = async (stream: ReadableStream<Uint8Array>, limit: number) => {
    const reader = stream.getReader()
    const chunks: Uint8Array[] = []
    let bytes = 0
    try {
      while (true) {
        const next = await reader.read()
        if (next.done) break
        bytes += next.value.byteLength
        if (bytes > limit) { failure = 'Git preview exceeds the output limit'; proc.kill(); break }
        chunks.push(next.value)
      }
    } finally { reader.releaseLock() }
    return Buffer.concat(chunks).toString('utf8')
  }
  try {
    const [stdout, stderr, code] = await Promise.all([read(proc.stdout, 2 * 1024 * 1024), read(proc.stderr, 8192), proc.exited])
    return { code: failure ? 2 : code, stderr: failure || stderr, stdout }
  } finally { clearTimeout(timer) }
}

const utf8Length = (text: string): number => new TextEncoder().encode(text).length

/** Parse unified-diff text into classified rows, honoring line/byte caps. */
export function parseUnifiedDiff(
  text: string,
  { maxBytes = DEFAULT_MAX_BYTES, maxLines = DEFAULT_MAX_LINES }: { maxBytes?: number | undefined; maxLines?: number | undefined } = {}
): { insertions: number; deletions: number; files: number; lines: DiffLine[]; truncated: boolean } {
  const lines: DiffLine[] = []
  let files = 0
  let insertions = 0
  let deletions = 0
  let bytes = 0
  let truncated = false
  let oldLine: number | undefined
  let newLine: number | undefined

  const push = (kind: DiffLineKind, row: string, numbers: Pick<DiffLine, 'oldLine' | 'newLine'> = {}): boolean => {
    if (lines.length >= maxLines || bytes >= maxBytes) {
      truncated = true
      return false
    }
    let rendered = row
    if (rendered.length > MAX_RENDERED_ROW_CHARS) {
      rendered = rendered.slice(0, MAX_RENDERED_ROW_CHARS - TRUNCATED_ROW_SUFFIX.length) + TRUNCATED_ROW_SUFFIX
      truncated = true
    }
    const remainingBytes = maxBytes - bytes
    if (utf8Length(rendered) + 1 > remainingBytes) {
      truncated = true
      return true
    }
    lines.push({ kind, text: rendered, ...numbers })
    bytes += utf8Length(rendered) + 1
    return true
  }

  for (const raw of text.split('\n')) {
    if (raw.startsWith('diff --git ')) {
      files += 1
      const path = raw.match(/^diff --git a\/.+ b\/(.+)$/)?.[1] ?? raw
      oldLine = undefined
      newLine = undefined
      if (!push('file', path)) break
      continue
    }
    if (raw.startsWith('@@')) {
      const range = raw.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/)
      oldLine = range?.[1] === undefined ? undefined : Number.parseInt(range[1], 10)
      newLine = range?.[2] === undefined ? undefined : Number.parseInt(range[2], 10)
      if (!push('hunk', raw)) break
      continue
    }
    if (raw.startsWith('+++') || raw.startsWith('---')) {
      // File-marker lines; the diff --git row already named the file.
      continue
    }
    if (raw.startsWith('+')) {
      insertions += 1
      if (!push('add', raw, { newLine })) break
      if (newLine !== undefined) newLine += 1
      continue
    }
    if (raw.startsWith('-')) {
      deletions += 1
      if (!push('del', raw, { oldLine })) break
      if (oldLine !== undefined) oldLine += 1
      continue
    }
    if (raw.startsWith(' ')) {
      if (!push('context', raw, { oldLine, newLine })) break
      if (oldLine !== undefined) oldLine += 1
      if (newLine !== undefined) newLine += 1
      continue
    }
    if (raw.startsWith('index ') || raw.startsWith('Binary ') || raw.startsWith('new file') || raw.startsWith('deleted file') || raw.startsWith('similarity') || raw.startsWith('rename ')) {
      if (!push('meta', raw)) break
      continue
    }
    if (raw.startsWith('\\')) {
      if (!push('meta', raw)) break
      continue
    }
    // Blank separator lines between sections.
    if (raw.trim() === '' && lines.length > 0) {
      if (!push('context', '')) break
    }
  }

  // Drop a trailing blank row for a cleaner panel bottom edge.
  while (lines.length > 0 && lines[lines.length - 1]!.text === '') {
    lines.pop()
  }

  return { deletions, files, insertions, lines, truncated }
}

/** Collect the bounded worktree diff for one repository. */
export async function collectGitDiff(options: CollectGitDiffOptions): Promise<GitDiffResult> {
  const path = options.path
  if (path !== undefined && (!path || path.length > 4096 || path.startsWith('/') || path.split('/').some(part => part === '..' || part === '.') || /[\0\r\n]/.test(path))) return { kind: 'error', message: 'Choose a workspace-relative file path without parent traversal' }
  const cwd = options.cwd.trim() || process.cwd()
  const run = options.run ?? defaultRunner
  const maxUntracked = options.maxUntracked ?? DEFAULT_MAX_UNTRACKED

  let inside: GitDiffRunOutput
  try {
    inside = await run(['rev-parse', '--is-inside-work-tree'], cwd)
  } catch (error) {
    return { kind: 'error', message: `could not run git: ${error instanceof Error ? error.message : String(error)}` }
  }
  if (inside.code !== 0 || inside.stdout.trim() !== 'true') {
    return { kind: 'error', message: 'not a git repository — the diff viewer needs a git worktree' }
  }

  const head = await run(['rev-parse', '--verify', 'HEAD'], cwd)
  const base = head.code === 0 ? 'HEAD' : EMPTY_TREE

  const diff = await run([...(path ? ['--literal-pathspecs'] : []), 'diff', '--no-color', '--no-ext-diff', base, '--', ...(path ? [path] : [])], cwd)
  if (diff.code !== 0) {
    return { kind: 'error', message: `git diff failed: ${diff.stderr.trim() || `exit code ${diff.code}`}` }
  }

  const untrackedOut = await run(['ls-files', '--others', '--exclude-standard', ...(options.includeUntracked ? ['-z'] : [])], cwd)
  if (untrackedOut.code !== 0) return { kind: 'error', message: 'Could not list untracked files: ' + (untrackedOut.stderr.trim() || 'Git failed') }
  const untrackedAll = untrackedOut.stdout.split(options.includeUntracked ? '\0' : '\n').filter(entry => Boolean(entry) && (!path || entry === path))
  const untracked = untrackedAll.slice(0, maxUntracked)
  const untrackedTruncated = untrackedAll.length > untracked.length

  const parsed = parseUnifiedDiff(diff.stdout, { maxBytes: options.maxBytes, maxLines: options.maxLines })
  if (options.includeUntracked) {
    for (const path of untracked) {
      // Direct argv and Git's symlink handling avoid shell execution and
      // following untracked links outside the workspace.
      const remainingLines = (options.maxLines ?? DEFAULT_MAX_LINES) - parsed.lines.length
      const remainingBytes = (options.maxBytes ?? DEFAULT_MAX_BYTES) - utf8Length(parsed.lines.map(line => line.text).join('\n'))
      if (remainingLines <= 1 || remainingBytes <= 0) { parsed.truncated = true; break }
      const preview = await run(['diff', '--no-index', '--no-color', '--no-ext-diff', '--', '/dev/null', path], cwd)
      const extra = preview.code <= 1
        ? parseUnifiedDiff(preview.stdout, { maxLines: remainingLines, maxBytes: remainingBytes })
        : { lines: [{ kind: 'file' as const, text: path }, { kind: 'meta' as const, text: `Preview unavailable: ${preview.stderr.trim()}` }], files: 1, insertions: 0, deletions: 0, truncated: false }
      if (!extra.lines.length) extra.lines.push({ kind: 'file', text: path }, { kind: 'meta', text: 'Empty untracked file' })
      else if (preview.code <= 1 && extra.lines.length < remainingLines && extra.lines.every(line => line.kind === 'file' || line.kind === 'meta') && !extra.lines.some(line => line.text.startsWith('Binary '))) {
        extra.lines.push({ kind: 'meta', text: 'Empty untracked file' })
      }
      parsed.lines.push(...extra.lines.map(line => line.kind === 'file' ? { ...line, text: path } : line))
      parsed.files += Math.max(1, extra.files)
      parsed.insertions += extra.insertions
      parsed.truncated ||= extra.truncated
    }
  }

  if (parsed.lines.length === 0 && untracked.length === 0) {
    return { kind: 'clean' }
  }

  return {
    kind: 'ok',
    diff: {
      deletions: parsed.deletions,
      files: parsed.files,
      insertions: parsed.insertions,
      lines: parsed.lines,
      truncated: parsed.truncated,
      untracked,
      untrackedTruncated
    }
  }
}
