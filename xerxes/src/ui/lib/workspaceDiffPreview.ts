// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { DiffLine, GitDiffResult } from './gitDiff.js'

/** Select one file, including against older daemons that ignore the path filter. */
export async function workspaceFileDiff(
  call: (method: string, params: Record<string, unknown>) => Promise<unknown>, path: string, untracked: boolean,
): Promise<GitDiffResult> {
  const result = await call('workspace.diff', { path }) as GitDiffResult
  if (!result || !['ok', 'clean', 'error'].includes(result.kind)) throw new Error('Invalid workspace diff response')
  if (result.kind !== 'ok') return result
  const start = result.diff.lines.findIndex(line => line.kind === 'file' && line.text === path)
  if (start >= 0) {
    const following = result.diff.lines.findIndex((line, index) => index > start && line.kind === 'file')
    const lines = result.diff.lines.slice(start, following < 0 ? undefined : following)
    return { kind: 'ok', diff: { ...result.diff, files: 1, lines, insertions: lines.filter(line => line.kind === 'add').length, deletions: lines.filter(line => line.kind === 'del').length } }
  }
  if (!untracked) return { kind: 'error', message: 'This runtime omitted the selected diff. Update the workspace runtime and retry.' }
  // Old runtimes may exhaust the overview before reaching new files. Their
  // existing bounded, workspace-checked preview can still show those additions.
  const preview = await call('workspace.filePreview', { path }) as { content?: unknown; truncated?: unknown; error?: unknown; ok?: unknown }
  if (preview?.ok === false || typeof preview?.content !== 'string') return { kind: 'error', message: typeof preview?.error === 'string' ? preview.error : 'File preview is unavailable. Refresh the working tree and retry.' }
  const contents = preview.content.replace(/\n$/, '').split('\n')
  const rows = preview.content ? contents.slice(0, 3997) : []
  const lines: DiffLine[] = [{ kind: 'file', text: path }, { kind: 'hunk', text: `@@ -0,0 +1,${contents.length} @@` }, ...rows.map((line, index) => ({ kind: 'add' as const, text: '+' + line.slice(0, 511), newLine: index + 1 }))]
  if (!preview.content) lines.push({ kind: 'meta', text: 'Empty untracked file' })
  return { kind: 'ok', diff: { files: 1, lines, insertions: rows.length, deletions: 0, truncated: preview.truncated === true || contents.length > rows.length && Boolean(preview.content) || rows.some(row => row.length > 511), untracked: [path], untrackedTruncated: false } }
}
