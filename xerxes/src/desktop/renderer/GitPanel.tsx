// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Source Control in the task rail, modelled on the editor view people already
 * know: branch and sync up top, a commit box, then Staged / Changes groups
 * with per-file actions, the selected file's diff beside them, and recent
 * history. Everything goes through the daemon's `git.*` methods, which run
 * in the session's repository — the renderer never touches git itself.
 *
 * Remote actions (push, pull, fetch) run only when clicked. Discard asks
 * first. "Review with agent" does not review anything here: it asks the chat
 * agent to, in its own words, with its own tools.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactElement } from 'react'

import type { ScmBranch, ScmCommit, ScmCommitDetail, ScmFile, ScmStatus } from '../../workspace/gitScm.js'
import { desktopCall, text, type RpcRecord } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { reflowContextText } from './ContextInspector.js'
import { store, type Snapshot } from './store.js'

export type ScmGroupKey = 'conflicts' | 'staged' | 'unstaged' | 'untracked'

export interface ScmSelection {
  readonly path: string
  readonly group: ScmGroupKey | 'commit'
  /** Set when the file was picked from a commit in Recent commits. */
  readonly commit?: { readonly hash: string; readonly short: string; readonly origPath?: string }
}

const STATUS_LABEL: Record<string, string> = {
  M: 'Modified', A: 'Added', D: 'Deleted', R: 'Renamed', C: 'Copied', T: 'Type changed', U: 'Conflict', '?': 'Untracked',
}

/** Letter shown in the row badge; untracked reads as "U" like the editor, conflicts as "!". */
export function statusLetter(file: ScmFile, group: ScmGroupKey): string {
  if (group === 'conflicts') return '!'
  if (file.status === '?') return 'U'
  return file.status
}

export function statusLabel(file: ScmFile, group: ScmGroupKey): string {
  return group === 'conflicts' ? 'Merge conflict' : STATUS_LABEL[file.status] ?? 'Changed'
}

/**
 * The chat prompt behind "Review with agent". It names the exact diff to read
 * so the agent reviews what you are about to commit, not the whole repo, and
 * it asks for findings rather than edits.
 */
export function reviewPrompt(status: Pick<ScmStatus, 'branch' | 'hasHead'>): string {
  const read = status.hasHead
    ? '`git diff HEAD` (staged and unstaged together) plus the untracked files from `git ls-files --others --exclude-standard`'
    : '`git ls-files --others --exclude-standard` and the new files themselves (there is no commit yet)'
  return [
    `Do a deep code review of all uncommitted changes${status.branch ? ` on branch \`${status.branch}\`` : ''} — staged, unstaged and new files alike. Respect .gitignore: files it matches are not part of the change, so leave them out.`,
    '',
    `Read them yourself with ${read}. Open the surrounding code wherever a change's correctness depends on it, and check the callers of anything whose behavior changed.`,
    '',
    'Report findings most severe first: correctness bugs, regressions, security problems, races, unhandled errors, and missing or weakened tests. For each, give file:line, what goes wrong and when, and a concrete fix. Skip style nits. If you find nothing serious, say so plainly.',
    '',
    'Do not modify any files.',
  ].join('\n')
}

export type PrimaryAction =
  | { readonly kind: 'commit'; readonly label: string; readonly title: string }
  | { readonly kind: 'push' | 'pull' | 'sync' | 'publish'; readonly label: string; readonly title: string }

/**
 * The main button follows the next useful step, like the editor: commit while
 * there is anything to commit; once the tree is clean, push, pull, sync or
 * publish the branch instead of offering a commit that cannot happen.
 */
export function primaryAction(repo: Pick<ScmStatus, 'upstream' | 'ahead' | 'behind' | 'branch' | 'hasHead' | 'detached'>, total: number): PrimaryAction {
  if (total > 0 || !repo.hasHead || repo.detached || !repo.branch) {
    return { kind: 'commit', label: 'Commit all', title: `Commit all ${total} changed file${total === 1 ? '' : 's'} (.gitignore respected)` }
  }
  const plural = (n: number) => `${n} commit${n === 1 ? '' : 's'}`
  if (!repo.upstream) return { kind: 'publish', label: 'Publish branch', title: `Push ${repo.branch} to the remote and track it` }
  if (repo.ahead > 0 && repo.behind > 0) return { kind: 'sync', label: `Sync ↓${repo.behind} ↑${repo.ahead}`, title: `Pull ${plural(repo.behind)} (fast-forward), then push ${plural(repo.ahead)} to ${repo.upstream}` }
  if (repo.ahead > 0) return { kind: 'push', label: `Push ↑${repo.ahead}`, title: `Push ${plural(repo.ahead)} to ${repo.upstream}` }
  if (repo.behind > 0) return { kind: 'pull', label: `Pull ↓${repo.behind}`, title: `Pull ${plural(repo.behind)} from ${repo.upstream} (fast-forward only)` }
  return { kind: 'commit', label: 'Commit all', title: 'Nothing to commit' }
}

function basename(path: string): string {
  return path.split('/').pop() || path
}

function dirname(path: string): string {
  const at = path.lastIndexOf('/')
  return at > 0 ? path.slice(0, at) : ''
}

const DRAFT_KEY = 'xerxes.git.draft.'

function useScm(snap: Snapshot) {
  const [repo, setRepo] = useState<ScmStatus | null | undefined>(undefined)
  const [reason, setReason] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const call = useCallback((method: string, params: RpcRecord = {}) => desktopCall(window.xerxes, snap.sessionKey, method, params), [snap.sessionKey])
  const refresh = useCallback(async () => {
    try {
      const result = await call('git.status')
      if (!alive.current) return
      const repository = result.repository as ScmStatus | null
      setRepo(repository ?? null)
      setReason(text(result.reason))
    } catch (failure) {
      if (alive.current) setError(failure instanceof Error ? failure.message : String(failure))
    }
  }, [call])
  /** Run one action; its fresh status replaces the list, its failure is shown verbatim. */
  const act = useCallback(async (label: string, method: string, params: RpcRecord = {}): Promise<RpcRecord | null> => {
    setBusy(label); setError('')
    try {
      const result = await call(method, params)
      if (alive.current && result.status) setRepo(result.status as ScmStatus)
      return result
    } catch (failure) {
      if (alive.current) setError(failure instanceof Error ? failure.message : String(failure))
      return null
    } finally { if (alive.current) setBusy('') }
  }, [call])
  return { repo, reason, error, setError, busy, call, refresh, act }
}

export function GitPanel({ snap, initialPath = '', onSnapshots, onReviewSent }: { snap: Snapshot; initialPath?: string; onSnapshots?: () => void; onReviewSent?: () => void }): ReactElement {
  const scm = useScm(snap)
  const { repo, refresh } = scm
  const [selected, setSelected] = useState<ScmSelection | null>(null)
  const [message, setMessage] = useState('')
  const [notice, setNotice] = useState('')
  const [branchesOpen, setBranchesOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(true)

  // Refresh on open, on window focus, when a turn settles or the agent edits a
  // file, and on a slow poll while visible — outside edits (your editor, a
  // terminal) have no event to listen for.
  useEffect(() => { void refresh() }, [refresh, snap.cwd])
  useEffect(() => { if (!snap.turnActive) void refresh() }, [snap.turnActive, snap.changes.length, refresh])
  useEffect(() => {
    const onFocus = () => { void refresh() }
    window.addEventListener('focus', onFocus)
    const timer = setInterval(() => { if (document.visibilityState === 'visible') void refresh() }, 8000)
    return () => { window.removeEventListener('focus', onFocus); clearInterval(timer) }
  }, [refresh])

  // The draft survives closing the rail and switching tabs, per repository.
  const draftKey = repo ? DRAFT_KEY + repo.root : ''
  useEffect(() => { if (draftKey) setMessage(sessionStorage.getItem(draftKey) ?? '') }, [draftKey])
  useEffect(() => { if (draftKey) { if (message) sessionStorage.setItem(draftKey, message); else sessionStorage.removeItem(draftKey) } }, [draftKey, message])

  const groups = useMemo(() => repo ? ([
    ['conflicts', 'Merge conflicts', repo.conflicts],
    ['staged', 'Staged changes', repo.staged],
    ['unstaged', 'Changes', [...repo.unstaged, ...repo.untracked]],
  ] as const) : [], [repo])

  // Keep the selection valid as the lists change; honour a deep link once.
  const linked = useRef('')
  useEffect(() => {
    if (!repo) return
    const find = (path: string): ScmSelection | null => {
      if (repo.conflicts.some(f => f.path === path)) return { path, group: 'conflicts' }
      if (repo.unstaged.some(f => f.path === path)) return { path, group: 'unstaged' }
      if (repo.untracked.some(f => f.path === path)) return { path, group: 'untracked' }
      if (repo.staged.some(f => f.path === path)) return { path, group: 'staged' }
      return null
    }
    if (initialPath && linked.current !== initialPath) {
      linked.current = initialPath
      const hit = find(initialPath)
      if (hit) { setSelected(hit); return }
    }
    setSelected(current => {
      // A file picked from history stays put while the working tree changes.
      if (current?.group === 'commit') return current
      if (current) {
        const same = (current.group === 'staged' ? repo.staged : current.group === 'conflicts' ? repo.conflicts : current.group === 'untracked' ? repo.untracked : repo.unstaged).some(f => f.path === current.path)
        if (same) return current
        const moved = find(current.path)
        if (moved) return moved
      }
      const first = repo.conflicts[0] ?? repo.staged[0] ?? repo.unstaged[0] ?? repo.untracked[0]
      return first ? find(first.path) : null
    })
  }, [repo, initialPath])

  if (repo === undefined) {
    // A runtime started before this app version has no git.* methods at all.
    if (/Unknown method: git\./.test(scm.error)) {
      return <div className="scm scm--empty">
        <div className="scm-empty"><Icon name="branch" size={22} /><strong>Update the workspace runtime</strong><p>The running runtime is older than this app and has no Git support yet. Updating restarts it once your running work finishes.</p></div>
        <button className="btn" onClick={() => { void store.restartDaemon().then(() => refresh()) }}>Update runtime</button>
      </div>
    }
    return <div className="scm scm--empty"><p className="studio-muted">Reading the repository…</p>{scm.error && <p className="studio-error" role="alert">{scm.error}</p>}</div>
  }
  if (repo === null) {
    return <div className="scm scm--empty">
      <div className="scm-empty"><Icon name="branch" size={22} /><strong>No git repository</strong><p>{scm.reason || 'This folder is not a git repository.'}</p></div>
      {onSnapshots && <button className="btn" onClick={onSnapshots}>Snapshots & restore</button>}
    </div>
  }

  const total = repo.counts.staged + repo.counts.unstaged + repo.counts.untracked + repo.counts.conflicts
  // Commit, Generate and Review all act on every change (.gitignore respected);
  // the Staged / Changes groups are for looking, not for choosing what ships.
  const canCommit = Boolean(message.trim()) && !scm.busy && repo.counts.conflicts === 0 && total > 0
  const primary = primaryAction(repo, total)
  const runPrimary = async (): Promise<void> => {
    if (primary.kind === 'commit') return commit()
    if (primary.kind === 'sync') {
      const pulled = await scm.act('Pulling…', 'git.pull')
      if (!pulled) return
    }
    if (primary.kind === 'pull') { if (await scm.act('Pulling…', 'git.pull')) setNotice('Pulled.'); return }
    const pushed = await scm.act(primary.kind === 'publish' ? 'Publishing…' : 'Pushing…', 'git.push')
    if (pushed) setNotice(primary.kind === 'publish' ? 'Branch published.' : primary.kind === 'sync' ? 'Synced.' : `Pushed to ${repo.upstream}.`)
  }

  const commit = async (): Promise<void> => {
    if (!canCommit) return
    const result = await scm.act('Committing…', 'git.commit', { message, all: true })
    if (!result) return
    const done = result.commit as ScmCommit | undefined
    setMessage('')
    setNotice(done ? `Committed ${done.short} — ${done.subject}` : 'Committed')
  }

  const generate = async (): Promise<void> => {
    const result = await scm.act('Drafting a message…', 'git.commitMessage')
    if (result && typeof result.message === 'string') setMessage(result.message)
  }

  const review = (): void => {
    void store.submit(reviewPrompt(repo))
    setNotice(snap.turnActive ? 'Review request sent — it runs when the current step settles.' : 'Review request sent to the chat.')
    // The Git view covers the conversation; bring it back so the review is visible.
    onReviewSent?.()
  }

  const discard = (files: readonly ScmFile[]): void => {
    if (!files.length) return
    const untracked = files.filter(f => f.status === '?').length
    const what = files.length === 1 ? `“${files[0]!.path}”` : `${files.length} files`
    const prompt = untracked === files.length
      ? `Delete ${what}? ${files.length === 1 ? 'It is' : 'They are'} untracked, so this cannot be undone.`
      : `Discard your unstaged changes to ${what}? This cannot be undone.${untracked ? ` ${untracked} untracked file${untracked === 1 ? '' : 's'} will be deleted.` : ''}`
    if (!window.confirm(prompt)) return
    void scm.act('Discarding…', 'git.discard', { paths: files.map(f => f.path) })
  }

  const onMessageKey = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); void commit() }
  }

  const sync = repo.upstream
    ? <>
        <button className="scm-tool" title={repo.behind ? `Pull ${repo.behind} commit${repo.behind === 1 ? '' : 's'} (fast-forward only)` : 'Pull'} disabled={Boolean(scm.busy)} onClick={() => void scm.act('Pulling…', 'git.pull').then(r => r && setNotice('Pulled.'))}>
          <Icon name="arrowDown" size={13} />{repo.behind > 0 && <span>{repo.behind}</span>}
        </button>
        <button className="scm-tool" title={repo.ahead ? `Push ${repo.ahead} commit${repo.ahead === 1 ? '' : 's'} to ${repo.upstream}` : `Push to ${repo.upstream}`} disabled={Boolean(scm.busy)} onClick={() => void scm.act('Pushing…', 'git.push').then(r => r && setNotice(`Pushed to ${repo.upstream}.`))}>
          <Icon name="arrowUp" size={13} />{repo.ahead > 0 && <span>{repo.ahead}</span>}
        </button>
      </>
    : repo.branch && repo.hasHead
      ? <button className="scm-tool scm-tool--label" title="Push this branch and track it on the remote" disabled={Boolean(scm.busy)} onClick={() => void scm.act('Publishing…', 'git.push').then(r => r && setNotice('Branch published.'))}><Icon name="cloud" size={13} /><span>Publish</span></button>
      : null

  return (
    <div className="scm">
      <div className="scm__side">
        <header className="scm__head">
          <button className="scm-branch" aria-expanded={branchesOpen} title={repo.detached ? 'Detached HEAD — pick a branch' : 'Switch or create a branch'} onClick={() => setBranchesOpen(value => !value)}>
            <Icon name="branch" size={14} /><span>{repo.detached ? 'Detached HEAD' : repo.branch ?? 'No branch'}</span><Icon name="caretDown" size={11} />
          </button>
          <span className="scm__tools">
            {sync}
            <button className="scm-tool" title="Fetch from remotes" disabled={Boolean(scm.busy)} onClick={() => void scm.act('Fetching…', 'git.fetch')}><Icon name="retry" size={13} /></button>
          </span>
        </header>
        {branchesOpen && <BranchList scm={scm} onDone={() => setBranchesOpen(false)} />}

        <div className="scm__commit">
          <textarea
            value={message}
            rows={Math.min(14, Math.max(6, message.split('\n').length + 1))}
            placeholder={repo.branch ? `Message (⌘⏎ to commit on ${repo.branch})` : 'Message (⌘⏎ to commit)'}
            aria-label="Commit message"
            spellCheck
            onChange={event => setMessage(event.target.value)}
            onKeyDown={onMessageKey}
          />
          <div className="scm__commit-row">
            <button className="btn" disabled={Boolean(scm.busy) || total === 0} title="Draft a message from all changes with your current model" onClick={() => void generate()}>
              <Icon name="spark" size={13} /> {scm.busy === 'Drafting a message…' ? 'Drafting…' : 'Generate'}
            </button>
            <button
              className="btn btn--solid"
              disabled={primary.kind === 'commit' ? !canCommit : Boolean(scm.busy)}
              title={primary.kind === 'commit' && repo.counts.conflicts ? 'Resolve merge conflicts first' : primary.title}
              onClick={() => void runPrimary()}
            >
              <Icon name={primary.kind === 'commit' ? 'check' : primary.kind === 'pull' ? 'arrowDown' : primary.kind === 'publish' ? 'cloud' : 'arrowUp'} size={13} /> {primary.label}
            </button>
          </div>
          <button className="scm-review" disabled={total === 0} onClick={review}>
            <Icon name="agent" size={13} /> Review all changes with the agent
          </button>
        </div>

        {(scm.busy || scm.error || notice) && (
          <div className="scm__status" role={scm.error ? 'alert' : 'status'}>
            {scm.busy ? <><span className="studio-spinner" /> {scm.busy}</> : scm.error ? <span className="scm__error">{scm.error}</span> : <span>{notice}</span>}
            {!scm.busy && (scm.error || notice) && <button className="scm-tool" aria-label="Dismiss" onClick={() => { scm.setError(''); setNotice('') }}><Icon name="close" size={11} /></button>}
          </div>
        )}

        <div className="scm__groups">
          {total === 0 && <div className="scm-clean"><Icon name="checkCircle" size={16} /> Nothing to commit — the working tree is clean.</div>}
          {groups.map(([key, title, files]) => files.length === 0 ? null : (
            <ScmGroup
              key={key}
              title={title}
              count={key === 'unstaged' ? repo.counts.unstaged + repo.counts.untracked : repo.counts[key]}
              files={files}
              group={key}
              selected={selected}
              busy={Boolean(scm.busy)}
              onSelect={setSelected}
              onStage={paths => void scm.act('Staging…', 'git.stage', { paths })}
              onUnstage={paths => void scm.act('Unstaging…', 'git.unstage', { paths })}
              onStageAll={() => void scm.act('Staging…', 'git.stage', { all: true })}
              onUnstageAll={() => void scm.act('Unstaging…', 'git.unstage', { all: true })}
              onDiscard={discard}
            />
          ))}
          {repo.truncated && <p className="studio-muted scm__more">Showing the first 2,000 files per group.</p>}
        </div>

        <details className="scm__history" open={historyOpen} onToggle={event => setHistoryOpen(event.currentTarget.open)}>
          <summary>Recent commits</summary>
          {historyOpen && <History scm={scm} head={repo.hasHead ? `${repo.branch}:${repo.ahead}:${repo.behind}:${repo.counts.staged}` : ''} selected={selected} onSelect={setSelected} />}
        </details>
        {onSnapshots && <button className="scm-link" onClick={onSnapshots}>Snapshots & restore…</button>}
      </div>

      <FileDiff scm={scm} selection={selected} repo={repo} />
    </div>
  )
}

function ScmGroup({ title, count, files, group, selected, busy, onSelect, onStage, onUnstage, onStageAll, onUnstageAll, onDiscard }: {
  title: string; count: number; files: readonly ScmFile[]; group: ScmGroupKey; selected: ScmSelection | null; busy: boolean
  onSelect: (selection: ScmSelection) => void; onStage: (paths: string[]) => void; onUnstage: (paths: string[]) => void
  onStageAll: () => void; onUnstageAll: () => void; onDiscard: (files: readonly ScmFile[]) => void
}): ReactElement {
  // File lists start folded so the commit box and history lead; a merge
  // conflict blocks the commit, so that group is always shown open.
  const [open, setOpen] = useState(group === 'conflicts')
  const staged = group === 'staged'
  return (
    <section className="scm-group">
      <header className="scm-group__head">
        <button className="scm-group__toggle" aria-expanded={open} onClick={() => setOpen(value => !value)}>
          <Icon name="chevron" size={12} /><span>{title}</span><span className="scm-count">{count}</span>
        </button>
        <span className="scm-group__actions">
          {group !== 'staged' && group !== 'conflicts' && <button className="scm-tool" title="Discard all changes" disabled={busy} onClick={() => onDiscard(files)}><Icon name="retry" size={13} /></button>}
          {staged
            ? <button className="scm-tool" title="Unstage all" disabled={busy} onClick={onUnstageAll}><Icon name="close" size={12} /></button>
            : <button className="scm-tool" title="Stage all" disabled={busy} onClick={onStageAll}><Icon name="plus" size={13} /></button>}
        </span>
      </header>
      {open && <ul className="scm-files">
        {files.map(file => {
          const rowGroup: ScmGroupKey = file.status === '?' ? 'untracked' : group
          const isSelected = selected?.path === file.path && (selected.group === rowGroup)
          const dir = dirname(file.path)
          return (
            <li key={`${rowGroup}:${file.path}`} className={`scm-file${isSelected ? ' is-selected' : ''}`}>
              <button className="scm-file__open" title={file.origPath ? `${file.origPath} → ${file.path}` : file.path} onClick={() => onSelect({ path: file.path, group: rowGroup })}>
                <span className="scm-file__name">{basename(file.path)}</span>
                {dir && <span className="scm-file__dir">{dir}</span>}
              </button>
              <span className="scm-file__actions">
                {!staged && group !== 'conflicts' && <button className="scm-tool" title={file.status === '?' ? 'Delete untracked file' : 'Discard changes'} disabled={busy} onClick={() => onDiscard([file])}><Icon name={file.status === '?' ? 'trash' : 'retry'} size={12} /></button>}
                {staged
                  ? <button className="scm-tool" title="Unstage" disabled={busy} onClick={() => onUnstage([file.path])}><Icon name="close" size={11} /></button>
                  : <button className="scm-tool" title={group === 'conflicts' ? 'Mark resolved (stage)' : 'Stage'} disabled={busy} onClick={() => onStage([file.path])}><Icon name={group === 'conflicts' ? 'check' : 'plus'} size={12} /></button>}
              </span>
              <span className={`scm-badge scm-badge--${statusLetter(file, rowGroup) === '!' ? 'conflict' : file.status === '?' ? 'untracked' : file.status}`} title={statusLabel(file, rowGroup)}>{statusLetter(file, rowGroup)}</span>
            </li>
          )
        })}
      </ul>}
    </section>
  )
}

function FileDiff({ scm, selection, repo }: { scm: ReturnType<typeof useScm>; selection: ScmSelection | null; repo: ScmStatus }): ReactElement {
  const [lines, setLines] = useState<RpcRecord[] | null>(null)
  const [error, setError] = useState('')
  // Re-read when the file's status changes (stage/unstage moves its diff).
  const version = selection ? JSON.stringify([repo.staged.find(f => f.path === selection.path), repo.unstaged.find(f => f.path === selection.path), repo.counts]) : ''
  useEffect(() => {
    if (!selection) { setLines(null); return }
    let current = true
    setError('')
    const params = selection.commit
      ? { path: selection.path, commit: selection.commit.hash, ...(selection.commit.origPath ? { orig_path: selection.commit.origPath } : {}) }
      : { path: selection.path, staged: selection.group === 'staged', untracked: selection.group === 'untracked' }
    void scm.call('git.diff', params)
      .then(result => { if (current) setLines(Array.isArray(result.lines) ? result.lines as RpcRecord[] : []) })
      .catch(failure => { if (current) { setLines([]); setError(failure instanceof Error ? failure.message : String(failure)) } })
    return () => { current = false }
  }, [selection?.path, selection?.group, selection?.commit?.hash, version, scm.call])
  if (!selection) return <section className="scm__diff scm__diff--empty"><p className="studio-muted">Select a file to see its changes.</p></section>
  const visible = (lines ?? []).filter(line => line.kind !== 'file')
  return (
    <section className="scm__diff change-review__preview" aria-label="Selected file diff">
      <header title={selection.path}>
        <span>{selection.path}</span>
        <em>{selection.commit ? `Commit ${selection.commit.short}` : selection.group === 'staged' ? 'Staged' : selection.group === 'untracked' ? 'New file' : selection.group === 'conflicts' ? 'Conflict' : 'Working tree'}</em>
      </header>
      <pre className="change-review__source" tabIndex={0} role="region" aria-label="Diff contents">
        {lines === null && <span role="status">Loading…</span>}
        {error && <span role="alert">{error}</span>}
        {lines !== null && !error && visible.length === 0 && <span>No textual changes (binary file or mode change).</span>}
        {visible.map((line, index) => (
          <span key={index} className={`studio-diff-${text(line.kind)}`}>
            <span className="diff-gutter" aria-hidden="true">{typeof line.oldLine === 'number' ? line.oldLine : ''}</span>
            <span className="diff-gutter" aria-hidden="true">{typeof line.newLine === 'number' ? line.newLine : ''}</span>
            <span className="diff-code">{text(line.text)}</span>
          </span>
        ))}
      </pre>
    </section>
  )
}

function BranchList({ scm, onDone }: { scm: ReturnType<typeof useScm>; onDone: () => void }): ReactElement {
  const [branches, setBranches] = useState<ScmBranch[] | null>(null)
  const [name, setName] = useState('')
  const [filter, setFilter] = useState('')
  useEffect(() => {
    let current = true
    void scm.call('git.branches').then(result => { if (current) setBranches(Array.isArray(result.branches) ? result.branches as ScmBranch[] : []) }).catch(failure => { if (current) { setBranches([]); scm.setError(failure instanceof Error ? failure.message : String(failure)) } })
    return () => { current = false }
  }, [scm.call])
  const switchTo = async (branch: string, create = false): Promise<void> => {
    const result = await scm.act(create ? 'Creating branch…' : 'Switching branch…', 'git.switch', { branch, create })
    if (result) onDone()
  }
  const shown = (branches ?? []).filter(branch => !filter || branch.name.toLowerCase().includes(filter.toLowerCase()))
  return (
    <div className="scm-branches" role="dialog" aria-label="Branches">
      <input value={filter} placeholder="Filter branches…" aria-label="Filter branches" onChange={event => setFilter(event.target.value)} autoFocus />
      <ul>
        {branches === null && <li className="studio-muted">Loading branches…</li>}
        {shown.map(branch => (
          <li key={branch.name}>
            <button disabled={branch.current || Boolean(scm.busy)} onClick={() => void switchTo(branch.name)}>
              <Icon name={branch.current ? 'check' : 'branch'} size={12} />
              <span>{branch.name}</span>
              <small>{branch.updated}</small>
            </button>
          </li>
        ))}
      </ul>
      <form className="scm-branches__new" onSubmit={event => { event.preventDefault(); if (name.trim()) void switchTo(name.trim(), true) }}>
        <input value={name} placeholder="New branch name" aria-label="New branch name" spellCheck={false} onChange={event => setName(event.target.value)} />
        <button className="btn" disabled={!name.trim() || Boolean(scm.busy)}>Create</button>
      </form>
    </div>
  )
}

function History({ scm, head, selected, onSelect }: { scm: ReturnType<typeof useScm>; head: string; selected: ScmSelection | null; onSelect: (selection: ScmSelection) => void }): ReactElement {
  const [commits, setCommits] = useState<ScmCommit[] | null>(null)
  const [open, setOpen] = useState('')
  const [detail, setDetail] = useState<{ commit: ScmCommitDetail; files: ScmFile[] } | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let current = true
    if (!head) { setCommits([]); return }
    void scm.call('git.log', { limit: 15 }).then(result => { if (current) setCommits(Array.isArray(result.commits) ? result.commits as ScmCommit[] : []) }).catch(() => { if (current) setCommits([]) })
    return () => { current = false }
  }, [head, scm.call])
  const toggle = (commit: ScmCommit): void => {
    if (open === commit.hash) { setOpen(''); return }
    setOpen(commit.hash); setDetail(null); setError('')
    void scm.call('git.show', { commit: commit.hash })
      .then(result => {
        const shown = { commit: result.commit as ScmCommitDetail, files: Array.isArray(result.files) ? result.files as ScmFile[] : [] }
        setDetail(shown)
        // Opening a commit shows its first file straight away, like the editor does.
        const first = shown.files[0]
        if (first) onSelect({ path: first.path, group: 'commit', commit: { hash: commit.hash, short: commit.short, ...(first.origPath ? { origPath: first.origPath } : {}) } })
      })
      .catch(failure => setError(failure instanceof Error ? failure.message : String(failure)))
  }
  if (commits === null) return <p className="studio-muted scm__history-note">Loading…</p>
  if (commits.length === 0) return <p className="studio-muted scm__history-note">No commits yet.</p>
  return (
    <ul className="scm-log">
      {commits.map(commit => {
        const expanded = open === commit.hash
        return (
          <li key={commit.hash} className={expanded ? 'is-open' : ''}>
            <button className="scm-log__row" aria-expanded={expanded} title={`${commit.hash}\n${commit.author}, ${commit.when}`} onClick={() => toggle(commit)}>
              <span className="scm-log__subject">{commit.subject}</span>
              <span className="scm-log__meta"><code>{commit.short}</code> · {commit.author} · {commit.when}</span>
            </button>
            {expanded && <div className="scm-log__detail">
              {error && <p className="scm__error">{error}</p>}
              {!detail && !error && <p className="studio-muted scm__history-note">Loading…</p>}
              {detail?.commit.body && <p className="scm-log__body">{reflowContextText(detail.commit.body)}</p>}
              {detail && detail.files.length === 0 && <p className="studio-muted scm__history-note">{detail.commit.parents > 1 ? 'Merge commit with no changes against its first parent.' : 'No file changes.'}</p>}
              {detail && detail.files.length > 0 && <ul className="scm-files">
                {detail.files.map(file => {
                  const isSelected = selected?.group === 'commit' && selected.commit?.hash === commit.hash && selected.path === file.path
                  const dir = dirname(file.path)
                  return (
                    <li key={file.path} className={`scm-file${isSelected ? ' is-selected' : ''}`}>
                      <button className="scm-file__open" title={file.origPath ? `${file.origPath} → ${file.path}` : file.path} onClick={() => onSelect({ path: file.path, group: 'commit', commit: { hash: commit.hash, short: commit.short, ...(file.origPath ? { origPath: file.origPath } : {}) } })}>
                        <span className="scm-file__name">{basename(file.path)}</span>
                        {dir && <span className="scm-file__dir">{dir}</span>}
                      </button>
                      <span className={`scm-badge scm-badge--${file.status}`} title={STATUS_LABEL[file.status] ?? 'Changed'}>{file.status}</span>
                    </li>
                  )
                })}
              </ul>}
            </div>}
          </li>
        )
      })}
    </ul>
  )
}
