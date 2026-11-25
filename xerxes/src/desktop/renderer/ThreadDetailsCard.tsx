// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Floating Thread Details Panel (`data-thread-details-card`).
 *
 * Top-right floating card:
 * - Section 1: Machine/Workspace row + folder switcher, Open in Files/Finder menu, + New worktree
 * - Section 2: Git branch row, Commit & push menu, Changes (+adds -dels) diff trigger
 * - Section 3 (conditional): Quiet 1-line links for active Immortals, Subagents, Plan, or Artifacts
 *
 * On wide windows (`mode === 'inline'`) it floats gently in the top-right margin beside the
 * centered conversation column; on narrower windows (`mode === 'popover'`) it dismisses on
 * outside click or Escape.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'
import { store, showsImmortalsForSession, type Snapshot } from './store.js'
import { desktopCall } from './desktopRpc.js'
import { localMachineLabel } from './runModeView.js'
import { T3Icon } from './T3Icons.js'
import { activityFleetRows, agentState } from './AgentRoster.js'
import { parseGoal } from './goalText.js'
import {
  ANTIGRAVITY_COMMIT_AND_PUSH_PROMPT,
  ANTIGRAVITY_PULL_REQUEST_PROMPT,
  isAntigravityAcpSession,
  prefillComposerPrompt,
  pullRequestPrompt,
} from './GitPanel.js'
import { swarmMemberCounts } from './swarmState.js'
import type { ScmStatus } from '../../workspace/gitScm.js'
import type { DesktopPanel } from './DesktopPanels.js'

interface GitRepoSummary {
  readonly branch: string | null
  readonly added: number
  readonly removed: number
  readonly files: number
}

function folderBase(cwd: string): string {
  return cwd.replaceAll('\\', '/').split('/').filter(Boolean).at(-1) || 'workspace'
}

function localHostLabel(): string {
  return localMachineLabel(true)
}

function revealInFolderLabel(): string {
  if (typeof navigator === 'undefined') return 'Reveal in Finder'
  const ua = `${navigator.platform ?? ''} ${navigator.userAgent ?? ''}`
  if (/Win/i.test(ua)) return 'Show in Explorer'
  if (/Mac/i.test(ua)) return 'Reveal in Finder'
  return 'Show in Files'
}

export function ThreadDetailsCard({
  snap,
  mode = 'popover',
  onClose,
  onNavigate,
  onOpenTerminal,
}: {
  readonly snap: Snapshot
  readonly mode?: 'inline' | 'popover'
  readonly onClose: () => void
  readonly onNavigate: (panel: DesktopPanel, filePath?: string) => void
  readonly onOpenTerminal: () => void
}): ReactElement {
  const cardRef = useRef<HTMLElement>(null)
  const [repo, setRepo] = useState<GitRepoSummary | null>(null)
  const [creatingWorktree, setCreatingWorktree] = useState(false)
  const [worktreeName, setWorktreeName] = useState('')
  const [openMenu, setOpenMenu] = useState<'open' | 'git' | null>(null)

  const refreshGit = useCallback(async () => {
    if (!snap.cwd || snap.connection !== 'online') return
    try {
      const result = await desktopCall(window.xerxes, snap.sessionKey, 'git.status', {})
      const status = result.repository as ScmStatus | null | undefined
      if (status !== undefined) store.syncBranch(status?.branch ?? '')
      if (!status) {
        setRepo(null)
        return
      }
      const files = status.counts.staged + status.counts.unstaged + status.counts.untracked + status.counts.conflicts
      setRepo({
        branch: status.branch,
        added: status.lines?.added ?? 0,
        removed: status.lines?.removed ?? 0,
        files,
      })
    } catch {
      setRepo(null)
    }
  }, [snap.cwd, snap.sessionKey, snap.connection])

  useEffect(() => {
    if (!snap.turnActive) void refreshGit()
  }, [refreshGit, snap.turnActive, snap.changes.length])

  useEffect(() => {
    const onFocus = (): void => { void refreshGit() }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [refreshGit])

  useEffect(() => {
    const hasNested = creatingWorktree || openMenu !== null
    if (mode === 'inline' && !hasNested) return
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        if (creatingWorktree) {
          setCreatingWorktree(false)
          setWorktreeName('')
          return
        }
        if (openMenu !== null) {
          setOpenMenu(null)
          return
        }
        if (mode !== 'inline') onClose()
      }
    }
    const onPointer = (event: MouseEvent): void => {
      const target = event.target as HTMLElement | null
      if (!target) return
      if (cardRef.current?.contains(target)) return
      if (target.closest('[data-thread-details-toggle="true"]')) return
      if (openMenu !== null) setOpenMenu(null)
      if (mode !== 'inline') onClose()
    }
    document.addEventListener('keydown', onKey, true)
    document.addEventListener('mousedown', onPointer)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      document.removeEventListener('mousedown', onPointer)
    }
  }, [mode, onClose, creatingWorktree, openMenu])

  const projectName = folderBase(snap.cwd)
  const hostLabel = snap.contextHost?.trim() || (snap.storageScope?.startsWith('ssh:') ? snap.storageScope.slice(4) : localHostLabel())
  const branchName = repo?.branch || snap.branch || ''
  const sessionAdds = snap.changes.reduce((sum, file) => sum + file.adds, 0)
  const sessionDels = snap.changes.reduce((sum, file) => sum + file.dels, 0)
  const totalAdds = repo && (repo.added > 0 || repo.removed > 0) ? repo.added : sessionAdds
  const totalDels = repo && (repo.added > 0 || repo.removed > 0) ? repo.removed : sessionDels
  const changedCount = repo ? repo.files : snap.changes.length

  const fleet = activityFleetRows(snap.fleet, snap.blocks)
  const workingAgents = fleet.filter(row => agentState(row.status).priority < 2).length
  const goal = parseGoal(snap.goal)
  const todos = snap.todos ?? snap.plan?.items.map((item, idx) => ({ id: String(idx), content: item.text, status: item.done ? 'completed' : 'pending' })) ?? []
  const doneTodos = todos.filter(item => item.status === 'completed').length
  const canSubmit = snap.connection === 'online' && !snap.turnActive && !snap.submissionPending

  const submitWorktree = (): void => {
    const name = worktreeName.trim()
    setCreatingWorktree(false)
    setWorktreeName('')
    if (name) void store.createWorktree(name)
  }

  const revealFolder = (): void => {
    setOpenMenu(null)
    if (snap.cwd) void window.xerxes.fileAction?.('reveal', '.', snap.cwd)
  }

  const isAntigravity = isAntigravityAcpSession(snap)
  const commitAndPushPrompt = (): string =>
    `Review the current git diff on branch ${branchName || 'current branch'}, stage the changes, write a concise conventional commit message, and commit and push.`

  return (
    <aside
      ref={cardRef}
      className={`t3-details-card t3-details-card--${mode}`}
      data-thread-details-card="true"
      data-thread-details-mode={mode}
      aria-label="Thread details"
    >
      {/* Section 1: Machine / Workspace, Open in Files/Finder, New worktree */}
      <section className="t3-dsec">
        <div className="t3-drow t3-drow--host">
          <button
            type="button"
            className="t3-drow__main"
            title={`${hostLabel} · ${snap.cwd || 'No workspace'}`}
            onClick={() => store.openWorkspaceMenu()}
          >
            <T3Icon name="monitor" size={14} className="t3-drow__ico" />
            <span className="t3-drow__label">{hostLabel}</span>
          </button>
          <button
            type="button"
            className="t3-drow__trailing"
            title={`Switch workspace (${projectName})`}
            aria-label="Switch workspace folder"
            onClick={() => store.openWorkspaceMenu()}
          >
            <T3Icon name="folder" size={14} />
          </button>
        </div>

        <div className="t3-drow-anchor">
          <button
            type="button"
            className="t3-drow"
            aria-haspopup="menu"
            aria-expanded={openMenu === 'open'}
            onClick={() => setOpenMenu(current => (current === 'open' ? null : 'open'))}
          >
            <T3Icon name="codeEditor" size={14} className="t3-drow__ico t3-drow__ico--accent" />
            <span className="t3-drow__label">Open in Files</span>
            <T3Icon name="chevronDown" size={12} className="t3-drow__chev" />
          </button>
          {openMenu === 'open' && (
            <div className="t3-dsubmenu" role="menu">
              <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); onNavigate('files') }}>
                <T3Icon name="codeEditor" size={13} />
                <span>Files &amp; Editor</span>
              </button>
              <button type="button" role="menuitem" onClick={revealFolder}>
                <T3Icon name="folder" size={13} />
                <span>{revealInFolderLabel()}</span>
              </button>
              <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); onOpenTerminal() }}>
                <T3Icon name="terminal" size={13} />
                <span>Open Terminal</span>
              </button>
              <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); onNavigate('browser') }}>
                <T3Icon name="globe" size={13} />
                <span>Open Browser</span>
              </button>
            </div>
          )}
        </div>

        <button
          type="button"
          className="t3-drow"
          aria-expanded={creatingWorktree}
          onClick={() => setCreatingWorktree(value => !value)}
          title="Create an isolated git worktree next to this repository"
        >
          <T3Icon name="plus" size={14} className="t3-drow__ico" />
          <span className="t3-drow__label">New worktree</span>
        </button>

        {creatingWorktree && (
          <form
            className="t3-details-wt"
            onSubmit={event => {
              event.preventDefault()
              submitWorktree()
            }}
          >
            <input
              type="text"
              autoFocus
              spellCheck={false}
              placeholder="Branch or worktree name"
              aria-label="New worktree name"
              value={worktreeName}
              onChange={event => setWorktreeName(event.target.value)}
            />
            <button
              type="button"
              className="t3-details-pill"
              onClick={() => { setCreatingWorktree(false); setWorktreeName('') }}
            >
              Cancel
            </button>
            <button type="submit" className="t3-details-pill t3-details-pill--primary" disabled={!worktreeName.trim()}>
              Create
            </button>
          </form>
        )}
      </section>

      {/* Section 2: Git Branch, Commit & Push, Changes (+adds -dels) */}
      <section className="t3-dsec">
        <button
          type="button"
          className="t3-drow"
          title={branchName ? `Branch: ${branchName} · Open Source Control` : 'Open Source Control'}
          onClick={() => onNavigate('review')}
        >
          <T3Icon name="gitBranch" size={14} className="t3-drow__ico" />
          <span className="t3-drow__label t3-drow__label--mono">{branchName || 'No branch'}</span>
          <T3Icon name="chevronRight" size={12} className="t3-drow__chev" />
        </button>

        <div className="t3-drow-anchor">
          <button
            type="button"
            className="t3-drow"
            aria-haspopup="menu"
            aria-expanded={openMenu === 'git'}
            onClick={() => setOpenMenu(current => (current === 'git' ? null : 'git'))}
          >
            <T3Icon name="gitCommitPush" size={14} className="t3-drow__ico" />
            <span className="t3-drow__label">Commit &amp; push</span>
            <T3Icon name="chevronDown" size={12} className="t3-drow__chev" />
          </button>
          {openMenu === 'git' && (
            <div className="t3-dsubmenu" role="menu">
              <button
                type="button"
                role="menuitem"
                disabled={!canSubmit}
                onClick={() => {
                  setOpenMenu(null)
                  if (isAntigravity) {
                    onClose()
                    prefillComposerPrompt(ANTIGRAVITY_COMMIT_AND_PUSH_PROMPT)
                    return
                  }
                  void store.submit(commitAndPushPrompt(), 'Commit & push')
                }}
              >
                <T3Icon name="gitCommitPush" size={13} />
                <span>Commit &amp; push</span>
              </button>
              <button
                type="button"
                role="menuitem"
                disabled={!canSubmit}
                onClick={() => {
                  setOpenMenu(null)
                  if (isAntigravity) {
                    onClose()
                    prefillComposerPrompt(ANTIGRAVITY_PULL_REQUEST_PROMPT)
                    return
                  }
                  void store.submit(pullRequestPrompt(branchName || null), 'Create a pull request')
                }}
              >
                <T3Icon name="gitPr" size={13} />
                <span>Create PR</span>
              </button>
              <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); onNavigate('review') }}>
                <T3Icon name="fileDiff" size={13} />
                <span>Open Source Control</span>
              </button>
            </div>
          )}
        </div>

        <button
          type="button"
          className="t3-drow"
          title={changedCount > 0 ? `${changedCount} changed file${changedCount === 1 ? '' : 's'} · Open Diff` : 'Open Diff'}
          onClick={() => onNavigate('review')}
        >
          <T3Icon name="fileDiff" size={14} className="t3-drow__ico" />
          <span className="t3-drow__label">Changes</span>
          {(totalAdds > 0 || totalDels > 0) ? (
            <span className="t3-drow__diff">
              <span className="add">+{totalAdds.toLocaleString()}</span>
              <span className="del">-{totalDels.toLocaleString()}</span>
            </span>
          ) : changedCount > 0 ? (
            <span className="t3-drow__meta">{changedCount}</span>
          ) : (
            <span className="t3-drow__meta">Clean</span>
          )}
          <T3Icon name="chevronRight" size={12} className="t3-drow__chev" />
        </button>
      </section>

      {/* Section 3: Active Xerxes Orchestration (only shown when live work exists) */}
      {(goal || todos.length > 0 || fleet.length > 0 || snap.swarm?.active || showsImmortalsForSession(snap)) && (
        <section className="t3-dsec">
          {(snap.swarm?.active || showsImmortalsForSession(snap)) && (
            <button type="button" className="t3-drow" onClick={() => onNavigate('swarm')}>
              <T3Icon name="network" size={14} className="t3-drow__ico" />
              <span className="t3-drow__label">Immortals</span>
              <span className="t3-drow__meta">
                {snap.swarm?.active ? `${swarmMemberCounts(snap.swarm.members).working} active` : 'Open'}
              </span>
            </button>
          )}
          {fleet.length > 0 && (
            <button type="button" className="t3-drow" onClick={() => onNavigate('subagents')}>
              <T3Icon name="bot" size={14} className="t3-drow__ico" />
              <span className="t3-drow__label">Subagents</span>
              <span className="t3-drow__meta">{workingAgents > 0 ? `${workingAgents} working` : `${fleet.length}`}</span>
            </button>
          )}
          {todos.length > 0 && (
            <button
              type="button"
              className={`t3-drow${snap.tab === 'plan' ? ' is-selected' : ''}`}
              aria-pressed={snap.tab === 'plan'}
              title={snap.tab === 'plan' ? 'Back to conversation' : 'Open Plan & to-dos'}
              onClick={() => store.setTab(snap.tab === 'plan' ? 'activity' : 'plan')}
            >
              <T3Icon name="checkCircle" size={14} className="t3-drow__ico" />
              <span className="t3-drow__label">Plan &amp; to-dos</span>
              <span className="t3-drow__meta">{doneTodos}/{todos.length}</span>
            </button>
          )}
          {goal && (
            <button type="button" className="t3-drow" onClick={() => onNavigate('activity')} title={goal.objective}>
              <T3Icon name="sparkles" size={14} className="t3-drow__ico" />
              <span className="t3-drow__label">{goal.objective}</span>
              <span className="t3-drow__meta">{goal.phase}</span>
            </button>
          )}
        </section>
      )}
    </aside>
  )
}
