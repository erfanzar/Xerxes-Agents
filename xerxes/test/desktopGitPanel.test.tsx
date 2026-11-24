// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { FileDiff, primaryAction, reviewPrompt, scmSelectionKey, statusLabel, statusLetter } from '../src/desktop/renderer/GitPanel.js'


test('the review request covers every uncommitted change, names how to read it, and forbids edits', () => {
  const prompt = reviewPrompt({ branch: 'feature/login', hasHead: true })
  expect(prompt).toContain('all uncommitted changes on branch `feature/login` — staged, unstaged and new files alike')
  expect(prompt).toContain('`git diff HEAD`')
  expect(prompt).toContain('git ls-files --others --exclude-standard')
  expect(prompt).toContain('file:line')
  expect(prompt).toContain('Do not modify any files.')
  expect(prompt).toContain('Respect .gitignore')
  expect(prompt).not.toContain('ignoring anything in .gitignore')
  expect(prompt).not.toContain('--cached')
  expect(reviewPrompt({ branch: 'main', hasHead: false })).toContain('there is no commit yet')
})

test('status badges read like the editor: untracked is U, conflicts are !', () => {
  expect(statusLetter({ path: 'a', status: '?' }, 'untracked')).toBe('U')
  expect(statusLetter({ path: 'a', status: 'M' }, 'conflicts')).toBe('!')
  expect(statusLetter({ path: 'a', status: 'R', origPath: 'b' }, 'staged')).toBe('R')
  expect(statusLabel({ path: 'a', status: 'D' }, 'unstaged')).toBe('Deleted')
  expect(statusLabel({ path: 'a', status: 'U' }, 'conflicts')).toBe('Merge conflict')
})

test('the main button commits only staged files when staged > 0, else commits all, and follows push/pull/sync/publish when clean', () => {
  const repo = { upstream: 'origin/main', ahead: 0, behind: 0, branch: 'main', hasHead: true, detached: false }
  expect(primaryAction({ ...repo, ahead: 1 }, 3)).toMatchObject({ kind: 'commit', label: 'Commit all', stagedOnly: false })
  expect(primaryAction({ ...repo, counts: { staged: 0 } }, 3)).toMatchObject({ kind: 'commit', label: 'Commit all', stagedOnly: false })
  expect(primaryAction({ ...repo, counts: { staged: 2 } }, 5)).toMatchObject({
    kind: 'commit',
    label: 'Commit staged (2)',
    stagedOnly: true,
  })
  expect(primaryAction({ ...repo, ahead: 1 }, 0)).toMatchObject({ kind: 'push', label: 'Push 1', title: 'Push 1 commit to origin/main' })
  expect(primaryAction({ ...repo, behind: 2 }, 0)).toMatchObject({ kind: 'pull', label: 'Pull 2' })
  expect(primaryAction({ ...repo, ahead: 1, behind: 2 }, 0)).toMatchObject({ kind: 'sync', label: 'Sync ↓2 ↑1' })
  expect(primaryAction({ ...repo, upstream: null }, 0)).toMatchObject({ kind: 'publish', label: 'Publish branch' })
  expect(primaryAction(repo, 0)).toMatchObject({ kind: 'commit', title: 'Nothing to commit' })
  // Nothing sensible to push from a detached head or before the first commit.
  expect(primaryAction({ ...repo, ahead: 3, detached: true, branch: null }, 0).kind).toBe('commit')
  expect(primaryAction({ ...repo, upstream: null, hasHead: false }, 0).kind).toBe('commit')
})

test('Create PR asks the agent to commit, branch off the default branch, push and open a non-draft PR', async () => {
  const { pullRequestPrompt } = await import('../src/desktop/renderer/GitPanel.js')
  const prompt = pullRequestPrompt('main')
  expect(prompt).toContain('current branch `main`')
  expect(prompt).toContain('gh auth status')
  expect(prompt).toContain('default branch, create a new branch')
  expect(prompt).toContain('Never force-push')
  expect(prompt).toContain('gh pr create')
  expect(prompt).toContain('not a draft')
  expect(pullRequestPrompt(null)).not.toContain('(current branch')
})

test('a large review runs as a find-then-verify workflow unless agents are off', () => {
  const prompt = reviewPrompt({ branch: 'vnext', hasHead: true })
  expect(prompt).toContain('git diff HEAD --stat')
  expect(prompt).toContain('Small change (a handful of files): review it yourself')
  expect(prompt).toContain('run the review as a Workflow')
  expect(prompt).toContain('not turned off for this conversation')
  expect(prompt).toContain('Phase "Verify"')
  expect(prompt).toContain('how many survived verification')
})

test('a pull request describes the change in sections and reports the checks it ran', async () => {
  const { pullRequestPrompt } = await import('../src/desktop/renderer/GitPanel.js')
  const prompt = pullRequestPrompt('feature/x')
  for (const section of ['**Summary**', '**Changes**', '**Behaviour changes**', '**How it was tested**', '**Risks and rollback**', '**Follow-ups**']) {
    expect(prompt).toContain(section)
  }
  expect(prompt).toContain("Run the project's own checks")
  expect(prompt).toContain('summarize it with a Workflow')
  expect(prompt).toContain('only claim what you verified')
})

test('scmSelectionKey distinguishes paths, staging groups and commits so FileDiff resets scroll across switches', () => {
  expect(scmSelectionKey(null)).toBe('')
  const unstaged = scmSelectionKey({ path: 'a.py', group: 'unstaged' })
  const staged = scmSelectionKey({ path: 'a.py', group: 'staged' })
  const otherFile = scmSelectionKey({ path: 'b.py', group: 'unstaged' })
  const commitA = scmSelectionKey({ path: 'a.py', group: 'commit', commit: { hash: '1111111', short: '1111111' } })
  const commitB = scmSelectionKey({ path: 'a.py', group: 'commit', commit: { hash: '2222222', short: '2222222' } })
  expect(new Set([unstaged, staged, otherFile, commitA, commitB]).size).toBe(5)
  expect(scmSelectionKey({ path: 'a.py', group: 'unstaged' })).toBe(unstaged)
})

test('FileDiff renders full-width state messages and resets horizontal/vertical scroll on file change', () => {
  const emptyRepo = { staged: [], unstaged: [], counts: { staged: 0, unstaged: 1, untracked: 0, conflicts: 0 } }
  const noopScm = { call: async () => ({ lines: [] }) }
  const html = renderToStaticMarkup(createElement(FileDiff, {
    scm: noopScm,
    selection: { path: 'libs/easydel/modeling_qwen3_next.py', group: 'unstaged' },
    repo: emptyRepo,
  }))
  expect(html).toContain('libs/easydel/modeling_qwen3_next.py')
  expect(html).toContain('Working tree')
  expect(html).toContain('class="change-review__state studio-muted"')

  const gitPanelTsx = readFileSync(join(import.meta.dir, '../src/desktop/renderer/GitPanel.tsx'), 'utf8')
  expect(gitPanelTsx).toContain('el.scrollTop = 0')
  expect(gitPanelTsx).toContain('el.scrollLeft = 0')
})

test('diff viewer stretches highlighted rows across horizontal overflow, keeps gutters sticky, and strips card borders in SCM', () => {
  const rendererDir = join(import.meta.dir, '../src/desktop/renderer')
  const atelier = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')
  const layout = readFileSync(join(rendererDir, 'layout.css'), 'utf8')
  const vibes = readFileSync(join(rendererDir, 'vibes.css'), 'utf8')

  // Every row in `.change-review__source` must size to max(100%, widest row) so
  // add/del/hunk backgrounds never cut off mid-line when scrolled horizontally.
  expect(atelier).toMatch(/\.atelier \.change-review__source\{[^}]*display:grid;grid-template-columns:minmax\(max-content,1fr\);align-content:start/)
  // Both line-number gutters stay sticky on the left with opaque composited fills.
  expect(atelier).toMatch(/\.diff-gutter\{[^}]*position:sticky;left:0/)
  expect(atelier).toMatch(/\.diff-gutter\+\.diff-gutter\{left:var\(--diff-gutter-w\)\}/)
  expect(atelier).toContain('.studio-diff-add>.diff-gutter{background-image:linear-gradient(var(--x-diff-add),var(--x-diff-add))}')
  expect(atelier).toContain('.studio-diff-del>.diff-gutter{background-image:linear-gradient(var(--x-diff-del),var(--x-diff-del))}')
  // Full-bleed SCM diff pane must not inherit generic markdown <pre> borders/radius.
  expect(layout).toContain('.scm__diff>pre.change-review__source{border:0;border-radius:0;')
  expect(vibes).toContain('.scm__diff>pre.change-review__source{--diff-bg:var(--x-screen);background:var(--diff-bg);border:0;border-radius:0}')
  // Inline tool diff cards also size rows to the widest line.
  expect(atelier).toMatch(/\.atelier \.filetool__code pre\{[^}]*display:grid;grid-template-columns:minmax\(max-content,1fr\)/)
  expect(layout).toMatch(/\.atelier \.filetool__code pre\{[^}]*display:grid;grid-template-columns:minmax\(max-content,1fr\)/)
})
