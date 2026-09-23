// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { reviewPrompt, statusLabel, statusLetter } from '../src/desktop/renderer/GitPanel.js'


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
