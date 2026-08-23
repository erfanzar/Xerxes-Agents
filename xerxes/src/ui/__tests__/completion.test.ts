// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  activeToken,
  applyCompletion,
  COMPLETION_MENU_MAX_ROWS,
  completionColumns,
  completionMenuRows,
  completionMeta,
  cycleIndex,
  rankCompletionItems,
  shouldRequestCompletion
} from '../lib/completion.js'

describe('completionMenuRows', () => {
  it('caps the menu however tall the terminal is', () => {
    expect(completionMenuRows(200, 120)).toBe(COMPLETION_MENU_MAX_ROWS)
  })

  it('never asks for more rows than there are items', () => {
    expect(completionMenuRows(3, 60)).toBe(3)
  })

  it('shrinks with the terminal and bottoms out at zero', () => {
    expect(completionMenuRows(40, 24)).toBe(5)
    // Below the chrome reserve there is no room at all; the menu hides
    // rather than shoving the composer and footer off-screen.
    expect(completionMenuRows(40, 19)).toBe(0)
    expect(completionMenuRows(40, 8)).toBe(0)
  })
})

describe('completionColumns', () => {
  it('sizes the name column to the widest item, not the visible window', () => {
    const { nameWidth } = completionColumns([{ display: '/a' }, { display: '/eternal-army' }], 96)

    expect(nameWidth).toBe('/eternal-army'.length)
  })

  it('clamps the name column so one long name cannot eat the row', () => {
    const { nameWidth } = completionColumns([{ display: `/${'x'.repeat(60)}` }], 200)

    expect(nameWidth).toBe(28)
  })

  it('drops the description column when it would be too clipped to inform', () => {
    expect(completionColumns([{ display: '/eternal-army-command' }], 34).metaWidth).toBe(0)
    expect(completionColumns([{ display: '/model' }], 96).metaWidth).toBeGreaterThan(0)
  })
})

describe('completionMeta', () => {
  it('strips the boilerplate every bundled skill description opens with', () => {
    expect(completionMeta('Use this skill whenever the user wants a chart.', 80)).toBe('the user wants a chart.')
  })

  it('keeps only the first sentence', () => {
    expect(completionMeta('First thing. Second thing. Third thing.', 80)).toBe('First thing.')
  })

  it('collapses whitespace and clamps to the column', () => {
    expect(completionMeta('alpha   beta\n gamma delta', 12)).toBe('alpha beta…')
  })

  it('is empty for missing text or no column', () => {
    expect(completionMeta(undefined, 40)).toBe('')
    expect(completionMeta('anything', 0)).toBe('')
  })
})

describe('shouldRequestCompletion', () => {
  it('triggers on a slash command being typed', () => {
    expect(shouldRequestCompletion('/prov')).toBe(true)
    expect(shouldRequestCompletion('/help now')).toBe(false) // has a space → not the name
  })
  it('triggers on path-like last tokens', () => {
    expect(shouldRequestCompletion('open ./src/ap')).toBe(true)
    expect(shouldRequestCompletion('see ~/notes')).toBe(true)
    expect(shouldRequestCompletion('look @src/x')).toBe(true)
    expect(shouldRequestCompletion('cat /etc/ho')).toBe(true)
  })
  it('does not trigger on plain prose', () => {
    expect(shouldRequestCompletion('just a sentence')).toBe(false)
    expect(shouldRequestCompletion('')).toBe(false)
  })
})

describe('activeToken', () => {
  it('is the whole draft for a slash command', () => {
    expect(activeToken('/prov')).toBe('/prov')
  })
  it('is the last whitespace token otherwise', () => {
    expect(activeToken('open ./src/ap')).toBe('./src/ap')
  })
})

describe('applyCompletion', () => {
  it('replaces the whole draft for a slash command', () => {
    expect(applyCompletion('/prov', '/provider')).toBe('/provider')
  })
  it('replaces only the trailing token for paths', () => {
    expect(applyCompletion('open ./src/ap', './src/app/')).toBe('open ./src/app/')
    expect(applyCompletion('cat @src/m', '@src/main.ts')).toBe('cat @src/main.ts')
  })
  it('appends when the draft ends with whitespace', () => {
    expect(applyCompletion('edit ', 'foo.ts')).toBe('edit foo.ts')
  })
})

describe('cycleIndex', () => {
  it('wraps in both directions', () => {
    expect(cycleIndex(0, 3, 1)).toBe(1)
    expect(cycleIndex(2, 3, 1)).toBe(0)
    expect(cycleIndex(0, 3, -1)).toBe(2)
    expect(cycleIndex(0, 0, 1)).toBe(0)
  })
})

describe('rankCompletionItems', () => {
  const item = (display: string, meta?: string): { display: string; group: string; meta?: string; text: string } => ({
    display,
    ...(meta ? { meta } : {}),
    group: 'skills',
    text: `/${display}`
  })

  // The mockup's ladder: "fuzzy: prefix → substring → skill body".
  it('orders exact, then prefix, then substring, then description matches', () => {
    const ranked = rankCompletionItems(
      [
        item('code-audit', 'runs a review of the tree'), // description-only hit
        item('pre-review', 'nothing to see here'), // substring-of-name hit
        item('review-pr', 'pull request checks'), // name-prefix hit
        item('review', 'review things') // exact hit
      ],
      'review'
    )

    expect(ranked.map(entry => entry.display)).toEqual(['review', 'review-pr', 'pre-review', 'code-audit'])
  })

  it('matches descriptions case-insensitively and never ahead of names', () => {
    const ranked = rankCompletionItems(
      [item('model', 'switch the active model'), item('docx', 'Word DOCUMENTS and templates')],
      'documents'
    )

    // The shouty description still lands in its tier, below nothing here but
    // above the total non-match.
    expect(ranked.map(entry => entry.display)).toEqual(['docx', 'model'])
  })

  it('ranks items matching neither name nor description below description matches', () => {
    const ranked = rankCompletionItems(
      [item('model', 'switch the active model'), item('docx', 'Word documents')],
      'documents'
    )

    expect(ranked.map(entry => entry.display)).toEqual(['docx', 'model'])
  })

  it('keeps the bare-slash view untouched when nothing has been typed', () => {
    const ranked = rankCompletionItems([item('undo'), item('btw', 'before the wind')], '')

    // Empty query: everything sits in one tier, so the group/length/name
    // tie-breaks still produce a scannable order — no description shuffling.
    expect(ranked.map(entry => entry.display)).toEqual(['btw', 'undo'])
  })
})
