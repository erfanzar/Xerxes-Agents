// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// The menu used to have no bounded height: a row cap of 10 over
// variable-height rows let word-wrapped skill descriptions grow it past the
// terminal and push the composer and footer off-screen. These tests pin the
// geometry, because a pure-logic test cannot catch a Yoga overflow.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import type { CompletionItem } from '../app/interfaces.js'
import { resetUiState } from '../app/uiStore.js'
import { CompletionMenu } from '../opentui/completionMenu.js'

const SKILL_META =
  'Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files) or Word templates (.dotx files). Triggers include: any mention of Word doc, word document, .docx, or requests to produce professional documents with formatting like tables of contents, headings, page numbers, or letterheads.'

const item = (display: string, meta?: string): CompletionItem => ({
  display,
  ...(meta ? { meta } : {}),
  text: display
})

const many = (n: number): CompletionItem[] =>
  Array.from({ length: n }, (_, i) => item(`/command-${String(i).padStart(2, '0')}`, SKILL_META))

const renderMenu = async ({
  compIdx = 0,
  completions,
  height = 24,
  width = 96
}: {
  compIdx?: number
  completions: CompletionItem[]
  height?: number
  width?: number
}) => {
  const setup = await testRender(
    <box flexDirection="column" height="100%" width="100%">
      <CompletionMenu compIdx={compIdx} completions={completions} width={width} />
    </box>,
    { height, width }
  )

  await setup.flush()

  const frame = setup.captureCharFrame()

  act(() => setup.renderer.destroy())

  return frame
}

/** Rows that actually carry a completion, ignoring the blank canvas. */
const contentLines = (frame: string) => frame.split('\n').filter(line => line.trim().length > 0)

describe('completion menu', () => {
  afterEach(() => {
    resetUiState()
  })

  it('stays inside the terminal with many long-description items', async () => {
    const frame = await renderMenu({ completions: many(40), height: 24, width: 96 })

    // The whole canvas is 24 rows; the menu must occupy a bounded slice of
    // it, not overflow. 8 rows max + 1 overflow footer.
    expect(contentLines(frame).length).toBeLessThanOrEqual(9)
  })

  it('shrinks its row count on a short terminal', async () => {
    const frame = await renderMenu({ completions: many(40), height: 20, width: 96 })

    expect(contentLines(frame).length).toBeLessThanOrEqual(5)
  })

  it('renders a multi-sentence skill description on a single row', async () => {
    const frame = await renderMenu({ completions: [item('/docx', SKILL_META)], height: 24, width: 96 })
    const rows = contentLines(frame)

    expect(rows).toHaveLength(1)
    expect(rows[0]).toContain('/docx')
    // Boilerplate prefix stripped, first sentence only, clamped.
    expect(rows[0]).not.toContain('Use this skill')
    expect(rows[0]).toContain('…')
  })

  it('aligns the description column regardless of command-name length', async () => {
    const frame = await renderMenu({
      completions: [item('/a', 'alpha description'), item('/eternal-army', 'beta description')],
      height: 24,
      width: 96
    })
    const rows = contentLines(frame)

    expect(rows).toHaveLength(2)
    expect(rows[0]!.indexOf('alpha description')).toBe(rows[1]!.indexOf('beta description'))
  })

  it('keeps the column aligned even when a name overflows it', async () => {
    const frame = await renderMenu({
      completions: [
        item('/a', 'alpha description'),
        item(`/${'x'.repeat(60)}`, 'beta description'),
        item('/mid', 'gamma description')
      ],
      height: 24,
      width: 96
    })
    const rows = contentLines(frame)

    expect(rows).toHaveLength(3)
    // The over-long name is clamped into the column rather than shoving its
    // own description sideways and breaking every other row's alignment.
    const at = (row: string, needle: string) => row.indexOf(needle)

    expect(at(rows[1]!, 'beta description')).toBe(at(rows[0]!, 'alpha description'))
    expect(at(rows[2]!, 'gamma description')).toBe(at(rows[0]!, 'alpha description'))
  })

  it('reports how much of the list is off-screen', async () => {
    const frame = await renderMenu({ compIdx: 20, completions: many(40), height: 30, width: 96 })

    expect(frame).toContain('21/40')
    expect(frame).toContain('↑')
    expect(frame).toContain('↓')
  })

  it('drops the description column but keeps names readable when narrow', async () => {
    // The column is dropped once it would be too clipped to inform, not at a
    // fixed terminal width: a short name still affords a description at 40
    // columns, a long one does not.
    const roomy = await renderMenu({
      completions: [item('/model', 'switch the active model')],
      height: 24,
      width: 40
    })

    expect(roomy).toContain('/model')
    expect(roomy).toContain('switch the active model')

    const cramped = await renderMenu({
      completions: [item('/eternal-army-command', 'switch the active model')],
      height: 24,
      width: 34
    })

    expect(cramped).toContain('/eternal-army-command')
    expect(cramped).not.toContain('switch')
  })

  it('renders nothing when there is no room for even one row', async () => {
    const frame = await renderMenu({ completions: many(10), height: 14, width: 96 })

    expect(contentLines(frame)).toHaveLength(0)
  })
})
