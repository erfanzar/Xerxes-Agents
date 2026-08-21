// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// The transcript used to render as one undifferentiated gray column: user,
// assistant, and tool rows all resolved to the same '#aeb4bb', because
// `themeForMode` smeared the interaction-mode accent over `accent`, `primary`,
// and `label`. These tests assert the painted pixels, not the token table, so
// a future regression in either the theme or the renderer is caught.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { resetUiState } from '../app/uiStore.js'
import { VOICE } from '../domain/roles.js'
import { buildToolTrailLine } from '../lib/text.js'
import { MessageLine } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import type { Msg } from '../types.js'

// `code` is the default mode and the one that used to flatten everything.
const theme = themeForMode(DEFAULT_THEME, 'code')

interface CapturedSpan {
  bg: string
  fg: string
  text: string
}

/** RGBA channel buffer as captureSpans reports it: four 0-255 bytes. */
interface ChannelBuffer {
  buffer: Record<string, number>
}

interface RawSpan {
  bg?: ChannelBuffer
  fg?: ChannelBuffer
  text: string
}

const hex = (channel: ChannelBuffer | undefined): string => {
  const b = channel?.buffer

  if (!b) {
    return ''
  }

  const byte = (i: number) =>
    (b[String(i)] ?? 0).toString(16).padStart(2, '0')

  return `#${byte(0)}${byte(1)}${byte(2)}`
}

const spansOf = async (msg: Msg): Promise<CapturedSpan[]> => {
  const setup = await testRender(
    <box flexDirection="column" height="100%" width="100%">
      <MessageLine msg={msg} t={theme} />
    </box>,
    { height: 12, width: 70 }
  )

  await setup.flush()

  const captured = (
    setup as unknown as { captureSpans: () => { lines: { spans: RawSpan[] }[] } }
  ).captureSpans()

  const flat: CapturedSpan[] = []

  for (const line of captured.lines) {
    for (const span of line.spans) {
      flat.push({ bg: hex(span.bg), fg: hex(span.fg), text: span.text })
    }
  }

  act(() => setup.renderer.destroy())

  return flat
}

const findText = (spans: CapturedSpan[], needle: string) => spans.find(s => s.text.includes(needle))

describe('transcript voices', () => {
  afterEach(() => {
    resetUiState()
  })

  it('paints the user turn with the Derafsh gold bar and warm body text', async () => {
    const spans = await spansOf({ role: 'user', text: 'refactor the gateway' })

    // Absolute values, not `VOICE.user(theme).bar` — comparing the render to
    // the same table the renderer reads would pass even if both went gray
    // together, which is precisely the regression being guarded.
    // The bar is a filled Box, so it lands as a background on a blank span.
    expect(spans.some(s => s.bg === '#d8ae58')).toBe(true)
    expect(findText(spans, 'refactor the gateway')?.fg).toBe('#f0e4cd')
    expect(spans.some(s => s.bg === VOICE.user(theme).bar)).toBe(true)
  })

  it('paints the tool glyph in lapis and leaves its body muted', async () => {
    const spans = await spansOf({
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [buildToolTrailLine('read_file', 'src/one.ts', false, '', 0.1)]
    })

    const glyph = findText(spans, VOICE.tool(theme).glyph)

    expect(glyph?.fg).toBe('#7ea9e0')
    expect(findText(spans, 'one.ts')?.fg).toBe(theme.color.muted)
    expect(glyph?.fg).not.toBe(theme.color.accent)
  })

  it('paints the system voice violet, glyph included', async () => {
    const spans = await spansOf({ role: 'system', text: 'context compacted' })

    expect(findText(spans, 'context compacted')?.fg).toBe('#a98ad4')
    expect(findText(spans, VOICE.system(theme).glyph)?.fg).toBe('#a98ad4')
  })

  it('leaves the assistant unmarked — no bar, no glyph, neutral body', async () => {
    const assistant = VOICE.assistant(theme)

    expect(assistant.bar).toBe('')
    expect(assistant.glyph).toBe('')
    expect(assistant.body).toBe(theme.color.text)
  })

  it('keeps every voice visually distinct in code mode', async () => {
    const bodies = [
      VOICE.user(theme).body,
      VOICE.assistant(theme).body,
      VOICE.tool(theme).body,
      VOICE.system(theme).body
    ]

    expect(new Set(bodies).size).toBe(bodies.length)
  })
})
