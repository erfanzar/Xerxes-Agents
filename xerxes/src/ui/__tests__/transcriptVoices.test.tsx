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

  it('marks the user turn with the prompt glyph on a filled band', async () => {
    const spans = await spansOf({ role: 'user', text: 'refactor the gateway' })

    // Absolute values, not `VOICE.user(theme).bar` — comparing the render to
    // the same table the renderer reads would pass even if both went gray
    // together, which is precisely the regression being guarded.
    // The band is marked by the prompt glyph in the brand blue, not by a
    // filled left bar — `❯ …` is what you typed, and two markers on one row
    // is what the canvas removed. (The band's ground is a Box background, not
    // a span attribute, so it is not observable here; the glyph is.)
    expect(findText(spans, '❯')?.fg).toBe(VOICE.user(theme).bar)
    expect(findText(spans, 'refactor the gateway')?.fg).toBe('#d7dce3')
  })

  it('opens a quiet read-only call with a faint ⏺, lapis name, green tick', async () => {
    const spans = await spansOf({
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [buildToolTrailLine('read_file', 'src/one.ts', false, '', 0.1)]
    })

    // Outcome glyph FIRST (anatomy element ④): dim for read-only calls…
    expect(findText(spans, '⏺')?.fg).toBe(theme.color.muted)
    // …the VERB sits on the ramp's secondary step…
    expect(findText(spans, 'Read File')?.fg).toBe('#8b949e')
    // …the verdict rides right after the summary, ok-green on success…
    expect(findText(spans, '✓')?.fg).toBe('#57ca85')
    // …and the TARGET is the one thing on the row you are actually reading,
    // so it takes `title`, a step above the verb rather than below it.
    expect(findText(spans, 'one.ts')?.fg).toBe(theme.ds.title)
  })

  it('tints a successful non-quiet call green from glyph through mark', async () => {
    const spans = await spansOf({
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [buildToolTrailLine('bash', 'bun run check', false, '', 22.6)]
    })

    expect(findText(spans, '⏺')?.fg).toBe('#57ca85')
    expect(findText(spans, 'Bash')?.fg).toBe('#8b949e')
    expect(findText(spans, '✓')?.fg).toBe('#57ca85')
  })

  it('paints failures red from glyph through mark', async () => {
    const spans = await spansOf({
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [buildToolTrailLine('bash', 'bun run test', true, '1 fail', 41)]
    })

    expect(findText(spans, '⏺')?.fg).toBe('#f47067')
    expect(findText(spans, '✗')?.fg).toBe('#f47067')
    // The name still reads as a name — only the outcome carries red.
    expect(findText(spans, 'Bash')?.fg).toBe('#8b949e')
  })

  it('paints the system voice violet, glyph included', async () => {
    const spans = await spansOf({ role: 'system', text: 'context compacted' })

    expect(findText(spans, 'context compacted')?.fg).toBe('#b39cf0')
    expect(findText(spans, VOICE.system(theme).glyph)?.fg).toBe('#b39cf0')
  })

  it('opens the assistant turn with a small accent ✦ and neutral prose', async () => {
    const assistant = VOICE.assistant(theme)

    // No bar — the rail owns the left edge; the ✦ is the only marker.
    expect(assistant.bar).toBe('')
    expect(assistant.glyph).toBe('✦')
    expect(assistant.glyphColor).toBe(theme.color.accent)
    // The ramp's `prose` step, one below the `title` the user's own words
    // get — see the voice table for why the two must not be the same.
    expect(assistant.body).toBe(theme.ds.prose)

    const spans = await spansOf({ role: 'assistant', text: 'Release gate — three commands in order.' })

    expect(findText(spans, '✦')?.fg).toBe('#b1b8c1')
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
