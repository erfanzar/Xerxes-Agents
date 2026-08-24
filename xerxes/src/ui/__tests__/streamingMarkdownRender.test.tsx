// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// StreamingMarkdown (chunked live render) must be visually identical to the
// settled whole-buffer <markdown> render, must stay fence-safe mid-stream,
// and must keep stabilized chunks memoized while only the tail re-parses.
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { describe, expect, it } from 'vitest'

import { VOICE } from '../domain/roles.js'
import { MessageLine, StreamingMarkdown } from '../opentui/messageLine.js'
import { Text } from '../opentui/primitives.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'

const theme = themeForMode(DEFAULT_THEME, 'code')

const FULL =
  '# Title\n\nFirst paragraph here.\n\n- one\n- two\n\n```ts\nconst x = 1\n```\n\nTail paragraph.'

type Setup = Awaited<ReturnType<typeof testRender>>

const settle = async (setup: Setup, marker: string): Promise<string> => {
  // Generous budget: under full-suite parallel load the native markdown
  // highlighter can take well over 300ms to paint the first frame.
  for (let pass = 0; pass < 100; pass++) {
    await Bun.sleep(20)
    await setup.flush()

    const frame = setup.captureCharFrame()

    if (frame.includes(marker)) {
      await setup.waitForVisualIdle()
      return setup.captureCharFrame()
    }
  }

  throw new Error(`timed out waiting for ${marker}`)
}

// Mirrors the settled AssistantMessage wrapper, so the comparison is
// apples-to-apples. The margin is scaffolding, not the thing under test — it
// must match whatever the settled block does (leadGap-less here) or the
// frames differ by a blank row for no real reason. That now includes the
// redesign's turn-opening ✦ column (rail gutter + glyph + blank before the
// prose), mirroring opentui/messageLine.tsx AssistantMessage exactly.
function StreamingBlock({ text }: { text: string }) {
  const assistantVoice = VOICE.assistant(theme)

  return (
    <box flexDirection="column" flexShrink={0}>
      <box flexDirection="row" flexShrink={0}>
        <box flexShrink={0} width={1} />
        <box flexShrink={0} width={1}>
          {assistantVoice.glyph ? (
            <Text color={assistantVoice.glyphColor}>{assistantVoice.glyph}</Text>
          ) : null}
        </box>
        <box flexShrink={0} width={1} />
        <box flexDirection="column" flexGrow={1} minWidth={0}>
          <StreamingMarkdown text={text} t={theme} />
        </box>
      </box>
    </box>
  )
}

describe('StreamingMarkdown', () => {
  it('renders the final buffer identically to a whole-buffer markdown message', async () => {
    const whole = await testRender(
      <box flexDirection="column">
        <MessageLine msg={{ role: 'assistant', text: FULL }} t={theme} />
      </box>,
      { height: 40, width: 60 }
    )
    const chunked = await testRender(
      <box flexDirection="column">
        <StreamingBlock text={FULL} />
      </box>,
      { height: 40, width: 60 }
    )

    try {
      const settledFrame = await settle(whole, 'Tail paragraph')
      const chunkedFrame = await settle(chunked, 'Tail paragraph')

      expect(chunkedFrame).toBe(settledFrame)
    } finally {
      act(() => whole.renderer.destroy())
      act(() => chunked.renderer.destroy())
    }
  })

  it('stays fence-safe and consistent while deltas grow the buffer', async () => {
    let pushDelta = (_text: string) => {}

    function Harness() {
      const [text, setText] = useState('Working on it')
      pushDelta = setText

      return (
        <box flexDirection="column">
          <StreamingBlock text={text} />
        </box>
      )
    }

    const setup = await testRender(<Harness />, { height: 40, width: 60 })

    try {
      await settle(setup, 'Working on it')

      // Grow the buffer in realistic deltas, including an open code fence
      // with a blank line inside it (the chunker must not split there).
      const deltas = [
        'Working on it.\n\nHere comes code:\n\n```ts\nconst x',
        'Working on it.\n\nHere comes code:\n\n```ts\nconst x = 1\n\nconst y = 2',
        'Working on it.\n\nHere comes code:\n\n```ts\nconst x = 1\n\nconst y = 2\n```\n\nDone.'
      ]

      for (const delta of deltas) {
        act(() => pushDelta(delta))
        await setup.flush()
        await Bun.sleep(10)
      }

      const final = await settle(setup, 'Done.')

      expect(final).toContain('Working on it.')
      expect(final).toContain('Here comes code:')
      expect(final).toContain('const x = 1')
      expect(final).toContain('const y = 2')
      // The blank line inside the fence survived as code content, and the
      // fence never rendered as broken inline markdown.
      expect(final).not.toContain('```')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('resets when a new turn reuses the mounted component', async () => {
    let pushDelta = (_text: string) => {}

    function Harness() {
      const [text, setText] = useState('first reply\n\nwith more')
      pushDelta = setText

      return (
        <box flexDirection="column">
          <StreamingBlock text={text} />
        </box>
      )
    }

    const setup = await testRender(<Harness />, { height: 20, width: 60 })

    try {
      await settle(setup, 'with more')

      act(() => pushDelta('brand new turn'))
      const frame = await settle(setup, 'brand new turn')

      expect(frame).not.toContain('first reply')
      expect(frame).not.toContain('with more')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
