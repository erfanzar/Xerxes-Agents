// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { buildToolTrailLine } from '../lib/text.js'
import { MessageLine } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'

const theme = themeForMode(DEFAULT_THEME, 'code')

describe('OpenTUI message lifecycle', () => {
  afterEach(resetUiState)

  it('keeps chronological thinking/tool phases while native markdown updates in place', async () => {
    let finishMarkdown = () => {}

    function Harness() {
      const [markdown, setMarkdown] = useState('# Partial heading')

      finishMarkdown = () => setMarkdown('# Final heading\n\n- first item\n- second item')

      return (
        <box flexDirection="column">
          <MessageLine
            msg={{
              kind: 'trail',
              role: 'system',
              text: '',
              thinking: 'First reasoning phase',
              tools: [buildToolTrailLine('read_file', 'src/one.ts', false, '', 0.1)]
            }}
            t={theme}
          />
          <MessageLine
            msg={{
              kind: 'trail',
              role: 'system',
              text: '',
              thinking: 'Second reasoning phase',
              tools: [buildToolTrailLine('read_file', 'src/two.ts', false, '', 0.2)]
            }}
            t={theme}
          />
          <MessageLine msg={{ role: 'assistant', text: markdown }} t={theme} />
        </box>
      )
    }

    const setup = await testRender(<Harness />, { height: 18, width: 80 })
    const waitForText = async (text: string) => {
      for (let pass = 0; pass < 30; pass++) {
        await Bun.sleep(10)
        await setup.flush()

        const frame = setup.captureCharFrame()

        if (frame.includes(text)) {
          return frame
        }
      }

      throw new Error(`timed out waiting for ${text}`)
    }

    try {
      await setup.flush()
      const partial = await waitForText('Partial heading')

      expect(partial).toContain('First reasoning phase')
      expect(partial).toContain('Read File src/one.ts')
      expect(partial).toContain('Second reasoning phase')
      expect(partial).toContain('Read File src/two.ts')
      expect(partial).toContain('Partial heading')

      act(finishMarkdown)
      const finished = await waitForText('Final heading')

      expect(finished).toContain('Final heading')
      expect(finished).toContain('first item')
      expect(finished).toContain('second item')
      expect(finished).not.toContain('Partial heading')
      expect(finished).toContain('First reasoning phase')
      expect(finished).toContain('Second reasoning phase')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('applies /details hidden to settled reasoning and tools while subagents stay in the agent panel', async () => {
    patchUiState({ detailsMode: 'hidden', detailsModeCommandOverride: true })

    const msg = {
      kind: 'trail' as const,
      role: 'system' as const,
      text: '',
      thinking: 'private reasoning detail',
      tools: [buildToolTrailLine('read_file', 'src/hidden.ts')],
      subagents: [
        {
          depth: 1,
          goal: 'hidden agent detail',
          id: 'child-1',
          index: 0,
          notes: [],
          parentId: null,
          status: 'completed' as const,
          taskCount: 1,
          thinking: [],
          toolCount: 0,
          tools: []
        }
      ]
    }
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={msg} t={theme} />
      </box>,
      { height: 10, width: 80 }
    )

    try {
      await setup.flush()
      const hidden = setup.captureCharFrame()

      expect(hidden).not.toContain('private reasoning detail')
      expect(hidden).not.toContain('src/hidden.ts')
      expect(hidden).not.toContain('hidden agent detail')

      act(() => patchUiState({ detailsMode: 'expanded', detailsModeCommandOverride: true }))
      await setup.flush()
      const expanded = setup.captureCharFrame()

      expect(expanded).toContain('private reasoning detail')
      expect(expanded).toContain('Read File src/hidden.ts')
      expect(expanded).not.toContain('hidden agent detail')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
