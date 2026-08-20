// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import { ReasoningPicker } from '../opentui/reasoningPicker.js'
import { DEFAULT_THEME } from '../theme.js'

describe('OpenTUI reasoning picker', () => {
  it('keeps the highlighted effort visible after moving beyond the first page', async () => {
    const levels = Array.from({ length: 12 }, (_, index) => ({
      description: `description ${index}`,
      effort: `effort-${index}`
    }))
    let resolveLevels!: (value: unknown) => void
    const levelsResponse = new Promise(resolve => (resolveLevels = resolve))
    const services = {
      gw: {} as GatewayClient,
      rpc: vi.fn(() => levelsResponse)
    } as unknown as GatewayServices
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ReasoningPicker onSelect={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await act(async () => {
        resolveLevels({ current: 'effort-0', default: 'effort-0', levels })
        await Bun.sleep(0)
      })
      await setup.flush()

      act(() => {
        for (let index = 0; index < 10; index += 1) {
          setup.renderer.keyInput.processParsedKey({
            ctrl: false,
            eventType: 'press',
            meta: false,
            name: 'down',
            option: false,
            raw: '\u001b[B',
            sequence: '\u001b[B',
            shift: false,
            source: 'raw'
          })
        }
      })
      await setup.flush()

      const frame = setup.captureCharFrame()
      expect(frame).toContain('effort-10')
      expect(frame).toContain('● effort-10')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
