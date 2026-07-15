// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { testRender } from '@opentui/react/test-utils'
import { act, useCallback, useRef, useState } from 'react'
import { describe, expect, it } from 'vitest'

import { resolveDeferredVirtualRange, useVirtualHistory } from '../hooks/useVirtualHistory.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'

const ITEMS = Array.from({ length: 100 }, (_, index) => ({ key: `row:${index}` }))

describe('virtual history lifecycle', () => {
  it('subscribes when the transcript scrollbox mounts after the welcome screen', async () => {
    let revealTranscript = () => {}
    let moveToTop = () => {}

    function Harness() {
      const [visible, setVisible] = useState(false)
      const scrollRef = useRef<ScrollBoxHandle | null>(null)
      const virtual = useVirtualHistory(scrollRef, visible ? ITEMS : [], 40, {
        coldStartCount: 25,
        estimate: 1,
        maxMounted: 20,
        overscan: 2
      })
      const attachScrollbox = useCallback(
        (_scrollbox: ScrollBoxRenderable | null) => {
          if (!_scrollbox) {
            virtual.setScrollHandle(null)

            return
          }

          const listeners = new Set<() => void>()
          let lastManualScrollAt = Date.now()
          let top = 70
          const notify = () => listeners.forEach(listener => listener())
          const scrollTo = (next: number) => {
            top = Math.max(0, next)
            lastManualScrollAt = Date.now()
            notify()
          }
          const handle: ScrollBoxHandle = {
            getFreshScrollHeight: () => ITEMS.length,
            getLastManualScrollAt: () => lastManualScrollAt,
            getPendingDelta: () => 0,
            getScrollHeight: () => ITEMS.length,
            getScrollTop: () => top,
            getViewportHeight: () => 10,
            getViewportTop: () => top,
            isSticky: () => false,
            scrollBy: delta => scrollTo(top + delta),
            scrollTo,
            scrollToBottom: () => scrollTo(ITEMS.length - 10),
            scrollToElement: () => {},
            setClampBounds: () => {},
            subscribe: listener => {
              listeners.add(listener)

              return () => listeners.delete(listener)
            }
          }

          moveToTop = () => handle.scrollTo(0)
          virtual.setScrollHandle(handle)
        },
        [virtual.setScrollHandle]
      )

      revealTranscript = () => setVisible(true)

      return (
        <box flexDirection="column">
          {visible ? (
            <scrollbox ref={attachScrollbox} style={{ height: 4 }}>
              <text>transcript</text>
            </scrollbox>
          ) : (
            <text>welcome</text>
          )}
          <text>{`range:${virtual.start}-${virtual.end}`}</text>
        </box>
      )
    }

    const setup = await testRender(<Harness />, { height: 8, width: 40 })

    try {
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('welcome')

      act(revealTranscript)
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('range:68-83')

      act(moveToTop)
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('range:0-14')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('falls back to the current non-empty range when deferred bounds only touch', () => {
    expect(
      resolveDeferredVirtualRange({
        deferredEnd: 40,
        deferredStart: 40,
        end: 52,
        itemCount: 100,
        start: 28,
        sticky: false
      })
    ).toEqual([28, 52])

    expect(
      resolveDeferredVirtualRange({
        deferredEnd: 0,
        deferredStart: 0,
        end: 0,
        itemCount: 0,
        start: 0,
        sticky: false
      })
    ).toEqual([0, 0])
  })
})
