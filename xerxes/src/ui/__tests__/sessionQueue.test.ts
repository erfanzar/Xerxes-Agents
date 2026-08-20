// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { useQueue } from '../hooks/useQueue.js'

describe('live-session composer queues', () => {
  it('keeps queued follow-ups with their owning session and restores them on return', async () => {
    let queue: ReturnType<typeof useQueue> | undefined

    const Probe = () => {
      queue = useQueue()
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })

    try {
      await setup.flush()
      if (!queue) throw new Error('queue hook did not mount')

      // Input authored before gateway.ready belongs to the first session.
      act(() => queue!.enqueue('startup follow-up'))
      act(() => queue!.activateSessionQueue('session-a'))
      act(() => queue!.enqueue('only for A'))
      await setup.flush()
      expect(queue.queuedDisplay).toEqual(['startup follow-up', 'only for A'])

      act(() => queue!.activateSessionQueue('session-b'))
      await setup.flush()
      expect(queue.queuedDisplay).toEqual([])

      act(() => queue!.enqueue('only for B'))
      act(() => queue!.activateSessionQueue('session-a'))
      await setup.flush()
      expect(queue.queuedDisplay).toEqual(['startup follow-up', 'only for A'])

      act(() => queue!.activateSessionQueue('session-b'))
      await setup.flush()
      expect(queue.queuedDisplay).toEqual(['only for B'])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
