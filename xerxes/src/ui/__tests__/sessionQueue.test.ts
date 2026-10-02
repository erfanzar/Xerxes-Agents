// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { queuedMessage } from '../domain/queuedMessage.js'
import { useQueue } from '../hooks/useQueue.js'

describe('live-session composer queues', () => {
  it('keeps image ownership through queue editing and switching sessions', async () => {
    let queue: ReturnType<typeof useQueue> | undefined
    const Probe = () => { queue = useQueue(); return null }
    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      const images = [{ data: 'aA==', mediaType: 'image/png', name: 'shot.png', path: '/tmp/shot.png', size: 1 }]
      act(() => queue!.activateSessionQueue('a'))
      act(() => queue!.enqueue('see'))
      queue!.queueRef.current[0]!.images = images
      act(() => queue!.replaceQ(0, 'inspect this'))
      act(() => queue!.activateSessionQueue('b'))
      expect(queue!.queueRef.current).toEqual([])
      act(() => queue!.activateSessionQueue('a'))
      await setup.flush()
      expect(queue!.queuedDisplay).toEqual(['inspect this · 1 image(s)'])
      expect(queue!.dequeue()).toMatchObject({ submitText: 'inspect this', images })
    } finally { act(() => setup.renderer.destroy()) }
  })

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

  it('moves a prompt typed during a session switch into the session being opened', async () => {
    let queue: ReturnType<typeof useQueue> | undefined
    const Probe = () => { queue = useQueue(); return null }
    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      act(() => queue!.activateSessionQueue('old'))
      act(() => queue!.enqueue('follow-up for old'))
      act(() => queue!.holdForSwitch(queuedMessage('typed while switching')))
      act(() => queue!.activateSessionQueue('new'))
      await setup.flush()
      expect(queue!.queuedDisplay).toEqual(['typed while switching'])

      act(() => queue!.activateSessionQueue('old'))
      await setup.flush()
      expect(queue!.queuedDisplay).toEqual(['follow-up for old'])
    } finally { act(() => setup.renderer.destroy()) }
  })

  it('keeps a prompt held by a failed switch with the session the user stayed in', async () => {
    let queue: ReturnType<typeof useQueue> | undefined
    const Probe = () => { queue = useQueue(); return null }
    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      act(() => queue!.activateSessionQueue('old'))
      act(() => queue!.holdForSwitch(queuedMessage('typed while a failed switch ran')))
      act(() => queue!.releaseSwitchHold())
      act(() => queue!.activateSessionQueue('later'))
      await setup.flush()
      expect(queue!.queuedDisplay).toEqual([])

      act(() => queue!.activateSessionQueue('old'))
      await setup.flush()
      expect(queue!.queuedDisplay).toEqual(['typed while a failed switch ran'])
    } finally { act(() => setup.renderer.destroy()) }
  })
})
