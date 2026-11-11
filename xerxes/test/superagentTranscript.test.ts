// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { resolveSessionKey } from '../src/hub/commandRouter.js'
import { TURN_METADATA_KEY } from '../src/hub/supervisor/tools.js'
import { disposeAll } from './hubHarness.js'
import { makeWorld, toolCall } from './hubSupervisorHarness.js'

afterEach(disposeAll)

test('tail session resolution requests summaries, never all active histories', async () => {
  const world = makeWorld()
  expect(await resolveSessionKey(world.backend, 'w1', 's1')).toBe('k1')
  expect(world.backendCalls[0]?.params).toEqual({ history_limit: 0 })
})

test('task_tail carries an opaque history cursor in both directions', async () => {
  const world = makeWorld()
  world.backendBehavior.run = method => method === 'session.active_list'
    ? { state: 'accepted', result: { sessions: [{ id: 's1', key: 'k1' }] } }
    : { state: 'accepted', result: { history: { actions: [{ messages: [{ role: 'assistant', content: 'older page' }] }], has_more: true, before: 'next-page' } } }
  const turn = { threadId: 'shared', requester: world.user('alice'), reader: world.reader('alice'), denial: undefined }
  const result: unknown = JSON.parse(await world.registry.execute(toolCall('task_tail', {
    workspace_id: 'w1', task_id: 's1', before: 'older-page', limit: 2,
  }), { metadata: { [TURN_METADATA_KEY]: turn } }))
  expect(result).toMatchObject({ steps: ['assistant: older page'], has_more: true, before: 'next-page' })
  expect(world.backendCalls.at(-1)?.params).toMatchObject({ session_key: 'k1', history_limit: 2, before: 'older-page' })
})
