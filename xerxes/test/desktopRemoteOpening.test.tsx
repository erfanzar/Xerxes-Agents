// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { RemoteWorkspaceGate, remoteOpeningState } from '../src/desktop/renderer/RemoteWorkspaceGate.js'

test('SSH restore state retains its workspace, failure, and resume identity', () => {
  const machine = { alias: 'worker', target: 'worker', workspacePath: '/srv/project' }
  expect(remoteOpeningState({ ok: true, machine, connecting: false, connected: false,
    resume_session_id: 'saved-123', error: 'Host key verification failed.' })).toEqual({
    machine, connecting: false, connected: false, resume_session_id: 'saved-123', error: 'Host key verification failed.',
  })
  expect(remoteOpeningState({ ok: true, connecting: true }).connecting).toBe(true)
  expect(remoteOpeningState({ ok: true, connected: true }).connected).toBe(true)
  expect(remoteOpeningState({ ok: true, error: 'Remote connection cancelled' }).error).toBe('Remote connection cancelled')
  expect(() => remoteOpeningState({ ok: false, error: 'Permission denied' })).toThrow('Permission denied')
  expect(() => remoteOpeningState(null)).toThrow('Could not read SSH connection status')
})

test('unbound saved SSH view renders progress and cancellation instead of local onboarding', () => {
  const html = renderToStaticMarkup(createElement(RemoteWorkspaceGate, { remote: async () => ({ ok: true }) }))
  expect(html).toContain('Connecting to SSH workspace')
  expect(html).toContain('Cancel connection')
  expect(html).not.toContain('Choose your first folder')
})
