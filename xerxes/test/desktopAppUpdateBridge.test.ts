// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { Store } from '../src/desktop/renderer/store.js'

afterEach(() => { delete (globalThis as { window?: unknown }).window })

// The app starts the store without a bridge, so it runs through the store's
// own adapter over window.xerxes. That adapter once lacked the update methods:
// Check for Updates… ran in the main process and its answer never arrived.
test('the update state and actions reach the window through the default bridge', async () => {
  let push: ((state: unknown) => void) | undefined
  const actions: string[] = []
  ;(globalThis as { window?: unknown }).window = {
    xerxes: {
      onEvent: () => () => {},
      call: () => new Promise(() => {}),
      getWorkspace: () => Promise.resolve(null),
      appUpdate: (action: string) => { actions.push(action); return Promise.resolve(action === 'check' ? { phase: 'current', version: '0.6.21' } : { phase: 'idle' }) },
      onAppUpdate: (handler: (state: unknown) => void) => { push = handler; return () => { push = undefined } },
    },
  }
  const store = new Store()
  store.start()
  await Bun.sleep(0)
  expect(actions).toContain('state')
  expect(push).toBeFunction()

  // The menu item's answer arrives as a broadcast from the main process.
  push!({ phase: 'current', version: '0.6.21' })
  expect(store.getSnapshot().appUpdate).toEqual({ phase: 'current', version: '0.6.21' })
  push!({ phase: 'available', installable: true, release: { version: '0.6.22', notes: 'Fixes', pageUrl: 'https://github.com/erfanzar/Xerxes-Agents/releases/tag/v0.6.22' } })
  expect(store.getSnapshot().appUpdate).toMatchObject({ phase: 'available', version: '0.6.22', installable: true })

  // And the prompt's buttons reach the main process.
  await store.appUpdateAction('dismiss')
  expect(actions).toContain('dismiss')
  await store.appUpdateAction('check')
  expect(store.getSnapshot().appUpdate).toEqual({ phase: 'current', version: '0.6.21' })
})

test('a desktop build without the update bridge reports it instead of doing nothing', async () => {
  ;(globalThis as { window?: unknown }).window = {
    xerxes: { onEvent: () => () => {}, call: () => new Promise(() => {}), getWorkspace: () => Promise.resolve(null) },
  }
  const store = new Store()
  store.start()
  await store.appUpdateAction('check')
  expect(store.getSnapshot().appUpdate).toMatchObject({ phase: 'error', message: expect.stringContaining('Update the desktop app') })
})
