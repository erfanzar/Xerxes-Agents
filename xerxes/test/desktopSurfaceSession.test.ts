// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { SurfaceSession } from '../src/desktop/main/surfaceSession.js'

test('a reload the renderer starts itself resumes the surface\'s own session every time', () => {
  const surface = new SurfaceSession('goal-session')
  expect(surface.current).toBe('goal-session')
  expect(surface.observe('initialize', { session: { id: 'goal-session' }, cwd: '/repo' })).toBe(false)
  // The error screen's "Reload window" asks again; a one-shot answer of null
  // let the workspace-wide preference (another view's session) win.
  expect(surface.current).toBe('goal-session')
  expect(surface.current).toBe('goal-session')
  expect(surface.cwd).toBe('/repo')
})

test('in-place navigation and rebinding move the surface; malformed results do not', () => {
  const surface = new SurfaceSession(null)
  expect(surface.current).toBeNull()
  expect(surface.observe('initialize', { session: { id: 'fresh', cwd: '/repo' } })).toBe(true)
  expect(surface.observe('session.open', { session: { id: 'other' } })).toBe(true)
  expect(surface.current).toBe('other')
  for (const result of [{}, { session: { id: '../escape' } }, { session: { id: 7 } }]) expect(surface.observe('initialize', result)).toBe(false)
  expect(surface.observe('session.list', { session: { id: 'listed' } })).toBe(false)
  expect(surface.current).toBe('other')
  expect(surface.cwd).toBe('/repo')
  surface.bind(null)
  expect(surface.current).toBeNull()
  expect(surface.cwd).toBeNull()
  surface.bind('picked')
  expect(surface.current).toBe('picked')
})

test('the main process answers every page load of a surface with that surface\'s session', async () => {
  const main = await readFile(join(import.meta.dir, '..', 'src', 'desktop', 'main.ts'), 'utf8')
  expect(main).toContain("handle('desktop:resume', () => session.current)")
  expect(main).not.toContain('resumeSession = null')
})
