// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { loadWindowLayout, parseWindowLayout, saveWindowLayout, visibleWindowBounds, type SavedWindow } from '../src/desktop/main/windowState.js'

const first: SavedWindow = { workspace: '/one', sessionId: 'session-one', remote: null, bounds: { x: 20, y: 40, width: 900, height: 700 }, maximized: false, fullscreen: false }
test('window layout round-trips separate sessions even in the same workspace', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-window-state-')), file = join(directory, 'windows.json')
  try {
    expect(loadWindowLayout(file)).toBeNull()
    const windows = [first, { ...first, sessionId: 'session-two', maximized: true, bounds: { ...first.bounds, x: 950 } }]
    saveWindowLayout(file, windows)
    expect(loadWindowLayout(file)).toEqual(windows)
    if (process.platform !== 'win32') expect(statSync(file).mode & 0o777).toBe(0o600)
    const previous = readFileSync(file, 'utf8')
    expect(() => saveWindowLayout(file, [{ ...first, workspace: 'relative' }])).toThrow('workspace')
    expect(readFileSync(file, 'utf8')).toBe(previous)
    saveWindowLayout(file, [])
    expect(loadWindowLayout(file)).toEqual([])
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
test('remote paths cannot become local workspaces on restoration', () => {
  const remote = { alias: 'work', target: 'user@host', workspacePath: '/remote/project' }
  expect(parseWindowLayout({ version: 1, windows: [{ ...first, remote }] })[0]).toMatchObject({ remote, workspace: null })
  expect(() => parseWindowLayout({ version: 1, windows: [{ ...first, remote: { ...remote, target: 'host; command' } }] })).toThrow('SSH')
  for (const patch of [{ sessionId: '../session' }, { bounds: { ...first.bounds, x: Infinity } }, { maximized: 'yes' }]) {
    expect(() => parseWindowLayout({ version: 1, windows: [{ ...first, ...patch }] })).toThrow()
  }
})
test('restored frames fit available monitors after resolution or dock changes', () => {
  const areas = [{ x: 0, y: 24, width: 1440, height: 856 }, { x: -1920, y: 24, width: 1920, height: 1056 }]
  expect(visibleWindowBounds({ x: -1800, y: 40, width: 1000, height: 700 }, areas)).toEqual({ x: -1800, y: 40, width: 1000, height: 700 })
  expect(visibleWindowBounds({ x: 5000, y: -2000, width: 2400, height: 1500 }, areas.slice(0, 1))).toEqual(areas[0]!)
  expect(visibleWindowBounds({ x: 1300, y: 700, width: 900, height: 700 }, areas.slice(0, 1))).toEqual({ x: 540, y: 180, width: 900, height: 700 })
})

test('workspace views retain their shared window and active selection on restoration', () => {
  const views=[{...first,windowGroup:'one',active:false},{...first,workspace:'/two',sessionId:'two',windowGroup:'one',active:true}]
  expect(parseWindowLayout({version:1,windows:views})).toEqual(views)
  expect(()=>parseWindowLayout({version:1,windows:[{...first,windowGroup:'invalid group'}]})).toThrow('group')
})
