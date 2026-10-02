// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { loadWindowLayout, MAX_RESTORED_VIEWS, MAX_SAVED_WINDOWS, parseWindowLayout, saveWindowLayout, visibleWindowBounds, WindowLayoutRecorder, type SavedWindow } from '../src/desktop/main/windowState.js'

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

test('quitting after the last window closed restores that window instead of an empty layout', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-window-state-')), file = join(directory, 'windows.json')
  try {
    const layout = new WindowLayoutRecorder()
    const base = { ...first, windowGroup: '1', active: false }, view = { ...first, sessionId: 'goal', windowGroup: '1', active: true }
    const other = { ...first, workspace: '/other', windowGroup: '2', active: true }
    layout.set(1, () => base)
    layout.set(2, () => view)
    layout.set(3, () => other)
    // Closing one of two windows forgets it: that close was deliberate.
    layout.closing('2')
    layout.delete(3)
    expect(layout.rows()).toEqual([base, view])
    // The last window's 'closed' teardown empties the live set before quit.
    layout.closing('1')
    layout.delete(1)
    layout.delete(2)
    saveWindowLayout(file, layout.rows())
    expect(loadWindowLayout(file)).toEqual([base, view])
    // A window opened afterwards (a macOS dock click) is the new layout.
    const reopened = { ...first, workspace: '/reopened', windowGroup: '9' }
    layout.set(9, () => reopened)
    expect(layout.rows()).toEqual([reopened])
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('a window that accumulated views keeps saving its active and most recently shown ones', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-window-state-')), file = join(directory, 'windows.json')
  try {
    const layout = new WindowLayoutRecorder()
    let active = 0
    for (let id = 0; id < 150; id++) layout.set(id, () => ({ ...first, sessionId: `s${id}`, windowGroup: 'host', active: id === active }))
    layout.used(3)
    active = 7
    const rows = layout.rows()
    expect(rows).toHaveLength(MAX_RESTORED_VIEWS)
    expect(rows.map(row => row.sessionId)).toEqual(['s3', 's7', ...Array.from({ length: MAX_RESTORED_VIEWS - 2 }, (_, index) => `s${150 - MAX_RESTORED_VIEWS + 2 + index}`)])
    // Past MAX_SAVED_WINDOWS rows the write threw, and the file froze for good.
    saveWindowLayout(file, rows)
    expect(loadWindowLayout(file)).toEqual(rows)
    expect(layout.mostRecent([1, 3, 2])).toBe(3)
    for (let group = 0; group < 15; group++) for (let id = 0; id < MAX_RESTORED_VIEWS; id++)
      layout.set(1000 + group * 100 + id, () => ({ ...first, windowGroup: `g${group}` }))
    expect(layout.rows()).toHaveLength(MAX_SAVED_WINDOWS)
    saveWindowLayout(file, layout.rows())
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('main.ts snapshots a closing window and lets File > Close retire a single view', () => {
  const main = readFileSync(join(import.meta.dir, '..', 'src', 'desktop', 'main.ts'), 'utf8')
  expect(main).toContain("window.once('close', () => windowStates.closing(String(window.id)))")
  expect(main).toContain("click: closeActiveSurface")
  expect(main).not.toContain("{ role: 'close' }")
  expect(main).toContain('window.contentView.removeChildView(view)')
})
