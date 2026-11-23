// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { requestWorkspaceSshForm } from '../src/desktop/renderer/DesktopPanels.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')

describe('T3Sidebar Add project (Local vs SSH) & Gemini Workspace SSH trust', () => {
  test('offers Local folder and Connect via SSH choices from the Add project toolbar button', () => {
    const sidebarSource = readFileSync(join(rendererDir, 'T3Sidebar.tsx'), 'utf8')
    const vibesCss = readFileSync(join(rendererDir, 'vibes.css'), 'utf8')

    expect(sidebarSource).toContain('aria-label="Add project"')
    expect(sidebarSource).toContain('aria-haspopup="menu"')
    expect(sidebarSource).toContain('aria-label="Add project source"')
    expect(sidebarSource).toContain('Local folder')
    expect(sidebarSource).toContain('Connect via SSH')
    expect(sidebarSource).toContain('requestWorkspaceSshForm()')
    expect(sidebarSource).toContain("onOpen('workspace')")
    expect(vibesCss).toContain('.t3-add-pop')
    expect(vibesCss).toContain('.t3-add-pop__option')
  })

  test('requestWorkspaceSshForm dispatches the workspace SSH add event when window is present', () => {
    const events: string[] = []
    const originalWindow = (globalThis as { window?: unknown }).window
    try {
      ;(globalThis as { window?: unknown }).window = {
        dispatchEvent(event: Event) {
          events.push(event.type)
          return true
        },
      }
      requestWorkspaceSshForm()
      expect(events).toEqual(['xerxes:workspace-add-ssh'])
    } finally {
      if (originalWindow === undefined) delete (globalThis as { window?: unknown }).window
      else (globalThis as { window?: unknown }).window = originalWindow
    }
  })

  test('desktop remote browse verifies first-time SSH host keys before batch browsing', () => {
    const mainSource = readFileSync(join(import.meta.dir, '../src/desktop/main.ts'), 'utf8')
    const browseIndex = mainSource.indexOf("else if (action === 'browse')")
    expect(browseIndex).toBeGreaterThan(-1)
    const browseSlice = mainSource.slice(browseIndex, browseIndex + 900)
    expect(browseSlice).toContain('verifyRemoteHost(')
    expect(browseSlice).toContain('confirmSshHostKey')
    expect(browseSlice).toContain('SSH host key was not trusted')
  })

  test('WorkspacePanel renders Gemini-styled SSH and Local tabs with inline errors and Save & connect', () => {
    const panelsSource = readFileSync(join(rendererDir, 'DesktopPanels.tsx'), 'utf8')
    const atelierCss = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')
    expect(panelsSource).toContain('ws-gemini')
    expect(panelsSource).toContain('SSH &amp; Remote')
    expect(panelsSource).toContain('Save &amp; connect')
    expect(panelsSource).toContain('Verifying SSH host key')
    expect(atelierCss).toContain('.workspace-sheet.studio-sheet')
    expect(atelierCss).toContain('--g-primary:#a8c7fa')
  })
})
