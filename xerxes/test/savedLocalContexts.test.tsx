// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { SavedLocalContexts } from '../src/desktop/main/savedLocal.js'
import { groupByWorkspace, sidebarSections, THIS_MAC } from '../src/desktop/renderer/workspaceGroups.js'
import { T3Sidebar } from '../src/desktop/renderer/T3Sidebar.js'
import { Store, type Snapshot } from '../src/desktop/renderer/store.js'
import { parseThreadInboxState } from '../src/desktop/renderer/threadInbox.js'

describe('SavedLocalContexts and This Mac sidebar navigation from SSH views', () => {
  test('SavedLocalContexts loads real local sessions per folder, includes subdir sessions, and resolves latest/specific clicks', async () => {
    let fetchCalls = 0
    const local = new SavedLocalContexts({
      workspaces: () => [
        '/Users/erfan/Documents/Projects/EasyDeL',
        '/Users/erfan/Documents/Projects/Xerxes-Agents',
        '/Users/erfan/Documents/Projects/evvan',
      ],
      fetchSessions: async () => {
        fetchCalls += 1
        return {
          saved: [
            {
              id: 'xrx-newest',
              cwd: '/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes',
              title: 'Fix sidebar contexts',
              status: 'idle',
            },
            {
              id: 'xrx-older',
              cwd: '/Users/erfan/Documents/Projects/Xerxes-Agents',
              title: 'Upgrade AGY ACP',
              status: 'idle',
            },
            {
              id: 'ed-1',
              cwd: '/Users/erfan/Documents/Projects/EasyDeL',
              title: 'Benchmark v5p',
              status: 'idle',
            },
            {
              id: 'other-1',
              cwd: '/Users/erfan/Documents/Projects/ScratchRepo',
              title: 'Ad-hoc experiment',
              status: 'idle',
            },
          ],
          live: [
            {
              id: 'xrx-newest',
              cwd: '/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes',
              title: 'Fix sidebar contexts',
              status: 'working',
            },
          ],
        }
      },
    })

    await local.refresh({ awaitInitial: true })
    expect(fetchCalls).toBe(1)

    const contexts = local.contexts()
    expect(contexts.map(ctx => ctx.workspace)).toEqual([
      '/Users/erfan/Documents/Projects/EasyDeL',
      '/Users/erfan/Documents/Projects/Xerxes-Agents',
      '/Users/erfan/Documents/Projects/evvan',
      '/Users/erfan/Documents/Projects/ScratchRepo',
    ])

    const xerxesCtx = contexts.find(ctx => ctx.workspace.endsWith('/Xerxes-Agents'))!
    expect(xerxesCtx.id).toBeLessThan(-999_999)
    expect(xerxesCtx.remote).toBe(false)
    expect(xerxesCtx.saved).toBe(true)
    expect(xerxesCtx.sessions.map(s => [s.id, s.status])).toEqual([
      ['xrx-newest', 'working'],
      ['xrx-older', 'idle'],
    ])

    const evvanCtx = contexts.find(ctx => ctx.workspace.endsWith('/evvan'))!
    expect(evvanCtx.sessions).toEqual([])

    // Selection on folder resumes its newest session; explicit session resolves to exact cwd.
    expect(local.selection(xerxesCtx.id)).toEqual({
      directory: '/Users/erfan/Documents/Projects/Xerxes-Agents',
      sessionId: 'xrx-newest',
    })
    expect(local.selection(xerxesCtx.id, 'xrx-newest')).toEqual({
      directory: '/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes',
      sessionId: 'xrx-newest',
    })
    expect(local.selection(evvanCtx.id)).toEqual({
      directory: '/Users/erfan/Documents/Projects/evvan',
    })

    // Sidebar keeps subdirectory sessions inside the saved parent project folder without splitting.
    const sections = sidebarSections([], contexts, 'prism-v5p-1024_1')
    const xerxesSections = sections.filter(sec => sec.host === THIS_MAC && sec.folder === 'Xerxes-Agents')
    expect(xerxesSections).toHaveLength(1)
    expect(xerxesSections[0]!.rows.map(r => r.id)).toEqual(['xrx-newest', 'xrx-older'])
    expect(sections.some(sec => sec.host === THIS_MAC && sec.folder === 'xerxes')).toBe(false)

    // Covered folders are skipped when an open local view already owns them.
    const remaining = local.contexts({
      coveredFolders: new Set(['/Users/erfan/Documents/Projects/Xerxes-Agents']),
      includeSessions: false,
    })
    expect(remaining.some(ctx => ctx.workspace.endsWith('/Xerxes-Agents'))).toBe(false)
    expect(remaining.every(ctx => ctx.sessions.length === 0)).toBe(true)

    local.dispose()
  })

  test('SavedLocalContexts degrades cleanly on fetch failure or timeout without throwing', async () => {
    const failing = new SavedLocalContexts({
      workspaces: () => ['/Users/erfan/Documents/Projects/Xerxes-Agents'],
      fetchSessions: async () => {
        throw new Error('local daemon offline')
      },
      firstLoadTimeoutMs: 25,
    })
    await failing.refresh({ awaitInitial: true })
    const contexts = failing.contexts()
    expect(contexts).toHaveLength(1)
    expect(contexts[0]!.sessions).toEqual([])
    failing.dispose()
  })

  test('T3Sidebar renders This Mac session counts & rows on an SSH view and wires empty folder click to activateContext', () => {
    const base = new Store().getSnapshot()
    const liveRow: Snapshot['live'][number] = {
      id: 'remote-1',
      title: 'EasyDeL Workspace Greeting',
      cwd: '/home/erfan/easydel-src',
      status: 'idle',
      turns: 2,
      current: true,
      age: 'now',
    }
    const contexts: Snapshot['contexts'] = [
      {
        id: -1_000_000,
        label: 'This Mac · Xerxes-Agents',
        workspace: '/Users/erfan/Documents/Projects/Xerxes-Agents',
        remote: false,
        saved: true,
        sessions: [
          {
            id: 'local-xrx-1',
            cwd: '/Users/erfan/Documents/Projects/Xerxes-Agents',
            title: 'AGY ACP compatibility',
            status: 'idle',
          },
        ],
      },
      {
        id: -1_000_001,
        label: 'This Mac · evvan',
        workspace: '/Users/erfan/Documents/Projects/evvan',
        remote: false,
        saved: true,
        sessions: [],
      },
    ]
    const snap: Snapshot = {
      ...base,
      connection: 'online',
      sessionKey: 'remote-1',
      workspace: '/home/erfan/easydel-src',
      remoteHost: 'prism-v5p-1024_1',
      live: [liveRow],
      workspaceDirectories: [],
      contexts,
    }
    const sections = sidebarSections(groupByWorkspace(snap.live, snap.workspaceDirectories), snap.contexts, snap.remoteHost)

    const html = renderToStaticMarkup(
      createElement(T3Sidebar, {
        snap,
        page: null,
        sections,
        allRows: [],
        filter: '',
        setFilter: () => {},
        inbox: parseThreadInboxState({}),
        undo: null,
        onUndo: () => {},
        onDismissUndo: () => {},
        onSetViewMode: () => {},
        onToggleProjectScope: () => {},
        onClearProjectScope: () => {},
        onTogglePin: () => {},
        onSettle: () => {},
        onSnooze: () => {},
        onToggleShelf: () => {},
        canGoBack: false,
        canGoForward: false,
        onGoBack: () => {},
        onGoForward: () => {},
        onOpen: () => {},
        sidebarToggle: null,
        runtimeStatus: null,
      }),
    )

    // Xerxes-Agents shows its real session count and session title under This Mac
    expect(html).toContain('data-project-folder="/Users/erfan/Documents/Projects/Xerxes-Agents"')
    expect(html).toContain('AGY ACP compatibility')
    expect(html).toContain('<span class="t3-folder__count">1</span>')

    // Empty saved local context advertises direct Open action (no dead collapse toggle)
    expect(html).toContain('Open This Mac · evvan (/Users/erfan/Documents/Projects/evvan)')
    const sidebarSource = readFileSync(
      join(import.meta.dir, '..', 'src', 'desktop', 'renderer', 'T3Sidebar.tsx'),
      'utf8',
    )
    expect(sidebarSource).toContain("if (section.kind === 'context' && folderRows.length === 0)")
    expect(sidebarSource).toContain('void store.activateContext(section.context.id)')
  })
})
