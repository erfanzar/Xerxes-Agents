// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { HubStatusSheet } from '../components/HubStatusSheet.js'
import { ServicesProvider } from '../context.js'
import { liveState, summary, workspace } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '')

describe('redesigned Mobile Hub & capabilities', () => {
  it('offers New task, Refresh, and live task search on Inbox', () => {
    const state = liveState([summary({ id: 's1', title: 'Audit auth', status: 'working' })], [workspace('w1', 'Mac · app')])
    const html = plain(render(state, '/'))
    expect(html).toContain('>New task<')
    expect(html).toContain('>Refresh<')
    expect(html).toContain('aria-label="Search tasks"')
    expect(html).toContain('aria-label="Hub and paired devices"')
  })

  it('offers quick prompt chips when XERXES thread is empty and Stop while working', () => {
    const empty = { ...liveState(), supervisor: { phase: 'ready' as const, messages: [], confirmations: [], busy: false } }
    const emptyHtml = plain(render(empty, '/xerxes'))
    expect(emptyHtml).toContain('What needs me right now?')
    expect(emptyHtml).toContain('Summarize active tasks')
    expect(emptyHtml).not.toContain('>Stop<')

    const working = { ...liveState(), supervisor: { phase: 'ready' as const, messages: [], confirmations: [], busy: false, working: true } }
    const workingHtml = plain(render(working, '/xerxes'))
    expect(workingHtml).toContain('>Stop<')
    expect(workingHtml).toContain('>Working<')
  })

  it('renders HubStatusSheet with workspaces and paired devices with Revoke only for other devices', () => {
    const base = liveState([], [workspace('w1', 'Mac · app', 'online'), workspace('w2', 'GPU Box', 'offline')])
    const state = {
      ...base,
      devices: {
        phase: 'ready' as const,
        canRevokeAll: true,
        list: [
          { id: 'dev_me', userId: 'u1', displayName: 'Erfan', label: 'This iPhone', status: 'active', createdAt: null, lastSeenAt: null, revokedAt: null, current: true },
          { id: 'dev_old', userId: 'u1', displayName: 'Erfan', label: 'Old iPad', status: 'active', createdAt: null, lastSeenAt: null, revokedAt: null, current: false },
        ],
      },
    }
    const html = plain(renderToStaticMarkup(
      <ServicesProvider services={makeServices(state, '/')}>
        <HubStatusSheet onClose={() => {}} />
      </ServicesProvider>,
    ))
    expect(html).toContain('1 of 2 workspaces reachable')
    expect(html).toContain('Mac · app')
    expect(html).toContain('GPU Box')
    expect(html).toContain('This iPhone')
    expect(html).toContain('This device')
    expect(html).toContain('Old iPad')
    expect(html).toContain('>Revoke<')
  })
})
