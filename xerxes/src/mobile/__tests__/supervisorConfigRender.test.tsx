// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { UNCONFIGURED_PROMPT } from '../../client/hubController.js'
import type { HubState } from '../../client/hubStore.js'
import { ModelSheet } from '../components/ModelChooser.js'
import { ServicesProvider } from '../context.js'
import { OPTIONS, liveState, machine, summary, withSupervisorConfig, workspace } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")

const MACHINES = { phase: 'ready' as const, list: [machine('m1', 'Mac', ['w1']), machine('m2', 'Build server', ['w2'], { health: 'offline' })] }
const CHOICE = { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'opus', effort: 'high' }

function owner(patch: Parameters<typeof withSupervisorConfig>[1] = {}): HubState {
  return { ...withSupervisorConfig(liveState(), { status: 'configured', canEdit: true, config: CHOICE, options: { phase: 'ready', options: { ...OPTIONS, model: 'opus' } }, ...patch }), machines: MACHINES }
}

describe('Xerxes tab model chooser', () => {
  it('shows XERXES · <model> · <effort> in the header', () => {
    expect(plain(render(owner(), '/xerxes'))).toContain('XERXES · opus · high')
    const noEffort = owner({ config: { ...CHOICE, effort: null } })
    expect(plain(render(noEffort, '/xerxes'))).toContain('XERXES · opus · default effort')
  })

  it('shows the chooser inline when unconfigured, with machine, profile, model and effort selects, and no env-var text', () => {
    const html = plain(render(owner({ status: 'unconfigured' as const, config: undefined as never }), '/xerxes'))
    expect(html).toContain("Choose XERXES's model")
    for (const label of ['Machine', 'Provider profile', 'Model', 'Thinking effort']) expect(html).toContain(`<span>${label}</span>`)
    expect(html.match(/<select/g)).toHaveLength(4)
    expect(html).toContain('>Save<')
    expect(html).toContain('Work · anthropic')
    expect(html).toContain('>Opus<')
    expect(html).toContain('Default (low)')
    expect(html).not.toContain('XERXES_HUB_SUPERVISOR')
    expect(html).not.toContain('Set these on the Hub machine')
  })

  it('does not show env-var instructions when the hub does not offer the supervisor', () => {
    const state: HubState = { ...liveState(), supervisor: { phase: 'unsupported', messages: [], confirmations: [], busy: false } }
    const html = plain(render(state, '/xerxes'))
    expect(html).toContain('Supervisor not available on this hub')
    expect(html).not.toContain('XERXES_HUB_SUPERVISOR')
  })

  it('opens the same chooser in a sheet, filled from the options and disabled Save until a model is picked', () => {
    const html = plain(renderToStaticMarkup(<ServicesProvider services={makeServices(owner(), '/xerxes')}><ModelSheet onClose={() => undefined} /></ServicesProvider>))
    expect(html).toContain('role="dialog"')
    expect(html).toContain('XERXES model')
    expect(html).toContain('>Mac<')
    expect(html).toContain('>Sonnet<')
    expect(html).toContain('>Opus<')
    expect(html).toMatch(/<option value="opus" selected/)
    expect(html).toMatch(/<option value="high"/)
    expect(html).not.toMatch(/<button type="submit"[^>]*disabled/)
    const blank = plain(renderToStaticMarkup(
      <ServicesProvider services={makeServices(owner({ status: 'unconfigured', config: undefined as never, options: { phase: 'loading' } }), '/xerxes')}><ModelSheet onClose={() => undefined} /></ServicesProvider>,
    ))
    expect(blank).toMatch(/<button type="submit"[^>]*disabled/)
  })

  it('shows save errors and an options error with a retry', () => {
    const html = plain(render(owner({ status: 'unconfigured' as const, config: undefined as never, saveError: 'The model is not available.', options: { phase: 'error', error: 'daemon offline' } }), '/xerxes'))
    expect(html).toContain('The model is not available.')
    expect(html).toContain('Could not load the choices: daemon offline')
    expect(html).toContain('Try again')
  })

  it('is read-only for non-owners: shows the current choice, no selects, no Save', () => {
    const html = plain(render(owner({ canEdit: false }), '/xerxes'))
    expect(html).toContain('Model (view)')
    expect(html).not.toContain('<select')
    expect(html).not.toContain('>Save<')
    const sheet = plain(renderToStaticMarkup(<ServicesProvider services={makeServices(owner({ canEdit: false }), '/xerxes')}><ModelSheet onClose={() => undefined} /></ServicesProvider>))
    expect(sheet).toContain('Only the Hub owner can change')
    for (const text of ['Mac', 'work', 'opus', 'high']) expect(sheet).toContain(`<dd>${text}</dd>`)
    expect(sheet).not.toContain('<select')
  })

  it('tells a non-owner that XERXES has no model yet', () => {
    const html = plain(render(owner({ status: 'unconfigured' as const, config: undefined as never, canEdit: false }), '/xerxes'))
    expect(html).toContain('The Hub owner chooses it')
    expect(html).not.toContain('<select')
    expect(html).not.toContain('>Save<')
  })

  it('points at the desktop app when no machine can run the model', () => {
    const state: HubState = { ...owner({ status: 'unconfigured' as const, config: undefined as never }), machines: { phase: 'ready', list: [machine('m1', 'Mac', ['w1'], { modelsSupported: false })] } }
    expect(plain(render(state, '/xerxes'))).toContain('link a machine')
  })

  it("turns an 'unconfigured' submit refusal into a prompt, beside the chooser", () => {
    const state = owner({ status: 'unconfigured' as const, config: undefined as never })
    const withError: HubState = { ...state, supervisor: { ...state.supervisor, error: UNCONFIGURED_PROMPT } }
    const html = plain(render(withError, '/xerxes'))
    expect(html).toContain("Choose XERXES's model first")
    expect(html).toContain('<select')
  })
})

describe('Inbox machine grouping', () => {
  const tasks = [summary({ id: 'a', title: 'Mac task', workspaceId: 'w1' }), summary({ id: 'b', title: 'Server task', workspaceId: 'w2' }), summary({ id: 'c', title: 'Loose task', workspaceId: 'w3' })]
  const spaces = [workspace('w1', 'Mac · app'), workspace('w2', 'Build server · api'), workspace('w3', 'Orphan')]

  it('groups by machine when there is more than one, with leftovers last', () => {
    const state: HubState = { ...liveState(tasks, spaces), machines: MACHINES }
    const html = plain(render(state, '/'))
    expect(html).toContain('aria-label="Machine Mac"')
    expect(html).toContain('aria-label="Machine Build server"')
    expect(html).toContain('>Other workspaces<')
    expect(html.indexOf('Machine Mac')).toBeLessThan(html.indexOf('Other workspaces'))
  })

  it('does not group with one machine or with none', () => {
    const one: HubState = { ...liveState(tasks, spaces), machines: { phase: 'ready', list: [machine('m1', 'Mac', ['w1', 'w2', 'w3'])] } }
    expect(plain(render(one, '/'))).not.toContain('aria-label="Machine')
    expect(plain(render(liveState(tasks, spaces), '/'))).not.toContain('aria-label="Machine')
  })
})
