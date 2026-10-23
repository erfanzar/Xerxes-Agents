// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { initialHubState, initialSupervisorConfig, type HubState } from '../src/client/hubStore.js'
import { SidebarViews } from '../src/desktop/renderer/App.js'
import { ActionConfirmationCard } from '../src/desktop/renderer/ActionConfirmationCard.js'
import {
  CONNECT_HUB_TITLE, ConnectHub, SUPERVISOR_OLD_HUB, SignInFailed, SupervisorView, modelChipText,
  type SupervisorViewProps,
} from '../src/desktop/renderer/SupervisorPage.js'
import { clearLegacyHubUrl } from '../src/desktop/renderer/hubPrefs.js'

const noop = (): void => {}
const live: HubState = { ...initialHubState(), connection: { state: 'live' } }

const configured: HubState['supervisorConfig'] = {
  ...initialSupervisorConfig(),
  phase: 'ready',
  status: 'configured',
  canEdit: true,
  config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'model-a', effort: 'high' },
}

function view(state: HubState, extra: Partial<SupervisorViewProps> = {}): string {
  return renderToStaticMarkup(createElement(SupervisorView, {
    state, hubLabel: 'hub.example.ts.net', now: Date.now(), onSend: async () => true, onRespond: noop, onReopen: noop,
    onOpenSettings: noop, chooserOpen: false, onChooserChange: noop, onChooserShown: noop, onOptions: noop, onSaveModel: async () => true,
    ...extra,
  }))
}

const ready = (patch: Partial<HubState> = {}): HubState => ({
  ...live,
  supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
  supervisorConfig: configured,
  ...patch,
})

test('the sidebar offers Tasks, Agents and XERXES, with the selected one pressed', () => {
  const html = renderToStaticMarkup(createElement(SidebarViews, { page: 'xerxes', open: noop }))
  expect(html).toContain('>Tasks<')
  expect(html).toContain('>Agents<')
  expect(html).toContain('>XERXES<')
  expect(html).toMatch(/aria-pressed="true" class="is-selected">XERXES</)
})

test('with no Hub saved the tab says Connect a Hub, points at Settings, and has no address form', () => {
  const html = renderToStaticMarkup(createElement(ConnectHub, { onOpenSettings: noop }))
  expect(html).toContain(CONNECT_HUB_TITLE)
  expect(html).toContain('Open Hub &amp; phone settings')
  expect(html).toContain('Nothing is shown until a Hub is connected')
  expect(html).not.toContain('<input')
  expect(html).not.toContain('Conversation')
})

test('outside the desktop app the tab says so instead of offering a setup', () => {
  const html = renderToStaticMarkup(createElement(ConnectHub, { unavailable: true, onOpenSettings: noop }))
  expect(html).toContain('only available in the Xerxes desktop app')
  expect(html).not.toContain('Open Hub')
})

test('a failed sign-in offers Try again, or Pair again when the Hub refused this desktop', () => {
  const retry = renderToStaticMarkup(createElement(SignInFailed, { message: 'The Hub did not answer: timeout', denied: false, onRetry: noop, onSignInAgain: noop, onOpenSettings: noop }))
  expect(retry).toContain('role="alert"')
  expect(retry).toContain('The Hub did not answer: timeout')
  expect(retry).toContain('>Try again<')
  expect(retry).not.toContain('Pair this desktop again')
  const denied = renderToStaticMarkup(createElement(SignInFailed, { message: 'pairing expired', denied: true, onRetry: noop, onSignInAgain: noop, onOpenSettings: noop }))
  expect(denied).toContain('Pair this desktop again')
  expect(denied).not.toContain('>Try again<')
})

test('the old env-var instructions are gone; an old Hub gets a plain update notice', () => {
  const state: HubState = { ...live, supervisor: { phase: 'unsupported', messages: [], confirmations: [], busy: false } }
  const html = view(state)
  expect(html).toContain(SUPERVISOR_OLD_HUB)
  expect(html).not.toContain('XERXES_HUB_SUPERVISOR')
})

test('the header shows the model chip with model and effort, and a Model button', () => {
  const html = view(ready())
  expect(html).toMatch(/class="supervisor__model"[^>]*>model-a · high</)
  expect(html).toContain('>Model<')
  expect(html).toContain('aria-expanded="false"')
  expect(html).not.toContain('aria-label="XERXES model" aria-busy')
})

test('the model chip states: unconfigured, offline, loading, unavailable, no effort', () => {
  expect(modelChipText({ ...initialSupervisorConfig(), status: 'unconfigured' })).toBe('No model chosen')
  expect(modelChipText(initialSupervisorConfig())).toBe('Model…')
  expect(modelChipText({ ...initialSupervisorConfig(), phase: 'error' })).toBe('Model unavailable')
  expect(modelChipText({ ...configured, status: 'invalid' })).toBe('Model unavailable')
  expect(modelChipText({ ...configured, status: 'machine_offline' })).toBe('model-a · high · machine offline')
  expect(modelChipText({ ...configured, config: { ...configured.config!, effort: null } })).toBe('model-a')
})

test('an unconfigured XERXES shows the chooser inline, disables the composer and never mentions env vars', () => {
  const state = ready({ supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', status: 'unconfigured', canEdit: true } })
  const html = view(state)
  expect(html).toContain('aria-label="XERXES model"')
  expect(html).toContain('XERXES has no model yet')
  expect(html).toContain('Choose a model first')
  expect(html).toMatch(/<button[^>]*aria-label="Send to XERXES"[^>]*disabled=""/)
  expect(html).not.toContain('XERXES_HUB_SUPERVISOR')
  expect(html).not.toContain('>Close<')
})

test('while XERXES works the page shows the task view\'s Thinking row with a clock', () => {
  const html = view(ready({ supervisor: { phase: 'ready', busy: true, messages: [], confirmations: [] } }))
  expect(html).toContain('class="streamstatus supervisor__thinking"')
  expect(html).toContain('Thinking…')
  expect(html).toContain('class="streamstatus__clock"')
  expect(view(ready())).not.toContain('Thinking…')
  // The submit request has returned but the Hub is still running the turn.
  const working = view(ready({ supervisor: { phase: 'ready', busy: false, working: true, messages: [], confirmations: [] } }))
  expect(working).toContain('Thinking…')
})

test('a XERXES reply is never a scroll box inside the conversation', async () => {
  const css = await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()
  expect(css).toContain('.supervisor__message--xerxes .md{max-height:none;overflow:visible;padding:0}')
})

test('XERXES replies render as Markdown and the page uses the task composer, not raw text in boxes', () => {
  const state = ready({
    supervisor: {
      phase: 'ready', busy: false, confirmations: [],
      messages: [
        { id: 'u1', role: 'user', text: 'hi what can u do?', at: new Date().toISOString(), source: 'text' },
        { id: 'x1', role: 'xerxes', text: '**Look**\n- List your tasks', at: new Date().toISOString(), source: 'text' },
      ],
    },
  })
  const html = view(state)
  expect(html).toContain('<strong>Look</strong>')
  expect(html).not.toContain('**Look**')
  expect(html).toContain('class="msg msg--user"')
  expect(html).toContain('class="composer__input"')
  expect(html).toContain('class="supervisor__stream"')
})

test('the chooser lists machine, profile, model and effort from the options, and saves only for the owner', () => {
  const options = {
    machineId: 'm1',
    profiles: [{ name: 'work', label: 'Work', provider: 'anthropic', defaultModel: 'model-a' }],
    profile: 'work',
    models: [{ id: 'model-a', label: 'Model A' }, { id: 'model-b', label: 'Model B' }],
    model: 'model-a',
    efforts: [{ effort: 'low' }, { effort: 'high', description: 'think harder' }],
    defaultEffort: 'low',
  }
  const state = ready({
    machines: { phase: 'ready', list: [{ id: 'm1', label: 'Mac', health: 'online', readOnly: false, modelsSupported: true, workspaceIds: ['w'] }] },
    supervisorConfig: { ...configured, options: { phase: 'ready', options } },
  })
  const html = view(state, { chooserOpen: true })
  expect(html).toContain('aria-expanded="true"')
  expect(html).toContain('>Close<')
  expect(html).toContain('Mac')
  expect(html).toContain('Work · anthropic')
  expect(html).toContain('Model B')
  expect(html).toContain('high — think harder')
  expect(html).toContain('Provider default (low)')
  expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Save</)
  // A non-owner sees the choice read-only.
  const readOnly = view({ ...state, supervisorConfig: { ...state.supervisorConfig, canEdit: false } }, { chooserOpen: true })
  expect(readOnly).toContain('Only the Hub owner can change')
  expect(readOnly).toMatch(/<select[^>]*disabled=""/)
  expect(readOnly).toMatch(/<button[^>]*disabled=""[^>]*>Save</)
})

test('chooser errors and a missing machine are stated, not hidden', () => {
  const state = ready({
    machines: { phase: 'ready', list: [] },
    supervisorConfig: { ...configured, saveError: 'The Hub did not accept that choice.', options: { phase: 'error', error: 'machine offline' } },
  })
  const html = view(state, { chooserOpen: true })
  expect(html).toContain('No machine is linked to this Hub yet')
  expect(html).toContain('The Hub did not accept that choice.')
  expect(html).toContain('machine offline')
})

test('legacy typed Hub address cleanup is safe without localStorage', () => {
  expect(() => clearLegacyHubUrl()).not.toThrow()
})

test('a pending confirmation renders a card with Confirm and Cancel, who asked, and the truth about what runs', () => {
  const state: HubState = {
    ...ready(),
    supervisor: {
      phase: 'ready', busy: false,
      messages: [{ id: 'm1', role: 'xerxes', text: 'I proposed a steer.', at: new Date().toISOString(), source: 'text' }],
      confirmations: [{ id: 'act-1', tool: 'task.steer', summary: 'Steer the task: focus on tests', targets: ['One / First task'], requestedBy: 'ALICE', state: 'pending', expiresAt: new Date(Date.now() + 60_000).toISOString() }],
    },
  }
  const html = view(state)
  expect(html).toContain('I proposed a steer.')
  expect(html).toContain('XERXES proposes: task.steer')
  expect(html).toContain('Requested by ALICE')
  expect(html).toContain('Nothing runs until you confirm')
  expect(html).toContain('>Confirm<')
  expect(html).toContain('>Cancel<')
})

test('card states are honest: failed shows the error, expired offers no buttons, offline disables them', () => {
  const base = { id: 'a', tool: 'task.stop', summary: 'Stop it', targets: [] as string[] }
  const failed = renderToStaticMarkup(createElement(ActionConfirmationCard, { item: { ...base, state: 'failed', message: 'Could not confirm whether this was applied.' }, canAct: true, onRespond: noop }))
  expect(failed).toContain('role="alert"')
  expect(failed).toContain('Could not confirm whether this was applied.')
  expect(failed).not.toContain('>Confirm<')
  const expired = renderToStaticMarkup(createElement(ActionConfirmationCard, { item: { ...base, state: 'expired' }, canAct: true, onRespond: noop }))
  expect(expired).toContain('Expired')
  expect(expired).not.toContain('>Confirm<')
  const offline = renderToStaticMarkup(createElement(ActionConfirmationCard, { item: { ...base, state: 'pending' }, canAct: false, onRespond: noop }))
  expect(offline).toMatch(/<button[^>]*disabled=""[^>]*>Confirm</)
})
