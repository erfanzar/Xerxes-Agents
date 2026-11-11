// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { SuperagentControlPanel } from '../src/client/SuperagentControlPanel.js'
import { createSuperagentControlClient } from '../src/client/superagentControl.js'
import { initialHubState } from '../src/client/hubStore.js'
import { SupervisorView } from '../src/desktop/renderer/SupervisorPage.js'

const raw = { control: { revision: 1, paused: false, killed: false, limits: { actions: 0 } },
  usage: { actions: 0, wakes: 1, modelCalls: 2, tokens: null, usd: null } }

test('desktop Safety is an owner-only entry point wired to the shared control client', () => {
  const client = createSuperagentControlClient({ allowed: () => true, request: async () => raw })
  const initial = initialHubState()
  const render = (owner: boolean) => renderToStaticMarkup(createElement(SupervisorView, {
    state: { ...initial, autopilot: { ...initial.autopilot, canEdit: owner } }, controlClient: client,
    hubLabel: 'local', now: 0, onSend: async () => true, onRespond() {}, onReopen() {}, onOpenSettings() {},
    chooserOpen: false, onChooserChange() {}, onChooserShown() {}, onOptions() {}, onSaveModel: async () => true,
  }))
  expect(render(true)).toContain('Safety: Owner controls')
  expect(render(false)).not.toContain('Safety: Owner controls')
})

test('owner controls show explicit unknown usage, zero caps and human-only mutation buttons', async () => {
  const client = createSuperagentControlClient({ allowed: () => true, request: async () => raw })
  await client.load()
  const html = renderToStaticMarkup(createElement(SuperagentControlPanel, { client, online: true }))
  for (const text of ['Pause', 'Kill', 'Daily budgets', 'Unknown', 'Daily actions limit', 'value="0"']) expect(html).toContain(text)
  expect(html).not.toContain('Confirm kill')
})

test('Pause and Resume remain separate targets when admission state changes', async () => {
  for (const paused of [false, true]) {
    const client = createSuperagentControlClient({ allowed: () => true, request: async () => ({ ...raw, control: { ...raw.control, paused } }) })
    await client.load()
    const html = renderToStaticMarkup(createElement(SuperagentControlPanel, { client, online: true }))
    const pause = html.match(/<button[^>]*>Pause<\/button>/)?.[0]
    const resume = html.match(/<button[^>]*>Resume new work<\/button>/)?.[0]
    expect(pause).toBeDefined(); expect(resume).toBeDefined()
    expect(pause?.includes('disabled')).toBe(paused)
    expect(resume?.includes('disabled')).toBe(!paused)
  }
})

test('offline owner controls disable changes and do not mutate while rendering', async () => {
  let writes = 0
  const client = createSuperagentControlClient({ allowed: () => true, request: async method => { if (method.endsWith('.set')) writes += 1; return raw } })
  await client.load()
  const html = renderToStaticMarkup(createElement(SuperagentControlPanel, { client, online: false }))
  expect(html).toContain('disabled=""'); expect(writes).toBe(0)
})
