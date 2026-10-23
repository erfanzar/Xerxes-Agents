// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { qrDataUrl, type CockpitPairing, type CockpitStatus } from '../src/desktop/hubShared.js'
import { EMPTY_STATUS, HubPhoneView, healthText, type HubPhoneViewProps } from '../src/desktop/renderer/HubPhonePanel.js'

const noop = (): void => {}
const SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="1" height="1"/></svg>'

const saved: CockpitStatus = {
  target: 'xerxes@209.38.102.184',
  origin: 'https://hub.example.ts.net',
  wsUrl: 'wss://hub.example.ts.net/api/hub/v1/ws',
  fingerprint: 'SHA256:abc',
  hubVersion: '1.2.3',
  linkedMachineId: null,
  hasSession: false,
}

function render(patch: Partial<HubPhoneViewProps> = {}): string {
  return renderToStaticMarkup(createElement(HubPhoneView, {
    available: true, status: EMPTY_STATUS, address: '', onAddress: noop, saving: false, onSave: noop, onForget: noop,
    link: { phase: 'idle' }, onLink: noop, machines: { phase: 'idle', machines: [] }, onRefreshMachines: noop,
    pair: { phase: 'idle' }, onPair: noop, onConfirm: noop,
    ...patch,
  }))
}

const pairing = (patch: Partial<CockpitPairing> = {}): CockpitPairing => ({
  pairingId: 'pair-1',
  url: 'https://hub.example.ts.net/pair#abc',
  comparisonCode: '123-456',
  expiresAt: '2026-10-22T12:00:00Z',
  confirmRequired: true,
  qrDataUrl: qrDataUrl(SVG) ?? null,
  ...patch,
})

test('no address: one SSH field with the example, a save button, and no link or pair controls', () => {
  const html = render()
  expect(html).toContain('Hub &amp; phone')
  expect(html).toContain('Hub SSH address')
  expect(html).toContain('placeholder="xerxes@209.38.102.184"')
  expect(html).toContain('No Hub is saved on this Mac yet')
  expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Save and connect</)
  expect(html).not.toContain('Link this Mac')
  expect(html).not.toContain('Pair a phone')
  expect(html).not.toContain('type="url"')
})

test('the field never asks for a web address: the origin is learned and only displayed', () => {
  const html = render({ status: saved, address: saved.target! })
  expect(html).not.toContain('https://hub.example.ts.net"')
  expect(html).toContain('hub.example.ts.net')
  expect(html).toContain('1.2.3')
  expect(html).toContain('SHA256:abc')
  expect(html).toContain('Forget this Hub')
})

test('outside the desktop app the panel says so and shows no form', () => {
  const html = render({ available: false })
  expect(html).toContain('only available in the Xerxes desktop app')
  expect(html).not.toContain('<input')
})

test('saving shows progress and disables the field; a save error is announced', () => {
  const saving = render({ status: saved, address: 'x@y', saving: true })
  expect(saving).toContain('Checking the Hub…')
  expect(saving).toMatch(/<input[^>]*disabled=""/)
  const failed = render({ status: EMPTY_STATUS, address: 'x@y', saveError: 'That Hub has not published a web address yet.' })
  expect(failed).toContain('role="alert"')
  expect(failed).toContain('That Hub has not published a web address yet.')
})

test('saved but not linked: Link this Mac is offered; no machine list yet', () => {
  const html = render({ status: saved, address: saved.target! })
  expect(html).toContain('>Link this Mac<')
  expect(html).toContain('Not linked yet')
  expect(html).not.toContain('Machines and projects')
})

test('linking shows progress lines and disables the button; a failure shows its message', () => {
  const running = render({ status: saved, link: { phase: 'running', lines: ['registering machine…', 'enrolling key'] } })
  expect(running).toContain('Linking…')
  expect(running).toContain('registering machine…')
  expect(running).toContain('enrolling key')
  expect(running).toMatch(/<button[^>]*disabled=""[^>]*>Linking…</)
  const starting = render({ status: saved, link: { phase: 'running', lines: [] } })
  expect(starting).toContain('Starting…')
  const failed = render({ status: saved, link: { phase: 'error', message: 'ssh: connect refused' } })
  expect(failed).toContain('role="alert"')
  expect(failed).toContain('ssh: connect refused')
})

test('linked: the machine and its projects show with their health', () => {
  const html = render({
    status: { ...saved, linkedMachineId: 'm1' },
    machines: {
      phase: 'ready',
      machines: [{
        id: 'm1', label: 'Erfan Mac', endpointKind: 'local', discover: true,
        projects: [
          { workspaceId: 'w1', projectDir: '/a', label: 'Erfan Mac · app', health: 'online' },
          { workspaceId: 'w2', projectDir: '/b', label: 'Erfan Mac · docs', health: 'offline' },
        ],
      }],
    },
  })
  expect(html).toContain('Linked to the Hub as m1')
  expect(html).toContain('Link this Mac again')
  expect(html).toContain('Erfan Mac · all projects')
  expect(html).toContain('Erfan Mac · app')
  expect(html).toContain('>Online<')
  expect(html).toContain('>Offline<')
  expect(html).toContain('hubphone__health--offline')
})

test('a project list that could not load says why, and an empty machine explains itself', () => {
  const failed = render({ status: { ...saved, linkedMachineId: 'm1' }, machines: { phase: 'error', machines: [], error: 'The Hub sent a machine list this app cannot read.' } })
  expect(failed).toContain('The Hub sent a machine list this app cannot read.')
  const empty = render({
    status: { ...saved, linkedMachineId: 'm1' },
    machines: { phase: 'ready', machines: [{ id: 'm1', label: 'Mac', endpointKind: 'local', discover: true, projects: [] }] },
  })
  expect(empty).toContain('No projects found yet')
})

test('pairing: the QR renders as an inert data-URL image, with the link, the code and the confirm button', () => {
  const html = render({ status: saved, pair: { phase: 'ready', pairing: pairing(), confirming: false } })
  expect(html).toMatch(/<img[^>]*class="hubphone__qr"[^>]*src="data:image\/svg\+xml;base64,[A-Za-z0-9+/=]+"/)
  expect(html).not.toContain('<svg')
  expect(html).toContain('value="https://hub.example.ts.net/pair#abc"')
  expect(html).toContain('123-456')
  expect(html).toContain('>The phone shows 123-456<')
  expect(html).toContain('New pairing link')
})

test('pairing without a QR falls back to the link; confirming and an error are shown; no-confirm hubs skip the button', () => {
  const noQr = render({ status: saved, pair: { phase: 'ready', pairing: pairing({ qrDataUrl: null }), confirming: false } })
  expect(noQr).not.toContain('<img')
  expect(noQr).toContain('No QR code was made')
  const confirming = render({ status: saved, pair: { phase: 'ready', pairing: pairing(), confirming: true, error: 'The code did not match.' } })
  expect(confirming).toMatch(/<button[^>]*disabled=""[^>]*>Confirming…</)
  expect(confirming).toContain('The code did not match.')
  const auto = render({ status: saved, pair: { phase: 'ready', pairing: pairing({ confirmRequired: false }), confirming: false } })
  expect(auto).not.toContain('The phone shows')
  expect(auto).toContain('No confirmation is needed')
})

test('pairing states: creating, confirmed and failed', () => {
  expect(render({ status: saved, pair: { phase: 'creating' } })).toContain('Creating link…')
  expect(render({ status: saved, pair: { phase: 'confirmed', deviceId: 'd1' } })).toContain('Phone paired')
  const failed = render({ status: saved, pair: { phase: 'error', message: 'ssh: timed out' } })
  expect(failed).toContain('role="alert"')
  expect(failed).toContain('ssh: timed out')
})

test('health labels are plain words', () => {
  expect(healthText('needs_operator')).toBe('Needs attention')
  expect(healthText('stale')).toBe('Not responding')
  expect(healthText('weird')).toBe('Unknown')
})
