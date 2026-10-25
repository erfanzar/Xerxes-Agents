// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { createMacPermissions, type MacPermissionsPort, type MediaAccessAnswer } from '../src/desktop/main/macPermissions.js'
import type { MacPermissionsState } from '../src/desktop/macPermissionsShared.js'
import {
  MacPermissionsNoticeView, MacPermissionsView, macPermissionsNoticeVisible,
  type MacPermissionsNoticeProps, type MacPermissionsViewProps,
} from '../src/desktop/renderer/MacPermissionsPanel.js'

const noop = (): void => {}

function stateOf(patch: { trusted?: boolean; screen?: MediaAccessAnswer; microphone?: MediaAccessAnswer; dismissed?: boolean; platform?: string } = {}): MacPermissionsState {
  const port: MacPermissionsPort = {
    platform: patch.platform ?? 'darwin',
    appLabel: 'Xerxes Agents',
    isTrustedAccessibilityClient: () => patch.trusted ?? true,
    getMediaAccessStatus: media => (media === 'screen' ? patch.screen ?? 'granted' : patch.microphone ?? 'granted'),
    askForMicrophone: async () => true,
    requestScreenCapture: async () => {},
    openExternal: async () => {},
    relaunch: noop,
    readDismissed: () => patch.dismissed ?? false,
    writeDismissed: noop,
    now: () => Date.UTC(2026, 9, 25, 12),
  }
  return createMacPermissions(port).check()
}

function view(patch: Partial<MacPermissionsViewProps> = {}): string {
  return renderToStaticMarkup(createElement(MacPermissionsView, {
    available: true, state: stateOf(), busy: null,
    onOpen: noop, onRequest: noop, onRecheck: noop, onRemind: noop, onRelaunch: noop,
    ...patch,
  }))
}

function notice(patch: Partial<MacPermissionsNoticeProps> = {}): string {
  return renderToStaticMarkup(createElement(MacPermissionsNoticeView, {
    state: stateOf(), hidden: false,
    onOpen: noop, onLater: noop, onDismiss: noop, onRelaunch: noop, onDetails: noop,
    ...patch,
  }))
}

test('the settings section lists every permission with its status, purpose and actions', () => {
  const html = view({ state: stateOf({ trusted: false, screen: 'not-determined', microphone: 'denied' }) })
  expect(html).toContain('macOS permissions')
  for (const label of ['Accessibility', 'Screen Recording', 'Automation', 'App Management', 'Microphone', 'Full Disk Access']) expect(html).toContain(label)
  expect(html).toContain('data-permission="accessibility"')
  expect(html).toContain('Not asked yet')
  expect(html).toContain('Cannot be checked')
  expect(html).toContain('Dictation in the composer.')
  // Open Settings on every row, Request only where macOS has a prompt to show.
  expect(html.match(/>Open Settings</g)).toHaveLength(6)
  expect(html.match(/>Request</g)).toHaveLength(2)
  expect(html).toContain('Re-check')
  expect(html).toContain('Relaunch Xerxes Agents')
  // Who needs the grant is said plainly, including the VS Code case.
  expect(html).toContain('VS Code needs them instead')
  expect(html).toContain('only after Xerxes Agents relaunches')
  // The unreadable ones explain why.
  expect(html).toContain('without asking another app')
  // No URL is ever rendered into the page.
  expect(html).not.toContain('x-apple.systempreferences')
})

test('all granted: no Request buttons and no relaunch note', () => {
  const html = view()
  expect(html).not.toContain('>Request<')
  expect(html).not.toContain('only after Xerxes Agents relaunches')
  expect(html.match(/>On</g)?.length).toBe(3)
})

test('outside the desktop app, before the first answer, and off macOS', () => {
  expect(view({ available: false })).toContain('only checked in the Xerxes desktop app')
  expect(view({ state: undefined })).toContain('Checking permissions…')
  const other = view({ state: stateOf({ platform: 'linux' }) })
  expect(other).toContain('only exist on macOS')
  expect(other).not.toContain('Open Settings')
})

test('busy disables the buttons; an error is announced', () => {
  const html = view({ state: stateOf({ trusted: false }), busy: 'recheck', error: 'LaunchServices said no' })
  expect(html).toContain('Checking…')
  expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Open Settings</)
  expect(html).toContain('role="alert"')
  expect(html).toContain('LaunchServices said no')
})

test('the notice shows when Accessibility or Screen Recording is off and offers the pane, relaunch and opt-out', () => {
  const html = notice({ state: stateOf({ trusted: false, screen: 'denied' }) })
  expect(html).toContain('Computer use needs Accessibility and Screen Recording for Xerxes Agents')
  expect(html).toContain('Accessibility and Screen Recording are off')
  expect(html).toContain('Open settings')
  expect(html).toContain('Open Screen Recording')
  expect(html).toContain('Relaunch now')
  expect(html).toContain('relaunched after Screen Recording')
  expect(html).toContain('Don&#x27;t remind me')
  expect(html).toContain('Not now')

  const accessibilityOnly = notice({ state: stateOf({ trusted: false }) })
  expect(accessibilityOnly).toContain('Accessibility is off')
  expect(accessibilityOnly).not.toContain('Relaunch now')
})

test('the notice hides when granted, dismissed, closed for now, or off macOS', () => {
  expect(notice()).toBe('')
  expect(notice({ state: stateOf({ trusted: false, dismissed: true }) })).toBe('')
  expect(notice({ state: stateOf({ trusted: false }), hidden: true })).toBe('')
  expect(notice({ state: stateOf({ trusted: false, platform: 'linux' }) })).toBe('')
  expect(notice({ state: undefined })).toBe('')
  // Microphone alone is not a computer-use problem.
  expect(macPermissionsNoticeVisible(stateOf({ microphone: 'denied' }), false)).toBe(false)
  expect(macPermissionsNoticeVisible(stateOf({ screen: 'not-determined' }), false)).toBe(true)
})
