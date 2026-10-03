// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Says, on the composer, that this SSH conversation's prompts run through
 * this computer's sign-in provider rather than on the host — never silently.
 * A dot pulses while a request is in flight; a lapsed access turns it into a
 * warning. Clicking opens the workspace panel with the full details.
 */

import type { ReactElement } from 'react'

import { useDesktopNavigation } from './DesktopPanels.js'
import { Icon } from './Icon.js'
import type { ProviderRelayView } from './store.js'

export function relaySummary(relay: ProviderRelayView, now = Date.now()): string {
  const route = relay.profile ? `${relay.profile}${relay.model ? ` (${relay.model})` : ''}` : 'its provider'
  const lines = [relay.live
    ? `Prompts for this task run on this computer through ${route}, not on ${relay.destination ?? 'the SSH host'}.`
    : relay.reconnecting
      ? `This task runs through ${route} on this computer. The link to ${relay.destination ?? 'the SSH host'} dropped; this window is restoring it and retries until it is back. A running turn waits and continues.`
      : `This task runs through ${route} on this computer, which is not connected to it right now. It continues once this window reconnects.`]
  lines.push(`${relay.requests} request${relay.requests === 1 ? '' : 's'} carried by this window${relay.inFlight ? ` · ${relay.inFlight} in flight` : ''}.`)
  if (relay.lastAt) lines.push(`Last request ${new Date(relay.lastAt).toLocaleTimeString()}.`)
  if (relay.lastError) lines.push(`Last error: ${relay.lastError}.`)
  if (!relay.live && relay.lastFailure) lines.push(`Last attempt: ${relay.lastFailure}`)
  if (relay.live && relay.expiresAt && relay.expiresAt > now) lines.push(`Access renews automatically; current grant ends ${new Date(relay.expiresAt).toLocaleTimeString()}.`)
  lines.push('Click to inspect or stop.')
  return lines.join('\n')
}

export function RelayBadge({ relay }: { relay: ProviderRelayView }): ReactElement {
  const open = useDesktopNavigation()
  const state = !relay.live ? relay.reconnecting ? 'reconnecting' : 'ended' : relay.inFlight > 0 ? 'busy' : 'live'
  return (
    <button
      className="cchip composer__text relaybadge"
      data-state={state}
      title={relaySummary(relay)}
      aria-label={relaySummary(relay)}
      onClick={() => open('workspace')}
    >
      <Icon name={state === 'ended' ? 'warning' : 'laptop'} size={14} />
      <span>{state === 'ended' ? 'Mac not connected' : state === 'reconnecting' ? 'Reconnecting to this Mac…' : 'via this Mac'}</span>
      {(state === 'busy' || state === 'reconnecting') && <span className="relaybadge__dot" aria-hidden="true" />}
    </button>
  )
}
