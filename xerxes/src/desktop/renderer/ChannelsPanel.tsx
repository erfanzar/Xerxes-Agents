// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Settings → Channels: the daemon's messaging gateways (telegram, discord,
 * slack, whatsapp, email, signal, …) with their configured/enabled state and
 * the enable/disable switches. Status is point-in-time `channel.list` state,
 * kept live by the daemon's `channel_status` broadcast — an enable here also
 * flips the same row in any other attached client.
 */

import { useEffect, useState, type ReactElement } from 'react'

import { store, type Snapshot } from './store.js'
import type { ChannelRow } from './types.js'

export function ChannelsCard({ snap }: { snap: Snapshot }): ReactElement {
  // Statuses are point-in-time daemon state, not pushed on open.
  useEffect(() => { store.loadChannels() }, [])
  const [busy, setBusy] = useState<string | null>(null)
  const channels = snap.channels

  const toggle = (row: ChannelRow): void => {
    setBusy(row.name)
    store.setChannelEnabled(row.name, !row.enabled)
    // The click lands immediately; the refreshed list (response + broadcast)
    // clears the pending look on the next snapshot.
    setTimeout(() => setBusy(current => (current === row.name ? null : current)), 600)
  }

  return (
    <>
      <h2 className="modal__title">Channels</h2>
      <p className="modal__sub">
        Messaging gateways the daemon answers chats through — telegram, discord, slack, whatsapp, email, signal. Credentials live outside this app; a gateway only runs when the daemon found its config at boot.
      </p>
      {!snap.channelsAvailable && (
        <div className="row">
          <span className="dot dot--idle" />
          <div className="row__main">
            <div className="row__t">Channel manager not configured</div>
            <div className="row__s">this daemon started without channel adapters — nothing to enable here</div>
          </div>
        </div>
      )}
      {snap.channelsAvailable && !snap.channelsConfigured && (
        <div className="row">
          <span className="dot dot--idle" />
          <div className="row__main">
            <div className="row__t">No channel credentials found</div>
            <div className="row__s">configure a gateway (token/secret env or config) and restart the daemon to list it here</div>
          </div>
        </div>
      )}
      <div className="rowlist">
        {channels.map(row => (
          <div className="row" key={row.name}>
            <span className={`dot ${row.enabled ? 'dot--live' : row.lastError ? 'dot--fail' : 'dot--idle'}`} />
            <div className="row__main">
              <div className="row__t">{row.name} <span className="chipbtn" style={{ marginLeft: 6 }}>{row.adapterName}</span></div>
              <div className="row__s">
                {row.enabled
                  ? 'enabled — receiving and sending'
                  : row.lastError
                    ? `disabled — last attempt failed: ${row.lastError}`
                    : 'disabled'}
                {row.lastOperation ? ` · last op: ${row.lastOperation}` : ''}
              </div>
            </div>
            <button
              className={`switch${row.enabled ? ' is-on' : ''}`}
              role="switch"
              aria-checked={row.enabled}
              aria-label={`${row.enabled ? 'Disable' : 'Enable'} ${row.name}`}
              disabled={busy === row.name || snap.connection !== 'online'}
              onClick={() => toggle(row)}
            />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 16 }}>
        <button className="btn btn--ghost" onClick={() => store.loadChannels()}>↻ Refresh</button>
      </div>
      <p className="modal__sub" style={{ marginTop: 14 }}>
        Status arrives live: the daemon broadcasts <code>channel_status</code> after every enable/disable, wherever it was asked from.
      </p>
    </>
  )
}
