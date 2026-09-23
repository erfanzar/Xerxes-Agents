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
import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'

export function ChannelsCard({ snap }: { snap: Snapshot }): ReactElement {
  // Statuses are point-in-time daemon state, not pushed on open.
  useEffect(() => { store.loadChannels() }, [])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const channels = snap.channels

  const toggle = (row: ChannelRow): void => {
    setBusy(row.name)
    setError('')
    void store.setChannelEnabled(row.name, !row.enabled)
      .catch(error => setError(desktopError(error)))
      .finally(() => setBusy(null))
  }

  return (
    <>
      <h2 className="modal__title">Channels</h2>
      {error && <p className="studio-error" role="alert">{error}</p>}
      <p className="modal__sub">
        Messaging gateways the daemon answers chats through — telegram, discord, slack, whatsapp, email, signal. Credentials live outside this app; a gateway only runs when the daemon found its config at boot.
      </p>
      {!snap.channelsAvailable && (
        <div className="row">
          <span className="dot dot--idle" />
          <div className="row__main">
            <div className="row__t">Channel manager not configured</div>
            <div className="row__s">This runtime started without channel adapters, so there is nothing to enable.</div>
          </div>
        </div>
      )}
      {snap.channelsAvailable && !snap.channelsConfigured && (
        <div className="row">
          <span className="dot dot--idle" />
          <div className="row__main">
            <div className="row__t">No channel credentials found</div>
            <div className="row__s">Configure a gateway (token or secret in env or config), then restart the runtime to list it here.</div>
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
              disabled={busy !== null || snap.connection !== 'online'}
              aria-busy={busy === row.name}
              onClick={() => toggle(row)}
            />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 16 }}>
        <button className="btn btn--ghost" onClick={() => store.loadChannels()}><Icon name="retry" size={13} /> Refresh</button>
      </div>
    </>
  )
}
