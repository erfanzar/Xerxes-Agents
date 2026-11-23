// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { relativeTime } from '../../client/format.js'
import { workspaceAvailable } from '../../client/hubStore.js'
import { useHubState, useNow, useServices } from '../context.js'
import { reconnectOf } from '../MobileApp.js'
import { Sheet } from './Sheet.js'

const HEALTH_LABEL: Readonly<Record<string, string>> = {
  online: 'Online',
  connecting: 'Connecting',
  stale: 'Not responding',
  offline: 'Offline',
  needs_operator: 'Needs attention',
}

export function HubStatusSheet({ onClose }: { readonly onClose: () => void }): ReactElement {
  const state = useHubState()
  const now = useNow()
  const { controller } = useServices()
  const [confirmRevokeId, setConfirmRevokeId] = useState<string | null>(null)

  const live = state.connection.state === 'live'
  const reconnect = reconnectOf(controller)

  useEffect(() => {
    if (live) void controller.loadDevices?.()
  }, [controller, live])

  const workspaces = state.workspaceOrder
    .map(id => state.workspaces[id])
    .filter((w): w is NonNullable<typeof w> => w !== undefined)
  const onlineCount = workspaces.filter(workspaceAvailable).length
  const devices = state.devices
  const activeDevices = devices.list.filter(device => device.status !== 'revoked')

  return (
    <Sheet title="Hub & devices" onClose={onClose}>
      <div className="hub-sheet">
        <section className="hub-section" aria-label="Hub connection">
          <div className="hub-card">
            <div className="hub-row-main">
              <p className="hub-row-title">
                <span className={live ? 'hub-dot hub-dot-live' : 'hub-dot hub-dot-off'} aria-hidden="true" />
                <strong>{live ? 'Connected to Hub' : `Connection: ${state.connection.state}`}</strong>
              </p>
              <p className="hub-meta">
                {onlineCount} of {workspaces.length} {workspaces.length === 1 ? 'workspace' : 'workspaces'} reachable
              </p>
            </div>
            <div className="hub-row-actions">
              <button type="button" className="btn" disabled={!live} onClick={() => { void controller.refreshInbox() }}>Refresh</button>
              {reconnect !== undefined ? <button type="button" className="btn" onClick={reconnect}>Reconnect</button> : null}
            </div>
          </div>
        </section>

        <section className="hub-section" aria-label="Connected workspaces">
          <div className="hub-section-head">
            <h3 className="section-title">Workspaces</h3>
          </div>
          {workspaces.length === 0 ? (
            <p className="empty">No workspaces linked to this Hub yet.</p>
          ) : (
            <ul className="hub-list">
              {workspaces.map(workspace => {
                const ok = workspaceAvailable(workspace)
                return (
                  <li key={workspace.id} className="hub-row">
                    <div className="hub-row-main">
                      <p className="hub-row-title">
                        <span className={ok ? 'hub-dot hub-dot-live' : 'hub-dot hub-dot-off'} aria-hidden="true" />
                        <strong>{workspace.label}</strong>
                      </p>
                      {workspace.readOnly ? <p className="hub-meta">View only (update remote Xerxes to send commands)</p> : null}
                    </div>
                    <span className="chip">{HEALTH_LABEL[workspace.health] ?? workspace.health}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </section>

        {devices.unsupported ? null : (
          <section className="hub-section" aria-label="Paired devices">
            <div className="hub-section-head">
              <h3 className="section-title">Paired devices</h3>
              <button type="button" className="btn btn-quiet hub-refresh-btn" disabled={!live || devices.phase === 'loading'} onClick={() => { void controller.loadDevices?.() }}>
                {devices.phase === 'loading' ? 'Loading…' : 'Refresh'}
              </button>
            </div>
            {devices.error !== undefined ? (
              <div role="alert" className="notice is-error">
                <p>Could not load devices: {devices.error}</p>
              </div>
            ) : null}
            {devices.phase === 'ready' && activeDevices.length === 0 ? (
              <p className="empty">No active devices reported.</p>
            ) : null}
            {activeDevices.length > 0 ? (
              <ul className="hub-list">
                {activeDevices.map(device => {
                  const busy = devices.revokingId === device.id
                  const confirming = confirmRevokeId === device.id
                  return (
                    <li key={device.id} className="hub-row">
                      <div className="hub-row-main">
                        <p className="hub-row-title">
                          <strong>{device.label}</strong>
                          {device.current ? <span className="chip">This device</span> : null}
                        </p>
                        <p className="hub-meta">
                          {device.displayName ? `${device.displayName} · ` : ''}
                          {device.lastSeenAt ? `Seen ${relativeTime(device.lastSeenAt, now)}` : 'Never seen yet'}
                        </p>
                      </div>
                      {!device.current && controller.revokeDevice !== undefined ? (
                        <div className="hub-row-actions">
                          {confirming ? (
                            <>
                              <button
                                type="button"
                                className="btn btn-danger"
                                disabled={!live || busy}
                                onClick={() => {
                                  setConfirmRevokeId(null)
                                  void controller.revokeDevice?.(device.id)
                                }}
                              >
                                {busy ? 'Revoking…' : 'Confirm'}
                              </button>
                              <button type="button" className="btn" disabled={busy} onClick={() => setConfirmRevokeId(null)}>Cancel</button>
                            </>
                          ) : (
                            <button type="button" className="btn" disabled={!live || busy} onClick={() => setConfirmRevokeId(device.id)}>
                              {busy ? 'Revoking…' : 'Revoke'}
                            </button>
                          )}
                        </div>
                      ) : null}
                    </li>
                  )
                })}
              </ul>
            ) : null}
          </section>
        )}
      </div>
    </Sheet>
  )
}
