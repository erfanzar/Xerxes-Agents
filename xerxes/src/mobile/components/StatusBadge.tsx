// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import type { StatusView } from '../../client/orbMapping.js'
import { AgentOrb, type OrbSize } from '../../desktop/renderer/AgentOrb.js'

const GLYPH: Readonly<Record<'failed' | 'offline' | 'unknown', string>> = { failed: '!', offline: '–', unknown: '?' }

/**
 * The status mark: an orb (the desktop AgentOrb over thinking-orbs, so the
 * motion vocabulary matches) or a static mark. Status is always spelled out in
 * text; colour and motion are never the only signal.
 */
export function StatusBadge({ view, size = 20 }: { view: StatusView; size?: OrbSize }): ReactElement {
  return (
    <span className={`status status-${view.tone}`} data-kind={view.kind}>
      {view.kind === 'orb'
        ? <AgentOrb state={view.state} size={size} live={view.live} />
        : <span className="status-mark" aria-hidden="true">{GLYPH[view.mark]}</span>}
      <span className="status-label">{view.label}</span>
      {view.outdated === true ? <span className="status-note">may be outdated</span> : null}
    </span>
  )
}
