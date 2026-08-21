// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
// Shared modal chrome for the picker overlays (model, reasoning, …). The
// centered backdrop + titled panel used to be copy-pasted per picker with
// subtly divergent props; keep one implementation so they stay in lockstep.
import type { ReactNode } from 'react'

import type { Theme } from '../theme.js'

export function ModalShell({
  children,
  height,
  panelHeight,
  panelWidth,
  t,
  title,
  width
}: {
  children: ReactNode
  height: number
  panelHeight: number
  panelWidth: number
  t: Theme
  title: string
  width: number
}) {
  const top = Math.max(0, Math.floor((height - panelHeight) / 2))

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height={height}
      left={0}
      paddingTop={top}
      position="absolute"
      top={0}
      width={width}
      zIndex={200}
    >
      <box
        backgroundColor={t.color.statusBg}
        flexDirection="column"
        flexShrink={0}
        height={panelHeight}
        paddingBottom={1}
        paddingTop={1}
        width={panelWidth}
      >
        <box flexDirection="row" flexShrink={0} justifyContent="space-between" paddingLeft={2} paddingRight={2}>
          <text fg={t.color.accent} flexShrink={0}>
            <b>{title}</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc
          </text>
        </box>
        {children}
      </box>
    </box>
  )
}

export function InfoRow({ children, color, pad = true }: { children: ReactNode; color: string; pad?: boolean }) {
  return (
    <box flexShrink={0} paddingLeft={pad ? 2 : 0} paddingRight={pad ? 2 : 0}>
      <text fg={color} flexShrink={0} truncate width="100%" wrapMode="none">
        {children}
      </text>
    </box>
  )
}
