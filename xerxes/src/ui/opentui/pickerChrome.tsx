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
  footer,
  headerRight,
  height,
  panelHeight,
  panelWidth,
  t,
  title,
  titleDetail,
  width
}: {
  children: ReactNode
  footer?: ReactNode
  headerRight?: ReactNode
  height: number
  panelHeight: number
  panelWidth: number
  t: Theme
  title: string
  titleDetail?: ReactNode
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
        backgroundColor={t.color.completionBg}
        borderColor={t.color.border}
        borderStyle="rounded"
        flexDirection="column"
        flexShrink={0}
        height={panelHeight}
        paddingBottom={1}
        paddingTop={1}
        width={panelWidth}
      >
        <box flexDirection="row" flexShrink={0} paddingLeft={2} paddingRight={2}>
          {/* The left title must yield to right-side state. A width="100%"
              text beside another text made the latter paint over the border
              in wide/tall terminals (the clipped `esc` seen in the picker). */}
          <box flexDirection="row" flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
            <text flexShrink={0} truncate width="100%" wrapMode="none">
              <span fg={t.color.accent}>✦ </span>
              <b>{title}</b>
              {titleDetail ? <span fg={t.color.muted}>{'  ›  '}{titleDetail}</span> : null}
            </text>
          </box>
          <box flexShrink={0}>
            {headerRight ?? (
              <text fg={t.color.muted} flexShrink={0}>
                esc
              </text>
            )}
          </box>
        </box>
        {children}
        {footer ? (
          <box
            backgroundColor={t.color.completionMetaBg}
            flexShrink={0}
            marginTop={1}
            paddingLeft={2}
            paddingRight={2}
          >
            {footer}
          </box>
        ) : null}
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
