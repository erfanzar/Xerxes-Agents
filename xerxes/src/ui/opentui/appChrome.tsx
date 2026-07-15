// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Quiet session chrome: mode/title above the transcript and workspace below.
// Context and model metadata live with the composer, where they are actionable.
import type { Theme } from '../theme.js'

import { Box, Span, Text } from './primitives.js'

export const displayModeLabel = (mode?: string): string => {
  const value = (mode || 'code').trim()

  return value ? value[0]!.toUpperCase() + value.slice(1) : 'Code'
}

export function SessionHeader({
  mode,
  sessionId,
  sessionTitle,
  t
}: {
  mode?: string
  sessionId?: null | string
  sessionTitle?: null | string
  t: Theme
}) {
  const title = sessionTitle?.trim()
  const id = sessionId?.trim()

  return (
    <Box flexDirection="row" flexShrink={0} paddingX={2} paddingY={1} width="100%">
      <Box flexGrow={1} flexShrink={1} overflow="hidden">
        <Text bold wrap="truncate-end">
          <Span bold color={t.color.accent}>
            {displayModeLabel(mode)}
          </Span>
          {title ? <Span color={t.color.text}>: {title}</Span> : null}
        </Text>
      </Box>
      {id ? (
        <Text color={t.color.muted} wrap="truncate-end">
          {id}
        </Text>
      ) : null}
    </Box>
  )
}

export function WorkspaceFooter({ cwdLabel, rightLabel, t }: { cwdLabel: string; rightLabel?: string; t: Theme }) {
  if (!cwdLabel && !rightLabel) {
    return null
  }

  return (
    <Box flexDirection="row" flexShrink={0} justifyContent="space-between" paddingBottom={1} paddingX={2} width="100%">
      <Text color={t.color.muted} wrap="truncate-end">
        {cwdLabel}
      </Text>
      {rightLabel ? <Text color={t.color.muted}>{rightLabel}</Text> : null}
    </Box>
  )
}
