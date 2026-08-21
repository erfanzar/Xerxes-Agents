// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'
import type { ReasoningLevelsResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayLayout.js'
import { InfoRow, ModalShell } from './pickerChrome.js'

const MAX_VISIBLE = 10
const MIN_PANEL_WIDTH = 44
const MAX_PANEL_WIDTH = 90

interface LevelRow {
  readonly description: string
  readonly effort: string
}

export interface ReasoningPickerProps {
  onCancel?: () => void
  onSelect: (effort: string) => void
  t?: Theme
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

/**
 * Effort selector for the active model.
 *
 * The rows are whatever the provider reports for the model in use. Nothing
 * here is a fixed menu: the set varies per model, and providers that only
 * switch thinking on and off — or decide it themselves — render as such
 * rather than as a graded scale that cannot be honored.
 */
export function ReasoningPicker({ onCancel, onSelect, t }: ReasoningPickerProps) {
  const gateway = useGateway()
  const theme = useStore($uiTheme)
  const activeTheme = t ?? theme
  const { height, width } = useTerminalDimensions()

  const [levels, setLevels] = useState<readonly LevelRow[]>([])
  const [current, setCurrent] = useState('')
  const [defaultEffort, setDefaultEffort] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [index, setIndex] = useState(0)

  const close = useCallback(() => {
    patchOverlayState({ reasoningPicker: false })
    onCancel?.()
  }, [onCancel])

  useEffect(() => {
    let cancelled = false
    void gateway
      .rpc<ReasoningLevelsResponse>('reasoning.levels', {})
      .then(raw => {
        if (cancelled) return
        const result = asRpcResult<ReasoningLevelsResponse>(raw)
        if (!result) {
          setError('invalid response: reasoning.levels')
          setLoading(false)
          return
        }
        const rows = (result.levels ?? []).map(level => ({
          description: level.description ?? '',
          effort: level.effort
        }))
        setLevels(rows)
        setCurrent(result.current ?? '')
        setDefaultEffort(result.default ?? '')
        setNote(result.note ?? '')
        // Open on the active effort so Enter is a no-op rather than a
        // surprise change to whatever happens to sit at the top.
        const activeIndex = rows.findIndex(row => row.effort === result.current)
        setIndex(activeIndex >= 0 ? activeIndex : 0)
        setLoading(false)
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        setError(cause instanceof Error ? cause.message : String(cause))
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [gateway])

  const handleKey = useCallback(
    (key: KeyEvent) => {
      const name = key.name ?? ''
      if (name === 'escape') {
        consume(key)
        close()
        return
      }
      if (loading || levels.length === 0) {
        return
      }
      if (name === 'up' || (key.ctrl && name === 'p')) {
        consume(key)
        setIndex(previous => (previous - 1 + levels.length) % levels.length)
        return
      }
      if (name === 'down' || (key.ctrl && name === 'n')) {
        consume(key)
        setIndex(previous => (previous + 1) % levels.length)
        return
      }
      if (name === 'return' || name === 'enter') {
        consume(key)
        const chosen = levels[index]
        if (chosen) {
          patchOverlayState({ reasoningPicker: false })
          onSelect(chosen.effort)
        }
      }
    },
    [close, index, levels, loading, onSelect]
  )

  useKeyboard(handleKey)

  const panelWidth = Math.max(
    1,
    Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, width - 6), Math.max(1, width - 2))
  )
  const visible = Math.max(1, Math.min(MAX_VISIBLE, levels.length || 1, Math.max(1, height - 12)))
  const panelHeight = Math.min(height, visible + 8)
  const { items: visibleLevels, offset } = windowItems(levels, index, visible)

  // Composed by the daemon: it depends on the provider's reasoning shape,
  // not just on whether the list came back live.
  const subtitle = note

  if (loading) {
    return (
      <ModalShell height={height} panelHeight={5} panelWidth={panelWidth} t={activeTheme} title="Reasoning effort" width={width}>
        <InfoRow color={activeTheme.color.muted}>asking the provider…</InfoRow>
        <InfoRow color={activeTheme.color.muted}>Esc close</InfoRow>
      </ModalShell>
    )
  }

  return (
    <ModalShell
      height={height}
      panelHeight={panelHeight}
      panelWidth={panelWidth}
      t={activeTheme}
      title="Reasoning effort"
      width={width}
    >
      {subtitle ? <InfoRow color={activeTheme.color.muted}>{subtitle}</InfoRow> : null}
      {error ? <InfoRow color={activeTheme.color.error}>error: {error}</InfoRow> : null}
      <InfoRow color={activeTheme.color.muted}>↑/↓ select · Enter apply · Esc cancel</InfoRow>

      {levels.length === 0 ? (
        <InfoRow color={activeTheme.color.muted}>nothing to select for this provider</InfoRow>
      ) : (
        visibleLevels.map((level, rowIndex) => {
          const selected = offset + rowIndex === index
          const marks = [
            level.effort === current ? 'active' : '',
            level.effort === defaultEffort ? 'default' : ''
          ].filter(Boolean)
          const suffix = marks.length ? ` (${marks.join(', ')})` : ''
          const label = level.description
            ? `${level.effort}${suffix} · ${level.description}`
            : `${level.effort}${suffix}`

          return (
            <box
              backgroundColor={selected ? activeTheme.color.completionCurrentBg : undefined}
              flexShrink={0}
              height={1}
              key={level.effort}
              paddingLeft={2}
              paddingRight={2}
              width="100%"
            >
              <text
                fg={selected ? activeTheme.color.accent : activeTheme.color.text}
                flexShrink={0}
                truncate
                width="100%"
                wrapMode="none"
              >
                {`${level.effort === current ? '*' : '●'} ${label}`}
              </text>
            </box>
          )
        })
      )}
    </ModalShell>
  )
}
