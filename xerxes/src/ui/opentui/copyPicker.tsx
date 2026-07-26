// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'
import type { CopyOutcome } from '../lib/copyText.js'
import { copyPreview, copyRoleLabel, copyTextToClipboard, formatCopyOutcome } from '../lib/copyText.js'
import type { Theme } from '../theme.js'

const MAX_VISIBLE = 12
const MIN_PANEL_WIDTH = 48
const MAX_PANEL_WIDTH = 96

export interface CopyPickerProps {
  copyFn?: (text: string) => Promise<CopyOutcome>
  onCancel?: () => void
  onCopied?: (message: string) => void
  t?: Theme
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

const windowItems = <T,>(items: readonly T[], selected: number, visible: number) => {
  if (visible <= 0) {
    return { items: [] as readonly T[], offset: 0 }
  }

  const offset = Math.max(0, Math.min(selected - Math.floor(visible / 2), items.length - visible))

  return { items: items.slice(offset, offset + visible), offset }
}

/**
 * Bare /copy picker: lists recent copyable transcript messages (both roles,
 * newest last) and copies the highlighted one through the same
 * native → OSC52 clipboard chain as /copy and Ctrl+O. Closing the overlay
 * (Enter, Esc, Ctrl+C, or a failed copy) restores the prior screen because
 * the panel is absolutely positioned above the transcript — no transcript
 * state is touched.
 */
export function CopyPicker({ copyFn, onCancel, onCopied, t: suppliedTheme }: CopyPickerProps) {
  const overlay = useStore($overlayState)
  const storeTheme = useStore($uiTheme)
  const { height, width } = useTerminalDimensions()
  const t = suppliedTheme ?? storeTheme
  const state = overlay.copyPicker
  const items = state?.items ?? []

  const [selected, setSelected] = useState(() => Math.max(0, items.length - 1))

  // Re-open with a fresh snapshot → start on the newest message again.
  useEffect(() => {
    setSelected(Math.max(0, items.length - 1))
  }, [state])

  const close = useCallback(() => {
    patchOverlayState({ copyPicker: null })
    onCancel?.()
  }, [onCancel])

  const report = useCallback(
    (message: string) => {
      patchOverlayState({ copyPicker: null })
      onCopied?.(message)
    },
    [onCopied]
  )

  useKeyboard(event => {
    if (!state) {
      return
    }

    const name = event.name.toLowerCase()
    const isEscape = name === 'escape'
    const isReturn = name === 'return' || name === 'enter' || name === 'kpenter'

    if (isEscape || (event.ctrl && name === 'c')) {
      consume(event)
      close()

      return
    }

    if (name === 'up' || name === 'k') {
      consume(event)
      setSelected(index => Math.max(0, index - 1))

      return
    }

    if (name === 'down' || name === 'j') {
      consume(event)
      setSelected(index => Math.min(Math.max(0, items.length - 1), index + 1))

      return
    }

    if (name === 'g') {
      consume(event)
      setSelected(event.shift ? Math.max(0, items.length - 1) : 0)

      return
    }

    if (isReturn) {
      consume(event)
      const target = items[selected]

      if (!target) {
        close()

        return
      }

      void (copyFn ?? copyTextToClipboard)(target.text)
        .then(outcome => report(formatCopyOutcome(outcome)))
        .catch((error: unknown) => report(`copy failed: ${String(error)}`))
    }
  })

  if (!state) {
    return null
  }

  const panelWidth = Math.max(Math.min(width - 4, MAX_PANEL_WIDTH), Math.min(MIN_PANEL_WIDTH, width - 2))
  const visible = Math.min(MAX_VISIBLE, items.length, Math.max(3, height - 8))
  const { items: rows, offset } = windowItems(items, selected, visible)
  const previewWidth = Math.max(12, panelWidth - 22)
  const panelHeight = rows.length + 3
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
            <b>Copy message</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc
          </text>
        </box>
        {rows.map((item, rowIndex) => {
          const index = offset + rowIndex
          const isSelected = index === selected
          const label = copyRoleLabel(item)
          const preview = copyPreview(item.text, previewWidth)

          return (
            <box
              backgroundColor={isSelected ? t.color.completionCurrentBg : undefined}
              flexShrink={0}
              height={1}
              key={`${item.role}:${item.ordinal}`}
              paddingLeft={2}
              paddingRight={2}
              width="100%"
            >
              <text
                fg={isSelected ? t.color.accent : item.role === 'user' ? t.color.text : t.color.muted}
                flexShrink={0}
                truncate
                width="100%"
                wrapMode="none"
              >
                {`${isSelected ? '›' : ' '} ${String(index + 1).padStart(2)}. ${label}: ${preview}`}
              </text>
            </box>
          )
        })}
        <box flexShrink={0} height={1} paddingLeft={2} paddingRight={2}>
          <text fg={t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
            ↑/↓ select · Enter copy · Esc close
          </text>
        </box>
      </box>
    </box>
  )
}
