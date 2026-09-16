// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useRef, useState } from 'react'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogFooter } from './dialogChrome.js'

export interface RecordField { key: string; label: string; help?: string }
/** A keyboard form with one roomy editor; values stay with the owning workflow. */
export function RecordForm({ t, fields, values, onChange, onSubmit, onBack, busy, submitLabel }: {
  t: Theme; fields: RecordField[]; values: Record<string, string>; onChange: (key: string, value: string) => void
  onSubmit: () => void; onBack: () => void; busy: boolean; submitLabel: string
}) {
  const [index, setIndex] = useState(0)
  const input = useRef<TextareaRenderable | null>(null)
  const field = fields[Math.min(index, fields.length - 1)]
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'tab' || key.name === 'escape' || (key.ctrl && key.name === 's')) {
      key.preventDefault(); key.stopPropagation()
      if (busy) return
      if (key.name === 'escape') onBack()
      else if (key.name === 'tab') setIndex(value => (value + (key.shift ? fields.length - 1 : 1)) % fields.length)
      else onSubmit()
    }
  })
  if (!field) return <Box flexDirection="column" flexGrow={1}><Text color={t.ds.secondary}>No input values required.</Text><DialogFooter t={t}><Text color={t.ds.secondary}>Ctrl+S {submitLabel} · Esc back</Text></DialogFooter></Box>
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <Text color={t.color.accent} wrap="wrap">{field.label} · {index + 1}/{fields.length}</Text>
    {field.help ? <Text color={t.ds.secondary} wrap="wrap">{field.help}</Text> : null}
    <textarea key={field.key} ref={input} initialValue={values[field.key] ?? ''} onContentChange={() => { if (input.current) onChange(field.key, input.current.plainText) }} focused={!busy} flexGrow={1} minHeight={1} focusedBackgroundColor={t.color.statusBg} focusedTextColor={t.color.text} />
    <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">Tab / Shift+Tab fields · Ctrl+S {submitLabel} · Esc back</Text></DialogFooter>
  </Box>
}
