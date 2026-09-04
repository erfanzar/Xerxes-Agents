// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useEffect, useState } from 'react'

import { patchOverlayState } from '../app/overlayStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { GLYPH } from '../domain/nocturne.js'
import type { Theme } from '../theme.js'
import { Box, Span, Text } from './primitives.js'

export interface MachineTarget {
  alias: string
  host: string
  user: string
  workspacePath: string
}

export interface MachinePickerProps {
  onSelect: (machine: MachineTarget | null) => void
  t: Theme
}

/** Overlay for selecting a remote machine to work on. */
export function MachinePicker({ onSelect, t }: MachinePickerProps) {
  const { width } = useTerminalDimensions()
  const [machines, setMachines] = useState<MachineTarget[]>([])
  const [selected, setSelected] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load machines from the registry. For now this is a stub — the real
    // implementation reads from ~/.xerxes/remote.json.
    const load = async () => {
      setLoading(false)
      setMachines([
        { alias: 'local', host: 'localhost', user: 'current', workspacePath: '.' },
      ])
    }
    void load()
  }, [])

  useKeyboard(key => {
    if (key.name === 'escape') {
      patchOverlayState({ machinePicker: false })
      onSelect(null)
      return
    }
    if (key.name === 'return') {
      const machine = machines[selected]
      if (machine) {
        patchOverlayState({ machinePicker: false })
        onSelect(machine)
      }
      return
    }
    if (key.name === 'up' || key.name === 'k') {
      setSelected(Math.max(0, selected - 1))
      return
    }
    if (key.name === 'down' || key.name === 'j') {
      setSelected(Math.min(machines.length - 1, selected + 1))
      return
    }
  })

  const panelWidth = Math.min(60, width - 8)

  return (
    <Box
      backgroundColor={t.color.completionBg}
      borderColor={t.color.border}
      borderStyle="round"
      flexDirection="column"
      paddingX={2}
      paddingY={1}
      width={panelWidth}
    >
      <Text bold color={t.color.text}>
        <Span color={t.color.accent}>{`${GLYPH.tool} `}</Span>
        Select Machine
      </Text>
      <Text color={t.color.muted} dimColor>
        Choose a remote workspace to work on
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {loading ? (
          <Text color={t.color.muted}>Loading…</Text>
        ) : machines.length === 0 ? (
          <Text color={t.color.muted}>No remote machines configured</Text>
        ) : (
          machines.map((machine, index) => (
            <Box
              backgroundColor={index === selected ? t.color.accent : undefined}
              flexDirection="row"
              key={machine.alias}
              paddingX={1}
            >
              <Text color={index === selected ? t.color.text : t.color.text}>
                {index === selected ? '▸ ' : '  '}
                {machine.alias}
                <Span color={index === selected ? t.color.text : t.color.muted}>
                  {`  ${machine.user}@${machine.host}:${machine.workspacePath}`}
                </Span>
              </Text>
            </Box>
          ))
        )}
      </Box>
      <Box marginTop={1}>
        <Text color={t.color.muted} dimColor>
          ↑↓ select · Enter confirm · Esc cancel
        </Text>
      </Box>
    </Box>
  )
}
