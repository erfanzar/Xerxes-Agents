// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { useKeyboard } from '@opentui/react'

import { patchOverlayState } from '../app/overlayStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { useStore } from '@nanostores/react'
import { $uiState } from '../app/uiStore.js'
import { GLYPH } from '../domain/nocturne.js'
import type { Theme } from '../theme.js'
import { Box, Span, Text } from './primitives.js'

export interface GoalOverlayProps {
  t: Theme
}

/** Full-screen overlay showing the current goal and todo list. */
export function GoalOverlay({ t }: GoalOverlayProps) {
  const goal = useStore($uiState).info?.goal
  const goalPhase = useStore($uiState).info?.goal_phase
  const todos = useTurnSelector(state => state.todos)

  useKeyboard(key => {
    if (key.name === 'escape') {
      patchOverlayState({ goal: false })
    }
  })

  const done = todos.filter(todo => todo.status === 'completed').length
  const inProgress = todos.filter(todo => todo.status === 'in_progress').length

  return (
    <Box
      alignItems="center"
      backgroundColor={t.color.completionBg}
      flexDirection="column"
      height="100%"
      justifyContent="center"
      width="100%"
    >
      <Box
        backgroundColor={t.color.completionBg}
        borderColor={t.color.border}
        borderStyle="round"
        flexDirection="column"
        maxWidth={80}
        paddingX={3}
        paddingY={2}
        width="90%"
      >
        <Text bold color={t.color.text}>
          <Span color={t.color.accent}>{`${GLYPH.tool} `}</Span>
          Goal & Todos
        </Text>

        {goal ? (
          <Box flexDirection="column" marginTop={2}>
            <Text color={t.ds.caption}>Goal</Text>
            <Text color={t.color.text} wrap="wrap">
              {goal}
            </Text>
            {goalPhase && goalPhase !== 'active' ? (
              <Text color={t.color.muted} dimColor>
                Status: {goalPhase}
              </Text>
            ) : null}
          </Box>
        ) : (
          <Box marginTop={2}>
            <Text color={t.color.muted}>
              No goal set. Use /goal to create one.
            </Text>
          </Box>
        )}

        {todos.length ? (
          <Box flexDirection="column" marginTop={2}>
            <Text color={t.ds.caption}>
              Todos ({done}/{todos.length} done{inProgress ? `, ${inProgress} in progress` : ''})
            </Text>
            {todos.map(todo => (
              <Box flexDirection="row" key={todo.id} marginTop={1}>
                <Text color={todo.status === 'completed' ? t.color.ok : todo.status === 'in_progress' ? t.color.accent : t.color.muted}>
                  {todo.status === 'completed' ? '✓' : todo.status === 'in_progress' ? '◌' : '◇'}
                </Text>
                <Text color={todo.status === 'completed' ? t.color.muted : t.color.text} wrap="wrap">
                  {' '}{todo.content}
                </Text>
              </Box>
            ))}
          </Box>
        ) : null}

        <Box marginTop={2}>
          <Text color={t.color.muted} dimColor>
            Esc close
          </Text>
        </Box>
      </Box>
    </Box>
  )
}
