// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Indented delegation panel for live subagent activity (lib/subagentTree.ts).

import { Box, Text } from '@xerxes/ink'

import { listSubagents, subagentSummary, type SubagentsState } from '../lib/subagentTree.js'
import { DEFAULT_THEME, type Theme } from '../theme.js'

export interface SubagentPanelProps {
  subagents: SubagentsState
  theme?: Theme
}

export function SubagentPanel({ subagents, theme = DEFAULT_THEME }: SubagentPanelProps) {
  const nodes = listSubagents(subagents)
  if (nodes.length === 0) {
    return null
  }
  return (
    <Box
      flexDirection="column"
      marginY={1}
      paddingLeft={1}
      borderStyle="single"
      borderColor={theme.color.system}
      borderLeft
      borderTop={false}
      borderRight={false}
      borderBottom={false}
    >
      <Text color={theme.color.system} bold>
        ⑂ delegation ({nodes.length})
      </Text>
      {nodes.map(node => (
        <Box key={node.agentId} flexDirection="row">
          <Text color={node.active ? theme.color.accent : theme.color.muted}>{node.active ? '◐ ' : '✓ '}</Text>
          <Text color={theme.color.toolName}>{node.subagentType || 'subagent'}</Text>
          <Text color={theme.color.muted}>
            {' '}
            {node.tools.length ? `[${node.tools.length} tools] ` : ''}
            {subagentSummary(node)}
          </Text>
        </Box>
      ))}
    </Box>
  )
}
