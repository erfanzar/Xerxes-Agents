// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'

import type { TurnState } from './turnStore.js'

const tail = (value: string | undefined, max = 120): string => {
  if (!value) {
    return ''
  }

  return value.length > max ? value.slice(-max) : value
}

export function liveTailScrollKey(state: TurnState): string {
  const lastSegment = state.streamSegments.at(-1)

  return [
    state.streaming.length,
    tail(state.streaming),
    state.streamPendingTools.length,
    tail(state.streamPendingTools.at(-1)),
    state.streamSegments.length,
    lastSegment?.role ?? '',
    lastSegment?.kind ?? '',
    lastSegment?.text.length ?? 0,
    tail(lastSegment?.text),
    state.reasoning.length,
    state.reasoningActive ? 1 : 0,
    state.reasoningStreaming ? 1 : 0,
    state.tools.map(tool => [tool.id, tool.name, tail(tool.context), tool.startedAt ?? 0].join(':')).join('|'),
    state.subagents
      .map(agent =>
        [
          agent.id,
          agent.status,
          agent.toolCount,
          agent.taskCount,
          tail(agent.tools.at(-1)),
          tail(agent.notes.at(-1)),
          tail(agent.summary),
          agent.durationSeconds ?? ''
        ].join(':')
      )
      .join('|'),
    state.todos.map(todo => [todo.id, todo.status, tail(todo.content)].join(':')).join('|'),
    state.turnTrail.length,
    tail(state.turnTrail.at(-1))
  ].join('\x1f')
}

export function shouldAutoScrollLiveTail(liveTailActive: boolean, scroll: ScrollBoxHandle | null): boolean {
  return liveTailActive && Boolean(scroll?.isSticky())
}
