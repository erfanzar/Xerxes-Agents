// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  type AgentTransitionRecord,
  type SessionRecord,
  type ToolCallRecord,
  type TurnRecord,
} from './models.js'

/** One chronologically sortable event in a replay timeline. */
export interface TimelineEvent {
  readonly data: Readonly<Record<string, unknown>>
  readonly eventType: 'agent_transition' | 'tool_call' | 'turn_end' | 'turn_start'
  readonly summary: string
  readonly timestamp: string
}

/** Read-only projection of a session for inspection, filtering, and rendering. */
export class ReplayView {
  readonly session: SessionRecord
  readonly turns: readonly TurnRecord[]

  constructor(session: SessionRecord, turns?: readonly TurnRecord[]) {
    this.session = session
    this.turns = turns ? [...turns] : [...session.turns]
  }

  filterByAgent(agentId: string): ReplayView {
    return new ReplayView(this.session, this.turns.filter(turn => turn.agentId === agentId))
  }

  getAgentTransitions(): AgentTransitionRecord[] {
    return [...this.session.agentTransitions]
  }

  getTimeline(): TimelineEvent[] {
    const events: TimelineEvent[] = []
    for (const turn of this.turns) {
      if (turn.startedAt) {
        events.push({
          timestamp: turn.startedAt,
          eventType: 'turn_start',
          summary: `Turn ${turn.turnId} started (agent=${turn.agentId})`,
          data: { turn_id: turn.turnId, agent_id: turn.agentId },
        })
      }
      for (const call of turn.toolCalls) {
        events.push({
          timestamp: turn.startedAt,
          eventType: 'tool_call',
          summary: `Tool call: ${call.toolName} (${call.status})`,
          data: { call_id: call.callId, tool_name: call.toolName, status: call.status },
        })
      }
      if (turn.endedAt) {
        events.push({
          timestamp: turn.endedAt,
          eventType: 'turn_end',
          summary: `Turn ${turn.turnId} ended (${turn.status})`,
          data: { turn_id: turn.turnId, status: turn.status },
        })
      }
    }
    for (const transition of this.session.agentTransitions) {
      events.push({
        timestamp: transition.timestamp,
        eventType: 'agent_transition',
        summary: `Agent switch: ${transition.fromAgent} -> ${transition.toAgent}`,
        data: {
          from_agent: transition.fromAgent,
          to_agent: transition.toAgent,
          reason: transition.reason,
        },
      })
    }
    return events.sort((left, right) => left.timestamp.localeCompare(right.timestamp))
  }

  getToolCalls(): ToolCallRecord[] {
    return this.turns.flatMap(turn => turn.toolCalls)
  }

  getTurn(indexOrId: number | string): TurnRecord | undefined {
    if (typeof indexOrId === 'number') return indexOrId >= 0 ? this.turns[indexOrId] : undefined
    return this.turns.find(turn => turn.turnId === indexOrId)
  }

  toMarkdown(): string {
    const lines = [
      `# Session ${this.session.sessionId}`,
      '',
      `- **Workspace:** ${this.session.workspaceId ?? 'N/A'}`,
      `- **Created:** ${this.session.createdAt}`,
      `- **Updated:** ${this.session.updatedAt}`,
      `- **Initial Agent:** ${this.session.agentId ?? 'N/A'}`,
      `- **Turns:** ${this.turns.length}`,
      `- **Tool Calls:** ${this.getToolCalls().length}`,
      '',
    ]
    if (this.session.agentTransitions.length > 0) {
      lines.push('## Agent Transitions', '')
      for (const transition of this.session.agentTransitions) {
        lines.push(
          `- [${transition.timestamp}] ${transition.fromAgent} -> ${transition.toAgent}${
            transition.reason ? ` (${transition.reason})` : ''
          }`,
        )
      }
      lines.push('')
    }
    lines.push('## Turns', '')
    for (const [index, turn] of this.turns.entries()) {
      lines.push(
        `### Turn ${index + 1}: ${turn.turnId}`,
        '',
        `- **Agent:** ${turn.agentId ?? 'N/A'}`,
        `- **Status:** ${turn.status}`,
        `- **Started:** ${turn.startedAt}`,
        `- **Ended:** ${turn.endedAt ?? 'N/A'}`,
      )
      if (turn.prompt) lines.push(`- **Prompt:** ${turn.prompt}`)
      if (turn.responseContent) {
        const preview = turn.responseContent.length > 200 ? `${turn.responseContent.slice(0, 200)}...` : turn.responseContent
        lines.push(`- **Response:** ${preview}`)
      }
      if (turn.error) lines.push(`- **Error:** ${turn.error}`)
      if (turn.toolCalls.length > 0) {
        lines.push(`- **Tool Calls (${turn.toolCalls.length}):**`)
        for (const call of turn.toolCalls) {
          const duration = call.durationMs ? ` (${Math.round(call.durationMs)}ms)` : ''
          lines.push(`  - \`${call.toolName}\` [${call.status}]${duration}`)
        }
      }
      lines.push('')
    }
    return lines.join('\n')
  }
}

/** Factory retained as the stable entry point for replay consumers. */
export class SessionReplay {
  static load(session: SessionRecord): ReplayView {
    return new ReplayView(session)
  }
}
