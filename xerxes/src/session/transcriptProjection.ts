// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { TranscriptEvent, TranscriptEventMessage } from './transcriptEventLog.js'

export interface ProjectedTool {
  readonly arguments: Readonly<Record<string, unknown>>
  readonly callId: string
  readonly name: string
  readonly ok?: boolean
  readonly result?: unknown
  readonly status: 'running' | 'completed'
}
export interface ProjectedTurn {
  readonly context?: Readonly<Record<string, unknown>>
  readonly ended: boolean
  readonly mode?: string
  readonly policies: readonly Readonly<Record<string, unknown>>[]
  readonly request?: Readonly<Record<string, unknown>>
  readonly stopReason?: string
  readonly tools: readonly ProjectedTool[]
  readonly turnId: string
  readonly usage?: Readonly<Record<string, number>>
}
export interface TranscriptProjection {
  readonly lastSequence: number
  readonly messages: readonly TranscriptEventMessage[]
  readonly turns: readonly ProjectedTurn[]
}

type MutableTool = { arguments: Readonly<Record<string, unknown>>; callId: string; name: string; ok?: boolean; result?: unknown; status: 'running' | 'completed' }
type MutableTurn = {
  context?: Readonly<Record<string, unknown>>; ended: boolean; mode?: string
  policies: Readonly<Record<string, unknown>>[]; request?: Readonly<Record<string, unknown>>
  stopReason?: string; tools: MutableTool[]; turnId: string; usage?: Readonly<Record<string, number>>
}

/** Rebuild user-visible transcript state from a typed event suffix. */
export function projectTranscriptEvents(input: readonly TranscriptEvent[]): TranscriptProjection {
  const current = input.filter(event => event.sequence !== undefined)
    .sort((left, right) => (left.sequence ?? 0) - (right.sequence ?? 0))
  let expected = current.length > 0 ? current[0]!.sequence! : 0
  const seen = new Set<number>()
  for (const event of current) {
    const sequence = event.sequence!
    if (seen.has(sequence)) throw new Error(`duplicate transcript event sequence ${sequence}`)
    if (sequence !== expected) throw new Error(`transcript event sequence gap: expected ${expected}, received ${sequence}`)
    seen.add(sequence); expected += 1
  }

  const messages: TranscriptEventMessage[] = []
  const turns = new Map<string, MutableTurn>()
  for (const event of current) {
    if (event.type === 'message_appended') {
      if (event.index === messages.length) messages.push(event.message)
      continue
    }
    const turn = turns.get(event.turnId) ?? { ended: false, policies: [], tools: [], turnId: event.turnId }
    turns.set(event.turnId, turn)
    switch (event.type) {
      case 'turn_started': if (event.details.mode !== undefined) turn.mode = event.details.mode; break
      case 'request_prepared': turn.request = { ...event.details }; break
      case 'context_composed': turn.context = { ...event.details }; break
      case 'policy_decided': turn.policies.push({ ...event.details }); break
      case 'tool_started': turn.tools.push({ ...event.details, status: 'running' }); break
      case 'tool_completed': {
        const tool = [...turn.tools].reverse().find(candidate => candidate.callId === event.details.callId)
        if (tool) {
          tool.status = 'completed'; tool.ok = event.details.ok
          if ('result' in event.details) tool.result = event.details.result
        }
        break
      }
      case 'turn_completed':
        turn.ended = true; turn.stopReason = event.details.stopReason
        if (event.details.usage !== undefined) turn.usage = { ...event.details.usage }
        break
    }
  }
  return {
    lastSequence: current.at(-1)?.sequence ?? 0,
    messages,
    turns: [...turns.values()],
  }
}
