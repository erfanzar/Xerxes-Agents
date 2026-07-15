// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SessionRecord } from './models.js'

export interface SessionSummaryOptions {
  readonly agentIds?: readonly string[]
  readonly charCount?: number
  readonly keyActions?: readonly string[]
  readonly outcome?: 'failure' | 'mixed' | 'success' | 'unknown'
  readonly sessionId: string
  readonly synopsis?: string
  readonly title?: string
  readonly turnCount?: number
}

/** Cheap, factual session summary suitable for history listings. */
export class SessionSummary {
  agentIds: string[]
  charCount: number
  keyActions: string[]
  outcome: 'failure' | 'mixed' | 'success' | 'unknown'
  sessionId: string
  synopsis: string
  title: string
  turnCount: number

  constructor(options: SessionSummaryOptions) {
    this.sessionId = options.sessionId
    this.title = options.title ?? ''
    this.synopsis = options.synopsis ?? ''
    this.keyActions = [...options.keyActions ?? []]
    this.outcome = options.outcome ?? 'unknown'
    this.turnCount = options.turnCount ?? 0
    this.agentIds = [...options.agentIds ?? []]
    this.charCount = options.charCount ?? 0
  }

  toRecord(): Record<string, unknown> {
    return {
      session_id: this.sessionId,
      title: this.title,
      synopsis: this.synopsis,
      key_actions: [...this.keyActions],
      outcome: this.outcome,
      turn_count: this.turnCount,
      agent_ids: [...this.agentIds],
      char_count: this.charCount,
    }
  }
}

export interface SessionSummarizerOptions {
  readonly llmClient?: (prompt: string) => string
}

/** Produces a deterministic heuristic summary, optionally refined by a caller-owned LLM. */
export class SessionSummarizer {
  private readonly llmClient: ((prompt: string) => string) | undefined

  constructor(options: SessionSummarizerOptions = {}) {
    this.llmClient = options.llmClient
  }

  summarize(session: SessionRecord): SessionSummary {
    const draft = this.deriveSynopsis(session)
    let synopsis = draft
    if (this.llmClient && session.turns.length > 0) {
      try {
        synopsis = this.refine(session, draft) || draft
      } catch {
        synopsis = draft
      }
    }
    return new SessionSummary({
      sessionId: session.sessionId,
      title: this.deriveTitle(session),
      synopsis,
      keyActions: this.collectTools(session),
      outcome: this.deriveOutcome(session),
      turnCount: session.turns.length,
      agentIds: this.distinctAgents(session),
      charCount: session.turns.reduce(
        (count, turn) => count + turn.prompt.length + (turn.responseContent?.length ?? 0),
        0,
      ),
    })
  }

  private collectTools(session: SessionRecord): string[] {
    const seen = new Set<string>()
    const tools: string[] = []
    for (const turn of session.turns) {
      for (const call of turn.toolCalls) {
        if (call.toolName && !seen.has(call.toolName)) {
          seen.add(call.toolName)
          tools.push(call.toolName)
        }
      }
    }
    return tools
  }

  private deriveOutcome(session: SessionRecord): 'failure' | 'mixed' | 'success' | 'unknown' {
    if (session.turns.length === 0) return 'unknown'
    if (session.turns.every(turn => turn.status === 'success')) return 'success'
    if (session.turns.every(turn => turn.status !== 'success')) return 'failure'
    return 'mixed'
  }

  private deriveSynopsis(session: SessionRecord): string {
    if (session.turns.length === 0) return 'Empty session.'
    const firstPrompt = session.turns[0]?.prompt.trim() ?? ''
    const lastResponse = [...session.turns].reverse().find(turn => turn.responseContent)?.responseContent?.trim() ?? ''
    const toolCount = session.turns.reduce((count, turn) => count + turn.toolCalls.length, 0)
    const sentences: string[] = []
    if (firstPrompt) sentences.push(`User asked: "${truncate(firstPrompt, 120)}".`)
    sentences.push(
      toolCount > 0
        ? `Agent used ${toolCount} tool call(s) across ${session.turns.length} turn(s).`
        : `Agent answered in ${session.turns.length} turn(s) without tools.`,
    )
    if (lastResponse) sentences.push(`Final answer: "${truncate(lastResponse, 200)}".`)
    return sentences.join(' ')
  }

  private deriveTitle(session: SessionRecord): string {
    const prompt = session.turns[0]?.prompt.trim()
    if (!prompt) return `Session ${session.sessionId.slice(0, 8)}`
    const words = prompt.split(/\s+/)
    if (words.length > 12) return `${words.slice(0, 10).join(' ')}…`
    return prompt.slice(0, 80)
  }

  private distinctAgents(session: SessionRecord): string[] {
    const seen = new Set<string>()
    const agents: string[] = []
    for (const turn of session.turns) {
      if (turn.agentId && !seen.has(turn.agentId)) {
        seen.add(turn.agentId)
        agents.push(turn.agentId)
      }
    }
    return agents
  }

  private refine(session: SessionRecord, draft: string): string {
    if (!this.llmClient) return ''
    const recent = session.turns.slice(-3)
      .map(turn => `- USER: ${turn.prompt.slice(0, 120)} | AGENT: ${(turn.responseContent ?? '').slice(0, 120)}`)
      .join('\n')
    return this.llmClient(
      'Rewrite this session synopsis as 1-3 short, neutral sentences. Preserve all factual claims; do not invent details.\n\n'
        + `Draft:\n${draft}\n\nRecent turns (newest last):\n${recent}`,
    ).trim()
  }
}

function truncate(value: string, length: number): string {
  const collapsed = value.split(/\s+/).filter(Boolean).join(' ')
  return collapsed.length <= length ? collapsed : `${collapsed.slice(0, length - 1).trimEnd()}…`
}
