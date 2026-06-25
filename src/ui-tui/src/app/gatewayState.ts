// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure reducer mapping daemon wire events (snake_case) onto render state. This
// is the Xerxes-native analogue of Xerxes' createGatewayEventHandler — kept
// pure so it is unit-testable without React or a live daemon.

import { normalizeEventType, type DisplayBlock, type QuestionItem } from '../gatewayTypes.js'
import { normalizeDisplayBlocks, summarizeResult } from '../lib/displayBlocks.js'
import { emptySubagents, reduceSubagentEvent, type SubagentsState } from '../lib/subagentTree.js'
import { type DetailsMode, resolveDetails } from '../lib/details.js'

export type Role = 'user' | 'assistant' | 'tool' | 'system' | 'think'

export interface TranscriptRow {
  id: number
  role: Role
  text: string
  /** Structured tool-result panels (diff/todo/background_task/brief/generic). */
  blocks?: DisplayBlock[]
}

export interface Session {
  model: string
  agentName: string
  gitBranch: string
  cwd: string
  contextLimit: number
  sessionId: string
}

export interface Status {
  contextTokens: number
  maxContext: number
  mode: string
  planMode: boolean
  reasoningEffort: string
}

export interface PendingApproval {
  id: string
  toolCallId: string
  action: string
  description: string
}

export interface PendingQuestion {
  id: string
  toolCallId: string
  questions: QuestionItem[]
}

export interface UiState {
  connected: boolean
  busy: boolean
  session: Session
  status: Status
  transcript: TranscriptRow[]
  /** Live assistant text for the in-flight turn (flushed on turn_end). */
  streaming: string
  /** Live reasoning text for the in-flight turn. */
  thinking: string
  /** Last transport/notice line for the status area. */
  notice: string
  /** A blocking approval prompt awaiting the user's decision, if any. */
  pendingApproval: PendingApproval | null
  /** A blocking clarify/question prompt awaiting the user's answer, if any. */
  pendingQuestion: PendingQuestion | null
  /** Messages typed while the agent was busy, awaiting drain. */
  queue: string[]
  /** Live subagent/delegation activity, keyed by agent id. */
  subagents: SubagentsState
  /** Tool/thinking detail visibility (/details). */
  details: DetailsMode
  nextId: number
}

export const initialState: UiState = {
  connected: false,
  busy: false,
  session: { model: '', agentName: 'Xerxes-Agents', gitBranch: '', cwd: '', contextLimit: 0, sessionId: '' },
  status: { contextTokens: 0, maxContext: 0, mode: 'code', planMode: false, reasoningEffort: 'off' },
  transcript: [],
  streaming: '',
  thinking: '',
  notice: '',
  pendingApproval: null,
  pendingQuestion: null,
  queue: [],
  subagents: emptySubagents,
  details: 'expanded',
  nextId: 1
}

export interface WireEvt {
  type: string
  payload: Record<string, unknown>
}

function pushRow(state: UiState, role: Role, text: string, blocks?: DisplayBlock[]): UiState {
  if (!text && !(blocks && blocks.length)) {
    return state
  }
  return {
    ...state,
    transcript: [...state.transcript, { id: state.nextId, role, text, ...(blocks && blocks.length ? { blocks } : {}) }],
    nextId: state.nextId + 1
  }
}

const str = (v: unknown, d = ''): string => (typeof v === 'string' ? v : d)
const num = (v: unknown, d = 0): number => (typeof v === 'number' ? v : d)
const bool = (v: unknown, d = false): boolean => (typeof v === 'boolean' ? v : d)

/** Apply one wire event (or a synthetic client event) to the UI state. */
export function reduce(state: UiState, evt: WireEvt): UiState {
  const type = normalizeEventType(evt.type)
  const p = evt.payload ?? {}

  switch (type) {
    // ── Client-only synthetics ───────────────────────────────────────
    case '__user':
      return pushRow(state, 'user', str(p.text))
    case '__clear':
      return { ...state, transcript: [], streaming: '', thinking: '', notice: 'cleared' }
    case '__enqueue':
      return { ...state, queue: [...state.queue, str(p.text)].filter(Boolean) }
    case '__dequeue':
      return { ...state, queue: state.queue.slice(1) }
    case '__notice':
      return { ...state, notice: str(p.text) }
    case '__details': {
      const mode = resolveDetails(str(p.arg), state.details)
      return { ...state, details: mode, notice: `details: ${mode}` }
    }
    case '__approval_done':
      return { ...state, pendingApproval: null }
    case '__question_done':
      return { ...state, pendingQuestion: null }

    case 'gateway.ready':
      return { ...state, connected: true, notice: 'connected' }
    case 'gateway.closed':
      return { ...state, connected: false, busy: false, notice: 'daemon disconnected' }
    case 'gateway.error':
      return { ...state, notice: `error: ${str(p.message)}` }
    case 'gateway.protocol_error':
      return { ...state, notice: 'protocol error' }

    case 'init_done':
      return {
        ...state,
        connected: true,
        session: {
          model: str(p.model, state.session.model),
          agentName: str(p.agent_name, state.session.agentName),
          gitBranch: str(p.git_branch, state.session.gitBranch),
          cwd: str(p.cwd, state.session.cwd),
          contextLimit: num(p.context_limit, state.session.contextLimit),
          sessionId: str(p.session_id, state.session.sessionId)
        }
      }

    case 'status_update':
      return {
        ...state,
        status: {
          contextTokens: num(p.context_tokens, state.status.contextTokens),
          maxContext: num(p.max_context, state.status.maxContext),
          mode: str(p.mode, state.status.mode),
          planMode: bool(p.plan_mode, state.status.planMode),
          reasoningEffort: str(p.reasoning_effort, state.status.reasoningEffort)
        }
      }

    case 'turn_begin':
      return { ...state, busy: true, streaming: '', thinking: '', subagents: emptySubagents }

    case 'subagent_event':
      return { ...state, subagents: reduceSubagentEvent(state.subagents, p) }

    case 'text_part':
      return { ...state, streaming: state.streaming + str(p.text) }

    case 'think_part':
      return { ...state, thinking: state.thinking + str(p.think) }

    case 'tool_call':
      return pushRow(state, 'tool', `${str(p.name, 'tool')}${p.arguments ? ` ${str(p.arguments)}` : ''}`)

    case 'tool_result': {
      const blocks = normalizeDisplayBlocks(p.display_blocks)
      const summary = summarizeResult(str(p.return_value), num(p.duration_ms))
      return pushRow(state, 'tool', blocks.length ? '' : summary, blocks)
    }

    case 'turn_end': {
      const flushed = state.streaming ? pushRow(state, 'assistant', state.streaming) : state
      return { ...flushed, busy: false, streaming: '', thinking: '' }
    }

    case 'step_interrupted':
      return { ...state, busy: false, notice: 'interrupted' }

    case 'notification':
      return pushRow({ ...state, notice: str(p.title) }, 'system', str(p.title) || str(p.body))

    case 'approval_request':
      return {
        ...state,
        notice: `approval requested: ${str(p.action)}`,
        pendingApproval: {
          id: str(p.id),
          toolCallId: str(p.tool_call_id),
          action: str(p.action),
          description: str(p.description)
        }
      }

    case 'question_request':
      return {
        ...state,
        notice: 'question requested',
        pendingQuestion: {
          id: str(p.id),
          toolCallId: str(p.tool_call_id),
          questions: Array.isArray(p.questions) ? (p.questions as QuestionItem[]) : []
        }
      }

    default:
      return state
  }
}
