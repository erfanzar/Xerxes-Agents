// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'

import { normalizeBtwEventPayload } from '../gatewayAdapter.js'
import type {
  BtwEventPayload,
  BtwEventToolInfo,
  SideQuestionHistoryTurn,
  SideQuestionResponse
} from '../gatewayTypes.js'
import { rpcErrorMessage } from '../lib/rpc.js'

import type { GatewayRpc } from './interfaces.js'
import { $uiSessionId, getUiState } from './uiStore.js'

const MAX_SIDE_QUESTION_ENTRIES = 60
const UNSCOPED_SESSION_KEY = '__unscoped__'

export type SideQuestionStatus = 'streaming' | 'done' | 'error'
export type SideQuestionToolActivity = BtwEventToolInfo

export interface SideQuestionEntry {
  readonly requestId: string
  readonly question: string
  readonly answer: string
  readonly model?: string
  readonly tools?: readonly SideQuestionToolActivity[]
  readonly status: SideQuestionStatus
  readonly error?: string
  readonly createdAt: number
}

export interface SideQuestionSessionSlice {
  readonly entries: readonly SideQuestionEntry[]
  readonly activeRequestId: string | null
  readonly selectedModel?: string
  readonly generation: number
}

export const EMPTY_SIDE_QUESTION_SLICE: SideQuestionSessionSlice = {
  entries: [],
  activeRequestId: null,
  generation: 0
}

export const $sideQuestionStore = atom<Record<string, SideQuestionSessionSlice>>({})

let nextClientRequestSeq = 0

export function createSideQuestionRequestId(): string {
  nextClientRequestSeq += 1
  return `btw-${Date.now().toString(36)}-${nextClientRequestSeq}`
}

/** Parse optional leading `--model <id>` or `-m <id>` flag from side question input. */
export function parseSideQuestionInput(raw: string): {
  readonly model?: string
  readonly question: string
} {
  const trimmed = raw.trim()
  if (!trimmed) {
    return { question: '' }
  }
  const eqMatch = /^(?:--model|-m)=("[^"]+"|'[^']+'|\S+)(?:\s+([\s\S]*))?$/.exec(trimmed)
  if (eqMatch) {
    const model = eqMatch[1]!.replace(/^["']|["']$/g, '').trim()
    const question = (eqMatch[2] ?? '').trim()
    return { ...(model ? { model } : {}), question }
  }
  const spMatch = /^(?:--model|-m)\s+("[^"]+"|'[^']+'|\S+)(?:\s+([\s\S]*))?$/.exec(trimmed)
  if (spMatch) {
    const model = spMatch[1]!.replace(/^["']|["']$/g, '').trim()
    const question = (spMatch[2] ?? '').trim()
    return { ...(model ? { model } : {}), question }
  }
  return { question: trimmed }
}

const resolveTargetKey = (sessionId?: string | null): string => {
  const explicit = typeof sessionId === 'string' ? sessionId.trim() : ''
  if (explicit) return explicit
  const current = getUiState().sid?.trim() ?? ''
  return current || UNSCOPED_SESSION_KEY
}

export function getSideQuestionSliceForSession(
  store: Readonly<Record<string, SideQuestionSessionSlice>>,
  sessionId?: string | null
): SideQuestionSessionSlice {
  const sid = typeof sessionId === 'string' ? sessionId.trim() : ''
  if (sid) {
    if (store[sid]) return store[sid]!
    return store[UNSCOPED_SESSION_KEY] ?? EMPTY_SIDE_QUESTION_SLICE
  }
  return store[UNSCOPED_SESSION_KEY] ?? Object.values(store).at(-1) ?? EMPTY_SIDE_QUESTION_SLICE
}

export const $activeSideQuestionSlice = computed([$sideQuestionStore, $uiSessionId], (store, sid) =>
  getSideQuestionSliceForSession(store, sid)
)

export function getActiveSideQuestionSlice(): SideQuestionSessionSlice {
  return $activeSideQuestionSlice.get()
}

export function setSideQuestionModel(model: string | undefined, sessionId?: string | null): void {
  const key = resolveTargetKey(sessionId)
  const current = $sideQuestionStore.get()
  const prev = current[key] ?? EMPTY_SIDE_QUESTION_SLICE
  const cleaned = typeof model === 'string' ? model.trim() : ''
  const nextSlice: SideQuestionSessionSlice = {
    entries: prev.entries,
    activeRequestId: prev.activeRequestId,
    ...(cleaned ? { selectedModel: cleaned } : {}),
    generation: prev.generation
  }
  $sideQuestionStore.set({
    ...current,
    [key]: nextSlice
  })
}

export function buildSideQuestionHistory(entries: readonly SideQuestionEntry[]): SideQuestionHistoryTurn[] {
  const history: SideQuestionHistoryTurn[] = []
  for (const entry of entries) {
    if (entry.status !== 'done') continue
    const q = entry.question.trim()
    const a = entry.answer.trim()
    if (!q || !a) continue
    history.push({ role: 'user', content: q }, { role: 'assistant', content: a })
  }
  return history
}

function capEntries(entries: readonly SideQuestionEntry[]): readonly SideQuestionEntry[] {
  return entries.length > MAX_SIDE_QUESTION_ENTRIES
    ? entries.slice(entries.length - MAX_SIDE_QUESTION_ENTRIES)
    : entries
}

function parseBtwEventPayload(rawEvent: unknown): BtwEventPayload {
  if (rawEvent && typeof rawEvent === 'object' && !Array.isArray(rawEvent)) {
    const record = rawEvent as Record<string, unknown>
    if (record.kind === 'tool') {
      const rawTool = record.tool && typeof record.tool === 'object' && !Array.isArray(record.tool)
        ? (record.tool as Record<string, unknown>)
        : null
      const name = rawTool && typeof rawTool.name === 'string' ? rawTool.name.trim() : ''
      const target = rawTool && typeof rawTool.target === 'string' && rawTool.target.trim()
        ? rawTool.target.trim()
        : undefined
      const status = rawTool?.status === 'running' || rawTool?.status === 'error' ? rawTool.status : 'done'
      const model = typeof record.model === 'string' && record.model.trim() ? record.model.trim() : undefined
      return {
        session_id: typeof record.session_id === 'string' ? record.session_id : '',
        request_id: typeof record.request_id === 'string' ? record.request_id : '',
        kind: 'tool',
        ...(model ? { model } : {}),
        ...(name ? { tool: { name, ...(target ? { target } : {}), status } } : {})
      }
    }
  }
  return normalizeBtwEventPayload(rawEvent)
}

function upsertToolActivity(
  existing: readonly SideQuestionToolActivity[] | undefined,
  nextTool: SideQuestionToolActivity
): readonly SideQuestionToolActivity[] {
  const list = [...(existing ?? [])]
  for (let i = list.length - 1; i >= 0; i -= 1) {
    const item = list[i]!
    if (item.name === nextTool.name && (item.target ?? '') === (nextTool.target ?? '') && item.status === 'running') {
      list[i] = nextTool
      return list
    }
  }
  list.push(nextTool)
  return list
}

export function applyBtwEvent(rawEvent: unknown, fallbackSessionId?: string | null): BtwEventPayload {
  const event = parseBtwEventPayload(rawEvent)
  const key = resolveTargetKey(event.session_id || fallbackSessionId)
  const requestId = event.request_id.trim()
  if (!requestId) return event

  const current = $sideQuestionStore.get()
  const prev = current[key] ?? EMPTY_SIDE_QUESTION_SLICE
  const existingIdx = prev.entries.findIndex(item => item.requestId === requestId)
  const existing = existingIdx >= 0 ? prev.entries[existingIdx] : undefined

  // Ignore late delta/end/tool events for an unknown request unless it starts via 'begin' or carries question text.
  if (!existing && event.kind !== 'begin' && !event.question) {
    return event
  }

  const base: SideQuestionEntry = existing ?? {
    requestId,
    question: event.question ?? '',
    answer: '',
    ...(event.model ? { model: event.model } : {}),
    status: 'streaming',
    createdAt: Date.now()
  }

  let updated: SideQuestionEntry = base
  const nextModel = event.model ?? base.model
  const nextQuestion = event.question || base.question

  if (event.kind === 'begin') {
    updated = {
      ...base,
      question: nextQuestion,
      ...(nextModel ? { model: nextModel } : {}),
      status: base.status === 'done' ? 'done' : 'streaming'
    }
  } else if (event.kind === 'tool' && event.tool) {
    updated = {
      ...base,
      question: nextQuestion,
      ...(nextModel ? { model: nextModel } : {}),
      tools: upsertToolActivity(base.tools, event.tool),
      status: base.status === 'done' ? 'done' : 'streaming'
    }
  } else if (event.kind === 'delta') {
    updated = {
      ...base,
      question: nextQuestion,
      answer: `${base.answer}${event.delta ?? ''}`,
      ...(nextModel ? { model: nextModel } : {}),
      status: 'streaming'
    }
  } else if (event.kind === 'end') {
    updated = {
      requestId: base.requestId,
      question: nextQuestion,
      answer: event.answer !== undefined && event.answer.length > 0 ? event.answer : base.answer,
      ...(nextModel ? { model: nextModel } : {}),
      ...(base.tools && base.tools.length > 0 ? { tools: base.tools } : {}),
      status: 'done',
      createdAt: base.createdAt
    }
  } else if (event.kind === 'error') {
    updated = {
      requestId: base.requestId,
      question: nextQuestion,
      answer: base.answer,
      ...(nextModel ? { model: nextModel } : {}),
      ...(base.tools && base.tools.length > 0 ? { tools: base.tools } : {}),
      status: 'error',
      error: event.error || 'Side question failed.',
      createdAt: base.createdAt
    }
  }

  const nextEntries =
    existingIdx >= 0
      ? prev.entries.map((item, idx) => (idx === existingIdx ? updated : item))
      : capEntries([...prev.entries, updated])

  const activeRequestId =
    updated.status === 'streaming'
      ? requestId
      : prev.activeRequestId === requestId
        ? null
        : prev.activeRequestId

  $sideQuestionStore.set({
    ...current,
    [key]: {
      entries: nextEntries,
      activeRequestId,
      ...(prev.selectedModel ? { selectedModel: prev.selectedModel } : {}),
      generation: prev.generation
    }
  })

  return event
}

function settleSideQuestionEntry(
  key: string,
  expectedGeneration: number,
  requestId: string,
  patch: { status: 'done'; answer?: string; model?: string } | { status: 'error'; error: string }
): void {
  const current = $sideQuestionStore.get()
  const prev = current[key]
  if (!prev || prev.generation !== expectedGeneration) return
  const idx = prev.entries.findIndex(entry => entry.requestId === requestId)
  if (idx < 0) return
  const base = prev.entries[idx]!

  const updated: SideQuestionEntry =
    patch.status === 'done'
      ? {
          requestId: base.requestId,
          question: base.question,
          answer: patch.answer && patch.answer.length > 0 ? patch.answer : base.answer,
          ...((patch.model || base.model) ? { model: patch.model || base.model } : {}),
          ...(base.tools && base.tools.length > 0 ? { tools: base.tools } : {}),
          status: 'done',
          createdAt: base.createdAt
        }
      : {
          requestId: base.requestId,
          question: base.question,
          answer: base.answer,
          ...(base.model ? { model: base.model } : {}),
          ...(base.tools && base.tools.length > 0 ? { tools: base.tools } : {}),
          status: 'error',
          error: patch.error,
          createdAt: base.createdAt
        }

  $sideQuestionStore.set({
    ...current,
    [key]: {
      entries: prev.entries.map((entry, i) => (i === idx ? updated : entry)),
      activeRequestId: prev.activeRequestId === requestId ? null : prev.activeRequestId,
      ...(prev.selectedModel ? { selectedModel: prev.selectedModel } : {}),
      generation: prev.generation
    }
  })
}

export async function askSideQuestion(
  rpc: GatewayRpc,
  params: { sessionId?: string | null; question: string; requestId?: string; model?: string }
): Promise<SideQuestionResponse | null> {
  const sid = typeof params.sessionId === 'string' && params.sessionId.trim()
    ? params.sessionId.trim()
    : getUiState().sid?.trim() ?? ''
  const parsed = parseSideQuestionInput(params.question)
  if (!parsed.question) {
    if (parsed.model) {
      setSideQuestionModel(parsed.model, sid)
    }
    return null
  }
  const trimmedQuestion = parsed.question

  const key = resolveTargetKey(sid)
  const requestId = params.requestId?.trim() || createSideQuestionRequestId()

  const before = $sideQuestionStore.get()
  const prev = before[key] ?? EMPTY_SIDE_QUESTION_SLICE
  const generation = prev.generation
  const history = buildSideQuestionHistory(prev.entries)
  const effectiveModel = params.model?.trim() || parsed.model || prev.selectedModel || ''

  const pendingEntry: SideQuestionEntry = {
    requestId,
    question: trimmedQuestion,
    answer: '',
    ...(effectiveModel ? { model: effectiveModel } : {}),
    status: 'streaming',
    createdAt: Date.now()
  }

  $sideQuestionStore.set({
    ...before,
    [key]: {
      entries: capEntries([...prev.entries.filter(item => item.requestId !== requestId), pendingEntry]),
      activeRequestId: requestId,
      ...(prev.selectedModel ? { selectedModel: prev.selectedModel } : {}),
      generation
    }
  })

  try {
    const res = await rpc<SideQuestionResponse>(
      'session.side_question',
      {
        ...(sid ? { session_id: sid } : {}),
        request_id: requestId,
        question: trimmedQuestion,
        ...(effectiveModel ? { model: effectiveModel } : {}),
        ...(history.length > 0 ? { history } : {})
      },
      { reportError: false }
    )

    if (!res) {
      settleSideQuestionEntry(key, generation, requestId, {
        status: 'error',
        error: 'Could not reach daemon for side question.'
      })
      return null
    }

    if (res.ok === false) {
      settleSideQuestionEntry(key, generation, requestId, {
        status: 'error',
        error: res.error || 'Side question request was refused.'
      })
      return res
    }

    settleSideQuestionEntry(key, generation, requestId, {
      status: 'done',
      ...(typeof res.answer === 'string' && res.answer ? { answer: res.answer } : {}),
      ...(typeof res.model === 'string' && res.model ? { model: res.model } : {})
    })
    return res
  } catch (cause) {
    settleSideQuestionEntry(key, generation, requestId, {
      status: 'error',
      error: rpcErrorMessage(cause)
    })
    return null
  }
}

export function clearSideQuestions(sessionId?: string | null): void {
  const key = resolveTargetKey(sessionId)
  const current = $sideQuestionStore.get()
  const prev = current[key] ?? EMPTY_SIDE_QUESTION_SLICE
  $sideQuestionStore.set({
    ...current,
    [key]: {
      entries: [],
      activeRequestId: null,
      ...(prev.selectedModel ? { selectedModel: prev.selectedModel } : {}),
      generation: prev.generation + 1
    }
  })
}

export function resetSideQuestionStore(): void {
  $sideQuestionStore.set({})
}
