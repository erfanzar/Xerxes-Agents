// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type FormEvent, type ReactElement } from 'react'

import { desktopCall, desktopError } from '../../desktop/renderer/desktopRpc.js'
import { Icon } from '../../desktop/renderer/Icon.js'
import { Markdown } from '../../desktop/renderer/markdown.js'
import { bareModelName } from '../../desktop/renderer/Overlays.js'
import { SideQuestionModelPicker } from '../../desktop/renderer/SideQuestion.js'
import {
  formatSideQuestionToolLabel,
  mergeSideQuestionTool,
  parseSideQuestionToolPayload,
  settleSideQuestionTools,
  type SideQuestionToolEvent,
} from '../../desktop/renderer/sideQuestionState.js'
import { store, type Snapshot } from '../../desktop/renderer/store.js'
import { XerxesThinkingMark } from '../../desktop/renderer/XerxesSprites.js'

export interface VscodeBtwEntry {
  readonly requestId: string
  readonly question: string
  readonly answer: string
  readonly model?: string
  readonly status: 'streaming' | 'done' | 'error'
  readonly error?: string
  readonly tools?: readonly SideQuestionToolEvent[]
}

export interface VscodeBtwHistoryTurn {
  readonly role: 'user' | 'assistant'
  readonly content: string
}

export interface ParsedVscodeBtwEvent {
  readonly sessionId: string
  readonly requestId: string
  readonly kind: 'begin' | 'delta' | 'tool' | 'end' | 'error'
  readonly question?: string
  readonly delta?: string
  readonly answer?: string
  readonly model?: string
  readonly error?: string
  readonly tool?: SideQuestionToolEvent
}

let nextVscodeBtwSeq = 0

export function createVscodeBtwRequestId(): string {
  nextVscodeBtwSeq += 1
  return `xv-btw-${Date.now().toString(36)}-${nextVscodeBtwSeq}`
}

export function stripLeadingBtwPrefix(raw: string): string {
  const trimmed = raw.trim()
  const match = trimmed.match(/^\/(?:btw|side)(?:\s+(.*))?$/i)
  return match ? (match[1] ?? '').trim() : trimmed
}

export function buildVscodeBtwHistory(entries: readonly VscodeBtwEntry[]): VscodeBtwHistoryTurn[] {
  const history: VscodeBtwHistoryTurn[] = []
  for (const entry of entries) {
    if (entry.status !== 'done') continue
    const q = entry.question.trim()
    const a = entry.answer.trim()
    if (!q || !a) continue
    history.push({ role: 'user', content: q }, { role: 'assistant', content: a })
  }
  return history
}

export function parseVscodeBtwEvent(payload: unknown): ParsedVscodeBtwEvent | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null
  const record = payload as Record<string, unknown>
  const requestId = typeof record.request_id === 'string' ? record.request_id.trim() : ''
  if (!requestId) return null
  const rawKind = typeof record.kind === 'string' ? record.kind : ''
  const kind: ParsedVscodeBtwEvent['kind'] =
    rawKind === 'begin' || rawKind === 'delta' || rawKind === 'tool' || rawKind === 'end' || rawKind === 'error'
      ? rawKind
      : typeof record.error === 'string' && record.error
        ? 'error'
        : typeof record.delta === 'string'
          ? 'delta'
          : 'end'
  const tool = kind === 'tool' ? parseSideQuestionToolPayload(record) : null
  if (kind === 'tool' && !tool) return null
  const sessionId = typeof record.session_id === 'string' ? record.session_id.trim() : ''
  const question = typeof record.question === 'string' && record.question.trim() ? record.question.trim() : undefined
  const model = typeof record.model === 'string' && record.model.trim() ? record.model.trim() : undefined
  const error = typeof record.error === 'string' && record.error.trim() ? record.error.trim() : undefined
  return {
    sessionId,
    requestId,
    kind,
    ...(question ? { question } : {}),
    ...(typeof record.delta === 'string' ? { delta: record.delta } : {}),
    ...(typeof record.answer === 'string' ? { answer: record.answer } : {}),
    ...(model ? { model } : {}),
    ...(error ? { error } : {}),
    ...(tool ? { tool } : {}),
  }
}

export function foldVscodeBtwEvent(
  entries: readonly VscodeBtwEntry[],
  ev: ParsedVscodeBtwEvent,
): readonly VscodeBtwEntry[] {
  const idx = entries.findIndex(item => item.requestId === ev.requestId)
  const existing = idx >= 0 ? entries[idx] : undefined
  if (!existing && ev.kind !== 'begin' && ev.kind !== 'tool' && !ev.question) {
    return entries
  }

  const base: VscodeBtwEntry = existing ?? {
    requestId: ev.requestId,
    question: ev.question ?? '',
    answer: '',
    ...(ev.model ? { model: ev.model } : {}),
    status: 'streaming',
  }

  const nextQuestion = ev.question || base.question
  const nextModel = ev.model ?? base.model
  let updated: VscodeBtwEntry = base

  if (ev.kind === 'begin') {
    updated = {
      ...base,
      question: nextQuestion,
      ...(nextModel ? { model: nextModel } : {}),
      status: base.status === 'done' ? 'done' : 'streaming',
    }
  } else if (ev.kind === 'delta') {
    updated = {
      ...base,
      question: nextQuestion,
      answer: `${base.answer}${ev.delta ?? ''}`,
      ...(nextModel ? { model: nextModel } : {}),
      status: 'streaming',
    }
  } else if (ev.kind === 'tool' && ev.tool) {
    updated = {
      ...base,
      question: nextQuestion,
      ...(nextModel ? { model: nextModel } : {}),
      tools: mergeSideQuestionTool(base.tools, ev.tool),
      status: 'streaming',
    }
  } else if (ev.kind === 'end') {
    const settledTools = settleSideQuestionTools(base.tools, 'done')
    updated = {
      requestId: base.requestId,
      question: nextQuestion,
      answer: ev.answer !== undefined && ev.answer.length > 0 ? ev.answer : base.answer,
      ...(nextModel ? { model: nextModel } : {}),
      ...(settledTools ? { tools: settledTools } : {}),
      status: 'done',
    }
  } else if (ev.kind === 'error') {
    const settledTools = settleSideQuestionTools(base.tools, 'error')
    updated = {
      requestId: base.requestId,
      question: nextQuestion,
      answer: base.answer,
      ...(nextModel ? { model: nextModel } : {}),
      ...(settledTools ? { tools: settledTools } : {}),
      status: 'error',
      error: ev.error || 'Side question failed.',
    }
  }

  if (idx >= 0) {
    return entries.map((item, i) => (i === idx ? updated : item))
  }
  const next = [...entries, updated]
  return next.length > 40 ? next.slice(next.length - 40) : next
}

export interface SideQuestionCardProps {
  readonly snap: Snapshot
  readonly entries: readonly VscodeBtwEntry[]
  readonly defaultModelPickerOpen?: boolean
  readonly onEntriesChange: (updater: (prev: readonly VscodeBtwEntry[]) => readonly VscodeBtwEntry[]) => void
  readonly onClear: () => void
  readonly onClose: () => void
}

export function SideQuestionCard({
  snap,
  entries,
  defaultModelPickerOpen = false,
  onEntriesChange,
  onClear,
  onClose,
}: SideQuestionCardProps): ReactElement {
  const [draft, setDraft] = useState('')
  const [copied, setCopied] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const generationRef = useRef(0)
  const autoLoadedRef = useRef(false)

  const online = snap.connection === 'online'
  const streaming = entries.some(item => item.status === 'streaming')
  const latestDone = [...entries].reverse().find(item => item.answer.trim().length > 0)
  const selectedModel = snap.sideQuestionModel ?? ''
  const storeSnap = store.getSnapshot()
  const models = snap.models.length > 0 ? snap.models : storeSnap.models
  const modelsLoading = Boolean(snap.modelsLoading || storeSnap.modelsLoading)
  const modelsError = snap.modelsError ?? storeSnap.modelsError ?? null

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (!online) {
      autoLoadedRef.current = false
      return
    }
    if (models.length === 0 && !modelsLoading && !modelsError && !autoLoadedRef.current) {
      autoLoadedRef.current = true
      void store.loadModels()
    }
  }, [online, models.length, modelsLoading, modelsError])

  useEffect(() => {
    const node = listRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [entries])

  const submit = async (event?: FormEvent): Promise<void> => {
    event?.preventDefault()
    const question = stripLeadingBtwPrefix(draft)
    if (!question || !online || streaming) return

    const requestId = createVscodeBtwRequestId()
    const history = buildVscodeBtwHistory(entries)
    const submitGeneration = generationRef.current
    const chosenModel = selectedModel.trim()
    setDraft('')

    onEntriesChange(prev => [
      ...prev.filter(item => item.requestId !== requestId),
      {
        requestId,
        question,
        answer: '',
        ...(chosenModel || snap.model ? { model: chosenModel || snap.model } : {}),
        status: 'streaming',
      },
    ])

    try {
      const raw = await desktopCall(window.xerxes, snap.sessionKey, 'session.side_question', {
        ...(snap.currentId ? { session_id: snap.currentId } : {}),
        request_id: requestId,
        question,
        ...(chosenModel ? { model: chosenModel } : {}),
        ...(history.length > 0 ? { history } : {}),
      })
      if (generationRef.current !== submitGeneration) return
      const ok = raw.ok !== false
      const answer = typeof raw.answer === 'string' ? raw.answer : ''
      const model = typeof raw.model === 'string' && raw.model.trim() ? raw.model.trim() : undefined
      const errText = typeof raw.error === 'string' && raw.error.trim() ? raw.error.trim() : undefined

      onEntriesChange(prev =>
        prev.map(item => {
          if (item.requestId !== requestId) return item
          if (!ok) {
            const settledTools = settleSideQuestionTools(item.tools, 'error')
            return {
              requestId: item.requestId,
              question: item.question,
              answer: item.answer,
              ...(model ?? item.model ? { model: model ?? item.model } : {}),
              ...(settledTools ? { tools: settledTools } : {}),
              status: 'error',
              error: errText || 'Could not answer side question.',
            }
          }
          const settledTools = settleSideQuestionTools(item.tools, 'done')
          return {
            requestId: item.requestId,
            question: item.question,
            answer: answer || item.answer,
            ...(model ?? item.model ? { model: model ?? item.model } : {}),
            ...(settledTools ? { tools: settledTools } : {}),
            status: 'done',
          }
        }),
      )
    } catch (failure) {
      if (generationRef.current !== submitGeneration) return
      const message = desktopError(failure)
      onEntriesChange(prev =>
        prev.map(item => {
          if (item.requestId !== requestId) return item
          const settledTools = settleSideQuestionTools(item.tools, 'error')
          return {
            requestId: item.requestId,
            question: item.question,
            answer: item.answer,
            ...(item.model ? { model: item.model } : {}),
            ...(settledTools ? { tools: settledTools } : {}),
            status: 'error',
            error: message,
          }
        }),
      )
    }
  }

  const copyLatest = (): void => {
    if (!latestDone?.answer) return
    void navigator.clipboard?.writeText(latestDone.answer).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1_400)
      },
      () => undefined,
    )
  }

  const clearAll = (): void => {
    generationRef.current += 1
    onClear()
    setDraft('')
    inputRef.current?.focus()
  }

  return (
    <section
      className="xv-btw"
      aria-label="Side question"
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.preventDefault()
          event.stopPropagation()
          onClose()
        }
      }}
    >
      <header className="xv-btw__head">
        <span className="xv-btw__title">Side question</span>
        <span className="xv-btw__badge">ephemeral · read-only</span>
        <span className="xv-btw__flex" />
        <SideQuestionModelPicker
          models={models}
          modelsLoading={modelsLoading}
          modelsError={modelsError}
          sessionModel={snap.model}
          selectedModel={selectedModel}
          online={online}
          compact
          defaultOpen={defaultModelPickerOpen}
          triggerClassName="xv-btw__model-btn"
          onSelectModel={model => store.setSideQuestionModel(model)}
          onRequestModels={() => {
            if (online) void store.loadModels(models.length === 0)
          }}
        />
        {latestDone && (
          <button
            type="button"
            className="xv-btw__action"
            title="Copy latest answer"
            onClick={copyLatest}
          >
            <Icon name={copied ? 'check' : 'copy'} size={12} />
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        )}
        {entries.length > 0 && (
          <button
            type="button"
            className="xv-btw__action"
            title="Clear side thread"
            onClick={clearAll}
          >
            <span>Clear</span>
          </button>
        )}
        <button
          type="button"
          className="xv-btw__close"
          aria-label="Close side question"
          title="Close (Esc — main task keeps running)"
          onClick={onClose}
        >
          <Icon name="close" size={13} />
        </button>
      </header>

      <div className="xv-btw__body" ref={listRef}>
        {entries.length === 0 ? (
          <p className="xv-btw__empty">
            Ask an aside about this task or codebase — even while a turn is running. Nothing here is saved to the conversation.
          </p>
        ) : (
          entries.map(entry => {
            const hasRunningTool = Boolean(entry.tools?.some(tool => tool.status === 'running'))
            return (
              <article key={entry.requestId} className="xv-btw__turn">
                <div className="xv-btw__q">{entry.question}</div>
                <div className="xv-btw__a">
                  {entry.tools && entry.tools.length > 0 && (
                    <div className="side-question__tools xv-btw__tools" aria-label="Read-only tools used">
                      {entry.tools.map((tool, index) => {
                        const formatted = formatSideQuestionToolLabel(tool)
                        return (
                          <span
                            key={`${tool.name}:${tool.target ?? ''}:${index}`}
                            className="side-question__tool-pill"
                            data-status={tool.status}
                            title={tool.target ? `${formatted.verb} ${tool.target}` : formatted.verb}
                          >
                            <span className="side-question__tool-dot" aria-hidden="true" />
                            <span className="side-question__tool-verb">{formatted.verb}</span>
                            {formatted.target && (
                              <span className="side-question__tool-target">{formatted.target}</span>
                            )}
                          </span>
                        )
                      })}
                    </div>
                  )}
                  {entry.status === 'streaming' && !entry.answer ? (
                    <span className="xv-btw__thinking side-question__thinking">
                      <XerxesThinkingMark size={13} active />
                      <span>{hasRunningTool ? 'Inspecting files…' : 'Thinking…'}</span>
                    </span>
                  ) : entry.answer ? (
                    <Markdown
                      text={entry.answer}
                      className="md--compact"
                      tail={entry.status === 'streaming' ? '▍' : undefined}
                    />
                  ) : null}
                  {entry.status === 'error' && (
                    <p className="xv-btw__err" role="alert">
                      {entry.error || 'Side question failed.'}
                    </p>
                  )}
                  {entry.status === 'done' && entry.model && (
                    <span className="xv-btw__model side-question__model" title={entry.model}>
                      {bareModelName(entry.model)}
                    </span>
                  )}
                </div>
              </article>
            )
          })
        )}
      </div>

      <form className="xv-btw__form" onSubmit={event => void submit(event)}>
        <input
          ref={inputRef}
          className="xv-btw__input"
          aria-label="Ask a side question"
          placeholder={
            !online
              ? 'Connect to runtime first…'
              : entries.length > 0
                ? 'Ask a follow-up side question…'
                : 'Ask aside (/btw) — never interrupts or pollutes chat…'
          }
          spellCheck={false}
          value={draft}
          onChange={event => setDraft(event.target.value)}
        />
        <button
          type="submit"
          className="xv-btw__send"
          disabled={!online || streaming || !stripLeadingBtwPrefix(draft)}
          title="Ask side question (Enter)"
          aria-label="Send side question"
        >
          <Icon name="arrowUp" size={14} />
        </button>
      </form>
    </section>
  )
}
