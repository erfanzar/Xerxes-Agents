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
  readonly entries?: readonly VscodeBtwEntry[]
  readonly defaultModelPickerOpen?: boolean
  readonly onEntriesChange?: (updater: (prev: readonly VscodeBtwEntry[]) => readonly VscodeBtwEntry[]) => void
  readonly onClear?: () => void
  readonly onClose?: () => void
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
  const resolvedEntries: readonly VscodeBtwEntry[] = entries ?? snap.sideQuestionTurns ?? []
  const streaming = entries
    ? resolvedEntries.some(item => item.status === 'streaming')
    : Boolean(snap.sideQuestionStreaming || resolvedEntries.some(item => item.status === 'streaming'))
  const cardError =
    !entries && snap.sideQuestionError && !resolvedEntries.some(item => item.status === 'error')
      ? snap.sideQuestionError
      : null
  const latestDone = [...resolvedEntries].reverse().find(item => item.answer.trim().length > 0)
  const selectedModel = snap.sideQuestionModel ?? ''
  const storeSnap = store.getSnapshot()
  const nativeModels = snap.models.length > 0 ? snap.models : storeSnap.models
  const acpBinding =
    snap.acpBinding?.connected && snap.acpBinding.transport !== 'native'
      ? snap.acpBinding
      : storeSnap.acpBinding?.connected && storeSnap.acpBinding.transport !== 'native'
        ? storeSnap.acpBinding
        : null
  const acpModels = acpBinding?.models?.map(item => ({
    id: item.id,
    label: item.name || item.id,
    provider: acpBinding.name,
  })) ?? []
  const models = acpBinding ? acpModels : nativeModels
  const effectiveSelectedModel =
    acpBinding &&
    selectedModel &&
    acpModels.length > 0 &&
    !acpModels.some(entry => entry.id === selectedModel)
      ? ''
      : selectedModel
  const sessionModel = acpBinding?.current_model || snap.model
  const modelsLoading = acpBinding ? false : Boolean(snap.modelsLoading || storeSnap.modelsLoading)
  const modelsError = acpBinding ? null : (snap.modelsError ?? storeSnap.modelsError ?? null)

  const closeCard = (): void => {
    if (onClose) onClose()
    else store.closeSideQuestion()
  }
  const closeRef = useRef(closeCard)
  closeRef.current = closeCard

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Intercept ⌘; / Ctrl+; in capture phase while the card is mounted so it
  // closes cleanly before GlobalKeys' bubble listener calls openSideQuestion().
  useEffect(() => {
    if (typeof window === 'undefined') return
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented) return
      if (typeof document !== 'undefined' && document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]')) return
      if (event.key === ';' && (event.metaKey || event.ctrlKey) && !event.altKey) {
        event.preventDefault()
        event.stopPropagation()
        closeRef.current()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [])

  useEffect(() => {
    if (!online || acpBinding) {
      autoLoadedRef.current = false
      return
    }
    if (models.length === 0 && !modelsLoading && !modelsError && !autoLoadedRef.current) {
      autoLoadedRef.current = true
      void store.loadModels()
    }
  }, [online, acpBinding, models.length, modelsLoading, modelsError])

  const pinnedRef = useRef(true)
  const lastScrollTopRef = useRef(0)
  const prevEntriesLenRef = useRef(resolvedEntries.length)

  useEffect(() => {
    const node = listRef.current
    if (!node) return
    const onWheel = (event: WheelEvent): void => {
      if (event.deltaY < 0 && node.scrollHeight > node.clientHeight + 2) {
        pinnedRef.current = false
      }
    }
    const onScroll = (): void => {
      const maxTop = Math.max(0, node.scrollHeight - node.clientHeight)
      const clampedTop = Math.min(Math.max(0, node.scrollTop), maxTop)
      const clampedPrev = Math.min(Math.max(0, lastScrollTopRef.current), maxTop)
      lastScrollTopRef.current = clampedTop
      if (maxTop <= 2 || maxTop - clampedTop <= 2) {
        pinnedRef.current = true
      } else if (clampedTop < clampedPrev - 1) {
        pinnedRef.current = false
      } else if (clampedTop > clampedPrev && maxTop - clampedTop <= 24) {
        pinnedRef.current = true
      }
    }
    node.addEventListener('wheel', onWheel, { passive: true })
    node.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      node.removeEventListener('wheel', onWheel)
      node.removeEventListener('scroll', onScroll)
    }
  }, [])

  useEffect(() => {
    const node = listRef.current
    if (!node) return
    if (resolvedEntries.length > prevEntriesLenRef.current) {
      pinnedRef.current = true
    }
    prevEntriesLenRef.current = resolvedEntries.length
    if (pinnedRef.current) {
      node.scrollTop = node.scrollHeight
      lastScrollTopRef.current = node.scrollTop
    }
  }, [resolvedEntries])

  const submit = async (event?: FormEvent): Promise<void> => {
    event?.preventDefault()
    const question = stripLeadingBtwPrefix(draft)
    if (!question || !online || streaming) return

    const chosenModel = effectiveSelectedModel.trim()
    setDraft('')

    if (!onEntriesChange) {
      await store.askSideQuestion(question, chosenModel || undefined)
      return
    }

    const requestId = createVscodeBtwRequestId()
    const history = buildVscodeBtwHistory(resolvedEntries)
    const submitGeneration = generationRef.current

    onEntriesChange(prev => [
      ...prev.filter(item => item.requestId !== requestId),
      {
        requestId,
        question,
        answer: '',
        ...(chosenModel || sessionModel ? { model: chosenModel || sessionModel } : {}),
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
    if (onClear) onClear()
    else store.clearSideQuestion()
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
          closeCard()
        }
      }}
    >
      <header className="xv-btw__head">
        <span className="xv-btw__title">Side question</span>
        <span className="xv-btw__badge">ephemeral · read-only</span>
        <span className="xv-btw__flex" />
        <div className="xv-btw__actions">
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
          {resolvedEntries.length > 0 && (
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
            onClick={closeCard}
          >
            <Icon name="close" size={13} />
          </button>
        </div>
      </header>

      <div className="xv-btw__bar">
        <SideQuestionModelPicker
          models={models}
          modelsLoading={modelsLoading}
          modelsError={modelsError}
          sessionModel={sessionModel}
          selectedModel={effectiveSelectedModel}
          online={online}
          compact
          defaultOpen={defaultModelPickerOpen}
          triggerClassName="xv-btw__model-btn"
          onSelectModel={model => store.setSideQuestionModel(model)}
          onRequestModels={() => {
            if (online && !acpBinding) void store.loadModels(models.length === 0)
          }}
        />
      </div>

      <div className="xv-btw__body" ref={listRef}>
        {resolvedEntries.length === 0 ? (
          <p className="xv-btw__empty">
            Ask an aside about this task or codebase — even while a turn is running. Nothing here is saved to the conversation.
          </p>
        ) : (
          resolvedEntries.map(entry => {
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
        {cardError && (
          <p className="xv-btw__err" role="alert">
            {cardError}
          </p>
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
              : resolvedEntries.length > 0
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
