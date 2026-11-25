// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Ephemeral Side Question (`/btw` / `/side` / `⌘;`) right-rail panel.
 *
 * Runs `session.side_question` concurrently alongside any active turn or swarm
 * via `QUEUE_BYPASS_METHODS`, supports choosing an independent model without
 * touching the main chat's model, renders read-only tool pills, and streams
 * live `'btw_event'` Markdown deltas without mutating session history.
 */

import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
} from 'react'

import { CopyButton, copyToClipboard } from './CopyButton.js'
import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { PickerLayer, bareModelName } from './Overlays.js'
import {
  applyBtwEventPayload,
  formatSideQuestionToolLabel,
  parseSideQuestionSlashCommand,
  type SideQuestionSlashAction,
  type SideQuestionToolEvent,
  type SideQuestionTurn,
} from './sideQuestionState.js'
import { store, type Snapshot } from './store.js'
import type { ModelChoice } from './types.js'
import { XerxesThinkingMark } from './XerxesSprites.js'

export {
  applyBtwEventPayload,
  formatSideQuestionToolLabel,
  parseSideQuestionSlashCommand,
  type SideQuestionSlashAction,
  type SideQuestionToolEvent,
  type SideQuestionTurn,
}

export interface SideQuestionModelOption {
  readonly id: string
  readonly label: string
  readonly provider: string
}

export interface SideQuestionModelPickerProps {
  readonly models?: readonly ModelChoice[] | undefined
  readonly modelsLoading?: boolean | undefined
  readonly modelsError?: string | null | undefined
  readonly sessionModel?: string | undefined
  readonly selectedModel?: string | undefined
  readonly online?: boolean | undefined
  readonly compact?: boolean | undefined
  readonly defaultOpen?: boolean | undefined
  readonly triggerClassName?: string | undefined
  readonly onSelectModel?: ((model: string) => void) | undefined
  readonly onRequestModels?: (() => void) | undefined
}

export interface SideQuestionViewProps {
  readonly turns: readonly SideQuestionTurn[]
  readonly online: boolean
  readonly streaming: boolean
  readonly turnActive: boolean
  readonly draft: string
  readonly copied: 'idle' | 'copied' | 'failed'
  readonly models?: readonly ModelChoice[]
  readonly modelsLoading?: boolean
  readonly modelsError?: string | null
  readonly sessionModel?: string
  readonly selectedModel?: string
  readonly defaultModelPickerOpen?: boolean
  readonly onSelectModel?: (model: string) => void
  readonly onRequestModels?: () => void
  readonly onChangeDraft: (value: string) => void
  readonly onSubmit: (event?: FormEvent) => void
  readonly onClear: () => void
  readonly onCopyAll: () => void
}

const STARTER_ASIDES: readonly string[] = [
  'Summarize what this session has changed so far',
  'Explain the architecture of the files in scope',
]

function sideQuestionShortcutLabel(): string {
  if (typeof navigator === 'undefined') return '⌘;'
  const ua = `${navigator.platform ?? ''} ${navigator.userAgent ?? ''}`
  return /Mac|iPhone|iPad|iPod/i.test(ua) ? '⌘;' : 'Ctrl+;'
}

export function buildSideQuestionModelChoices(
  models: readonly ModelChoice[] | undefined,
  selectedModel: string,
): readonly SideQuestionModelOption[] {
  const seen = new Set<string>()
  const rows: SideQuestionModelOption[] = []
  for (const choice of models ?? []) {
    const id = choice.id.trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    const label = choice.label?.trim() || bareModelName(id)
    const rawProvider = choice.provider?.trim() ?? ''
    const provider = rawProvider && rawProvider !== 'Models'
      ? rawProvider
      : id.includes('/')
        ? id.split('/')[0]!
        : ''
    rows.push({ id, label, provider })
  }
  const cleanSelected = selectedModel.trim()
  if (cleanSelected && !seen.has(cleanSelected)) {
    seen.add(cleanSelected)
    rows.unshift({
      id: cleanSelected,
      label: bareModelName(cleanSelected),
      provider: cleanSelected.includes('/') ? cleanSelected.split('/')[0]! : 'custom',
    })
  }
  return rows
}

export function SideQuestionModelPicker({
  models,
  modelsLoading = false,
  modelsError = null,
  sessionModel = '',
  selectedModel = '',
  online = true,
  compact = false,
  defaultOpen = false,
  triggerClassName,
  onSelectModel,
  onRequestModels,
}: SideQuestionModelPickerProps): ReactElement {
  const [open, setOpen] = useState(defaultOpen)
  const [query, setQuery] = useState('')
  const [customDraft, setCustomDraft] = useState('')
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const cleanSelected = selectedModel.trim()
  const cleanSession = sessionModel.trim()
  const sessionModelShort = cleanSession ? bareModelName(cleanSession) : 'active'

  const modelOptions = useMemo(
    () => buildSideQuestionModelChoices(models, cleanSelected),
    [models, cleanSelected],
  )
  const activeChoice = useMemo(
    () => (cleanSelected ? modelOptions.find(opt => opt.id === cleanSelected) : undefined),
    [modelOptions, cleanSelected],
  )
  const triggerLabel = cleanSelected
    ? activeChoice?.label ?? bareModelName(cleanSelected)
    : cleanSession
      ? sessionModelShort
      : 'Session default'

  const showSearch = modelOptions.length > 6
  const needle = query.trim().toLowerCase()
  const filteredOptions = useMemo(() => {
    if (!needle) return modelOptions
    return modelOptions.filter(
      opt =>
        opt.id.toLowerCase().includes(needle)
        || opt.label.toLowerCase().includes(needle)
        || opt.provider.toLowerCase().includes(needle),
    )
  }, [modelOptions, needle])
  const showSessionDefault = !needle
    || 'session default'.includes(needle)
    || cleanSession.toLowerCase().includes(needle)
    || sessionModelShort.toLowerCase().includes(needle)

  useLayoutEffect(() => {
    if (!open || typeof window === 'undefined') return
    const popup = popoverRef.current
    const anchor = triggerRef.current
    if (!popup || !anchor) return
    const place = (): void => {
      const rect = anchor.getBoundingClientRect()
      popup.style.position = 'fixed'
      popup.style.right = 'auto'
      popup.style.bottom = 'auto'
      popup.style.maxWidth = 'calc(100vw - 24px)'
      popup.style.maxHeight = `${Math.max(160, window.innerHeight - 24)}px`
      const width = popup.getBoundingClientRect().width || 284
      const height = popup.getBoundingClientRect().height || 240
      const preferredLeft = compact ? rect.right - width : rect.left
      const left = Math.max(12, Math.min(preferredLeft, window.innerWidth - width - 12))
      const aboveTop = rect.top - height - 6
      const belowTop = rect.bottom + 6
      const top = aboveTop >= 12
        ? aboveTop
        : belowTop + height <= window.innerHeight - 12
          ? belowTop
          : Math.max(12, Math.min(aboveTop, window.innerHeight - height - 12))
      popup.style.left = `${left}px`
      popup.style.top = `${top}px`
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    observer?.observe(popup)
    observer?.observe(anchor)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open, compact, filteredOptions.length, modelsLoading])

  useEffect(() => {
    if (!open) return
    if (showSearch) searchInputRef.current?.focus()
    else popoverRef.current?.focus()
  }, [open, showSearch])

  useEffect(() => {
    if (!open || typeof window === 'undefined') return
    const onEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onEscape, true)
    return () => window.removeEventListener('keydown', onEscape, true)
  }, [open])

  const close = (): void => {
    setOpen(false)
    triggerRef.current?.focus()
  }

  const pick = (modelId: string): void => {
    onSelectModel?.(modelId)
    setOpen(false)
    setQuery('')
    setCustomDraft('')
    triggerRef.current?.focus()
  }

  const applyCustom = (): void => {
    const clean = customDraft.trim()
    if (!clean) return
    pick(clean)
  }

  const toggleOpen = (): void => {
    if (open) {
      setOpen(false)
      return
    }
    setQuery('')
    setCustomDraft('')
    setOpen(true)
    onRequestModels?.()
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={`side-question__model-btn${cleanSelected ? ' is-custom' : ''}${triggerClassName ? ` ${triggerClassName}` : ''}`}
        aria-label="Side question model"
        aria-haspopup="dialog"
        aria-expanded={open}
        title={
          cleanSelected
            ? `Side question model: ${cleanSelected} (change without affecting main chat)`
            : `Side question model: Session default (${sessionModelShort})`
        }
        onClick={toggleOpen}
      >
        <Icon name="spark" size={compact ? 10 : 11} />
        <span className="side-question__model-label">{triggerLabel}</span>
        <span className="side-question__model-caret" aria-hidden="true">
          <Icon name="caretDown" size={9} />
        </span>
      </button>

      {open && (
        <PickerLayer>
          <div className="backdrop backdrop--clear" onClick={close} />
          <div
            ref={popoverRef}
            className="btw-model-pop"
            role="dialog"
            aria-label="Side question model"
            tabIndex={-1}
            onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                event.stopPropagation()
                close()
                return
              }
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Home' || event.key === 'End') {
                const target = event.target as HTMLElement | null
                if (target?.classList.contains('btw-model-pop__custom-input')) return
                const items = Array.from(popoverRef.current?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? [])
                if (items.length === 0) return
                event.preventDefault()
                const cur = items.indexOf(document.activeElement as HTMLButtonElement)
                const next =
                  event.key === 'Home' ? 0
                  : event.key === 'End' ? items.length - 1
                  : event.key === 'ArrowDown' ? (cur < 0 ? 0 : (cur + 1) % items.length)
                  : (cur <= 0 ? items.length - 1 : cur - 1)
                items[next]?.focus()
              }
            }}
          >
            <header className="btw-model-pop__head">
              <span className="btw-model-pop__title">Side question model</span>
              <button
                type="button"
                className="btw-model-pop__close"
                aria-label="Close side question model picker"
                onClick={close}
              >
                <Icon name="close" size={11} />
              </button>
            </header>

            {showSearch && (
              <label className="btw-model-pop__search">
                <Icon name="search" size={12} />
                <input
                  ref={searchInputRef}
                  type="search"
                  className="btw-model-pop__filter"
                  aria-label="Filter side question models"
                  placeholder="Filter models…"
                  spellCheck={false}
                  value={query}
                  onChange={event => setQuery(event.target.value)}
                />
              </label>
            )}

            <div
              className="btw-model-pop__list"
              role="listbox"
              aria-label="Available side question models"
            >
              {showSessionDefault && (
                <button
                  type="button"
                  role="option"
                  aria-selected={!cleanSelected}
                  className={`btw-model-pop__item btw-model-pop__item--default${!cleanSelected ? ' is-active' : ''}`}
                  title={`Follow the active session model (${sessionModelShort})`}
                  onClick={() => pick('')}
                >
                  <span className="btw-model-pop__check" aria-hidden="true">
                    {!cleanSelected ? <Icon name="check" size={12} /> : null}
                  </span>
                  <span className="btw-model-pop__name">Session default</span>
                  <span className="btw-model-pop__tag">{sessionModelShort}</span>
                </button>
              )}

              {showSessionDefault && (filteredOptions.length > 0 || modelsLoading) && (
                <div className="btw-model-pop__sep" role="separator" />
              )}

              {modelsLoading && modelOptions.length === 0 && (
                <p className="btw-model-pop__status" role="status">
                  Loading models…
                </p>
              )}

              {!modelsLoading && modelOptions.length === 0 && (
                <p className="btw-model-pop__empty">
                  {modelsError
                    ? modelsError
                    : online
                      ? 'No discovered models yet. Enter a custom model ID below.'
                      : 'Offline. Enter a custom model ID below.'}
                </p>
              )}

              {filteredOptions.length === 0 && modelOptions.length > 0 && (
                <p className="btw-model-pop__empty">
                  No matching models. Enter a custom model ID below.
                </p>
              )}

              {filteredOptions.map(opt => {
                const isActive = cleanSelected === opt.id
                return (
                  <button
                    key={opt.id}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    className={`btw-model-pop__item${isActive ? ' is-active' : ''}`}
                    title={opt.id}
                    onClick={() => pick(opt.id)}
                  >
                    <span className="btw-model-pop__check" aria-hidden="true">
                      {isActive ? <Icon name="check" size={12} /> : null}
                    </span>
                    <span className="btw-model-pop__name">{opt.label}</span>
                    {opt.provider && (
                      <span className="btw-model-pop__tag">{opt.provider}</span>
                    )}
                  </button>
                )
              })}
            </div>

            <div className="btw-model-pop__custom">
              <input
                type="text"
                className="btw-model-pop__custom-input"
                aria-label="Custom side question model ID"
                placeholder="Custom model ID…"
                spellCheck={false}
                value={customDraft}
                onChange={event => setCustomDraft(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    event.stopPropagation()
                    applyCustom()
                  } else if (event.key === 'Escape') {
                    event.preventDefault()
                    event.stopPropagation()
                    close()
                  }
                }}
              />
              <button
                type="button"
                className="btw-model-pop__custom-submit"
                aria-label="Apply custom side question model"
                title="Use custom model ID (↵)"
                disabled={!customDraft.trim()}
                onClick={applyCustom}
              >
                <Icon name="enter" size={12} />
              </button>
            </div>
          </div>
        </PickerLayer>
      )}
    </>
  )
}

export const SIDE_QUESTION_SCROLL_PIN_THRESHOLD = 48

export function isSideQuestionNearBottom(
  metrics: { readonly scrollTop: number; readonly clientHeight: number; readonly scrollHeight: number },
  threshold = SIDE_QUESTION_SCROLL_PIN_THRESHOLD,
): boolean {
  const maxTop = Math.max(0, metrics.scrollHeight - metrics.clientHeight)
  if (maxTop <= 2) return true
  const clampedTop = Math.min(Math.max(0, metrics.scrollTop), maxTop)
  return maxTop - clampedTop <= threshold
}

export function SideQuestionView(props: SideQuestionViewProps): ReactElement {
  const {
    turns,
    online,
    streaming,
    turnActive,
    draft,
    copied,
    models,
    modelsLoading = false,
    modelsError = null,
    sessionModel = '',
    selectedModel = '',
    defaultModelPickerOpen = false,
    onSelectModel,
    onRequestModels,
    onChangeDraft,
    onSubmit,
    onClear,
    onCopyAll,
  } = props

  const threadRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const pinnedRef = useRef(true)
  const [pinned, setPinned] = useState(true)
  const lastScrollTopRef = useRef(0)
  const prevTurnsLenRef = useRef(turns.length)

  const setPinnedState = (next: boolean): void => {
    if (pinnedRef.current !== next) {
      pinnedRef.current = next
      setPinned(next)
    }
  }

  useEffect(() => {
    const el = threadRef.current
    if (!el) return
    const onWheel = (event: WheelEvent): void => {
      if (event.deltaY < 0 && el.scrollHeight > el.clientHeight + 2) {
        setPinnedState(false)
      }
    }
    const onScroll = (): void => {
      const maxTop = Math.max(0, el.scrollHeight - el.clientHeight)
      const clampedTop = Math.min(Math.max(0, el.scrollTop), maxTop)
      const clampedPrev = Math.min(Math.max(0, lastScrollTopRef.current), maxTop)
      lastScrollTopRef.current = clampedTop
      if (maxTop <= 2 || maxTop - clampedTop <= 2) {
        setPinnedState(true)
      } else if (clampedTop < clampedPrev - 1) {
        setPinnedState(false)
      } else if (clampedTop > clampedPrev && isSideQuestionNearBottom(el, SIDE_QUESTION_SCROLL_PIN_THRESHOLD)) {
        setPinnedState(true)
      }
    }
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('scroll', onScroll)
    }
  }, [])

  useEffect(() => {
    const el = threadRef.current
    if (!el) return
    if (turns.length === 0 || turns.length > prevTurnsLenRef.current) {
      setPinnedState(true)
    }
    prevTurnsLenRef.current = turns.length
    if (pinnedRef.current) {
      el.scrollTop = el.scrollHeight
      lastScrollTopRef.current = el.scrollTop
    }
  }, [turns.length, turns.at(-1)?.answer, turns.at(-1)?.status, turns.at(-1)?.tools?.length])

  const jumpToLatest = (): void => {
    const el = threadRef.current
    setPinnedState(true)
    if (!el) return
    el.scrollTop = el.scrollHeight
    lastScrollTopRef.current = el.scrollTop
  }

  return (
    <section className="side-question" aria-label="Side question">
      <header className="side-question__bar">
        <div className="side-question__meta">
          <span className="side-question__badge">Ephemeral · read-only</span>
          {turnActive && <span className="side-question__parallel">Main task continues</span>}
        </div>
        {turns.length > 0 && (
          <div className="side-question__actions">
            <button
              type="button"
              className="rsec__icon"
              data-copied={copied}
              aria-label={copied === 'copied' ? 'Copied' : 'Copy side question thread'}
              title={copied === 'copied' ? 'Copied' : 'Copy thread'}
              onClick={onCopyAll}
            >
              <Icon name={copied === 'copied' ? 'check' : 'copy'} size={13} />
            </button>
            <button
              type="button"
              className="rsec__text"
              aria-label="Clear side questions"
              title="Clear ephemeral thread"
              disabled={streaming}
              onClick={onClear}
            >
              Clear
            </button>
          </div>
        )}
      </header>

      <div className="side-question__thread" ref={threadRef} role="log" aria-label="Side question responses">
        {turns.length === 0 ? (
          <div className="side-question__empty">
            <p className="side-question__lead">
              Ask a quick question about this conversation or workspace without interrupting a running task or adding to the chat history.
            </p>
            <div className="side-question__starters">
              {STARTER_ASIDES.map(prompt => (
                <button
                  key={prompt}
                  type="button"
                  className="side-question__starter"
                  disabled={!online || streaming}
                  onClick={() => {
                    onChangeDraft(prompt)
                    inputRef.current?.focus()
                  }}
                >
                  <span>{prompt}</span>
                  <Icon name="arrow" size={12} />
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map(turn => {
            const hasRunningTool = Boolean(turn.tools?.some(tool => tool.status === 'running'))
            return (
              <article
                key={turn.requestId}
                className="side-question__turn"
                data-status={turn.status}
              >
                <div className="side-question__q">{turn.question}</div>
                <div className="side-question__a">
                  {turn.tools && turn.tools.length > 0 && (
                    <div className="side-question__tools" aria-label="Read-only tools used">
                      {turn.tools.map((tool, index) => {
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
                  {turn.answer ? (
                    <Markdown
                      text={turn.answer}
                      className="md--agent side-question__md"
                      tail={turn.status === 'streaming' ? '▍' : undefined}
                    />
                  ) : turn.status === 'streaming' ? (
                    <p className="side-question__thinking" role="status">
                      <XerxesThinkingMark size={13} active />
                      <span>{hasRunningTool ? 'Inspecting files…' : 'Reading context…'}</span>
                    </p>
                  ) : turn.status === 'done' ? (
                    <p className="side-question__empty-reply">No response was returned.</p>
                  ) : null}
                  {turn.status === 'error' && (
                    <p className="side-question__error studio-error" role="alert">
                      {turn.error || 'Side question failed'}
                    </p>
                  )}
                  {turn.status === 'streaming' && turn.answer && (
                    <div className="side-question__turn-foot">
                      <span className="side-question__streaming-tag">
                        <XerxesThinkingMark size={11} active />
                        {turn.model ? bareModelName(turn.model) : 'Streaming…'}
                      </span>
                    </div>
                  )}
                  {turn.status === 'done' && turn.answer && (
                    <div className="side-question__turn-foot">
                      {turn.model && (
                        <span className="side-question__model" title={turn.model}>
                          {bareModelName(turn.model)}
                        </span>
                      )}
                      <CopyButton icon text={turn.answer} label="Copy answer" />
                    </div>
                  )}
                </div>
              </article>
            )
          })
        )}
        {turns.length > 0 && !pinned && (
          <button
            type="button"
            className="rsec__text side-question__jump"
            aria-label="Jump to latest side question output"
            onClick={jumpToLatest}
          >
            <Icon name="arrowDown" size={11} /> Latest
          </button>
        )}
      </div>

      <form
        className="side-question__composer"
        aria-label="Ask a side question"
        onSubmit={onSubmit}
      >
        <div className="side-question__composer-box">
          <textarea
            ref={inputRef}
            rows={2}
            className="side-question__input"
            aria-label="Side question prompt"
            placeholder={
              !online
                ? 'Connect to workspace runtime…'
                : streaming
                  ? 'Answering side question…'
                  : 'Ask a quick question…'
            }
            value={draft}
            disabled={!online || streaming}
            onChange={event => onChangeDraft(event.target.value)}
            onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
              if (event.nativeEvent.isComposing) return
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                onSubmit()
              } else if (event.key === 'Escape') {
                // Never let Escape inside Side Question bubble to GlobalKeys turn-stop!
                event.preventDefault()
                event.stopPropagation()
                if (draft) onChangeDraft('')
                else event.currentTarget.blur()
              }
            }}
          />
          <div className="side-question__toolbar">
            <SideQuestionModelPicker
              models={models}
              modelsLoading={modelsLoading}
              modelsError={modelsError}
              sessionModel={sessionModel}
              selectedModel={selectedModel}
              online={online}
              defaultOpen={defaultModelPickerOpen}
              onSelectModel={onSelectModel}
              onRequestModels={onRequestModels}
            />
            <span className="side-question__shortcut" title="Toggle Side Question panel">{sideQuestionShortcutLabel()}</span>
            <button
              type="submit"
              className="side-question__send"
              aria-label="Send side question"
              title="Ask aside (⏎)"
              disabled={!online || streaming || !draft.trim()}
            >
              <Icon name="arrowUp" size={14} />
            </button>
          </div>
        </div>
      </form>
    </section>
  )
}

export interface SideQuestionPanelProps {
  readonly snap: Snapshot
  readonly initialDraft?: string
  readonly defaultModelPickerOpen?: boolean
  readonly onAsk?: (question: string, model?: string) => Promise<boolean> | boolean
  readonly onClear?: () => void
  readonly onSelectModel?: (model: string) => void
  readonly onLoadModels?: () => void
}

export function SideQuestionPanel({
  snap,
  initialDraft = '',
  defaultModelPickerOpen = false,
  onAsk,
  onClear,
  onSelectModel,
  onLoadModels,
}: SideQuestionPanelProps): ReactElement {
  const [draft, setDraft] = useState(initialDraft)
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle')
  const copiedTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const autoLoadedRef = useRef(false)
  useEffect(() => () => clearTimeout(copiedTimer.current), [])

  const turns = snap.sideQuestionTurns ?? []
  const streaming = Boolean(snap.sideQuestionStreaming)
  const online = snap.connection === 'online'
  const selectedModel = snap.sideQuestionModel ?? ''
  const storeSnap = store.getSnapshot()
  const nativeModels = snap.models.length > 0 ? snap.models : storeSnap.models
  const acpBinding =
    snap.acpBinding?.connected && snap.acpBinding.transport !== 'native'
      ? snap.acpBinding
      : storeSnap.acpBinding?.connected && storeSnap.acpBinding.transport !== 'native'
        ? storeSnap.acpBinding
        : null
  const acpModels: ModelChoice[] = acpBinding?.models?.map(item => ({
    id: item.id,
    label: item.name || item.id,
    provider: acpBinding.name,
  })) ?? []
  // When connected to an external ACP runtime (e.g. Antigravity / Gemini ACP), keep /btw
  // strictly isolated to that runtime's models — never mix in native Xerxes provider models.
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

  useEffect(() => {
    if (!online || acpBinding) {
      autoLoadedRef.current = false
      return
    }
    if (models.length === 0 && !modelsLoading && !modelsError && !autoLoadedRef.current) {
      autoLoadedRef.current = true
      if (onLoadModels) onLoadModels()
      else void store.loadModels()
    }
  }, [online, acpBinding, models.length, modelsLoading, modelsError, onLoadModels])

  const requestModels = (): void => {
    if (!online || acpBinding) return
    if (onLoadModels) {
      onLoadModels()
      return
    }
    void store.loadModels(models.length === 0)
  }

  const submit = async (event?: FormEvent): Promise<void> => {
    event?.preventDefault()
    const clean = draft.trim()
    if (!clean || !online || streaming) return
    setDraft('')
    const chosen = effectiveSelectedModel.trim() || undefined
    const ok = await (onAsk ? onAsk(clean, chosen) : store.askSideQuestion(clean, chosen))
    if (!ok) setDraft(clean)
  }

  const copyAll = (): void => {
    const formatted = turns
      .map(turn => `Q: ${turn.question}\n\n${turn.answer || turn.error || ''}`.trim())
      .join('\n\n---\n\n')
    if (!formatted) return
    void copyToClipboard(formatted).then(ok => {
      setCopied(ok ? 'copied' : 'failed')
      clearTimeout(copiedTimer.current)
      copiedTimer.current = setTimeout(() => setCopied('idle'), 1400)
    })
  }

  return (
    <SideQuestionView
      turns={turns}
      online={online}
      streaming={streaming}
      turnActive={snap.turnActive}
      draft={draft}
      copied={copied}
      models={models}
      modelsLoading={modelsLoading}
      modelsError={modelsError}
      sessionModel={sessionModel}
      selectedModel={effectiveSelectedModel}
      defaultModelPickerOpen={defaultModelPickerOpen}
      onSelectModel={model => (onSelectModel ? onSelectModel(model) : store.setSideQuestionModel(model))}
      onRequestModels={requestModels}
      onChangeDraft={setDraft}
      onSubmit={event => void submit(event)}
      onClear={() => (onClear ? onClear() : store.clearSideQuestion())}
      onCopyAll={copyAll}
    />
  )
}
