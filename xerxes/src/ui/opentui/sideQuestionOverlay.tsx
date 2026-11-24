// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { focusComposer } from '../app/composerFocus.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import {
  $sideQuestionStore,
  applyBtwEvent,
  askSideQuestion,
  clearSideQuestions,
  getSideQuestionSliceForSession,
  parseSideQuestionInput,
  setSideQuestionModel,
  type SideQuestionEntry
} from '../app/sideQuestionStore.js'
import { $uiSessionId, $uiState } from '../app/uiStore.js'
import type { GatewayEvent, ModelOptionsResponse } from '../gatewayTypes.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { isPageDownKey, isPageUpKey } from '../lib/pageKeys.js'
import type { Theme } from '../theme.js'

import { DialogEmpty, DialogFooter, DialogHeader } from './dialogChrome.js'
import { StreamingMarkdown } from './messageLine.js'
import { OVERLAY_PANEL_SPECS, overlayPanelSize, windowItems } from './overlayLayout.js'
import { Box, Span, Text } from './primitives.js'

export interface SideQuestionModelOption {
  readonly id: string
  readonly label: string
  readonly provider?: string
}

export function collectSideQuestionModelOptions(
  response: ModelOptionsResponse | null | undefined,
  sessionModel: string,
  selectedModel: string
): readonly SideQuestionModelOption[] {
  const seen = new Set<string>()
  const out: SideQuestionModelOption[] = [
    {
      id: '',
      label: sessionModel ? `Session default (${sessionModel})` : 'Session default'
    }
  ]
  const add = (value: string | undefined, provider?: string) => {
    const trimmed = typeof value === 'string' ? value.trim() : ''
    if (!trimmed || seen.has(trimmed)) return
    seen.add(trimmed)
    out.push({
      id: trimmed,
      label: trimmed,
      ...(provider ? { provider } : {})
    })
  }
  add(selectedModel)
  add(sessionModel)
  add(response?.model)
  for (const provider of response?.providers ?? []) {
    const providerName = provider.name || provider.slug
    add(provider.configured_model, providerName)
    for (const model of provider.models ?? []) {
      add(model, providerName)
    }
  }
  return out
}

export function filterSideQuestionModels(
  options: readonly SideQuestionModelOption[],
  query: string
): readonly SideQuestionModelOption[] {
  const needle = query.trim()
  if (!needle) return options
  const ranked = fuzzyRank(options, needle, item => `${item.label} ${item.provider ?? ''}`)
  if (ranked.length > 0) return ranked.map(entry => entry.item)
  const lower = needle.toLowerCase()
  return options.filter(
    item =>
      item.label.toLowerCase().includes(lower) ||
      item.id.toLowerCase().includes(lower) ||
      (item.provider?.toLowerCase().includes(lower) ?? false)
  )
}

function SideQuestionTurnRow({ entry, t }: { entry: SideQuestionEntry; t: Theme }) {
  return (
    <Box flexDirection="column" flexShrink={0} marginBottom={1}>
      <Text wrap="wrap">
        <Span bold color={t.color.accent}>
          {'You  '}
        </Span>
        <Span bold color={t.color.text}>
          {entry.question}
        </Span>
      </Text>
      <Box flexDirection="column" minWidth={0} paddingLeft={2}>
        {entry.tools && entry.tools.length > 0 ? (
          <Text color={t.ds.secondary} wrap="wrap">
            {entry.tools
              .map(tool => {
                const icon = tool.status === 'running' ? '…' : tool.status === 'error' ? '!' : '✓'
                return `[${tool.name}${tool.target ? ` ${tool.target}` : ''} ${icon}]`
              })
              .join(' ')}
          </Text>
        ) : null}
        {entry.status === 'streaming' && !entry.answer ? (
          <Text color={t.ds.secondary}>Thinking…</Text>
        ) : entry.answer ? (
          <Box flexDirection="column" flexShrink={0}>
            <StreamingMarkdown t={t} text={entry.answer} />
            {entry.status === 'streaming' ? <Text color={t.color.accent}>▍</Text> : null}
          </Box>
        ) : null}
        {entry.status === 'error' ? (
          <Text color={t.color.warn} wrap="wrap">
            {`! ${entry.error || 'Side question failed.'}`}
          </Text>
        ) : null}
        {entry.status === 'done' && entry.model ? (
          <Text color={t.ds.meta} wrap="truncate-end">
            {entry.model}
          </Text>
        ) : null}
      </Box>
    </Box>
  )
}

export function SideQuestionOverlay({ sessionId, t }: { sessionId?: string | null; t: Theme }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const uiSid = useStore($uiSessionId)
  const uiState = useStore($uiState)
  const mainTurnBusy = uiState.busy
  const sessionModel = uiState.info?.model?.trim() ?? ''
  const store = useStore($sideQuestionStore)

  const sid = sessionId?.trim() || uiSid?.trim() || ''
  const slice = getSideQuestionSliceForSession(store, sid)
  const entries = slice.entries
  const isStreaming = slice.activeRequestId !== null
  const selectedModel = slice.selectedModel ?? ''

  const size = overlayPanelSize(terminal, {
    ...OVERLAY_PANEL_SPECS.sideQuestion,
    maxHeight: 36,
    minHeight: 14
  })
  const compact = size.width < 64 || size.height < 20

  const [composing, setComposing] = useState<boolean>(() => entries.length === 0)
  const [draft, setDraft] = useState('')
  const [tailOffset, setTailOffset] = useState(0)
  const [localError, setLocalError] = useState('')
  const [modelPickerOpen, setModelPickerOpen] = useState(false)
  const [modelFilter, setModelFilter] = useState('')
  const [modelCursor, setModelCursor] = useState(0)
  const [modelChoices, setModelChoices] = useState<readonly SideQuestionModelOption[] | null>(null)
  const [loadingModels, setLoadingModels] = useState(false)
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)

  const fallbackChoices = useMemo(
    () => collectSideQuestionModelOptions(null, sessionModel, selectedModel),
    [sessionModel, selectedModel]
  )
  const availableModels = modelChoices ?? fallbackChoices
  const filteredModels = useMemo(
    () => filterSideQuestionModels(availableModels, modelFilter),
    [availableModels, modelFilter]
  )

  useEffect(() => {
    if (entries.length === 0 && !composing && !modelPickerOpen) {
      setComposing(true)
    }
  }, [entries.length, composing, modelPickerOpen])

  useEffect(() => {
    if (modelCursor >= filteredModels.length) {
      setModelCursor(Math.max(0, filteredModels.length - 1))
    }
  }, [filteredModels.length, modelCursor])

  // Subscribe directly to live 'btw_event' stream frames when a GatewayClient emitter is present.
  useEffect(() => {
    const gw = gateway?.gw as unknown as {
      on?: (event: string, fn: (ev: GatewayEvent) => void) => void
      off?: (event: string, fn: (ev: GatewayEvent) => void) => void
    } | undefined
    if (!gw?.on || !gw?.off) return
    const onBtwEvent = (ev: GatewayEvent) => {
      if (ev.type === 'btw_event') {
        const eventSid = ev.payload?.session_id || ev.session_id || sid
        if (!sid || !eventSid || eventSid === sid) {
          applyBtwEvent(ev.payload, sid)
        }
      }
    }
    gw.on('btw_event', onBtwEvent)
    return () => {
      gw.off?.('btw_event', onBtwEvent)
    }
  }, [gateway?.gw, sid])

  const closeOverlay = () => {
    patchOverlayState({ sideQuestion: false })
    focusComposer()
  }

  const handleClear = () => {
    clearSideQuestions(sid)
    setDraft('')
    setLocalError('')
    setTailOffset(0)
    setModelPickerOpen(false)
    setComposing(true)
  }

  const openModelPicker = () => {
    setModelPickerOpen(true)
    setModelFilter('')
    const existingIdx = availableModels.findIndex(item => item.id === selectedModel)
    setModelCursor(existingIdx >= 0 ? existingIdx : 0)

    if (modelChoices || loadingModels) return
    if (!gateway) {
      setModelChoices(fallbackChoices)
      return
    }
    setLoadingModels(true)
    void gateway
      .rpc<ModelOptionsResponse>('model.options', {}, { reportError: false })
      .then(res => {
        const choices = collectSideQuestionModelOptions(res, sessionModel, selectedModel)
        setModelChoices(choices)
        const idx = choices.findIndex(item => item.id === selectedModel)
        setModelCursor(idx >= 0 ? idx : 0)
      })
      .catch(() => {
        setModelChoices(fallbackChoices)
      })
      .finally(() => {
        setLoadingModels(false)
      })
  }

  const commitModelSelection = (option: SideQuestionModelOption | undefined) => {
    if (option) {
      setSideQuestionModel(option.id || undefined, sid)
    } else if (modelFilter.trim()) {
      const custom = modelFilter.trim()
      setSideQuestionModel(custom === 'default' ? undefined : custom, sid)
    }
    setModelPickerOpen(false)
    setModelFilter('')
  }

  const submitQuestion = () => {
    if (!gateway) {
      setLocalError('Connect to a daemon to ask a side question.')
      return
    }
    if (isStreaming) {
      return
    }
    const raw = draft.trim()
    if (!raw) {
      return
    }
    const parsed = parseSideQuestionInput(raw)
    setLocalError('')
    setDraft('')
    if (!parsed.question && parsed.model) {
      setSideQuestionModel(parsed.model === 'default' ? undefined : parsed.model, sid)
      return
    }
    setTailOffset(0)
    setComposing(false)
    const explicitModel = parsed.model || selectedModel || undefined
    void askSideQuestion(gateway.rpc, {
      sessionId: sid,
      question: parsed.question,
      ...(explicitModel ? { model: explicitModel } : {})
    })
  }

  useKeyboard(event => {
    if (event.eventType === 'release') return
    const key = event.name
    const seq = event.sequence ?? ''

    if (modelPickerOpen) {
      event.preventDefault()
      event.stopPropagation()

      if (key === 'escape') {
        setModelPickerOpen(false)
        setModelFilter('')
        return
      }

      if (key === 'return' || key === 'enter' || key === 'kpenter') {
        commitModelSelection(filteredModels[modelCursor])
        return
      }

      if (key === 'up' || (key === 'tab' && event.shift)) {
        if (filteredModels.length > 0) {
          setModelCursor(idx => (idx - 1 + filteredModels.length) % filteredModels.length)
        }
        return
      }

      if (key === 'down' || (key === 'tab' && !event.shift)) {
        if (filteredModels.length > 0) {
          setModelCursor(idx => (idx + 1) % filteredModels.length)
        }
        return
      }

      if (isPageUpKey(event)) {
        setModelCursor(idx => Math.max(0, idx - 6))
        return
      }

      if (isPageDownKey(event)) {
        setModelCursor(idx => Math.min(Math.max(0, filteredModels.length - 1), idx + 6))
        return
      }

      if (key === 'home') {
        setModelCursor(0)
        return
      }

      if (key === 'end') {
        setModelCursor(Math.max(0, filteredModels.length - 1))
        return
      }

      if (key === 'backspace' || key === 'delete') {
        setModelFilter(prev => prev.slice(0, -1))
        setModelCursor(0)
        return
      }

      if (!event.ctrl && !event.meta && !event.super && seq && !/[\x00-\x1f\x7f]/u.test(seq)) {
        setModelFilter(prev => prev + seq)
        setModelCursor(0)
      }
      return
    }

    if (key === 'escape') {
      event.preventDefault()
      event.stopPropagation()
      if (composing && draft.length > 0 && entries.length > 0) {
        setDraft('')
        setComposing(false)
        return
      }
      closeOverlay()
      return
    }

    if (event.ctrl && key === 'l') {
      event.preventDefault()
      event.stopPropagation()
      handleClear()
      return
    }

    if ((event.meta || event.option) && key === 'm') {
      event.preventDefault()
      event.stopPropagation()
      openModelPicker()
      return
    }

    if (composing) {
      if (key === 'tab' && event.shift) {
        event.preventDefault()
        event.stopPropagation()
        openModelPicker()
        return
      }
      if (key === 'tab' && entries.length > 0 && !draft) {
        event.preventDefault()
        event.stopPropagation()
        setComposing(false)
      }
      return
    }

    if (
      (key === 'm' && !event.ctrl && !event.meta) ||
      seq === 'm' ||
      (key === 'tab' && event.shift)
    ) {
      event.preventDefault()
      event.stopPropagation()
      openModelPicker()
      return
    }

    if (
      (key === 'c' && !event.ctrl && !event.meta) ||
      seq === 'c'
    ) {
      event.preventDefault()
      event.stopPropagation()
      handleClear()
      return
    }

    if (
      key === 'i' ||
      key === 'return' ||
      key === 'enter' ||
      key === 'tab' ||
      seq === '/'
    ) {
      event.preventDefault()
      event.stopPropagation()
      setComposing(true)
      return
    }

    if (isPageUpKey(event)) {
      event.preventDefault()
      event.stopPropagation()
      setTailOffset(prev => prev + 6)
      scrollRef.current?.scrollBy(-6)
      return
    }

    if (isPageDownKey(event)) {
      event.preventDefault()
      event.stopPropagation()
      setTailOffset(prev => Math.max(0, prev - 6))
      scrollRef.current?.scrollBy(6)
      return
    }

    if (key === 'home') {
      event.preventDefault()
      event.stopPropagation()
      setTailOffset(999)
      scrollRef.current?.scrollTo(0)
      return
    }

    if (key === 'end') {
      event.preventDefault()
      event.stopPropagation()
      setTailOffset(0)
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      return
    }

    if (key === 'up' || key === 'down' || seq === 'k' || seq === 'j') {
      event.preventDefault()
      event.stopPropagation()
      const up = key === 'up' || seq === 'k'
      setTailOffset(prev => Math.max(0, prev + (up ? 2 : -2)))
      scrollRef.current?.scrollBy(up ? -2 : 2)
    }
  })

  const activeModelLabel = selectedModel
    ? selectedModel
    : sessionModel
      ? `session (${sessionModel})`
      : 'session default'

  const pickerVisibleCount = Math.max(4, size.height - 10)
  const pickerWindow = windowItems(filteredModels, modelCursor, Math.min(pickerVisibleCount, Math.max(1, filteredModels.length)))

  return (
    <box
      alignItems="center"
      backgroundColor={t.ds.scrim}
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={185}
    >
      <Box
        backgroundColor={t.color.overlayBg}
        borderColor={t.color.border}
        borderStyle="round"
        flexDirection="column"
        height={entries.length === 0 && !modelPickerOpen ? Math.min(20, size.height) : size.height}
        paddingX={1}
        width={size.width}
      >
        <DialogHeader
          subtitle={`Answers from the current conversation snapshot without interrupting or saving to chat · model: ${activeModelLabel}`}
          t={t}
          title={
            <>
              {'Side Question'}
              <Span color={t.ds.secondary}>{' · ephemeral · read-only'}</Span>
              {selectedModel ? <Span color={t.color.accent}>{` · ${selectedModel}`}</Span> : null}
              {mainTurnBusy ? <Span color={t.color.accent}>{' · main turn running'}</Span> : null}
            </>
          }
        />

        {localError ? (
          <Text color={t.color.warn} wrap="truncate-end">
            {localError}
          </Text>
        ) : null}

        {modelPickerOpen ? (
          <Box flexDirection="column" flexGrow={1} minHeight={0}>
            <Box
              backgroundColor={t.color.completionMetaBg}
              flexDirection="row"
              flexShrink={0}
              justifyContent="space-between"
              marginBottom={1}
              paddingX={1}
            >
              <Text wrap="truncate-end">
                <Span bold color={t.color.accent}>{'/ '}</Span>
                <Span color={modelFilter ? t.color.text : t.ds.secondary}>
                  {modelFilter || 'Type to filter models…'}
                </Span>
              </Text>
              <Text color={t.ds.meta}>
                {loadingModels ? 'loading…' : `${filteredModels.length}/${availableModels.length}`}
              </Text>
            </Box>

            <Box flexDirection="column" flexGrow={1} minHeight={0} overflow="hidden">
              {filteredModels.length === 0 ? (
                <Box flexDirection="column" paddingX={1}>
                  <Text color={t.ds.secondary} wrap="truncate-end">
                    {modelFilter.trim()
                      ? `No matching catalog model — press Enter to use "${modelFilter.trim()}"`
                      : 'No models available.'}
                  </Text>
                </Box>
              ) : (
                pickerWindow.items.map((option, idx) => {
                  const absoluteIdx = pickerWindow.offset + idx
                  const highlighted = absoluteIdx === modelCursor
                  const active = option.id === selectedModel
                  return (
                    <Box
                      backgroundColor={highlighted ? t.color.completionCurrentBg : undefined}
                      flexDirection="row"
                      flexShrink={0}
                      justifyContent="space-between"
                      key={option.id || '__default__'}
                      onClick={() => commitModelSelection(option)}
                      paddingX={1}
                    >
                      <Text
                        color={highlighted ? t.color.accent : active ? t.ds.title : t.color.text}
                        wrap="truncate-end"
                      >
                        {`${highlighted ? '› ' : '  '}${active ? '✓ ' : '  '}${option.label}`}
                      </Text>
                      {option.provider ? (
                        <Text color={t.ds.meta} wrap="truncate-end">
                          {option.provider}
                        </Text>
                      ) : null}
                    </Box>
                  )
                })
              )}
            </Box>
          </Box>
        ) : entries.length === 0 ? (
          <DialogEmpty
            description="Ask a quick question about this task or codebase (use --model <id> or Shift+Tab to choose a model) while your main turn keeps running."
            symbol="◇"
            t={t}
            title="Ephemeral side thread"
          />
        ) : (
          <scrollbox
            contentOptions={{ flexDirection: 'column' }}
            ref={scrollRef}
            stickyScroll={tailOffset === 0}
            stickyStart="bottom"
            style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
          >
            {entries.map(entry => (
              <SideQuestionTurnRow entry={entry} key={entry.requestId} t={t} />
            ))}
          </scrollbox>
        )}

        {!modelPickerOpen ? (
          <Box flexDirection="column" flexShrink={0} marginTop={compact ? 0 : 1}>
            {composing ? (
              <input
                backgroundColor={t.color.completionMetaBg}
                focused={!isStreaming}
                onInput={setDraft}
                onSubmit={submitQuestion}
                placeholder={
                  isStreaming
                    ? 'Answering side question…'
                    : entries.length > 0
                      ? 'Ask a follow-up (or --model <id> <question>)…'
                      : 'Ask a side question (--model <id> optional, never added to chat)…'
                }
                textColor={t.color.text}
                value={draft}
              />
            ) : (
              <Box
                backgroundColor={t.color.completionMetaBg}
                flexShrink={0}
                onMouseDown={() => setComposing(true)}
                paddingX={1}
              >
                <Text color={t.ds.secondary} wrap="truncate-end">
                  {isStreaming
                    ? 'Streaming answer… (Esc close — main turn continues)'
                    : `› Press Enter or i to ask · m choose model (${activeModelLabel})`}
                </Text>
              </Box>
            )}
          </Box>
        ) : null}

        <DialogFooter t={t}>
          <Text color={t.ds.secondary} wrap="truncate-end">
            {modelPickerOpen
              ? 'Type to filter · ↑↓/Tab/⇧Tab move · Enter select · Esc back'
              : composing
                ? compact
                  ? 'Enter ask · ⇧Tab model · ^L clear · Esc close'
                  : `Enter ask · Shift+Tab model · Ctrl+L clear${entries.length > 0 && !draft ? ' · Tab browse' : ''} · Esc close`
                : compact
                  ? 'Enter/i ask · m model · ↑↓ scroll · c clear · Esc close'
                  : 'Enter/i follow-up · m model picker · ↑↓/PgUp/PgDn scroll · c clear · Esc close'}
          </Text>
        </DialogFooter>
      </Box>
    </box>
  )
}
