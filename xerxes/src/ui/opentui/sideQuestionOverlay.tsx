// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

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
import { isPageDownKey, isPageUpKey } from '../lib/pageKeys.js'
import type { Theme } from '../theme.js'

import { DialogEmpty, DialogFooter, DialogHeader } from './dialogChrome.js'
import { OVERLAY_PANEL_SPECS, overlayPanelSize } from './overlayLayout.js'
import { Box, Span, Text } from './primitives.js'

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
      <Box flexDirection="column" paddingLeft={2}>
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
          <Text color={t.color.text} wrap="wrap">
            {entry.answer}
            {entry.status === 'streaming' ? <Span color={t.color.accent}>{' ▍'}</Span> : null}
          </Text>
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

function collectModelCandidates(
  response: ModelOptionsResponse | null | undefined,
  sessionModel: string,
  selectedModel: string
): readonly string[] {
  const seen = new Set<string>()
  const out: string[] = ['']
  const add = (value: string | undefined) => {
    const trimmed = typeof value === 'string' ? value.trim() : ''
    if (!trimmed || seen.has(trimmed)) return
    seen.add(trimmed)
    out.push(trimmed)
  }
  add(selectedModel)
  add(sessionModel)
  add(response?.model)
  for (const provider of response?.providers ?? []) {
    add(provider.configured_model)
    for (const model of provider.models ?? []) {
      add(model)
    }
  }
  return out
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
  const [modelChoices, setModelChoices] = useState<readonly string[] | null>(null)
  const [loadingModels, setLoadingModels] = useState(false)
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)

  useEffect(() => {
    if (entries.length === 0 && !composing) {
      setComposing(true)
    }
  }, [entries.length, composing])

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
    setComposing(true)
  }

  const cycleModel = () => {
    if (loadingModels) return
    const applyNext = (choices: readonly string[]) => {
      if (choices.length <= 1) {
        setSideQuestionModel(undefined, sid)
        return
      }
      const idx = choices.indexOf(selectedModel)
      const next = choices[(idx + 1) % choices.length] ?? ''
      setSideQuestionModel(next || undefined, sid)
    }
    if (modelChoices) {
      applyNext(modelChoices)
      return
    }
    if (!gateway) {
      const fallback = collectModelCandidates(null, sessionModel, selectedModel)
      setModelChoices(fallback)
      applyNext(fallback)
      return
    }
    setLoadingModels(true)
    void gateway
      .rpc<ModelOptionsResponse>('model.options', {}, { reportError: false })
      .then(res => {
        const choices = collectModelCandidates(res, sessionModel, selectedModel)
        setModelChoices(choices)
        applyNext(choices)
      })
      .catch(() => {
        const fallback = collectModelCandidates(null, sessionModel, selectedModel)
        setModelChoices(fallback)
        applyNext(fallback)
      })
      .finally(() => {
        setLoadingModels(false)
      })
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

    if (composing) {
      if (key === 'tab' && entries.length > 0 && !draft) {
        event.preventDefault()
        event.stopPropagation()
        setComposing(false)
      }
      return
    }

    if (
      (key === 'm' && !event.ctrl && !event.meta) ||
      seq === 'm'
    ) {
      event.preventDefault()
      event.stopPropagation()
      cycleModel()
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

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
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
        height={entries.length === 0 ? Math.min(20, size.height) : size.height}
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

        {entries.length === 0 ? (
          <DialogEmpty
            description="Ask a quick question about this task or codebase (use --model <id> to override model) while your main turn keeps running."
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
                  : `› Press Enter or i to ask · m model (${activeModelLabel})`}
              </Text>
            </Box>
          )}
        </Box>

        <DialogFooter t={t}>
          <Text color={t.ds.secondary} wrap="truncate-end">
            {composing
              ? compact
                ? 'Enter ask · --model <id> · ^L clear · Esc close'
                : `Enter ask · --model <id> · Ctrl+L clear${entries.length > 0 && !draft ? ' · Tab scroll/model' : ''} · Esc close`
              : compact
                ? 'Enter/i ask · m model · ↑↓ scroll · c clear · Esc close'
                : 'Enter/i follow-up · m cycle model · ↑↓/PgUp/PgDn scroll · c clear · Esc close'}
          </Text>
        </DialogFooter>
      </Box>
    </box>
  )
}
