// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Native OpenTUI view consuming the controller's AppLayoutProps contract:
// scrollbox transcript (native sticky-scroll), a native <textarea>
// composer, approval/confirm/clarify prompts, and compact application chrome.
import type { KeyBinding, KeyEvent, ScrollBoxRenderable, TextareaRenderable, TextRenderable } from '@opentui/core'
import { useBlur, useFocus, useKeyboard, usePaste, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { type MutableRefObject, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { AppLayoutProps, Notice } from '../app/interfaces.js'
import { setInputSelection } from '../app/inputSelectionStore.js'
import { isLiveTailActive, liveTailScrollKey, shouldAutoScrollLiveTail } from '../app/liveTailScroll.js'
import { $isBlocked, $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiState, $uiTheme } from '../app/uiStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { $spawnHistory } from '../app/spawnHistoryStore.js'
import {
  DERAFSH_ANIMATION_FRAME_COUNT,
  DERAFSH_ANIMATION_FRAME_MS,
  DERAFSH_KAVIANI_GLYPH,
  DERAFSH_KAVIANI_WIDTH,
  derafshAnimationEnabled,
  derafshCompactGradientFrame,
  derafshGradientFrame,
  derafshKaviani
} from '../banner.js'
import { agentSidebarWidth, shouldShowAgentSidebar } from '../domain/agentPanelLayout.js'
import { sectionMode } from '../domain/details.js'
import { completionToApplyOnSubmit } from '../domain/slash.js'
import { shouldShowStartupWelcome, startupComposerWidth } from '../domain/startupLayout.js'
import {
  isProviderPrompt,
  providerPromptCancelAnswer,
  providerPromptChoices,
  providerPromptIsSecret,
  providerPromptTitle
} from '../domain/providerPrompt.js'
import { ctxBarColor, sessionDisplayTitle, usageCounts } from '../domain/statusFormat.js'
import { unarchivedToolLines } from '../lib/liveProgress.js'
import { compactProgressRows, type CompactProgressRow } from '../lib/progressRows.js'
import { isYoloEnabled } from '../lib/statusSnapshot.js'
import { fmtK, formatToolCall, inlineToolDisplay } from '../lib/text.js'
import { useTerminalFocus } from '../lib/terminalRuntime.opentui.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'
import type { Theme } from '../theme.js'

import { AgentPanel, AgentPanelHotkey, AgentPanelOverlay, collectAgentPanelRecords } from './agentPanel.js'
import { displayModeLabel, SessionHeader, WorkspaceFooter } from './appChrome.js'
import { MessageLine } from './messageLine.js'
import { ModelPicker } from './modelPicker.js'
import { Box, Span, Text } from './primitives.js'
import { SessionPicker } from './sessionPicker.js'

const TEXTAREA_KEY_BINDINGS: KeyBinding[] = [
  { name: 'return', action: 'submit' },
  { name: 'return', shift: true, action: 'newline' },
  { name: 'kpenter', action: 'submit' },
  { name: 'kpenter', shift: true, action: 'newline' },
  { name: 'linefeed', action: 'submit' }
]

const decodePaste = (bytes: Uint8Array): string => new TextDecoder().decode(bytes)

// ── Live streaming turn ─────────────────────────────────────────────────

function StreamingAssistant() {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const streaming = useTurnSelector(s => s.streaming)
  const segments = useTurnSelector(s => s.streamSegments)
  const tools = useTurnSelector(s => s.tools)
  const pendingTools = useTurnSelector(s => s.streamPendingTools)
  const unsettledTools = unarchivedToolLines(segments, pendingTools)

  const anything = streaming || segments.length || tools.length || unsettledTools.length

  if (!anything) {
    return ui.busy ? <WaitingLine /> : null
  }

  return (
    <Box flexDirection="column" flexShrink={0}>
      {segments.map((segment, index) => (
        <MessageLine key={`segment:${index}`} msg={segment} t={t} />
      ))}

      {unsettledTools.length ? (
        <MessageLine msg={{ kind: 'trail', role: 'system', text: '', tools: unsettledTools }} t={t} />
      ) : null}

      {tools.map(tool => (
        <Box flexShrink={0} key={tool.id} paddingLeft={3}>
          <Text color={t.color.muted} wrap="truncate-end">
            → {inlineToolDisplay(formatToolCall(tool.name, tool.context))}
          </Text>
        </Box>
      ))}

      {streaming ? <MessageLine msg={{ role: 'assistant', text: streaming }} t={t} /> : null}
    </Box>
  )
}

const WAITING_FRAMES = ['◇', '◈', '◆', '◈'] as const

function WaitingLine() {
  const t = useStore($uiTheme)
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setFrame(current => (current + 1) % WAITING_FRAMES.length), 180)
    timer.unref?.()

    return () => clearInterval(timer)
  }, [])

  return (
    <Box flexShrink={0} marginTop={1} paddingLeft={3}>
      <Text color={t.color.muted}>
        <Span color={t.color.accent}>{WAITING_FRAMES[frame]} </Span>
        Planning next moves
      </Text>
    </Box>
  )
}

function progressToneColor(tone: CompactProgressRow['tone'], t: Theme): string {
  if (tone === 'error') {
    return t.color.error
  }
  if (tone === 'warn') {
    return t.color.warn
  }
  if (tone === 'success') {
    return t.color.ok
  }

  return t.color.muted
}

function CompactLiveProgress({ show }: { show: boolean }) {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const activity = useTurnSelector(state => state.activity)
  const outcome = useTurnSelector(state => state.outcome)
  const todos = useTurnSelector(state => state.todos)
  const turnTrail = useTurnSelector(state => state.turnTrail)
  const activityVisible =
    sectionMode('activity', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'
  const toolsVisible = sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'
  const rows = useMemo(
    () => compactProgressRows({ activity, outcome, todos, turnTrail }, { activityVisible, toolsVisible }),
    [activity, activityVisible, outcome, todos, toolsVisible, turnTrail]
  )

  if (!show || !rows.length) {
    return null
  }

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} paddingLeft={3}>
      {rows.map((row, index) => {
        const color = progressToneColor(row.tone, t)
        const glyph = row.kind === 'todo' ? '◇' : row.kind === 'outcome' ? '✓' : row.kind === 'activity' ? '·' : '→'

        return (
          <Text color={color} key={`${row.kind}:${index}:${row.text}`} wrap="truncate-end">
            <Span color={color}>{glyph} </Span>
            {row.text}
          </Text>
        )
      })}
    </Box>
  )
}

// ── Prompt overlays (approval / confirm / clarify) ─────────────────────────

const APPROVAL_OPTS = ['once', 'session', 'always', 'deny'] as const
const APPROVAL_LABELS = { once: 'Allow once', session: 'Allow this session', always: 'Always allow', deny: 'Deny' }

function InlinePromptPanel({ accent, children }: { accent: string; children: ReactNode }) {
  const t = useStore($uiTheme)

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="row" flexShrink={0} marginBottom={1} marginTop={1}>
      <Box backgroundColor={accent} flexShrink={0} width={1} />
      <Box flexDirection="column" flexGrow={1} flexShrink={0} paddingX={2} paddingY={1}>
        {children}
      </Box>
    </Box>
  )
}

function PromptPanelGap() {
  return <Box flexShrink={0} height={1} />
}

function NoticeBanner({ notice, t }: { notice: Notice | null; t: Theme }) {
  if (!notice?.text) {
    return null
  }

  const color =
    notice.level === 'error'
      ? t.color.error
      : notice.level === 'warn'
        ? t.color.warn
        : notice.level === 'success'
          ? t.color.statusGood
          : t.color.accent

  return (
    <Box flexDirection="row" flexShrink={0} marginBottom={1} paddingX={2}>
      <Box backgroundColor={color} flexShrink={0} width={1} />
      <Box backgroundColor={t.color.completionBg} flexGrow={1} flexShrink={1} paddingX={1}>
        <Text color={color} wrap="truncate-end">
          {notice.text}
        </Text>
      </Box>
    </Box>
  )
}

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

function ProviderPromptOverlay({ actions }: Pick<AppLayoutProps, 'actions'>) {
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const { height, width } = useTerminalDimensions()
  const clarify = isProviderPrompt(overlay.clarify) ? overlay.clarify : null
  const [customValue, setCustomValue] = useState(false)
  const [maskedValue, setMaskedValue] = useState('')
  const [selected, setSelected] = useState(0)
  const inputRef = useRef<TextareaRenderable | null>(null)

  const choices = clarify ? providerPromptChoices(clarify) : []
  const cancelAnswer = clarify ? providerPromptCancelAnswer(clarify) : ''
  const allowFreeform = clarify?.allowFreeform !== false
  const typing = Boolean(clarify && (customValue || choices.length === 0))
  const masked = Boolean(clarify && typing && providerPromptIsSecret(clarify))
  const rowCount = Math.max(1, Math.min(10, height - 13, choices.length + (allowFreeform ? 1 : 0)))
  const totalRows = choices.length + (allowFreeform ? 1 : 0)
  const offset = Math.max(0, Math.min(selected - Math.floor(rowCount / 2), totalRows - rowCount))
  const panelWidth = Math.max(34, Math.min(84, width - 4))

  useEffect(() => {
    setCustomValue(false)
    setMaskedValue('')
    setSelected(0)
    inputRef.current?.clear()
  }, [clarify?.requestId])

  useKeyboard(event => {
    if (!clarify) {
      return
    }

    const submitCancel = () => actions.answerClarify(cancelAnswer)

    if (typing) {
      if (event.name === 'escape') {
        if (customValue && choices.length) {
          setCustomValue(false)
        } else {
          submitCancel()
        }
        consumeKey(event)

        return
      }

      if (!masked) {
        return
      }

      if (event.name === 'return' || event.name === 'enter' || event.name === 'kpenter' || event.name === 'linefeed') {
        actions.answerClarify(maskedValue)
      } else if (event.name === 'backspace' || event.name === 'delete') {
        setMaskedValue(value => value.slice(0, -1))
      } else if (!event.ctrl && !event.meta && !event.super && event.sequence >= ' ' && event.sequence.length === 1) {
        setMaskedValue(value => value + event.sequence)
      } else {
        return
      }

      consumeKey(event)

      return
    }

    const quick = Number.parseInt(event.sequence ?? '', 10)

    if (event.name === 'escape') {
      submitCancel()
    } else if (event.name === 'up') {
      setSelected(current => Math.max(0, current - 1))
    } else if (event.name === 'down') {
      setSelected(current => Math.min(totalRows - 1, current + 1))
    } else if (quick >= 1 && quick <= Math.min(9, choices.length)) {
      actions.answerClarify(choices[quick - 1]!)
    } else if (event.name === 'return' || event.name === 'enter' || event.name === 'kpenter') {
      const choice = choices[selected]

      if (choice) {
        actions.answerClarify(choice)
      } else if (allowFreeform) {
        setCustomValue(true)
      }
    } else {
      return
    }

    consumeKey(event)
  })

  usePaste(event => {
    if (!clarify || !masked) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    setMaskedValue(value => value + decodePaste(event.bytes))
  })

  if (!clarify) {
    return null
  }

  const title = providerPromptTitle(clarify.questionId)
  const question =
    clarify.questionId === 'action' ? 'Switch profiles or manage provider connections.' : clarify.question
  const rows = [...choices, ...(allowFreeform ? ['Type a custom value…'] : [])]

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={190}
    >
      <box
        backgroundColor={t.color.statusBg}
        flexDirection="column"
        flexShrink={0}
        paddingBottom={2}
        paddingLeft={2}
        paddingRight={2}
        paddingTop={2}
        width={panelWidth}
      >
        <box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
          <text fg={t.color.accent} flexShrink={0}>
            <b>{title}</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc close
          </text>
        </box>

        <text fg={t.color.text} flexShrink={0} wrapMode="word">
          {question}
        </text>

        {typing ? (
          <>
            <box
              backgroundColor={t.color.completionCurrentBg}
              flexDirection="row"
              flexShrink={0}
              marginTop={1}
              minHeight={3}
              paddingLeft={1}
              paddingRight={1}
              paddingTop={1}
            >
              <text fg={t.color.accent} flexShrink={0}>
                ›{' '}
              </text>
              {masked ? (
                <text fg={t.color.text} flexShrink={0}>
                  {'•'.repeat(Math.min(maskedValue.length, panelWidth - 8)) || 'API key'}
                </text>
              ) : (
                <textarea
                  focused
                  focusedBackgroundColor={t.color.completionCurrentBg}
                  focusedTextColor={t.color.text}
                  keyBindings={TEXTAREA_KEY_BINDINGS}
                  maxHeight={5}
                  minHeight={1}
                  onSubmit={() => {
                    actions.answerClarify(inputRef.current?.plainText.trim() ?? '')
                    inputRef.current?.clear()
                  }}
                  placeholder={clarify.placeholder || 'Type a value…'}
                  placeholderColor={t.color.muted}
                  ref={inputRef}
                  style={{ flexGrow: 1, flexShrink: 0 }}
                  wrapMode="word"
                />
              )}
            </box>
            <text fg={t.color.muted} flexShrink={0} marginTop={1}>
              Enter continue · Esc {customValue && choices.length ? 'back' : 'cancel setup'}
            </text>
          </>
        ) : (
          <>
            <box flexDirection="column" flexShrink={0} marginTop={1}>
              {rows.slice(offset, offset + rowCount).map((choice, index) => {
                const absoluteIndex = offset + index
                const active = absoluteIndex === selected

                return (
                  <box
                    backgroundColor={active ? t.color.completionCurrentBg : undefined}
                    flexShrink={0}
                    height={1}
                    key={`${absoluteIndex}:${choice}`}
                    paddingLeft={1}
                    paddingRight={1}
                    width="100%"
                  >
                    <text
                      fg={active ? t.color.accent : absoluteIndex < choices.length ? t.color.text : t.color.muted}
                      flexShrink={0}
                      truncate
                      width="100%"
                      wrapMode="none"
                    >
                      {active ? '›' : ' '}{' '}
                      {absoluteIndex < choices.length && absoluteIndex < 9 ? `${absoluteIndex + 1}. ` : ''}
                      {choice}
                    </text>
                  </box>
                )
              })}
            </box>
            <text fg={t.color.muted} flexShrink={0} marginTop={1}>
              ↑/↓ navigate · Enter select · Esc cancel setup
            </text>
          </>
        )}
      </box>
    </box>
  )
}

function PromptZone({ actions }: Pick<AppLayoutProps, 'actions'>) {
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const [sel, setSel] = useState(0)
  const [customClarify, setCustomClarify] = useState(false)
  const [maskedValue, setMaskedValue] = useState('')
  const clarifyRef = useRef<TextareaRenderable | null>(null)

  const approval = overlay.approval
  const confirm = overlay.confirm
  const clarify = overlay.clarify
  const secret = overlay.secret
  const sudo = overlay.sudo
  const providerClarify = isProviderPrompt(clarify)
  const clarifyChoices = providerClarify ? [] : (clarify?.choices ?? [])
  const clarifyCancelAnswer = clarifyChoices.find(choice => choice.trim().toLowerCase() === 'cancel') ?? ''
  const typingClarify = Boolean(clarify && !providerClarify && (customClarify || clarifyChoices.length === 0))

  const opts = approval
    ? approval.allowPermanent === false
      ? APPROVAL_OPTS.filter(o => o !== 'always')
      : APPROVAL_OPTS
    : []

  useKeyboard(event => {
    const name = event.name

    if (secret || sudo) {
      if (name === 'escape') {
        secret ? actions.answerSecret('') : actions.answerSudo('')
      } else if (name === 'return' || name === 'kpenter' || name === 'linefeed') {
        secret ? actions.answerSecret(maskedValue) : actions.answerSudo(maskedValue)
      } else if (name === 'backspace' || name === 'delete') {
        setMaskedValue(value => value.slice(0, -1))
      } else if (!event.ctrl && !event.meta && !event.super && event.sequence >= ' ' && event.sequence.length === 1) {
        setMaskedValue(value => value + event.sequence)
      } else {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      return
    }

    if (approval) {
      if (name === 'escape') {
        actions.answerApproval('deny')
      } else {
        const n = Number.parseInt(event.sequence ?? '', 10)

        if (n >= 1 && n <= opts.length) {
          actions.answerApproval(opts[n - 1]!)
        } else if (name === 'up') {
          setSel(s => Math.max(0, s - 1))
        } else if (name === 'down') {
          setSel(s => Math.min(opts.length - 1, s + 1))
        } else if (name === 'return' || name === 'enter') {
          actions.answerApproval(opts[sel]!)
        } else {
          return
        }
      }

      event.preventDefault()
      event.stopPropagation()

      return
    }

    if (confirm) {
      const lower = (event.sequence ?? '').toLowerCase()

      if (name === 'escape' || lower === 'n') {
        patchOverlayState({ confirm: null })
      } else if (lower === 'y' || name === 'return' || name === 'enter') {
        patchOverlayState({ confirm: null })
        confirm.onConfirm()
      }

      return
    }

    if (clarify && !providerClarify) {
      if (typingClarify) {
        if (name === 'escape') {
          clarifyChoices.length ? setCustomClarify(false) : actions.answerClarify('')
          event.preventDefault()
          event.stopPropagation()
        }

        return
      }

      const n = Number.parseInt(event.sequence ?? '', 10)

      if (name === 'escape') {
        actions.answerClarify(clarifyCancelAnswer)
      } else if (n >= 1 && n <= clarifyChoices.length) {
        actions.answerClarify(clarifyChoices[n - 1]!)
      } else if (name === 'up') {
        setSel(s => Math.max(0, s - 1))
      } else if (name === 'down') {
        setSel(s => Math.min(clarifyChoices.length, s + 1))
      } else if (name === 'return' || name === 'enter') {
        const choice = clarifyChoices[sel]

        choice ? actions.answerClarify(choice) : setCustomClarify(true)
      }
    }
  })

  usePaste(event => {
    if (!secret && !sudo) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    setMaskedValue(value => value + decodePaste(event.bytes))
  })

  useEffect(() => {
    setSel(0)
    setCustomClarify(false)
    setMaskedValue('')
  }, [approval, confirm, clarify, secret, sudo])

  if (approval) {
    return (
      <InlinePromptPanel accent={t.color.warn}>
        <Box alignItems="center" flexDirection="row" flexShrink={0} justifyContent="space-between">
          <Text bold color={t.color.warn}>
            Approval required
          </Text>
          <Text color={t.color.muted}>Esc denies</Text>
        </Box>
        <PromptPanelGap />
        <Text color={t.color.muted}>Requested action</Text>
        <Text color={t.color.text} wrap="wrap">
          {approval.description}
        </Text>
        {approval.command ? (
          <Box
            backgroundColor={t.color.completionCurrentBg}
            flexShrink={0}
            marginTop={1}
            paddingX={1}
            paddingY={1}
          >
            <Text color={t.color.text} wrap="wrap">
              {approval.command.slice(0, 320)}
            </Text>
          </Box>
        ) : null}
        <PromptPanelGap />
        <Text color={t.color.muted}>Permission scope</Text>
        <Box flexDirection="column" flexShrink={0} gap={1} marginTop={1}>
          {opts.map((o, i) => (
            <Box
              alignItems="center"
              backgroundColor={sel === i ? t.color.selectionBg : undefined}
              flexDirection="row"
              flexShrink={0}
              key={o}
              minHeight={2}
              paddingX={1}
            >
              <Text color={sel === i ? t.color.warn : t.color.muted}>
                {sel === i ? '›' : ' '} {i + 1}. {APPROVAL_LABELS[o]}
              </Text>
            </Box>
          ))}
        </Box>
        <PromptPanelGap />
        <Text color={t.color.muted}>↑/↓ move · Enter allow · 1-{opts.length} quick select</Text>
      </InlinePromptPanel>
    )
  }

  if (confirm) {
    const accent = confirm.danger ? t.color.error : t.color.warn

    return (
      <InlinePromptPanel accent={accent}>
        <Text bold color={accent}>
          {confirm.title}
        </Text>
        {confirm.detail ? (
          <>
            <PromptPanelGap />
            <Text color={t.color.text} wrap="wrap">
              {confirm.detail}
            </Text>
          </>
        ) : null}
        <PromptPanelGap />
        <Text color={t.color.muted}>Y/Enter confirm · N/Esc cancel</Text>
      </InlinePromptPanel>
    )
  }

  if (clarify && !providerClarify) {
    return (
      <InlinePromptPanel accent={t.color.accent}>
        <Text bold color={t.color.text} wrap="wrap">
          {clarify.question}
        </Text>
        <PromptPanelGap />
        {typingClarify ? (
          <>
            <Box backgroundColor={t.color.completionCurrentBg} paddingX={1} paddingY={1}>
              <Text color={t.color.accent}>› </Text>
              <textarea
                focused
                focusedTextColor={t.color.text}
                keyBindings={TEXTAREA_KEY_BINDINGS}
                maxHeight={6}
                minHeight={1}
                onSubmit={() => {
                  const answer = clarifyRef.current?.plainText.trim() ?? ''

                  if (answer) {
                    actions.answerClarify(answer)
                    clarifyRef.current?.clear()
                  }
                }}
                placeholder="Type your answer…"
                placeholderColor={t.color.muted}
                ref={clarifyRef}
                style={{ flexGrow: 1, flexShrink: 0 }}
                wrapMode="word"
              />
            </Box>
            <PromptPanelGap />
            <Text color={t.color.muted}>
              Enter send · Shift+Enter newline · Esc {clarifyChoices.length ? 'back' : 'cancel'}
            </Text>
          </>
        ) : (
          <>
            {[...clarifyChoices, 'Other (type your answer)'].map((choice, i) => (
              <Box backgroundColor={sel === i ? t.color.selectionBg : undefined} key={i} paddingX={1}>
                <Text color={sel === i ? t.color.label : t.color.muted}>
                  {i + 1} {sel === i ? '●' : '○'} {choice}
                </Text>
              </Box>
            ))}
            <PromptPanelGap />
            <Text color={t.color.muted}>↑/↓ select · Enter confirm · 1-{clarifyChoices.length} quick · Esc cancel</Text>
          </>
        )}
      </InlinePromptPanel>
    )
  }

  if (secret || sudo) {
    return (
      <InlinePromptPanel accent={t.color.warn}>
        <Text bold color={t.color.warn}>
          {secret?.prompt ?? 'sudo password required'}
        </Text>
        {secret ? <Text color={t.color.muted}>for {secret.envVar}</Text> : null}
        <PromptPanelGap />
        <Text color={t.color.text}>› {'•'.repeat(Math.min(maskedValue.length, 48)) || ' '}</Text>
        <PromptPanelGap />
        <Text color={t.color.muted}>Enter submit · Esc/Ctrl+C cancel</Text>
      </InlinePromptPanel>
    )
  }

  return null
}

// ── Composer ───────────────────────────────────────────────────────────────

const PROMPT_LOADING_FRAMES = [
  { active: 0, forward: true },
  { active: 1, forward: true },
  { active: 2, forward: true },
  { active: 2, forward: false },
  { active: 1, forward: false },
  { active: 0, forward: false }
] as const

function PromptModeLabel({ busy, label }: { busy: boolean; label: string }) {
  const t = useStore($uiTheme)
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    if (!busy) {
      setFrame(0)

      return
    }

    const timer = setInterval(() => setFrame(current => (current + 1) % PROMPT_LOADING_FRAMES.length), 120)
    timer.unref?.()

    return () => clearInterval(timer)
  }, [busy])

  if (!busy) {
    return (
      <Text bold color={t.color.accent}>
        {displayModeLabel(label)}
      </Text>
    )
  }

  const step = PROMPT_LOADING_FRAMES[frame] ?? PROMPT_LOADING_FRAMES[0]

  return (
    <Text>
      {[0, 1, 2].map(index => {
        const distance = step.forward ? step.active - index : index - step.active
        const active = distance >= 0 && distance < 2

        return (
          <Span color={distance === 0 ? t.color.accent : t.color.border} key={index}>
            {active ? '■' : '⬝'}
          </Span>
        )
      })}
    </Text>
  )
}

function ContextMeter() {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const { max, used } = usageCounts(ui.usage)

  if (max <= 0) {
    return null
  }

  const remaining = Math.max(0, max - used)
  const usedPct = Math.min(100, (used / max) * 100)
  const remainingPct = Math.max(0, Math.round(100 - usedPct))

  return (
    <Text>
      <Span color={ctxBarColor(usedPct, t)}>{remainingPct}%</Span>
      <Span color={t.color.muted}> {fmtK(remaining)}</Span>
    </Text>
  )
}

function CompletionMenu({ composer }: Pick<AppLayoutProps, 'composer'>) {
  const t = useStore($uiTheme)
  const completions = composer.completions

  if (!completions.length) {
    return null
  }

  const windowSize = Math.min(10, completions.length)
  const start = Math.max(0, Math.min(composer.compIdx - Math.floor(windowSize / 2), completions.length - windowSize))

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} paddingY={1}>
      {completions.slice(start, start + windowSize).map((item, i) => {
        const active = start + i === composer.compIdx

        return (
          <Box
            backgroundColor={active ? t.color.selectionBg : t.color.completionBg}
            flexDirection="row"
            flexShrink={0}
            key={`${item.text}:${item.display}`}
            minHeight={1}
            paddingX={2}
          >
            <Text bold color={active ? t.color.accent : t.color.label}>
              {active ? '▸ ' : '  '}
              {item.display}
            </Text>
            {item.meta ? <Text color={t.color.muted}> {item.meta}</Text> : null}
          </Box>
        )
      })}
    </Box>
  )
}

function QueuePanel({ composer }: Pick<AppLayoutProps, 'composer'>) {
  const t = useStore($uiTheme)

  if (!composer.queuedDisplay.length) {
    return null
  }

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} paddingX={2} paddingY={1}>
      {composer.queuedDisplay.map((message, index) => (
        <Text color={index === composer.queueEditIdx ? t.color.accent : t.color.text} key={index} wrap="truncate-end">
          {index === composer.queueEditIdx ? '✎ ' : '→ '}
          {message}
        </Text>
      ))}
      <Box flexShrink={0} height={1} />
      <Text color={t.color.text}>
        <Span color={t.color.accent}>Enter </Span>
        <Span color={t.color.muted}>send now</Span> · <Span color={t.color.accent}>↑ </Span>
        <Span color={t.color.muted}>edit</Span> · <Span color={t.color.accent}>Esc </Span>
        <Span color={t.color.muted}>cancel</Span>
      </Text>
    </Box>
  )
}

function Composer({ composer }: Pick<AppLayoutProps, 'composer'>) {
  const ui = useStore($uiState)
  const isBlocked = useStore($isBlocked)
  const t = useStore($uiTheme)
  const ref = useRef<TextareaRenderable | null>(null)

  const modelLabel = ui.info?.model || 'choose model with /provider'
  const modeLabel = ui.info?.mode || 'code'
  const yoloEnabled = isYoloEnabled(ui.info?.permission_mode)
  const narrow = composer.cols < 76

  const syncInputSelection = useCallback(() => {
    const textarea = ref.current

    if (!textarea || isBlocked) {
      setInputSelection(null)

      return
    }

    const cursor = textarea.cursorOffset

    setInputSelection({
      clear: () => {},
      collapseToEnd: () => {
        if (ref.current) {
          ref.current.cursorOffset = ref.current.plainText.length
        }
      },
      end: cursor,
      start: cursor,
      value: textarea.plainText
    })
  }, [isBlocked])

  useEffect(() => {
    const textarea = ref.current

    if (textarea && textarea.plainText !== composer.input) {
      textarea.setText(composer.input)
      textarea.cursorOffset = composer.input.length
    }

    syncInputSelection()
  }, [composer.input, syncInputSelection])

  useEffect(() => () => setInputSelection(null), [])

  const applyDraft = useCallback(
    (value: string, cursor = value.length) => {
      const textarea = ref.current

      if (textarea && textarea.plainText !== value) {
        textarea.setText(value)
      }

      if (textarea) {
        textarea.cursorOffset = Math.max(0, Math.min(cursor, value.length))
      }

      composer.updateInput(value)
      syncInputSelection()
    },
    [composer, syncInputSelection]
  )

  const onSubmit = () => {
    const value = ref.current?.plainText ?? ''
    const row = composer.completions[composer.compIdx]
    const completion = completionToApplyOnSubmit(value, row?.text, composer.compReplace)

    if (completion !== null) {
      applyDraft(completion)

      return
    }

    composer.submit(value)
    ref.current?.clear()
    composer.updateInput('')
    syncInputSelection()
  }

  const onContentChange = () => {
    composer.updateInput(ref.current?.plainText ?? '')
    syncInputSelection()
  }

  usePaste(event => {
    const textarea = ref.current

    if (isBlocked || !textarea) {
      return
    }

    event.preventDefault()
    event.stopPropagation()

    const value = textarea.plainText
    const cursor = textarea.cursorOffset

    void Promise.resolve(
      composer.handleTextPaste({ bracketed: true, cursor, hotkey: false, text: decodePaste(event.bytes), value })
    ).then(result => {
      if (result) {
        applyDraft(result.value, result.cursor)
      }
    })
  })

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} width="100%">
      <QueuePanel composer={composer} />
      <CompletionMenu composer={composer} />
      <Box
        alignItems="flex-start"
        backgroundColor={t.color.completionCurrentBg}
        flexDirection="row"
        flexShrink={0}
        gap={2}
        minHeight={3}
        paddingX={2}
        paddingY={1}
      >
        <PromptModeLabel busy={ui.busy} label={modeLabel} />
        <Box flexGrow={1} flexShrink={1} minWidth={1}>
          <textarea
            focused={!isBlocked}
            focusedBackgroundColor={t.color.completionCurrentBg}
            focusedTextColor={t.color.text}
            keyBindings={TEXTAREA_KEY_BINDINGS}
            maxHeight={10}
            minHeight={1}
            onContentChange={onContentChange}
            onCursorChange={syncInputSelection}
            onSubmit={onSubmit}
            placeholder={
              ui.busy
                ? 'Queue a follow-up… (esc to interrupt)'
                : composer.empty
                  ? 'What are we building?'
                  : 'Message Xerxes…'
            }
            placeholderColor={t.color.muted}
            ref={ref}
            style={{ flexGrow: 1, flexShrink: 0 }}
            textColor={t.color.text}
            wrapMode="word"
          />
        </Box>
      </Box>
      <Box
        alignItems="center"
        flexDirection="row"
        flexShrink={0}
        height={1}
        justifyContent="space-between"
        paddingX={2}
      >
        <Box alignItems="center" flexDirection="row" flexShrink={1} gap={1} height={1} overflow="hidden">
          <Text color={t.color.text} wrap="truncate-end">
            {modelLabel}
          </Text>
          {yoloEnabled ? (
            <Text bold color={t.color.warn}>
              YOLO ON
            </Text>
          ) : null}
          <ContextMeter />
        </Box>
        <Box alignItems="center" flexDirection="row" flexShrink={0} gap={1} height={1}>
          {ui.busy ? (
            <Text color={t.color.text}>
              Enter <Span color={t.color.muted}>queue</Span> · Esc{' '}
              <Span color={t.color.muted}>{composer.queuedDisplay.length ? 'clear queue' : 'interrupt'}</Span>
            </Text>
          ) : composer.completions.length ? (
            <Text color={t.color.text}>
              Tab <Span color={t.color.muted}>accept</Span> · ↑↓ <Span color={t.color.muted}>navigate</Span> · Esc{' '}
              <Span color={t.color.muted}>dismiss</Span>
            </Text>
          ) : narrow ? (
            <Text color={t.color.muted}>Tab modes</Text>
          ) : (
            <Text color={t.color.text}>
              @ <Span color={t.color.muted}>files</Span> · Shift+Enter <Span color={t.color.muted}>new line</Span> · Tab{' '}
              <Span color={t.color.muted}>modes</Span>
            </Text>
          )}
        </Box>
      </Box>
    </Box>
  )
}

// ── Floating pager (/help, /status, …) ─────────────────────────────────────

function PagerOverlay({ composer }: Pick<AppLayoutProps, 'composer'>) {
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const pager = overlay.pager

  if (pager) {
    const size = composer.pagerPageSize
    const slice = pager.lines.slice(pager.offset, pager.offset + size)
    const atEnd = pager.offset + size >= pager.lines.length

    return (
      <box
        alignItems="center"
        backgroundColor="#000000cc"
        flexDirection="column"
        height="100%"
        justifyContent="center"
        left={0}
        position="absolute"
        top={0}
        width="100%"
        zIndex={150}
      >
        <Box backgroundColor={t.color.statusBg} flexDirection="column" maxWidth={110} minWidth={48} padding={2}>
          {pager.title ? (
            <Box justifyContent="space-between" marginBottom={1}>
              <Text bold color={t.color.primary}>
                {pager.title}
              </Text>
              <Text color={t.color.muted}>esc</Text>
            </Box>
          ) : null}
          {slice.map((line, i) => (
            <Text color={t.color.text} key={i} wrap="truncate-end">
              {line || ' '}
            </Text>
          ))}
          <Box marginTop={1}>
            <Text color={t.color.muted}>
              {atEnd
                ? `end · ↑↓/jk · b/PgUp back · g top · Esc/q close (${pager.lines.length} lines)`
                : `↑↓/jk · Space/PgDn page · g/G top/bottom · Esc/q close (${Math.min(pager.offset + size, pager.lines.length)}/${pager.lines.length})`}
            </Text>
          </Box>
        </Box>
      </box>
    )
  }

  return null
}

// ── Startup welcome ─────────────────────────────────────────────────────────

function useDerafshAnimation(
  enabled: boolean,
  compact: boolean,
  colors: Theme['color'],
  linesRef: MutableRefObject<Array<TextRenderable | null>>
): void {
  const terminalFocused = useRef(true)

  useBlur(() => {
    terminalFocused.current = false
  })
  useFocus(() => {
    terminalFocused.current = true
  })

  useEffect(() => {
    if (!enabled) {
      return
    }

    let frame = 0
    const timer = setInterval(() => {
      if (!terminalFocused.current) {
        return
      }

      frame = (frame + 1) % DERAFSH_ANIMATION_FRAME_COUNT
      const next = compact ? derafshCompactGradientFrame(colors, frame) : derafshGradientFrame(colors, frame)

      // Updating React state here made React 19 reconcile the centered home
      // layout on every animation tick. In Apple Terminal those commits could
      // be captured between native frames, periodically leaving only the
      // Derafsh visible. The mark's geometry and text never change, so update
      // the stable OpenTUI text renderables in place and let their fg setters
      // coalesce one native redraw without touching the surrounding layout.
      for (let index = 0; index < next.length; index += 1) {
        const line = linesRef.current[index]

        if (line) {
          line.fg = next[index]![0]
        }
      }
    }, DERAFSH_ANIMATION_FRAME_MS)
    timer.unref?.()

    return () => clearInterval(timer)
  }, [colors, compact, enabled, linesRef])
}

function StartupWelcome({ cols, rows }: { cols: number; rows: number }) {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const markFits = cols >= DERAFSH_KAVIANI_WIDTH + 4
  const showFullMark = markFits && rows >= 32
  const useGradient = !t.bannerLogo
  const showCompactMark = useGradient && markFits && !showFullMark && rows >= 22
  const showMark = showFullMark || showCompactMark
  const animationEnabled = useGradient && showMark && derafshAnimationEnabled()
  const markLinesRef = useRef<Array<TextRenderable | null>>([])
  const mark = useMemo(() => {
    if (showCompactMark) {
      return derafshCompactGradientFrame(t.color, 0)
    }
    return useGradient ? derafshGradientFrame(t.color, 0) : derafshKaviani(t.color, t.bannerLogo || undefined)
  }, [showCompactMark, t.bannerLogo, t.color, useGradient])

  useDerafshAnimation(animationEnabled, showCompactMark, t.color, markLinesRef)

  return (
    <Box alignItems="center" flexDirection="column" flexShrink={0}>
      {showMark ? (
        <Box alignItems="center" flexDirection="column" flexShrink={0}>
          {mark.map(([color, line], index) => (
            <text
              fg={color || t.color.warn}
              flexShrink={0}
              key={index}
              ref={(renderable: TextRenderable | null) => {
                markLinesRef.current[index] = renderable
              }}
            >
              {line || ' '}
            </text>
          ))}
        </Box>
      ) : (
        <Text bold color={t.color.warn}>
          {DERAFSH_KAVIANI_GLYPH}
        </Text>
      )}
      <Text bold color={t.color.primary}>
        {t.brand.name}
      </Text>
      {!ui.info?.model?.trim() ? <Text color={t.color.muted}>Choose a model with /provider</Text> : null}
    </Box>
  )
}

/** Keep stream-cadence scrolling out of the heavyweight app controller. */
function LiveTailFollower({ scrollRef }: { scrollRef: AppLayoutProps['transcript']['scrollRef'] }) {
  const active = useTurnSelector(isLiveTailActive)
  const changeKey = useTurnSelector(liveTailScrollKey)
  const terminalFocused = useTerminalFocus()
  const sync = useCallback(() => {
    const scroll = scrollRef.current

    if (!shouldAutoScrollLiveTail(active, scroll)) {
      return
    }

    queueMicrotask(() => {
      if (shouldAutoScrollLiveTail(active, scrollRef.current)) {
        scrollRef.current?.scrollToBottom()
      }
    })
  }, [active, scrollRef])

  useEffect(sync, [changeKey, sync])
  useEffect(() => {
    if (terminalFocused) {
      sync()
    }
  }, [sync, terminalFocused])

  return null
}

function openTuiScrollAdapter(scrollbox: ScrollBoxRenderable): ScrollBoxHandle {
  let clampMin: number | undefined
  let clampMax: number | undefined
  let lastManualScrollAt = 0

  const clamp = (value: number) =>
    Math.max(clampMin ?? 0, Math.min(clampMax ?? Number.POSITIVE_INFINITY, Math.max(0, value)))
  const markManual = () => {
    lastManualScrollAt = Date.now()
  }

  return {
    getFreshScrollHeight: () => scrollbox.scrollHeight,
    getLastManualScrollAt: () => lastManualScrollAt,
    getPendingDelta: () => 0,
    getScrollHeight: () => scrollbox.scrollHeight,
    getScrollTop: () => scrollbox.scrollTop,
    getViewportHeight: () => scrollbox.viewport.height,
    getViewportTop: () => scrollbox.scrollTop,
    isSticky: () => scrollbox.scrollTop >= Math.max(0, scrollbox.scrollHeight - scrollbox.viewport.height - 1),
    scrollBy: delta => {
      markManual()
      scrollbox.scrollTo(clamp(scrollbox.scrollTop + delta))
    },
    scrollTo: y => {
      markManual()
      scrollbox.scrollTo(clamp(y))
    },
    scrollToBottom: () => scrollbox.scrollTo(Math.max(0, scrollbox.scrollHeight - scrollbox.viewport.height)),
    scrollToElement: element => {
      const id = typeof element === 'object' && element && 'id' in element ? String(element.id) : ''

      if (id) {
        scrollbox.scrollChildIntoView(id)
      }
    },
    setClampBounds: (min, max) => {
      clampMin = min
      clampMax = max
    },
    subscribe: listener => {
      const notify = () => listener()

      scrollbox.verticalScrollBar.on('change', notify)
      scrollbox.on('layout-changed', notify)

      return () => {
        scrollbox.verticalScrollBar.off('change', notify)
        scrollbox.off('layout-changed', notify)
      }
    }
  }
}

function InfoOverlay({ kind }: { kind: 'pluginsHub' | 'skillsHub' }) {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const close = () => patchOverlayState({ [kind]: false })

  useKeyboard(event => {
    if (event.name === 'escape' || event.sequence === 'q') {
      event.preventDefault()
      event.stopPropagation()
      close()
    }
  })

  const title = kind === 'skillsHub' ? 'Native skills' : 'Native plugins'

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={180}
    >
      <box
        backgroundColor={t.color.statusBg}
        flexDirection="column"
        flexShrink={0}
        maxWidth={90}
        minWidth={42}
        padding={2}
      >
        <box flexDirection="row" flexShrink={0} justifyContent="space-between">
          <text fg={t.color.accent} flexShrink={0}>
            <b>{title}</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc
          </text>
        </box>
        {kind === 'skillsHub' ? (
          <>
            <text fg={t.color.muted} flexShrink={0}>
              Run /skills to discover skills available to this session.
            </text>
            <text fg={t.color.muted} flexShrink={0}>
              Run /skill &lt;name&gt; to activate one.
            </text>
          </>
        ) : (
          <>
            <text fg={t.color.muted} flexShrink={0}>
              Run /plugins to inspect loaded native plugins and commands.
            </text>
            <text fg={t.color.muted} flexShrink={0}>
              Plugin mutation is not exposed by the native daemon.
            </text>
          </>
        )}
        <text fg={t.color.muted} flexShrink={0}>
          Esc/q close
        </text>
      </box>
    </box>
  )
}

// ── Layout root ─────────────────────────────────────────────────────────────

export function AppLayout({
  actions,
  composer,
  progress,
  status,
  transcript
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'progress' | 'status' | 'transcript'>) {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const overlay = useStore($overlayState)
  const liveAgents = useTurnSelector(state => state.subagents)
  const hasLiveTurn = useTurnSelector(state =>
    Boolean(
      state.streaming ||
        state.streamSegments.length ||
        state.streamPendingTools.length ||
        state.tools.length ||
        state.reasoning ||
        state.activity.length ||
        state.turnTrail.length ||
        state.todos.length ||
        state.outcome ||
        state.subagents.length
    )
  )
  const spawnHistory = useStore($spawnHistory)
  const { height, width } = useTerminalDimensions()
  const scrollboxRef = useCallback(
    (scrollbox: ScrollBoxRenderable | null) => {
      transcript.virtualHistory.setScrollHandle(scrollbox ? openTuiScrollAdapter(scrollbox) : null)
    },
    [transcript.virtualHistory.setScrollHandle]
  )
  const visibleRows = transcript.virtualRows.slice(transcript.virtualHistory.start, transcript.virtualHistory.end)
  const firstUserMessage = transcript.historyItems.find(message => message.role === 'user')?.text
  const sessionTitle = sessionDisplayTitle(ui.sessionTitle, firstUserMessage)
  const agentCount = useMemo(
    () => collectAgentPanelRecords(liveAgents, spawnHistory).length,
    [liveAgents, spawnHistory]
  )
  const showAgentSidebar = shouldShowAgentSidebar(width, agentCount)
  const sidebarWidth = agentSidebarWidth(width)
  const agentHotkeyBlocked = Boolean(
    overlay.approval ||
    overlay.clarify ||
    overlay.confirm ||
    overlay.modelPicker ||
    overlay.pager ||
    overlay.pluginsHub ||
    overlay.secret ||
    overlay.sessions ||
    overlay.skillsHub ||
    overlay.sudo
  )
  const footerAgentHint = showAgentSidebar ? undefined : 'F6 agents'
  const welcomeRightLabel = [footerAgentHint, ui.info?.version ? `v${ui.info.version}` : undefined]
    .filter(Boolean)
    .join(' · ')
  const pendingInteraction = Boolean(
    overlay.approval || overlay.clarify || overlay.confirm || overlay.secret || overlay.sudo
  )
  const showStartupWelcome = shouldShowStartupWelcome({
    busy: ui.busy,
    hasLiveTurn,
    pendingInteraction,
    transcriptEmpty: composer.empty
  })

  return (
    <Box
      backgroundColor={t.color.statusBg}
      flexDirection="column"
      flexGrow={1}
      height="100%"
      minHeight={0}
      position="relative"
      width="100%"
    >
      <LiveTailFollower scrollRef={transcript.scrollRef} />
      <AgentPanelHotkey
        disabled={agentHotkeyBlocked}
        onToggle={agents => patchOverlayState({ agents })}
        open={overlay.agents}
      />
      <Box flexDirection="row" flexGrow={1} minHeight={0} width="100%">
        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} minWidth={0}>
          {showStartupWelcome ? (
            <>
              <Box alignItems="center" flexDirection="column" flexGrow={1} minHeight={0} paddingX={2}>
                <Box flexGrow={1} minHeight={0} />
                {composer.completions.length ? null : <StartupWelcome cols={composer.cols} rows={height} />}
                <Box flexShrink={1} height={1} minHeight={0} />
                <Box
                  flexDirection="column"
                  flexShrink={0}
                  maxWidth={startupComposerWidth(composer.cols)}
                  width="100%"
                >
                  <PromptZone actions={actions} />
                  <Composer composer={composer} />
                </Box>
                <Box flexShrink={1} height={2} minHeight={0} />
                <Box flexGrow={1} minHeight={0} />
              </Box>
              <NoticeBanner notice={ui.notice} t={t} />
              <WorkspaceFooter cwdLabel={status.cwdLabel} rightLabel={welcomeRightLabel || undefined} t={t} />
            </>
          ) : (
            <Box flexDirection="column" flexGrow={1} minHeight={0}>
              <SessionHeader
                mode={ui.info?.mode}
                sessionId={ui.sid ?? ui.info?.session_id}
                sessionTitle={sessionTitle}
                t={t}
              />
              <Box flexDirection="column" flexGrow={1} gap={1} minHeight={0} paddingX={2} paddingY={1}>
                <scrollbox
                  ref={scrollboxRef}
                  stickyScroll
                  stickyStart="bottom"
                  style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
                  viewportCulling
                >
                  <Box flexDirection="column">
                    {transcript.virtualHistory.topSpacer > 0 ? (
                      <Box flexShrink={0} height={transcript.virtualHistory.topSpacer} />
                    ) : null}
                    {visibleRows.map(row => (
                      <box
                        flexDirection="column"
                        flexShrink={0}
                        key={row.key}
                        ref={transcript.virtualHistory.measureRef(row.key)}
                      >
                        <MessageLine msg={row.msg} t={t} />
                      </box>
                    ))}
                    <StreamingAssistant />
                    <CompactLiveProgress show={progress.showProgressArea} />
                    {transcript.virtualHistory.bottomSpacer > 0 ? (
                      <Box flexShrink={0} height={transcript.virtualHistory.bottomSpacer} />
                    ) : null}
                  </Box>
                </scrollbox>
                <Box flexDirection="column" flexShrink={0}>
                  <PromptZone actions={actions} />
                  <Composer composer={composer} />
                </Box>
              </Box>
              <NoticeBanner notice={ui.notice} t={t} />
              <WorkspaceFooter cwdLabel={status.cwdLabel} rightLabel={footerAgentHint} t={t} />
            </Box>
          )}
        </Box>
        {showAgentSidebar ? (
          <Box
            flexDirection="column"
            flexShrink={0}
            minHeight={0}
            paddingBottom={1}
            paddingRight={1}
            paddingTop={1}
            width={sidebarWidth}
          >
            <AgentPanel history={spawnHistory} liveAgents={liveAgents} t={t} />
          </Box>
        ) : null}
      </Box>

      <PagerOverlay composer={composer} />
      <ProviderPromptOverlay actions={actions} />

      {overlay.modelPicker ? <ModelPicker onSelect={actions.onModelSelect} /> : null}
      {overlay.sessions ? <SessionPicker actions={actions} /> : null}
      {overlay.agents ? (
        <AgentPanelOverlay
          history={spawnHistory}
          liveAgents={liveAgents}
          onClose={() => patchOverlayState({ agents: false })}
          t={t}
        />
      ) : null}
      {overlay.skillsHub ? <InfoOverlay kind="skillsHub" /> : null}
      {overlay.pluginsHub ? <InfoOverlay kind="pluginsHub" /> : null}
    </Box>
  )
}
