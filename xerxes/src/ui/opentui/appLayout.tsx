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

import type { AppLayoutActions, AppLayoutProps, Notice, SessionTab } from '../app/interfaces.js'
import { $attachments, attachmentsTotalBytes } from '../app/attachmentsStore.js'
import { focusComposer, registerComposerFocusTarget } from '../app/composerFocus.js'
import { setInputSelection } from '../app/inputSelectionStore.js'
import { isLiveTailActive, liveTailScrollKey, shouldAutoScrollLiveTail } from '../app/liveTailScroll.js'
import { $isBlocked, $overlayState, overlayBlocksBackgroundHotkeys, patchOverlayState } from '../app/overlayStore.js'
import { $panelWidthDelta, withPanelWidthDelta } from '../app/panelSizeStore.js'
import { $uiState, $uiTheme } from '../app/uiStore.js'
import { $turnLive, getTurnPulse, getTurnState, useTurnSelector } from '../app/turnStore.js'
import { $spawnHistory, spawnHistoryForSession } from '../app/spawnHistoryStore.js'
import {
  DERAFSH_ANIMATION_FRAME_COUNT,
  DERAFSH_ANIMATION_FRAME_MS,
  DERAFSH_KAVIANI_GLYPH,
  DERAFSH_KAVIANI_WIDTH,
  derafshAnimationEnabled,
  derafshCompactGradientFrame,
  derafshGradientFrame,
  derafshGradientRamp,
  derafshKaviani,
  WORDMARK_ROWS,
  wordmarkRows
} from '../banner.js'
import { agentSidebarWidth, shouldMountAgentSidebar, shouldShowAgentSidebar } from '../domain/agentPanelLayout.js'
import { densityFor, GLYPH, stateSkin, wrapWithContinuation } from '../domain/nocturne.js'
import { chipKey, type StartChip, startWithChips } from '../domain/startWith.js'
import { agentGroup } from '../lib/agentGroups.js'
import { useRepoPulse } from '../hooks/useRepoPulse.js'
import type { RepoPulse } from '../lib/repoPulse.js'
import { busyInputLabels } from '../domain/busyInputLabels.js'
import { sectionMode } from '../domain/details.js'
import { VOICE } from '../domain/roles.js'
import { completionToApplyOnSubmit } from '../domain/slash.js'
import { activeToken } from '../lib/completion.js'
import { shouldShowStartupWelcome, contentColumnWidth } from '../domain/startupLayout.js'
import {
  isProviderPrompt,
  providerPromptCancelAnswer,
  providerPromptChoices,
  providerPromptIsSecret,
  providerPromptTitle
} from '../domain/providerPrompt.js'
import { sessionDisplayTitle, sessionTelemetryLine, usageCounts, writePolicyLabel } from '../domain/statusFormat.js'
import { formatBytes } from '../lib/imageAttachment.js'
import { describeLiveness, type LivenessPhase, livenessGlyph, livenessTokens, livenessVerb, pulsingAccentColor } from '../lib/liveness.js'
import { unarchivedToolLines } from '../lib/liveProgress.js'
import { compactProgressRows, type CompactProgressRow } from '../lib/progressRows.js'
import { getActiveSkin } from '../lib/skinEngine.js'
import { compactStatusNumber, formatStatusDuration, isYoloEnabled } from '../lib/statusSnapshot.js'
import { fmtK, formatToolCall, toolTrailParts } from '../lib/text.js'
import { useTerminalFocus } from '../lib/terminalRuntime.opentui.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'
import { themeForMode, type Theme } from '../theme.js'

import { AgentPanel, AgentPanelHotkey, AgentPanelOverlay, collectAgentPanelRecords } from './agentPanel.js'
import { displayModeLabel, SessionHeader, SessionTabStrip, SessionTelemetryRow, WorkspaceFooter } from './appChrome.js'
import { CompletionMenu } from './completionMenu.js'
import { CopyPicker } from './copyPicker.js'
import { DiffPanelHotkey, DiffPanelOverlay } from './diffPanel.js'
import { TerminalPanelHotkey, TerminalPanelOverlay } from './terminalPanel.js'
import { MessageLine, StreamingMarkdown } from './messageLine.js'
import { ModelPicker } from './modelPicker.js'
import { OVERLAY_PANEL_SPECS, overlayPanelWidth, responsivePanelWidth } from './overlayLayout.js'
import { rebasePasteResult } from './pasteRebase.js'
import { ReasoningPicker } from './reasoningPicker.js'
import { Box, Span, Text } from './primitives.js'
import { SessionPicker } from './sessionPicker.js'

/**
 * Rows the session chrome takes before the transcript gets any: header, tab
 * strip, column padding, prompt zone, composer well (including the v2 border
 * ring — a rounded border costs one row per edge), hint row and footer.
 * 14 → 16 when the ring landed.
 */
const TRANSCRIPT_CHROME_ROWS = 16

const TEXTAREA_KEY_BINDINGS: KeyBinding[] = [
  { name: 'return', action: 'submit' },
  { name: 'return', shift: true, action: 'newline' },
  { name: 'kpenter', action: 'submit' },
  { name: 'kpenter', shift: true, action: 'newline' },
  { name: 'linefeed', action: 'submit' }
]

const decodePaste = (bytes: Uint8Array): string => new TextDecoder().decode(bytes)

// ── Live streaming turn ─────────────────────────────────────────────────

function StreamingAssistant({ cols }: { cols: number }) {
  const t = useStore($uiTheme)
  // Live rows get the same column budget as settled history so dotted
  // leaders and fold math do not restyle the moment the turn lands.
  //
  // This took `useTerminalDimensions().width` instead, which is NOT the same
  // number: settled rows measure against `composer.cols`, the session width
  // with the agent sidebar already subtracted. So the moment agents spawned
  // and the rail mounted, every live row sized its dotted leader for a column
  // ~40 cells wider than the box it was painted into, overflowed, and got a
  // `...` truncation marker stamped through the middle of the run.
  const streaming = useTurnSelector(s => s.streaming)
  const segments = useTurnSelector(s => s.streamSegments)
  const tools = useTurnSelector(s => s.tools)
  const pendingTools = useTurnSelector(s => s.streamPendingTools)
  const unsettledTools = unarchivedToolLines(segments, pendingTools)

  const anything = streaming || segments.length || tools.length || unsettledTools.length

  // While the turn has produced nothing renderable yet, the floating pill at
  // the transcript end (LiveProgressPill) is the single liveness surface.
  if (!anything) {
    return null
  }

  return (
    <Box flexDirection="column" flexShrink={0}>
      {segments.map((segment, index) => (
        <MessageLine cols={cols} key={`segment:${index}`} msg={segment} msgKey={`live-segment:${index}`} t={t} />
      ))}

      {unsettledTools.length ? (
        <MessageLine cols={cols} msg={{ kind: 'trail', role: 'system', text: '', tools: unsettledTools }} t={t} />
      ) : null}

      {/* Same shape as a settled ToolStep so an in-flight call does not
          restyle the moment it finishes — only the duration and mark are
          added. It carries no mark, which is what distinguishes "running". */}
      {tools.map(tool => {
        const { args, name } = toolTrailParts(formatToolCall(tool.name, tool.context))

        return (
          <Box flexShrink={0} key={tool.id} paddingLeft={3}>
            <Text color={t.color.muted} wrap="truncate-end">
              <Span color={t.color.muted}>{`${VOICE.tool(t).glyph} `}</Span>
              <Span bold color={t.color.toolName}>
                {name}
              </Span>
              {args ? <Span color={t.color.muted}>{`  ${args}`}</Span> : null}
            </Text>
          </Box>
        )
      })}

      {streaming ? (
        // Same wrapper as a settled AssistantMessage; StreamingMarkdown keeps
        // the growing buffer out of a full re-parse per delta. The gap is
        // conditional for the same reason the settled block's is: it
        // separates the prose from tool rows above it, but must not open the
        // band with a stray blank row when the prose is all there is.
        <Box
          flexDirection="column"
          flexShrink={0}
          marginTop={segments.length || unsettledTools.length || tools.length ? 1 : 0}
          paddingLeft={3}
        >
          <StreamingMarkdown text={streaming} t={t} />
        </Box>
      ) : null}
    </Box>
  )
}

/** Fast enough to read as motion, slow enough to stay off the render budget. */
const LIVE_INDICATOR_TICK_MS = 120

/** Verb lists ride along with the skin's branding, so ask the active skin. */
const activeSpinnerVerbs = (): string[] => {
  try {
    return getActiveSkin().spinnerVerbs()
  } catch {
    // A broken $XERXES_HOME/skins entry must cost the user their verbs, not
    // their only proof that the turn is still alive.
    return ['working']
  }
}

/**
 * Mockup 02's progress pill: one quiet row that floats at the transcript end —
 * "✻ verb elapsed · N tools · Xk tok (esc interrupt)" — and disappears the
 * moment the turn settles.
 *
 * It replaces the old inline activity line as THE live indicator: same turn
 * store, same stalled detection (the glyph goes hollow and warn-tinted when
 * the stream goes quiet), and the same discipline of painting stable
 * renderables from an interval instead of reconciling React eight times a
 * second. The detailed rows above it remain only for what a single row cannot
 * say (see CompactLiveProgress).
 */
export function LiveProgressPill() {
  const t = useStore($uiTheme)
  const live = useStore($turnLive)
  const glyphRef = useRef<TextRenderable | null>(null)
  const labelRef = useRef<TextRenderable | null>(null)
  const verbs = useMemo(activeSpinnerVerbs, [])
  const tone = useMemo<Record<LivenessPhase, string>>(
    () => ({ stalled: t.color.warn, streaming: t.color.accent, tool: t.color.accent }),
    [t]
  )

  useEffect(() => {
    if (!live) {
      return
    }

    const paint = () => {
      const turn = getTurnState()
      const pulse = getTurnPulse()
      const liveness = describeLiveness({
        lastDeltaAt: pulse.lastDeltaAt,
        now: Date.now(),
        startedAt: pulse.startedAt,
        toolCount: turn.tools.length
      })
      const tokens = livenessTokens(turn)
      const glyph = glyphRef.current
      const label = labelRef.current

      if (glyph) {
        // ✻ is the frozen "working" glyph; a stalled swap keeps the hollow ring.
        // While the turn streams, the glyph sweeps cyan → purple → cyan so
        // "alive" reads from across the room, not only on inspection.
        glyph.content = liveness.phase === 'stalled' ? `${livenessGlyph(liveness.phase, liveness.intensity)} ` : '✻ '
        glyph.fg = liveness.phase === 'stalled' ? tone.stalled : pulsingAccentColor(Date.now() - pulse.startedAt)
      }

      if (label) {
        const parts = [
          `${livenessVerb(verbs, liveness.elapsedMs)} ${formatStatusDuration(liveness.elapsedMs / 1000)}`,
          `${turn.tools.length} tool${turn.tools.length === 1 ? '' : 's'}`
        ]

        if (tokens > 0) {
          parts.push(`${compactStatusNumber(tokens)} tok`)
        }

        parts.push('(esc interrupt)')
        label.content = parts.join(' · ')
        label.fg = liveness.phase === 'stalled' ? t.color.warn : t.color.muted
      }
    }

    paint()

    const timer = setInterval(paint, LIVE_INDICATOR_TICK_MS)
    timer.unref?.()

    return () => clearInterval(timer)
  }, [live, t.color.warn, t.color.muted, tone, verbs])

  if (!live) {
    return null
  }

  return (
    <Box alignSelf="flex-start" backgroundColor={t.color.completionBg} flexShrink={0} marginTop={1} paddingX={2}>
      <text
        fg={tone.streaming}
        flexShrink={0}
        ref={(renderable: TextRenderable | null) => {
          glyphRef.current = renderable
        }}
      >
        {'  '}
      </text>
      <text
        fg={t.color.muted}
        flexShrink={0}
        ref={(renderable: TextRenderable | null) => {
          labelRef.current = renderable
        }}
        truncate
        wrapMode="none"
      >
        {' '}
      </text>
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
  // The pill owns verb/elapsed/tool-count while the turn runs, so the
  // detailed rows stay only for what one row cannot say: the todo checklist,
  // the outcome, and warnings/errors. Settled turns fall back to the full
  // list so a post-turn outcome summary survives completion exactly as before.
  const live = useStore($turnLive)
  const rows = useMemo(
    () => compactProgressRows({ activity, outcome, todos, turnTrail }, { activityVisible, toolsVisible }),
    [activity, activityVisible, outcome, todos, toolsVisible, turnTrail]
  )
  const visibleRows = useMemo(
    () =>
      live
        ? rows.filter(row => row.kind === 'todo' || row.kind === 'outcome' || (row.kind === 'activity' && row.tone !== 'info'))
        : rows,
    [live, rows]
  )

  if (!show || !visibleRows.length) {
    return null
  }

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} paddingLeft={3}>
      {visibleRows.map((row, index) => {
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
const APPROVAL_LABELS = { once: 'run it once', session: 'allow for this session', always: 'always allow this exact shape', deny: 'deny and tell the agent why' }
/** The letter each answer actually answers to, printed as its own cap. */
const APPROVAL_HOTKEY = { once: 'y', session: 'a', always: 'A', deny: 'n' }
/**
 * What each answer costs you next time. The canvas puts this on the right of
 * every option row for the same reason the home chips carry their counts:
 * the choice should be informed before the keypress, not after it.
 */
const APPROVAL_CONSEQUENCE = {
  once: 'asks again next time',
  session: 'until this session ends',
  always: 'writes a rule to your policy',
  deny: '⎋ denies silently'
}

export type ApprovalKeyChoice = 'deny' | 'once' | 'session'

/**
 * Direct letter bindings for the approval card (mockup 10): y approves once,
 * a approves for this session — deliberately the same session-scoped option
 * the list offers, never the permanent "always" — and n denies. Esc denies
 * separately in the key handler; numbers keep their quick-select role.
 */
export const APPROVAL_KEY_CHOICES: Readonly<Record<string, ApprovalKeyChoice>> = {
  a: 'session',
  n: 'deny',
  y: 'once'
}

export const approvalKeyChoice = (key: string): null | ApprovalKeyChoice =>
  APPROVAL_KEY_CHOICES[key.toLowerCase()] ?? null

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
  const panelWidth = responsivePanelWidth(width, { max: 84, min: 34 })

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
                  {'•'.repeat(Math.min(maskedValue.length, Math.max(0, panelWidth - 8))) || 'API key'}
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

// Exported for the approval-key test harness; AppLayout is its only runtime consumer.
export function PromptZone({ actions }: Pick<AppLayoutProps, 'actions'>) {
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  // The prompt zone renders inside the composer's reading column, so its own
  // wrapping has to measure against the same width the column does.
  const { width: composerColumns } = useTerminalDimensions()
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
      const letter = approvalKeyChoice(event.sequence ?? '')

      if (name === 'escape') {
        actions.answerApproval('deny')
      } else if (letter) {
        actions.answerApproval(letter)
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
    // Three statements, always in this order: what will run, who asked, why
    // you are being asked. Anything else on the card is a distraction at the
    // moment of consent — which is why the description, the scope note and
    // the keys are each folded into one of the three rather than added
    // alongside them.
    //
    // The tint comes from the policy verdict, not from how the command feels.
    // Amber is ask; there is no "this looks scary" tier. (Red is deny, and a
    // denied call never reaches this card: the daemon answers it without
    // asking, so there is no red branch to write until a wire verdict says
    // otherwise.)
    const tool = approval.command.trim().split(/\s+/)[0] ?? ''
    const scope = writePolicyLabel(ui.info?.permission_mode)

    return (
      <InlinePromptPanel accent={t.color.warn}>
        <Box alignItems="center" flexDirection="row" flexShrink={0} justifyContent="space-between">
          <Text wrap="truncate-end">
            <Span color={t.color.warn}>{`${GLYPH.state} `}</Span>
            <Span bold color={t.ds.title}>
              permission
            </Span>
            {tool ? (
              <>
                <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
                <Span color={t.ds.secondary}>{tool}</Span>
              </>
            ) : null}
          </Text>
          <Text color={t.color.warn}>POLICY: ASK</Text>
        </Box>

        <PromptPanelGap />
        <Text color={t.ds.caption}>WHAT WILL RUN</Text>
        {/* Verbatim, in a code box, never summarised. A paraphrase is a
            different command. */}
        {approval.command ? (
          <Box backgroundColor={t.ds.sunken} flexDirection="column" flexShrink={0} marginTop={1} paddingX={1}>
            {/* Wrapped by us, not by the renderer, so every continuation
                carries `↳`. A soft wrap and a second command look identical
                in a monospace column, and here that is the difference
                between running one thing and running two. */}
            {wrapWithContinuation(approval.command.slice(0, 320), contentColumnWidth(composerColumns) - 8).map(
              (segment, index) => (
                <Text color={index ? t.ds.caption : t.ds.strong} key={index}>
                  {index ? (
                    <>
                      <Span color={t.ds.caption}>{segment.slice(0, 2)}</Span>
                      <Span color={t.ds.strong}>{segment.slice(2)}</Span>
                    </>
                  ) : (
                    segment
                  )}
                </Text>
              )
            )}
          </Box>
        ) : null}
        {ui.info?.cwd ? (
          <Text color={t.ds.meta} wrap="truncate-end">
            {`cwd ${ui.info.cwd}`}
          </Text>
        ) : null}

        <PromptPanelGap />
        <Text color={t.ds.caption}>WHO ASKED</Text>
        <Text color={t.ds.prose} wrap="wrap">
          {approval.description}
        </Text>

        <PromptPanelGap />
        <Text color={t.ds.caption}>WHY YOU ARE SEEING THIS</Text>
        <Text color={t.ds.prose} wrap="wrap">
          {`Interaction mode is ${ui.info?.mode || 'code'} — ${scope}.`}
        </Text>
        {approval.allowPermanent === false ? (
          <Text color={t.ds.meta} wrap="wrap">
            The daemon will not honour a permanent allow for this call, so
            &quot;always&quot; is not offered.
          </Text>
        ) : (
          <Text color={t.ds.meta} wrap="wrap">
            &quot;always&quot; is scoped to this exact command shape, not to the {tool || 'tool'} tool. A different
            path, flag or argument asks again.
          </Text>
        )}

        <PromptPanelGap />
        {/* One row per answer, each stating its own consequence — the choice
            has to be informed before the keypress, same rule as the home
            chips. Deny is the ⎋ default. */}
        <Box flexDirection="column" flexShrink={0}>
          {opts.map((option, index) => (
            <Box
              alignItems="center"
              backgroundColor={sel === index ? t.color.selectionBg : undefined}
              flexDirection="row"
              flexShrink={0}
              justifyContent="space-between"
              key={option}
              paddingX={1}
            >
              <Text wrap="truncate-end">
                <Span color={sel === index ? t.color.warn : t.ds.separator}>{sel === index ? '› ' : '  '}</Span>
                <Span color={t.ds.secondary}>{APPROVAL_HOTKEY[option]}</Span>
                <Span color={sel === index ? t.ds.title : t.ds.secondary}>{` ${APPROVAL_LABELS[option]}`}</Span>
              </Text>
              <Text color={t.ds.meta} wrap="truncate-end">
                {APPROVAL_CONSEQUENCE[option]}
              </Text>
            </Box>
          ))}
        </Box>
        <PromptPanelGap />
        <Text color={t.ds.caption}>
          <Span color={t.ds.secondary}>↑↓</Span> move
          <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
          <Span color={t.ds.secondary}>⏎</Span> choose
          <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
          <Span color={t.ds.secondary}>⎋</Span> deny and go back
        </Text>
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

/**
 * The composer identity chip: the active permission mode in gold.
 *
 * Idle only. While a turn runs the chip disappears — the ◆ activity line in
 * the status row below owns that moment, and a second diamond beside the
 * textarea read as two competing indicators, not one system.
 */
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

/** Pending /image attachments indicator: one muted line above the textarea. */
function AttachmentsPanel() {
  const attachments = useStore($attachments)
  const t = useStore($uiTheme)

  if (!attachments.length) {
    return null
  }

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} paddingX={2}>
      <Text color={t.color.muted} wrap="truncate-end">
        {'📎 '}
        {attachments.map(item => item.name).join(', ')}
        {` · ${formatBytes(attachmentsTotalBytes(attachments))} · next message only · /image clear to drop`}
      </Text>
    </Box>
  )
}

export function Composer({ composer }: Pick<AppLayoutProps, 'composer'>) {
  const ui = useStore($uiState)
  const isBlocked = useStore($isBlocked)
  const t = useStore($uiTheme)
  const ref = useRef<TextareaRenderable | null>(null)

  // Share the live textarea with the global double-space refocus gesture.
  useEffect(() => {
    registerComposerFocusTarget(ref.current)
    return () => registerComposerFocusTarget(null)
  }, [])

  const modelLabel = ui.info?.model || 'choose model with /provider'
  const modeLabel = ui.info?.mode || 'code'
  // 'code' is the assumed mode; every other one changes what the turn is
  // allowed to do and deserves to read as an exception rather than as chrome.
  const modeIsDefault = modeLabel === 'code'
  const yoloEnabled = isYoloEnabled(ui.info?.permission_mode)
  const narrow = composer.cols < 76
  // Below this the identity alone (model + YOLO + context + tokens) already
  // fills the row, so any hint would have to eat into it. Showing nothing
  // beats showing a hint welded onto a half-truncated token count.
  const cramped = composer.cols < 52

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

  const applyPasteResult = useCallback(
    (result: null | { cursor: number; value: string }, captured: { cursor: number; value: string }) => {
      const textarea = ref.current

      if (!result || !textarea) {
        return
      }

      const rebased = rebasePasteResult(captured, result, {
        cursor: textarea.cursorOffset,
        value: textarea.plainText
      })

      applyDraft(rebased.value, rebased.cursor)
    },
    [applyDraft]
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
    ).then(result => applyPasteResult(result, { cursor, value }))
  })

  // Ctrl+V smart paste: terminals only deliver Cmd+V / bracketed paste when
  // the clipboard carries TEXT — an image-only clipboard produces no event at
  // all, so a chord the TUI can actually receive is the only way to paste
  // images. The hotkey path pastes clipboard text normally and attaches a
  // clipboard image (with visible feedback) when there is no usable text.
  useKeyboard(event => {
    if (event.name !== 'v' || !event.ctrl || event.meta || event.super || event.shift) {
      return
    }

    const textarea = ref.current

    if (isBlocked || !textarea) {
      return
    }

    event.preventDefault()

    const value = textarea.plainText
    const cursor = textarea.cursorOffset

    void Promise.resolve(
      composer.handleTextPaste({ bracketed: false, cursor, hotkey: true, text: '', value })
    ).then(result => applyPasteResult(result, { cursor, value }))
  })

  // Say what Enter will actually do. The mode is configurable and defaults to
  // steer, so a hardcoded "queue" label misreported the common case.
  const busyLabels = busyInputLabels(ui.busyInputMode, composer.queuedDisplay.length)
  // What the turn is doing right now, read from the same turn state the live
  // indicator uses so the two can never disagree.
  const latestActivity = useTurnSelector(s => s.activity.at(-1)?.text ?? '')
  const liveActivity = latestActivity.trim() || 'working'

  return (
    <Box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} width="100%">
      {/* The menu renders inside the composer's reading column, not the full
          terminal, so its column math must use the same measure — otherwise
          it lays out for a width it does not have and the renderer truncates
          the descriptions mid-string. */}
      <CompletionMenu
        compIdx={composer.compIdx}
        completions={composer.completions}
        query={activeToken(composer.input)}
        width={contentColumnWidth(composer.cols)}
      />
      {/* The v2 input well: one rounded ring around everything that belongs
          to the prompt — queued lines, attachments, the textarea, and the
          identity row. Gold while you can type, hairline while an overlay
          owns the screen. Cramped terminals keep the two rows the border
          would cost. */}
      <Box
        flexDirection="column"
        flexShrink={0}
        {...(cramped
          ? {}
          : {
              borderColor: isBlocked ? t.color.border : t.color.brandGold,
              borderStyle: 'round' as const
            })}
      >
      <QueuePanel composer={composer} />
      <AttachmentsPanel />
      <Box
        alignItems="flex-start"
        backgroundColor={t.color.completionCurrentBg}
        flexDirection="row"
        flexShrink={0}
        gap={1}
        minHeight={1}
        paddingX={1}
      >
        {/* The prompt glyph, and only the prompt glyph. A mode chip used to
            sit here as well, which stated the mode twice on one screen — the
            row below now owns that, so this column is free to be what the
            canvas draws: the mark that says you are typing. */}
        <Text color={t.color.accent}>{t.brand.prompt}</Text>
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
              // The canvas prints "reply, or ⎋ to interrupt the current turn"
              // on a WORKING session. Showing it whenever the transcript is
              // non-empty made an idle composer claim a turn was running and
              // offer to interrupt something that had already finished.
              ui.busy
                ? busyLabels.placeholder
                : composer.empty
                  ? 'describe a task, paste a stack trace, or press / for commands'
                  : 'reply, or press / for commands'
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
        // A guaranteed gutter between the identity and the hints. Without it
        // `space-between` lets the two groups meet with no separator the
        // moment the row is tight, which renders as "YOLO ONTab modes" —
        // reading as corrupted text rather than as a truncation.
        // No rule between the input and this row: a per-side border inside a
        // box that already has a full frame makes OpenTUI paint the edge
        // THROUGH the text, so the identity row came out as
        // `╰─◆─code─mode─·─…─╯`. The canvas's hairline is a ground change
        // here instead, which a terminal can actually draw.
        backgroundColor={t.ds.chrome}
        gap={2}
        height={1}
        justifyContent="space-between"
        paddingX={1}
      >
        {/* Two tiers, not one row of everything.
            The model and permission mode barely change all session; the
            context percentage and token counts change every few seconds.
            Packing both into one line is what made them collide at narrow
            widths — the collision was the symptom, this is the cause.
            While a turn runs the row answers the question you actually have
            (what is it doing, how much context is left); when nothing is
            running it states identity and gets out of the way. */}
        {/* Mode, model and write policy, stated together: those three decide
            what the next ⏎ is allowed to do, and knowing two of them is not
            knowing. The context read-out moved to the session header, where
            it answers "what is happening" instead of competing here. */}
        <Box alignItems="center" flexDirection="row" flexShrink={1} gap={1} height={1} overflow="hidden">
          {ui.busy ? (
            <>
              <Text color={t.color.system}>{GLYPH.mode}</Text>
              {/* The mode stays on screen while the turn runs. It used to be
                  swapped out for the activity, so the one moment you need to
                  know you are in plan or objective mode — something is running
                  and behaving in a way you did not ask for — was the one moment
                  it was hidden. A non-default mode is coloured, because that is
                  the state worth catching the eye. */}
              <Text wrap="truncate-end">
                <Span color={modeIsDefault ? t.ds.meta : t.color.warn}>{modeLabel}</Span>
                <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
                <Span color={t.ds.title}>{liveActivity}</Span>
              </Text>
            </>
          ) : (
            <Text wrap="truncate-end">
              <Span color={t.color.accent}>{`${GLYPH.mode} `}</Span>
              <Span color={t.ds.secondary}>{`${modeLabel} mode`}</Span>
              <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
              <Span color={t.ds.meta}>{modelLabel}</Span>
              {narrow ? null : (
                <>
                  <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
                  <Span color={yoloEnabled ? t.color.warn : t.ds.meta}>
                    {writePolicyLabel(ui.info?.permission_mode)}
                  </Span>
                </>
              )}
            </Text>
          )}
        </Box>
        <Box alignItems="center" flexDirection="row" flexShrink={0} gap={1} height={1}>
          {ui.busy ? (
            <Text color={t.color.text}>
              Enter <Span color={t.color.muted}>{busyLabels.enter}</Span> · Esc{' '}
              <Span color={t.color.muted}>{busyLabels.escape}</Span>
            </Text>
          ) : composer.completions.length ? (
            <Text color={t.color.text}>
              Tab <Span color={t.color.muted}>accept</Span> · ↑↓ <Span color={t.color.muted}>navigate</Span> · Esc{' '}
              <Span color={t.color.muted}>dismiss</Span>
            </Text>
          ) : cramped ? null : narrow ? (
            <Text color={t.ds.caption}>
              <Span color={t.ds.secondary}>tab</Span> mode
            </Text>
          ) : (
            // Lowercase keys, exactly as typed, in the footer's own order:
            // what changes the turn, then what edits the draft, then send.
            <Text color={t.ds.caption}>
              <Span color={t.ds.secondary}>tab</Span> mode
              <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
              <Span color={t.ds.secondary}>/</Span> commands
              <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
              <Span color={t.ds.secondary}>@</Span> files
              <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
              <Span color={t.ds.secondary}>⇧⏎</Span> newline
              <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
              <Span color={t.ds.secondary}>⏎</Span> send
            </Text>
          )}
        </Box>
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
        <Box
          backgroundColor={t.color.statusBg}
          flexDirection="column"
          padding={2}
          width={overlayPanelWidth(composer.cols, OVERLAY_PANEL_SPECS.pager)}
        >
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
                ? `end · ↑↓/jk · b/⌃b back · g top · Esc/q close (${pager.lines.length} lines)`
                : `↑↓/jk · Space/⌃f page · g/G top/bottom · Esc/q close (${Math.min(pager.offset + size, pager.lines.length)}/${pager.lines.length})`}
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

/**
 * The Derafsh mark, strictly for the empty startup state.
 *
 * It lives ONLY on the welcome screen. Once a session has any transcript
 * content the mark must never reappear — not as a filler under a short
 * conversation, not centered in spare space. Decorative branding inside an
 * active conversation competes with the work on screen and reads as stale
 * layout, so this renders null the moment `contentRows > 0`.
 */
function TranscriptWatermark({ cols, contentRows, rows }: { cols: number; contentRows: number; rows: number }) {
  const t = useStore($uiTheme)
  if (contentRows > 0) {
    return null
  }
  const compact = derafshCompactGradientFrame(t.color, 0)
  const fitsWidth = cols >= DERAFSH_KAVIANI_WIDTH + 4
  // Chrome the transcript never gets: header, tab strip, column padding,
  // prompt zone, composer, hint row and footer.
  const viewport = rows - TRANSCRIPT_CHROME_ROWS
  const marginTop = Math.max(2, Math.floor((viewport - compact.length) / 2))
  const fits = compact.length + marginTop <= viewport

  if (!fitsWidth || !fits || !derafshAnimationEnabled()) {
    return null
  }

  return (
    <Box alignItems="center" flexDirection="column" flexShrink={0} marginTop={marginTop}>
      {compact.map(([, line], index) => (
        // One flat dim colour rather than the welcome screen's gradient: this
        // sits behind an empty prompt, and must never compete with it.
        <Text color={t.color.turnRail} key={index}>
          {line}
        </Text>
      ))}
    </Box>
  )
}

/**
 * One START WITH chip: a key cap, a state mark, what it does, and — the whole
 * point — what is true right now that makes it worth pressing.
 *
 * Filling the composer rather than submitting is deliberate. A chip is an
 * entry point, not a command button: you still read what it wrote and still
 * press ⏎, so a mis-hit costs a glance instead of a turn.
 */
function StartChipRow({
  chip,
  cols,
  composer,
  index,
  t
}: {
  chip: StartChip
  cols: number
  composer: AppLayoutProps['composer']
  index: number
  t: Theme
}) {
  const key = chipKey(index)
  const skin = stateSkin(chip.tone, t.ds)
  const density = densityFor(cols)

  // A bordered pill on the card ground, exactly as the canvas draws it. The
  // first pass rendered a flat row, which read as a list item rather than as
  // something you press — and the whole point of the band is that these are
  // entry points.
  return (
    <Box
      backgroundColor={t.ds.card}
      borderColor={t.ds.hairline}
      borderStyle="round"
      flexShrink={0}
      onClick={() => {
        composer.updateInput(chip.prompt)
        focusComposer()
      }}
      paddingX={1}
    >
      <Text wrap="truncate-end">
        {key ? <Span color={t.ds.secondary}>{`${key} `}</Span> : null}
        <Span color={skin.dot}>{`${GLYPH.tool} `}</Span>
        {chip.command ? <Span color={t.color.accent}>{`${chip.command} `}</Span> : null}
        <Span color={t.ds.title}>{chip.label}</Span>
        {density.goals ? (
          <>
            <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
            <Span color={t.ds.meta}>{chip.consequence}</Span>
          </>
        ) : null}
      </Text>
    </Box>
  )
}

/**
 * Digit shortcuts for the chips.
 *
 * Bound only while the draft is empty, which is the one moment a bare digit
 * cannot be the start of a sentence you are typing. The canvas draws these
 * key caps on the chips, and a key cap the product does not honour is worse
 * than no key cap at all — but so is a shortcut that eats the "1" in
 * "1.2.8 broke my build", so both halves of that have to hold.
 */
function StartChipKeys({
  chips,
  composer,
  enabled
}: {
  chips: readonly StartChip[]
  composer: AppLayoutProps['composer']
  enabled: boolean
}) {
  useKeyboard((event: KeyEvent) => {
    if (!enabled || event.ctrl || event.meta || event.super || event.option) {
      return
    }

    const index = chips.findIndex((_, position) => chipKey(position) === event.name)

    if (index < 0) {
      return
    }

    // Both, not just preventDefault: the textarea is focused on this screen
    // and would otherwise append the digit after the prompt we just wrote.
    event.preventDefault()
    event.stopPropagation()
    composer.updateInput(chips[index]!.prompt)
    focusComposer()
  })

  return null
}

export function StartupWelcome({
  cols,
  composer,
  pulse,
  rows
}: {
  cols: number
  composer: AppLayoutProps['composer']
  /** Working-tree state, polled once for the whole screen by `AppLayout`. */
  pulse: RepoPulse
  rows: number
}) {
  const ui = useStore($uiState)
  const t = useStore($uiTheme)
  const liveAgents = useTurnSelector(state => state.subagents)
  const markFits = cols >= DERAFSH_KAVIANI_WIDTH + 4
  // The canvas gives the mark about a third of the screen, not half. At the
  // reference 150x40 the full 20-row mark plus a six-row wordmark leaves no
  // room for the chips, so the half-height mark is the DEFAULT here and the
  // full one is reserved for terminals tall enough to spend the rows.
  const showFullMark = markFits && rows >= 48
  const useGradient = !t.bannerLogo
  const showCompactMark = useGradient && markFits && !showFullMark && rows >= 22
  const showMark = showFullMark || showCompactMark
  const animationEnabled = useGradient && showMark && derafshAnimationEnabled()
  // The wordmark is letter-spaced and tinted across the same Derafsh ramp the
  // boot mark animates. It follows brand.name, so a skin that renames the
  // agent inherits the treatment; [...name] keeps any astral character in a
  // custom name intact.
  const nameLetters = [...t.brand.name]
  // Block letters when the name can be drawn and the rows exist; otherwise
  // the letter-spaced form. A wordmark that has to compete with a 20-row mark
  // above it cannot be one row tall.
  const blockWordmark = useMemo(() => wordmarkRows(t.brand.name), [t.brand.name])
  const showBlockWordmark = blockWordmark.length > 0 && cols >= blockWordmark[0]!.length + 4
  const wordmarkRamp = useMemo(
    () => derafshGradientRamp(t.color, showBlockWordmark ? WORDMARK_ROWS : nameLetters.length),
    [nameLetters.length, showBlockWordmark, t.color]
  )
  const markLinesRef = useRef<Array<TextRenderable | null>>([])
  const mark = useMemo(() => {
    if (showCompactMark) {
      return derafshCompactGradientFrame(t.color, 0)
    }
    return useGradient ? derafshGradientFrame(t.color, 0) : derafshKaviani(t.color, t.bannerLogo || undefined)
  }, [showCompactMark, t.bannerLogo, t.color, useGradient])

  // "A chip with nothing true to say is not shown" — so the list is derived
  // from live signals every render rather than being a constant.
  const chips = useMemo(
    () =>
      startWithChips({
        agentsNeedingInput: liveAgents.filter(agent => agentGroup(agent.status) === 'input').length,
        agentsWorking: liveAgents.filter(agent => agentGroup(agent.status) === 'working').length,
        hasModel: Boolean(ui.info?.model?.trim()),
        pulse
      }),
    [liveAgents, pulse, ui.info?.model]
  )
  // Rows the chips may take without pushing the composer off a short screen.
  // The composer is the one thing on this screen that never degrades, so it
  // is subtracted first and the chips get whatever is left.
  const markRows = showMark ? mark.length : 1
  const wordmarkRowCount = showBlockWordmark ? WORDMARK_ROWS : 1
  // Bordered pills cost three rows per wrapped line, and roughly two fit on a
  // reading column at the reference width.
  const chipsPerLine = Math.max(1, Math.floor(contentColumnWidth(cols) / 52))
  const chipLines = Math.max(0, Math.floor((rows - markRows - wordmarkRowCount - 14) / 3))
  const visibleChips = chips.slice(0, Math.max(0, chipLines * chipsPerLine))

  useDerafshAnimation(animationEnabled, showCompactMark, t.color, markLinesRef)

  return (
    // Flush left, whitespace on the right: the mark, the wordmark, the
    // tagline and the chips all start on the same column, so the eye walks
    // one edge down the screen instead of re-centering on every band.
    <Box flexDirection="column" flexShrink={0} width="100%">
      <StartChipKeys chips={visibleChips} composer={composer} enabled={!composer.input} />
      {showMark ? (
        <Box flexDirection="column" flexShrink={0}>
          {mark.map(([color, line], index) => (
            <text
              fg={color || t.color.accent}
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
        <Text bold color={t.color.accent}>
          {DERAFSH_KAVIANI_GLYPH}
        </Text>
      )}
      <Box flexShrink={0} height={1} />
      {showBlockWordmark ? (
        // One ramp stop per ROW, so the gradient runs down the wordmark the
        // way the canvas runs it across the word.
        <Box flexDirection="column" flexShrink={0}>
          {blockWordmark.map((row, index) => (
            <Text color={wordmarkRamp[index]} key={index}>
              {row}
            </Text>
          ))}
        </Box>
      ) : nameLetters.length ? (
        <Text bold>
          {nameLetters.map((letter, index) => (
            <Span color={wordmarkRamp[index]} key={index}>
              {letter}
              {index < nameLetters.length - 1 ? ' ' : ''}
            </Span>
          ))}
        </Text>
      ) : null}
      <Text color={t.ds.meta} wrap="truncate-end">
        {t.brand.welcome}
      </Text>
      {visibleChips.length ? (
        <>
          <Box flexShrink={0} height={1} />
          <Text color={t.ds.caption}>START WITH</Text>
          {/* A wrapping cluster, not a stacked list: the canvas lays the chips
              out as pills that flow across the column, so a screen with four
              live signals reads as a cluster rather than as a menu. */}
          <Box flexDirection="row" flexShrink={0} flexWrap="wrap" gap={1}>
            {visibleChips.map((chip, index) => (
              <StartChipRow chip={chip} cols={cols} composer={composer} index={index} key={chip.id} t={t} />
            ))}
          </Box>
        </>
      ) : null}
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
  const t = useStore($uiTheme)
  const { width } = useTerminalDimensions()
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
        padding={2}
        width={overlayPanelWidth(width, OVERLAY_PANEL_SPECS.info)}
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

/**
 * Claude Code-style agent navigation, extended to the mockup 04 gesture: with
 * an empty composer, Left walks one tab to the left, and Left on the LEFTMOST
 * tab (or with a single tab) leaves the attached session for the agent view
 * without cancelling its work — the "two Left presses" backgrounding flow.
 * The picker owns selection and Right re-enters the highlighted session.
 *
 * Left is only claimed when it has no editing job to do: the composer input
 * must be empty (Left still moves the caret in text) and no overlay may own
 * the keyboard. Anything else falls through to the textarea untouched.
 */
export function SessionTabsHotkey({
  actions,
  activeId,
  composerEmpty,
  disabled,
  onOpenAgentView,
  tabs
}: {
  actions: Pick<AppLayoutActions, 'activateLiveSession' | 'newLiveSession'>
  activeId: null | string
  busy: boolean
  composerEmpty: boolean
  disabled: boolean
  onOpenAgentView: () => void
  tabs: readonly SessionTab[]
}) {
  useKeyboard(event => {
    if (disabled || !composerEmpty || event.name !== 'left') {
      return
    }
    if (event.ctrl || event.meta || event.super || event.shift) {
      return
    }

    // A tab to the left wins over the agent view: the first Left detaches one
    // step, and only the Left that has nowhere left to go backgrounds the
    // session. An untracked active id (the startup screen) has no tabs at
    // all, so it falls through to the view exactly as before.
    const activeIndex = activeId ? tabs.findIndex(tab => tab.id === activeId) : -1
    const leftNeighbor = activeIndex > 0 ? tabs[activeIndex - 1] : undefined

    consumeKey(event)

    if (leftNeighbor) {
      actions.activateLiveSession(leftNeighbor.id)
    } else {
      onOpenAgentView()
    }
  })

  return null
}

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
  const allSpawnHistory = useStore($spawnHistory)
  const spawnHistory = useMemo(
    () => spawnHistoryForSession(allSpawnHistory, ui.sid),
    [allSpawnHistory, ui.sid]
  )
  const { height, width } = useTerminalDimensions()
  const scrollboxRef = useCallback(
    (scrollbox: ScrollBoxRenderable | null) => {
      transcript.virtualHistory.setScrollHandle(scrollbox ? openTuiScrollAdapter(scrollbox) : null)
    },
    [transcript.virtualHistory.setScrollHandle]
  )
  const visibleRows = transcript.virtualRows.slice(transcript.virtualHistory.start, transcript.virtualHistory.end)
  const sessionTitle = sessionDisplayTitle(ui.sessionTitle)
  const agentCount = useMemo(
    () => collectAgentPanelRecords(liveAgents, spawnHistory).length,
    [liveAgents, spawnHistory]
  )
  const agentSidebarFits = shouldShowAgentSidebar(width, agentCount)
  const showAgentSidebar = shouldMountAgentSidebar(width, agentCount, overlay.agents)
  const panelWidthDelta = useStore($panelWidthDelta)
  const sidebarWidth = withPanelWidthDelta(agentSidebarWidth(width), width)
  void panelWidthDelta
  // Derived from the overlayStore policy table (OVERLAY_BLOCKS_BACKGROUND_HOTKEYS):
  // every overlay blocks the background hotkeys except `agents` itself, whose
  // F6 chord must stay live so it can also close the agents overlay.
  const agentHotkeyBlocked = overlayBlocksBackgroundHotkeys(overlay)
  // Keyed off whether the rail *fits*, not whether it is showing right now, so
  // opening the overlay does not swap the footer text underneath the backdrop
  // and swap it back on close.
  // The long hint only survives when the whole right side (hints + version
  // + provider dot) fits beside a truncated path; otherwise the flex-end clip
  // beheads it ("..terminals"). Bare keys are the mockup's narrow tier.
  const compactFooterHints = composer.cols < 100
  const footerAgentHint = agentSidebarFits
    ? undefined
    : compactFooterHints
      ? 'F6 · F7 · F8'
      : 'F6 agents · F7 diff · F8 terminals'
  // Compact tier drops the version too: at 80 cols the right side (hints +
  // version + provider dot) overflows by a character and the flex-end clip
  // beheads the F-keys. The wide footer keeps it.
  const welcomeRightLabel = [
    footerAgentHint,
    !compactFooterHints && ui.info?.version ? `v${ui.info.version}` : undefined
  ]
    .filter(Boolean)
    .join(' · ')
  // One poll for the whole screen: the statusbar's branch and dirty count and
  // the home chips' file totals are the same question asked twice.
  const pulse = useRepoPulse(ui.info?.cwd ?? '')
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
        // Clearing the inspect target on every toggle keeps a click from
        // sticking: reopening with F6 later should land on the list.
        onToggle={agents => patchOverlayState({ agents, agentsInspectId: null })}
        open={overlay.agents}
        resizeEnabled={overlay.agents || showAgentSidebar}
      />
      <DiffPanelHotkey
        disabled={agentHotkeyBlocked || overlay.agents}
        onToggle={diff => patchOverlayState({ diff })}
        open={overlay.diff}
      />
      <TerminalPanelHotkey
        disabled={agentHotkeyBlocked || overlay.agents}
        onToggle={terminals => patchOverlayState({ terminals })}
        open={overlay.terminals}
      />
      <SessionTabsHotkey
        actions={actions}
        activeId={ui.sid ?? ui.info?.session_id ?? null}
        busy={ui.busy}
        composerEmpty={!composer.input && composer.inputBuf.length === 0}
        disabled={agentHotkeyBlocked}
        onOpenAgentView={() => patchOverlayState({ sessions: true })}
        tabs={ui.sessionTabs}
      />
      <Box flexDirection="row" flexGrow={1} minHeight={0} width="100%">
        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} minWidth={0}>
          {showStartupWelcome ? (
            <>
              {/* Flush left, not centred. The canvas is explicit that content
                  hugs the left edge with the whitespace on the right, and the
                  welcome column is the same reading column the composer
                  below it uses — so the mark, the chips and the caret all
                  start on one line down the screen. */}
              <Box alignItems="flex-start" flexDirection="column" flexGrow={1} minHeight={0} paddingX={2}>
                <Box flexGrow={1} minHeight={0} />
                {composer.completions.length ? null : (
                  <Box
                    flexDirection="column"
                    flexShrink={0}
                    maxWidth={contentColumnWidth(composer.cols)}
                    width="100%"
                  >
                    <StartupWelcome cols={composer.cols} composer={composer} pulse={pulse} rows={height} />
                  </Box>
                )}
                <Box flexShrink={1} height={1} minHeight={0} />
                <Box
                  flexDirection="column"
                  flexShrink={0}
                  maxWidth={contentColumnWidth(composer.cols)}
                  width="100%"
                >
                  <PromptZone actions={actions} />
                  <Composer composer={composer} />
                </Box>
                <Box flexShrink={1} height={2} minHeight={0} />
                <Box flexGrow={1} minHeight={0} />
              </Box>
              <NoticeBanner notice={ui.notice} t={t} />
              <WorkspaceFooter
                cwdLabel={status.cwdLabel}
                providerModel={ui.info?.model}
                pulse={pulse}
                rightLabel={welcomeRightLabel || undefined}
                t={t}
              />
            </>
          ) : (
            <Box flexDirection="column" flexGrow={1} minHeight={0}>
              <SessionHeader
                busy={ui.busy}
                contextMax={usageCounts(ui.usage).max}
                contextUsed={usageCounts(ui.usage).used}
                mode={ui.info?.mode}
                sessionId={ui.sid ?? ui.info?.session_id}
                sessionTitle={sessionTitle}
                t={t}
              />
              <SessionTelemetryRow line={sessionTelemetryLine(ui.usage)} t={t} />
              <SessionTabStrip
                activeId={ui.sid ?? ui.info?.session_id ?? null}
                onNewTab={() => actions.newLiveSession()}
                onSelect={id => actions.activateLiveSession(id)}
                tabs={ui.sessionTabs}
                t={t}
                width={composer.cols}
              />
              <Box flexDirection="column" flexGrow={1} gap={1} minHeight={0} paddingX={2} paddingY={1}>
                <scrollbox
                  ref={scrollboxRef}
                  stickyScroll
                  stickyStart="bottom"
                  style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
                  viewportCulling
                >
                  {/* One reading column for the whole transcript, shared with
                      the composer below it. Capping only the prose would leave
                      tool rows and the user band running to the full terminal
                      width — a ragged right edge that reads as less designed,
                      not more. */}
                  {/* Top-anchored, with the composer pinned below — the
                      canvas layout. Bottom-anchoring this was a worse trade:
                      it moved the dead space from under the content to ABOVE
                      it, so a tall terminal opened on a header, forty empty
                      rows, and the conversation huddled at the bottom. The
                      scrollbox's sticky-bottom already keeps the newest row
                      in view once the transcript overflows. */}
                  <Box
                    flexDirection="column"
                    maxWidth={contentColumnWidth(composer.cols)}
                  >
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
                        <MessageLine
                          cols={composer.cols}
                          leadGap={row.leadGap}
                          msg={row.msg}
                          msgKey={row.key}
                          rail={row.rail}
                          t={t}
                          turnSeconds={row.turnSeconds}
                          turnTools={row.turnTools}
                        />
                      </box>
                    ))}
                    <StreamingAssistant cols={composer.cols} />
                    <CompactLiveProgress show={progress.showProgressArea} />
                    {/* Mockup 02: the quiet progress pill floats at the very end
                        of the live tail and unmounts on completion. */}
                    <LiveProgressPill />
                    <TranscriptWatermark
                      cols={composer.cols}
                      contentRows={transcript.virtualHistory.totalHeight}
                      rows={height}
                    />
                    {transcript.virtualHistory.bottomSpacer > 0 ? (
                      <Box flexShrink={0} height={transcript.virtualHistory.bottomSpacer} />
                    ) : null}
                  </Box>
                </scrollbox>
                {/* Same reading column as the welcome screen, so sending the
                    first message no longer snaps the input from 104 columns to
                    the full terminal width. Left-aligned rather than centered:
                    the transcript above it is left-aligned at this same
                    padding, and centering would offset the two. */}
                <Box
                  flexDirection="column"
                  flexShrink={0}
                  maxWidth={contentColumnWidth(composer.cols)}
                  width="100%"
                >
                  <PromptZone actions={actions} />
                  <Composer composer={composer} />
                </Box>
              </Box>
              <NoticeBanner notice={ui.notice} t={t} />
              <WorkspaceFooter
                cwdLabel={status.cwdLabel}
                providerModel={ui.info?.model}
                pulse={pulse}
                rightLabel={footerAgentHint}
                t={t}
              />
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
            <AgentPanel
              history={spawnHistory}
              liveAgents={liveAgents}
              onInspect={agentId => patchOverlayState({ agents: true, agentsInspectId: agentId })}
              t={t}
            />
          </Box>
        ) : null}
      </Box>

      <PagerOverlay composer={composer} />
      <ProviderPromptOverlay actions={actions} />

      {overlay.modelPicker ? <ModelPicker onSelect={actions.onModelSelect} /> : null}
      {overlay.reasoningPicker ? <ReasoningPicker onSelect={actions.onReasoningSelect} /> : null}
      {overlay.sessions ? <SessionPicker actions={actions} /> : null}
      {overlay.copyPicker ? <CopyPicker onCopied={actions.sys} /> : null}
      {overlay.diff ? (
        <DiffPanelOverlay cwd={ui.info?.cwd} onClose={() => patchOverlayState({ diff: false })} t={t} />
      ) : null}
      {overlay.agents ? (
        <AgentPanelOverlay
          history={spawnHistory}
          initialInspectId={overlay.agentsInspectId}
          liveAgents={liveAgents}
          onClose={() => patchOverlayState({ agents: false, agentsInspectId: null })}
          t={t}
        />
      ) : null}
      {overlay.terminals ? (
        <TerminalPanelOverlay onClose={() => patchOverlayState({ terminals: false })} t={t} />
      ) : null}
      {overlay.skillsHub ? <InfoOverlay kind='skillsHub' /> : null}
      {overlay.pluginsHub ? <InfoOverlay kind='pluginsHub' /> : null}
    </Box>
  )
}
