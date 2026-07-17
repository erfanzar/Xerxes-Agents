// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { providerPromptCancelAnswer } from '../domain/providerPrompt.js'
import type {
  ApprovalRespondResponse,
  ConfigSetResponse,
  SecretRespondResponse,
  SudoRespondResponse,
  VoiceRecordResponse
} from '../gatewayTypes.js'
import { isAction, isCopyShortcut, isMac, isVoiceToggleKey } from '../lib/platform.js'
import { computePrecisionWheelStep, initPrecisionWheel } from '../lib/precisionWheel.js'
import { forceRedraw, useInput } from '../lib/terminalRuntime.opentui.js'
import { computeWheelStep, initWheelAccelForHost } from '../lib/wheelAccel.js'

import { getInputSelection } from './inputSelectionStore.js'
import type { InputHandlerContext, InputHandlerResult } from './interfaces.js'
import { $isBlocked, $overlayState, clearApprovalOverlay, patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const isCtrl = (key: { ctrl: boolean }, ch: string, target: string) => key.ctrl && ch.toLowerCase() === target
const MODE_CYCLE = ['code', 'researcher', 'plan', 'objective'] as const

const nextInteractionMode = (current: string | undefined): string => {
  const idx = MODE_CYCLE.indexOf((current || 'code') as (typeof MODE_CYCLE)[number])

  return MODE_CYCLE[(idx + 1) % MODE_CYCLE.length] ?? 'code'
}

/**
 * Approval / clarify / confirm overlays mount their own `useInput` handlers
 * for the in-prompt keys (arrows, numbers, Enter, sometimes Esc).  The global
 * input handler used to early-return for any other key while one of those
 * overlays was up, which silently disabled transcript scrolling — the user
 * couldn't read context above the prompt that the prompt itself was asking
 * about.  Returns true when the key is a transcript-scroll input that should
 * fall through to the global scroll handlers even while a prompt is active.
 *
 * Modifier-held wheel (precision mode) is included — a user who wants to
 * scroll a single line at a time during a prompt expects it to work.
 */
export function shouldFallThroughForScroll(key: {
  downArrow: boolean
  pageDown: boolean
  pageUp: boolean
  shift: boolean
  upArrow: boolean
  wheelDown: boolean
  wheelUp: boolean
}): boolean {
  if (key.wheelUp || key.wheelDown) {
    return true
  }

  if (key.pageUp || key.pageDown) {
    return true
  }

  if (key.shift && (key.upArrow || key.downArrow)) {
    return true
  }

  return false
}

export function applyVoiceRecordResponse(
  response: null | VoiceRecordResponse,
  starting: boolean,
  voice: Pick<InputHandlerContext['voice'], 'setProcessing' | 'setRecording'>,
  sys: (text: string) => void
) {
  if (!starting || response?.status === 'recording') {
    return
  }

  voice.setRecording(false)

  if (response?.status === 'busy') {
    voice.setProcessing(true)
    sys('voice: still transcribing; try again shortly')
  } else {
    voice.setProcessing(false)
  }
}

export function useInputHandlers(ctx: InputHandlerContext): InputHandlerResult {
  const { actions, composer, gateway, terminal, voice, wheelStep } = ctx
  const { actions: cActions, refs: cRefs, state: cState } = composer

  const overlay = useStore($overlayState)
  const isBlocked = useStore($isBlocked)
  // Reserve rows for the centered pager's title, footer, and vertical padding.
  // A larger slice can exceed the viewport, which makes Yoga center the whole
  // panel offscreen and leaves only the dimming scrim visible.
  const pagerPageSize = Math.max(5, (terminal.stdout?.rows ?? 24) - 10)
  const scrollIdleTimer = useRef<null | ReturnType<typeof setTimeout>>(null)

  // Wheel accel ported from claude-code: inter-event timing drives step size,
  // direction flips reset. wheelStep (WHEEL_SCROLL_STEP) is the base; final
  // rows = wheelStep × accelMult. State mutates in place across renders.
  const wheelAccelRef = useRef(initWheelAccelForHost())

  const precisionWheelRef = useRef(initPrecisionWheel())

  useEffect(() => () => clearTimeout(scrollIdleTimer.current ?? undefined), [])

  const scrollTranscript = (delta: number) => {
    if (getUiState().busy) {
      turnController.boostStreamingForScroll()
      clearTimeout(scrollIdleTimer.current ?? undefined)
      scrollIdleTimer.current = setTimeout(() => {
        scrollIdleTimer.current = null
        turnController.relaxStreaming()
      }, TYPING_IDLE_MS)
    }

    terminal.scrollWithSelection(delta)
  }

  const copySelection = () => {
    // The terminal selection service owns the platform-specific copy path.
    terminal.selection.copySelection()
  }

  const clearSelection = () => {
    terminal.selection.clearSelection()
  }

  const cancelOverlayFromCtrlC = () => {
    if (overlay.clarify) {
      return actions.answerClarify(providerPromptCancelAnswer(overlay.clarify))
    }

    if (overlay.approval) {
      const requestId = overlay.approval.requestId

      return gateway
        .rpc<ApprovalRespondResponse>('approval.respond', {
          choice: 'deny',
          request_id: requestId,
          session_id: getUiState().sid
        })
        .then(r => r && (clearApprovalOverlay(requestId), patchTurnState({ outcome: 'denied' })))
    }

    if (overlay.confirm) {
      return patchOverlayState({ confirm: null })
    }

    if (overlay.sudo) {
      return gateway
        .rpc<SudoRespondResponse>('sudo.respond', { password: '', request_id: overlay.sudo.requestId })
        .then(r => r && (patchOverlayState({ sudo: null }), actions.sys('sudo cancelled')))
    }

    if (overlay.secret) {
      return gateway
        .rpc<SecretRespondResponse>('secret.respond', { request_id: overlay.secret.requestId, value: '' })
        .then(r => r && (patchOverlayState({ secret: null }), actions.sys('secret entry cancelled')))
    }

    if (overlay.modelPicker) {
      return patchOverlayState({ modelPicker: false })
    }

    if (overlay.skillsHub) {
      return patchOverlayState({ skillsHub: false })
    }

    if (overlay.pluginsHub) {
      return patchOverlayState({ pluginsHub: false })
    }

    if (overlay.sessions) {
      return patchOverlayState({ sessions: false })
    }

    if (overlay.agents) {
      return patchOverlayState({ agents: false })
    }
  }

  const cycleQueue = (dir: 1 | -1) => {
    const len = cRefs.queueRef.current.length

    if (!len) {
      return false
    }

    const index = cState.queueEditIdx === null ? (dir > 0 ? 0 : len - 1) : (cState.queueEditIdx + dir + len) % len

    cActions.setQueueEdit(index)
    cActions.setHistoryIdx(null)
    cActions.setInput(cRefs.queueRef.current[index]?.displayText ?? '')

    return true
  }

  const cycleHistory = (dir: 1 | -1) => {
    const h = cRefs.historyRef.current
    const cur = cState.historyIdx

    if (dir < 0) {
      if (!h.length) {
        return
      }

      if (cur === null) {
        cRefs.historyDraftRef.current = cState.input
      }

      const index = cur === null ? h.length - 1 : Math.max(0, cur - 1)

      cActions.setHistoryIdx(index)
      cActions.setQueueEdit(null)
      cActions.setInput(h[index] ?? '')

      return
    }

    if (cur === null) {
      return
    }

    const next = cur + 1

    if (next >= h.length) {
      cActions.setHistoryIdx(null)
      cActions.setInput(cRefs.historyDraftRef.current)
    } else {
      cActions.setHistoryIdx(next)
      cActions.setInput(h[next] ?? '')
    }
  }

  // The retired gateway exposed a microphone capture RPC. The Bun daemon
  // deliberately does not: it can forward `/voice` UI controls, but
  // this TUI has no native audio host port. Keep the shortcut discoverable
  // without optimistically changing state or calling a rejected RPC.
  const voiceRecordToggle = () => {
    actions.sys('voice capture is unavailable in this native Bun TUI; recording shortcuts are disabled.')
  }

  useInput((ch, key) => {
    const live = getUiState()

    if (isBlocked) {
      // When approval/clarify/confirm overlays are active, their own useInput
      // handlers must receive keystrokes (arrow keys, numbers, Enter).  Only
      // intercept Ctrl+C here so the user can deny/dismiss — all other keys
      // fall through to the component-level handlers.
      //
      // Scroll inputs (wheel / PageUp / PageDown / Shift+↑↓) are special:
      // they must reach the transcript scroll handlers below even with a
      // prompt up.  Long-thread context the prompt is asking about often
      // lives above the visible viewport, and being unable to read it while
      // answering felt like the prompt had locked the entire UI.  Explicitly
      // skip the prompt-overlay early-return for scroll keys so they fall
      // through to the wheel / PageUp / Shift+arrow handlers below.
      const promptOverlay = overlay.approval || overlay.clarify || overlay.confirm
      const fallThroughForScroll = promptOverlay && shouldFallThroughForScroll(key)

      if (promptOverlay && !fallThroughForScroll) {
        if (isCtrl(key, ch, 'c')) {
          cancelOverlayFromCtrlC()
        }

        return
      }

      if (overlay.pager) {
        if (key.escape || isCtrl(key, ch, 'c') || ch === 'q') {
          return patchOverlayState({ pager: null })
        }

        const move = (delta: number | 'top' | 'bottom') =>
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)
            const step = delta === 'top' ? -lines.length : delta === 'bottom' ? lines.length : delta
            const next = Math.max(0, Math.min(offset + step, max))

            return next === offset ? prev : { ...prev, pager: { ...prev.pager, offset: next } }
          })

        if (key.upArrow || ch === 'k') {
          return move(-1)
        }

        if (key.downArrow || ch === 'j') {
          return move(1)
        }

        if (key.pageUp || ch === 'b') {
          return move(-pagerPageSize)
        }

        if (ch === 'g') {
          return move('top')
        }

        if (ch === 'G') {
          return move('bottom')
        }

        if (key.return || ch === ' ' || key.pageDown) {
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)

            // Auto-close only when already at the last page — otherwise clamp
            // to `max` so the offset matches what the line/page-back handlers
            // can reach (prevents a snap-back jump on the next ↑/↓/PgUp).
            return offset >= max
              ? { ...prev, pager: null }
              : { ...prev, pager: { ...prev.pager, offset: Math.min(offset + pagerPageSize, max) } }
          })
        }

        return
      }

      if (isCtrl(key, ch, 'c')) {
        cancelOverlayFromCtrlC()
      } else if (key.escape && overlay.sessions) {
        patchOverlayState({ sessions: false })
      }

      // When a prompt overlay is up and the user pressed a scroll key, fall
      // through to the global scroll handlers below instead of returning.
      // Otherwise nothing above this comment matched, and there's nothing
      // useful to do for an arbitrary key while blocked.
      if (!fallThroughForScroll) {
        return
      }
    }

    if (cState.completions.length && cState.input && cState.historyIdx === null && (key.upArrow || key.downArrow)) {
      const len = cState.completions.length

      cActions.setCompIdx(i => (key.upArrow ? (i - 1 + len) % len : (i + 1) % len))

      return
    }

    if (key.wheelUp || key.wheelDown) {
      const dir: -1 | 1 = key.wheelUp ? -1 : 1
      const now = Date.now()
      // Modifier-held wheel = precision mode: one row per frame, no accel.
      // Smooth mice / trackpads emit tiny same-frame bursts; coalesce those
      // without the old 80ms throttle that made opt-scroll feel stepped.
      // SGR/X10 mouse encoding only carries shift/meta/ctrl bits; Cmd on
      // macOS is intercepted by the terminal, so we honor Option (meta) on
      // Mac / Alt (meta) on Win+Linux / Ctrl as a portable fallback. Shift
      // is reserved for selection extension.
      const hasModifier = key.meta || key.ctrl
      const precision = computePrecisionWheelStep(precisionWheelRef.current, dir, hasModifier, now)

      if (precision.active) {
        // Entering precision mode must discard any accelerated wheel state;
        // otherwise the next normal wheel event inherits stale momentum.
        if (precision.entered) {
          wheelAccelRef.current = initWheelAccelForHost()
        }

        return precision.rows ? scrollTranscript(dir * wheelStep) : undefined
      }

      // 0 = direction-flip bounce deferred; skip the no-op scroll.
      const rows = computeWheelStep(wheelAccelRef.current, dir, now)

      return rows ? scrollTranscript(dir * rows * wheelStep) : undefined
    }

    if (key.shift && key.upArrow) {
      return scrollTranscript(-1)
    }

    if (key.shift && key.downArrow) {
      return scrollTranscript(1)
    }

    if (key.pageUp || key.pageDown) {
      // Half-viewport keeps 50% visual continuity.
      const viewport = terminal.scrollRef.current?.getViewportHeight() ?? Math.max(6, (terminal.stdout?.rows ?? 24) - 8)
      const step = Math.max(4, Math.floor(viewport / 2))

      return scrollTranscript(key.pageUp ? -step : step)
    }

    // Escape-based voice bindings (ctrl/alt/super+escape) must win before the
    // generic Esc handlers below; otherwise queue-edit cancel / selection-clear
    // would swallow the chord and /voice would advertise a shortcut that never
    // actually toggles recording in those UI states.
    if (key.escape && isVoiceToggleKey(key, ch, voice.recordKey)) {
      return voiceRecordToggle()
    }

    // Queue-edit cancel beats selection-clear for plain Esc: the queue header
    // explicitly promises "Esc cancel", so honoring it takes priority over the
    // implicit selection-dismissal convention. Without an active edit, fall through.
    if (key.escape && cState.queueEditIdx !== null) {
      return cActions.clearIn()
    }

    if (key.escape && terminal.hasSelection) {
      return clearSelection()
    }

    if (key.escape && cState.completions.length) {
      return cActions.dismissCompletions()
    }

    if (key.escape && live.busy && live.sid) {
      // Grok's active-turn Escape is unconditional. If follow-ups are queued,
      // Escape cancels those too; otherwise the footer's advertised
      // "Esc interrupt" action would silently do nothing in OpenTUI.
      if (cRefs.queueRef.current.length) {
        cRefs.queueRef.current.splice(0)
        cActions.syncQueue()
        cActions.setQueueEdit(null)
        cActions.setInput('')
      }

      return turnController.interruptTurn({
        gw: gateway.gw,
        sid: live.sid,
        sys: actions.sys
      })
    }

    if (key.upArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null

      const noLineAbove =
        !cState.input || (cursor !== null && cState.input.lastIndexOf('\n', Math.max(0, cursor - 1)) < 0)

      if (noLineAbove) {
        cycleQueue(1) || cycleHistory(-1)

        return
      }
    }

    if (key.downArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null
      const noLineBelow = !cState.input || (cursor !== null && cState.input.indexOf('\n', cursor) < 0)

      if (noLineBelow || cState.historyIdx !== null) {
        cycleQueue(-1) || cycleHistory(1)

        return
      }
    }

    if (isCopyShortcut(key, ch)) {
      if (terminal.hasSelection) {
        return copySelection()
      }

      const inputSel = getInputSelection()

      if (inputSel && inputSel.end > inputSel.start) {
        inputSel.clear()

        return
      }

      // On macOS, Cmd+C with no selection is a no-op (Ctrl+C below handles interrupt).
      // On non-macOS, isAction uses Ctrl, so fall through to interrupt/clear/exit.
      if (isMac) {
        return
      }
    }

    if (isCtrl(key, ch, 'x') && cState.queueEditIdx !== null) {
      cActions.removeQueue(cState.queueEditIdx)

      return cActions.clearIn()
    }

    if (isCtrl(key, ch, 'x')) {
      return patchOverlayState({ sessions: true })
    }

    if (key.ctrl && ch.toLowerCase() === 'c') {
      if (live.busy && live.sid) {
        return turnController.interruptTurn({
          gw: gateway.gw,
          sid: live.sid,
          sys: actions.sys
        })
      }

      if (cState.input || cState.inputBuf.length) {
        return cActions.clearIn()
      }

      return actions.die()
    }

    if (isAction(key, ch, 'd')) {
      return actions.die()
    }

    if (isAction(key, ch, 'l')) {
      clearSelection()
      forceRedraw(terminal.stdout ?? process.stdout)

      return
    }

    if (isVoiceToggleKey(key, ch, voice.recordKey)) {
      return voiceRecordToggle()
    }

    // Cmd/Ctrl+G, plus Alt+G fallback for VSCode/Cursor (they bind the
    // primary keystroke to "Find Next" before the TUI sees it; Alt+G
    // arrives as meta+g across platforms).
    if (ch.toLowerCase() === 'g' && (isAction(key, ch, 'g') || key.meta)) {
      return void cActions.openEditor().catch((err: unknown) => {
        actions.sys(err instanceof Error ? `failed to open editor: ${err.message}` : 'failed to open editor')
      })
    }

    // Tab cycles interaction modes without spending a model turn. The
    // completion menu owns Tab while it is visible; Shift+Tab remains a
    // backwards-compatible alias for existing Xerxes muscle memory.
    if (key.tab && !cState.completions.length) {
      if (!live.sid) {
        return void actions.sys('mode switch needs an active session')
      }

      const target = nextInteractionMode(getUiState().info?.mode)
      return void gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'mode', session_id: live.sid, value: target })
        .then(r => {
          if (r) {
            const next = r.value || target
            patchUiState(state => ({ ...state, info: state.info ? { ...state.info, mode: next } : state.info }))
          }
        })
    }

    if (key.tab && cState.completions.length) {
      const row = cState.completions[cState.compIdx]

      if (row?.text) {
        const text =
          cState.input.startsWith('/') && row.text.startsWith('/') && cState.compReplace > 0
            ? row.text.slice(1)
            : row.text

        cActions.setInput(cState.input.slice(0, cState.compReplace) + text)
      }

      return
    }

    if (isAction(key, ch, 'k') && cRefs.queueRef.current.length && live.sid) {
      const next = cActions.dequeue()

      if (next) {
        cActions.setQueueEdit(null)
        actions.dispatchQueuedSubmission(next)
      }
    }
  })

  return { pagerPageSize }
}
