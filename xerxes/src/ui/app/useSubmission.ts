// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { type QueuedMessage, queuedMessage, queuedUserMessage } from '../domain/queuedMessage.js'
import { completionToApplyOnSubmit, looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { PromptSubmitResponse, SessionSteerResponse, ShellExecResponse } from '../gatewayTypes.js'
import { pasteTokenHash, readPasteSnippet } from '../lib/pasteStore.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { hasInterpolation, INTERPOLATION_RE } from '../protocol/interpolation.js'
import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import type { Msg } from '../types.js'

import type { ComposerActions, ComposerRefs, ComposerState, PasteSnippet } from './interfaces.js'
import { promptSubmitImages, restoreAttachments, takeAttachments } from './attachmentsStore.js'
import { decideSubmit } from './queue.js'
import { turnController } from './turnController.js'
import { getTurnPulse } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const SESSION_BUSY_RE = /session busy|waiting for model response/i

const isSessionBusyError = (e: unknown) => SESSION_BUSY_RE.test(rpcErrorMessage(e))

export const steerWasAccepted = (response: null | SessionSteerResponse): boolean =>
  response?.ok === true || response?.status === 'queued'

/**
 * Resolve collapsed paste tokens back to their text.
 *
 * Live composer snippets win because they need no disk read, but a token whose
 * snippet is gone — recalled from history, replayed from the queue, evicted by
 * `trimSnips` — still resolves through the hash stamped inside the token. Only
 * a token with neither backing survives into the prompt verbatim.
 */
export const expandSnips = (snips: PasteSnippet[]) => {
  const byLabel = new Map<string, string[]>()

  for (const { label, text } of snips) {
    const hit = byLabel.get(label)
    hit ? hit.push(text) : byLabel.set(label, [text])
  }

  return (value: string) =>
    value.replace(PASTE_SNIPPET_RE, tok => {
      const live = byLabel.get(tok)?.shift()

      if (live !== undefined) {
        return live
      }

      const hash = pasteTokenHash(tok)

      return (hash ? readPasteSnippet(hash) : null) ?? tok
    })
}

const spliceMatches = (text: string, matches: RegExpMatchArray[], results: string[]) =>
  matches.reduceRight((acc, m, i) => acc.slice(0, m.index!) + results[i] + acc.slice(m.index! + m[0].length), text)

export function useSubmission(opts: UseSubmissionOptions) {
  const {
    appendMessage,
    composerActions,
    composerRefs,
    composerState,
    gw,
    maybeGoodVibes,
    removeMessage,
    setLastUserMsg,
    slashRef,
    submitRef,
    sys
  } = opts

  const typingIdleTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (typingIdleTimer.current) {
      clearTimeout(typingIdleTimer.current)
      typingIdleTimer.current = null
    }

    if (!composerState.input && !composerState.inputBuf.length) {
      turnController.relaxStreaming()

      return
    }

    if (getUiState().busy) {
      turnController.boostStreamingForTyping()
    }

    typingIdleTimer.current = setTimeout(() => {
      typingIdleTimer.current = null
      turnController.relaxStreaming()
    }, TYPING_IDLE_MS)

    return () => {
      if (typingIdleTimer.current) {
        clearTimeout(typingIdleTimer.current)
        typingIdleTimer.current = null
      }
    }
  }, [composerState.input, composerState.inputBuf])

  const submitPrompt = useCallback(
    (message: QueuedMessage) => {
      const sid = getUiState().sid

      if (!sid) {
        return sys('session not ready yet')
      }

      // Pending /image attachments apply to exactly this submit: take them
      // out of the store now, and only restore them when the daemon never
      // accepted the turn (session-busy rollback re-queues the text).
      const images = takeAttachments()
      const userMessage = queuedUserMessage(message)

      turnController.clearStatusTimer()
      maybeGoodVibes(message.submitText)
      setLastUserMsg(message.displayText)
      appendMessage(userMessage)
      patchUiState({ busy: true, status: 'running…' })
      turnController.bufRef = ''
      turnController.interrupted = false

      gw.request<PromptSubmitResponse>('prompt.submit', {
        session_id: sid,
        submission_id: message.submissionId,
        text: message.submitText,
        display_text: message.displayText,
        ...(images.length ? { images: promptSubmitImages(images) } : {})
      }).catch(
        (e: unknown) => {
          // A user can switch to another live session while this request is
          // still settling. Never roll an old request back into the newly
          // active transcript/queue or clear that session's busy state.
          if (getUiState().sid !== sid) {
            return
          }

          if (isSessionBusyError(e)) {
            // The daemon has not accepted this as a turn yet. Roll back the
            // optimistic bubble and restore only the queue preview; the next
            // settle edge will dispatch it and append one fresh user bubble.
            removeMessage(userMessage)
            restoreAttachments(images)
            composerRefs.queueRef.current.unshift(message)
            composerActions.syncQueue()
            patchUiState({ busy: true, status: 'queued for next turn' })

            return
          }

          // The daemon rejected the submit rather than accepting a turn.
          // Undo the optimistic echo and return its one-shot attachments so
          // the composer remains an honest, retryable representation.
          removeMessage(userMessage)
          restoreAttachments(images)
          sys(`error: ${rpcErrorMessage(e)}`)
          patchUiState({ busy: false, status: 'ready' })
        }
      )
    },
    [appendMessage, composerActions, composerRefs.queueRef, gw, maybeGoodVibes, removeMessage, setLastUserMsg, sys]
  )

  const send = useCallback(
    (text: string) => {
      const submitText = expandSnips(composerState.pasteSnips)(text)

      // Native Bun accepts prompt text directly; pending /image attachments
      // are taken from the attachments store inside submitPrompt and ride the
      // same prompt.submit frame as `images`.
      submitPrompt(queuedMessage(text, submitText))
    },
    [composerState.pasteSnips, submitPrompt]
  )

  const shellExec = useCallback(
    (cmd: string) => {
      const sid = getUiState().sid
      appendMessage({ role: 'user', text: `!${cmd}` })
      patchUiState({ busy: true, status: 'running…' })

      gw.request<ShellExecResponse>('shell.exec', { command: cmd })
        .then(raw => {
          if (getUiState().sid !== sid) {
            return
          }

          const r = asRpcResult<ShellExecResponse>(raw)

          if (!r) {
            return sys('error: invalid response: shell.exec')
          }

          const out = [r.stdout, r.stderr].filter(Boolean).join('\n').trim()

          if (out) {
            sys(out)
          }

          if (r.code !== 0 || !out) {
            sys(`exit ${r.code}`)
          }
        })
        .catch((e: unknown) => {
          if (getUiState().sid === sid) {
            sys(`error: ${rpcErrorMessage(e)}`)
          }
        })
        .finally(() => {
          if (getUiState().sid === sid) {
            patchUiState({ busy: false, status: 'ready' })
          }
        })
    },
    [appendMessage, gw, sys]
  )

  const interpolate = useCallback(
    (text: string, then: (result: string) => void) => {
      patchUiState({ status: 'interpolating…' })
      const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

      Promise.all(
        matches.map(m =>
          gw
            .request<ShellExecResponse>('shell.exec', { command: m[1]! })
            .then(raw => {
              const r = asRpcResult<ShellExecResponse>(raw)

              return [r?.stdout, r?.stderr].filter(Boolean).join('\n').trim()
            })
            .catch(() => '(error)')
        )
      ).then(results => then(spliceMatches(text, matches, results)))
    },
    [gw]
  )

  const sendQueued = useCallback(
    (message: QueuedMessage) => {
      if (message.submitText.startsWith('!')) {
        return shellExec(message.submitText.slice(1).trim())
      }

      if (hasInterpolation(message.submitText)) {
        const sid = getUiState().sid
        patchUiState({ busy: true })

        return interpolate(message.submitText, submitText => {
          // Interpolation may outlive a live-session switch. The authored
          // prompt belongs to the session that started expansion and must
          // never be submitted into whichever tab happens to be active when
          // shell substitution finishes.
          if (!sid || getUiState().sid !== sid) {
            return
          }

          submitPrompt(queuedMessage(message.displayText, submitText))
        })
      }

      submitPrompt(message)
    },
    [interpolate, shellExec, submitPrompt]
  )

  // Honors `display.busy_input_mode` from config.yaml (CLI parity):
  //   - 'steer'     (default): inject into the current turn via session.steer;
  //                   falls back to queue when steer is rejected (no agent /
  //                   no tool window). Never cancels the live turn.
  //   - 'queue'     (legacy): append to queueRef; drains on busy → false
  //   - 'interrupt' (opt-in): queue the text + interrupt; the busy→false
  //                   settle edge drains it once (desktop parity).
  //                   No optimistic send → no duplicate bubble / race note.
  //
  // `opts.fallbackToFront` re-inserts at the queue head (queue-edit picks keep
  // their position); the mainline submit path appends.
  const handleBusyInput = useCallback(
    (message: QueuedMessage, opts: { fallbackToFront?: boolean } = {}) => {
      const live = getUiState()
      const mode = live.busyInputMode

      const enqueueText = () => {
        if (opts.fallbackToFront) {
          composerRefs.queueRef.current.unshift(message)
          composerActions.syncQueue()
        } else {
          composerActions.enqueue(message.submitText, message.displayText)
        }
      }

      if (mode === 'queue') {
        return composerActions.enqueue(message.submitText, message.displayText)
      }

      if (mode === 'steer' && live.sid) {
        const sid = live.sid
        // Anchor on the liveness epoch, not the wall-clock start: a turn that
        // ends and restarts within the steer RPC window can share a
        // `startedAt` millisecond, but never an epoch.
        const turnEpoch = getTurnPulse().epoch
        const fallback = (queuedNote: string, settledNote: string) => {
          if (getUiState().sid !== sid) {
            return
          }

          // The turn can settle before the steer acknowledgement arrives. A
          // queue inserted after busy became false has no settle edge left to
          // drain it, so dispatch it immediately as the next turn instead.
          if (!getUiState().busy) {
            sys(settledNote)
            sendQueued(message)

            return
          }

          enqueueText()
          sys(queuedNote)
        }

        gw.request<SessionSteerResponse>('session.steer', { session_id: sid, text: message.submitText })
          .then(raw => {
            if (getUiState().sid !== sid) {
              return
            }

            const r = asRpcResult<SessionSteerResponse>(raw)

            if (!steerWasAccepted(r)) {
              fallback(
                'steer rejected — message queued for next turn',
                'steer rejected after the turn settled — sending as next turn'
              )

              return
            }

            // A steer is still authored user input. The daemon intentionally
            // hides its internal replay marker, so the TUI owns this one
            // visible, standard user transcript block.
            if (getUiState().busy && getTurnPulse().epoch === turnEpoch) {
              turnController.recordUserSteer(message.displayText)
            } else {
              // A same-session turn may finish (or a new one may begin)
              // before the response frame is observed. Do not attach the
              // acknowledged steer to that unrelated live progress buffer.
              appendMessage(queuedUserMessage(message))
            }
            setLastUserMsg(message.displayText)
          })
          .catch(() =>
            fallback(
              'steer failed — message queued for next turn',
              'steer failed after the turn settled — sending as next turn'
            )
          )

        return
      }

      // 'interrupt': queue + interrupt; the daemon's settle edge
      // (message.complete) drains the queue exactly once.
      enqueueText()

      if (live.sid) {
        turnController.interruptTurn({ gw, sid: live.sid, sys })
      }
    },
    [appendMessage, composerActions, composerRefs, gw, sendQueued, setLastUserMsg, sys]
  )

  const dispatchQueuedSubmission = useCallback(
    (message: QueuedMessage) => {
      if (getUiState().busy) {
        return handleBusyInput(message, { fallbackToFront: true })
      }

      sendQueued(message)
    },
    [handleBusyInput, sendQueued]
  )

  const dispatchSubmission = useCallback(
    (full: string) => {
      if (!full.trim()) {
        return
      }

      if (looksLikeSlashCommand(full)) {
        appendMessage({ kind: 'slash', role: 'system', text: full })
        composerActions.pushHistory(full)
        slashRef.current(full)
        composerActions.clearIn()

        return
      }

      if (full.startsWith('!')) {
        composerActions.clearIn()

        return shellExec(full.slice(1).trim())
      }

      const live = getUiState()
      const message = queuedMessage(full, expandSnips(composerState.pasteSnips)(full))

      if (!live.sid) {
        composerActions.pushHistory(full)
        composerActions.enqueue(message.submitText, message.displayText)
        composerActions.clearIn()

        return
      }

      const editIdx = composerRefs.queueEditRef.current
      composerActions.clearIn()

      if (editIdx !== null) {
        composerActions.replaceQueue(editIdx, message.submitText, message.displayText)
        const picked = composerRefs.queueRef.current.splice(editIdx, 1)[0]
        composerActions.syncQueue()
        composerActions.setQueueEdit(null)

        if (!picked || !live.sid) {
          return
        }

        if (getUiState().busy) {
          // 'interrupt' / 'steer' should reach the live turn instead of
          // silently going back to the queue.  handleBusyInput resolves
          // mode-specific behavior (interrupt-and-send, steer, or queue).
          if (getUiState().busyInputMode === 'queue') {
            composerRefs.queueRef.current.unshift(picked)

            return composerActions.syncQueue()
          }

          return handleBusyInput(picked, { fallbackToFront: true })
        }

        return dispatchQueuedSubmission(picked)
      }

      composerActions.pushHistory(full)

      if (getUiState().busy) {
        return handleBusyInput(message)
      }

      sendQueued(message)
    },
    [
      appendMessage,
      composerActions,
      composerRefs,
      composerState.pasteSnips,
      dispatchQueuedSubmission,
      handleBusyInput,
      sendQueued,
      shellExec,
      slashRef
    ]
  )

  const submit = useCallback(
    (value: string) => {
      if (composerState.completions.length) {
        const row = composerState.completions[composerState.compIdx]
        const next = completionToApplyOnSubmit(value, row?.text, composerState.compReplace)

        if (next !== null) {
          return composerActions.setInput(next)
        }
      }

      if (!value.trim() && !composerState.inputBuf.length) {
        const live = getUiState()
        const decision = decideSubmit('', live.busy, composerRefs.queueRef.current.length)

        if (decision.kind === 'interrupt' && live.sid) {
          // Match the queue panel's Grok-style "Enter send now" contract:
          // interrupt the current run and keep busy until its settle event
          // drains the oldest queued message exactly once.
          return turnController.interruptTurn({ gw, sid: live.sid, sys })
        }

        if (decision.kind === 'drain' && live.sid) {
          const next = composerActions.dequeue()

          composerActions.syncQueue()

          if (next) {
            composerActions.setQueueEdit(null)
            dispatchQueuedSubmission(next)
          }
        }

        return
      }

      if (value.endsWith('\\')) {
        composerActions.setInputBuf(prev => [...prev, value.slice(0, -1)])

        return composerActions.setInput('')
      }

      dispatchSubmission([...composerState.inputBuf, value].join('\n'))
    },
    [appendMessage, composerActions, composerRefs, composerState, dispatchQueuedSubmission, dispatchSubmission, gw, sys]
  )

  submitRef.current = submit

  return { dispatchQueuedSubmission, dispatchSubmission, send, sendQueued, submit }
}

export interface UseSubmissionOptions {
  appendMessage: (msg: Msg) => void
  composerActions: ComposerActions
  composerRefs: ComposerRefs
  composerState: ComposerState
  gw: GatewayClient
  maybeGoodVibes: (text: string) => void
  removeMessage: (msg: Msg) => void
  setLastUserMsg: (value: string) => void
  slashRef: MutableRefObject<(cmd: string) => boolean>
  submitRef: MutableRefObject<(value: string) => void>
  sys: (text: string) => void
}
