// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { writeFileSync } from 'node:fs'

import { type RefObject, useCallback, useRef } from 'react'

import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { subagentProgressFromSnapshot } from '../domain/subagentProgress.js'
import { ZERO } from '../domain/usage.js'
import { looksLikeInternalUserPrompt } from '../gatewayAdapter.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCreateResponse,
  SessionInflightTurn,
  SessionResumeResponse,
  SessionTitleResponse,
  SetupStatusResponse,
  SubagentSnapshotPayload
} from '../gatewayTypes.js'
import { capTranscriptHistory } from '../lib/messages.js'
import { asRpcResult } from '../lib/rpc.js'
import { releaseTerminalCaches } from '../lib/terminalRuntime.opentui.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'
import type { Msg, PanelSection, SessionInfo, SubagentProgress, Usage } from '../types.js'

import type { ComposerActions, GatewayRpc, StateSetter } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { pushSnapshot } from './spawnHistoryStore.js'
import { turnController } from './turnController.js'
import { beginTurnPulse, patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const usageFrom = (info: null | SessionInfo): Usage => (info?.usage ? { ...ZERO, ...info.usage } : ZERO)

const statusFromLiveSession = (status?: string, running = false) => {
  if (status === 'waiting') {
    return 'waiting for input…'
  }

  if (status === 'starting') {
    return 'starting agent…'
  }

  return running || status === 'working' ? 'running…' : 'ready'
}

export const writeActiveSessionFile = (sessionId: null | string, file = process.env.XERXES_TUI_ACTIVE_SESSION_FILE) => {
  if (!file || !sessionId) {
    return
  }

  try {
    writeFileSync(file, JSON.stringify({ session_id: sessionId }), { mode: 0o600 })
  } catch {
    // Best-effort shell epilogue hint only; never break live session changes.
  }
}

export const liveSessionInflightMessages = (inflight?: null | SessionInflightTurn): Msg[] => {
  const user = String(inflight?.user ?? '').trim()

  // Internal prompts (skill activation, compaction, steers) are runtime
  // scaffolding; the reattached transcript must not show them as user rows.
  return user && !looksLikeInternalUserPrompt(user) ? [{ role: 'user', text: user }] : []
}

export const hydrateLiveSessionInflight = (inflight?: null | SessionInflightTurn) => {
  const assistant = String(inflight?.assistant ?? '')

  if (inflight?.streaming) {
    // Restore the turn's work so far — thinking and tool rows — before the
    // streaming text so both live surfaces render the mid-turn state.
    turnController.hydrateInflightTrail({ thinking: inflight.thinking, tools: inflight.tools })
  }

  if (!assistant && !inflight?.streaming) {
    return
  }

  turnController.hydrateStreamingText(assistant)
}

/** Children the daemon's manifest still reports as unfinished. */
const NON_TERMINAL_SNAPSHOT_STATUSES = new Set<SubagentProgress['status']>(['queued', 'running'])

/**
 * Rehydrate the session's persisted subagent manifest on reattach.
 *
 * Split by state, because the two halves belong in different places:
 *
 *  - finished children are history — a folded trail card in the transcript
 *    plus a spawn-history snapshot, which is what the whole manifest used to
 *    become;
 *  - children still working are LIVE, and are restored into the turn state so
 *    the F6 rail shows its WORKING count again and every subsequent
 *    `subagent.*` event (all update-only) has a row to land on. Archiving them
 *    is what made a fan-out you walked away from come back frozen.
 *
 * The live half only applies to a session that is actually mid-turn. An idle
 * session whose manifest still says "running" is describing children orphaned
 * by a daemon restart, and re-animating those would show work that will never
 * report again as permanently in flight.
 *
 * Returns the card to append, or null.
 */
const subagentTrailFromSnapshots = (
  snapshots: SubagentSnapshotPayload[] | undefined,
  sessionId: string,
  running: boolean
): Msg | null => {
  if (!snapshots?.length) {
    return null
  }

  const subagents = snapshots.map((row, index) => subagentProgressFromSnapshot(row, index))
  const live = running ? subagents.filter(item => NON_TERMINAL_SNAPSHOT_STATUSES.has(item.status)) : []
  const liveIds = new Set(live.map(item => item.id))
  const archived = liveIds.size ? subagents.filter(item => !liveIds.has(item.id)) : subagents

  turnController.hydrateSubagents(live)

  if (!archived.length) {
    return null
  }

  pushSnapshot(archived, { sessionId, startedAt: null })

  return { kind: 'trail', role: 'system', subagents: archived, text: '' }
}

/** Keep the live elapsed clock continuous across a mid-turn reattach. */
const seedTurnClock = (
  inflight: null | SessionInflightTurn | undefined,
  running: boolean,
  setTurnStartedAt?: StateSetter<null | number>
) => {
  const startedAt = inflight?.started_at
  if (!running || typeof startedAt !== 'number' || !Number.isFinite(startedAt) || startedAt <= 0) {
    return
  }
  const startedMs = startedAt * 1000

  // Open the liveness window with the real turn start before busy flips true;
  // the $turnLive listener keeps an already-open window instead of restarting it.
  beginTurnPulse(startedMs)
  setTurnStartedAt?.(startedMs)
}

/**
 * Drop the exchange the daemon just undid: its user row and everything the
 * turn rendered after it. Trimming by role from the tail stopped at the
 * `/undo` echo (and at a turn's role:'system' tool trail), so nothing was
 * removed while the session the model sees had already lost it. The echo of
 * the command being run is kept.
 */
export const trimLastExchange = (items: Msg[]): Msg[] => {
  const start = items.findLastIndex(item => item.role === 'user')

  if (start < 0) {
    return items
  }

  let keepFrom = items.length

  while (keepFrom > start + 1 && items[keepFrom - 1]?.kind === 'slash') {
    keepFrom--
  }

  return [...items.slice(0, start), ...items.slice(keepFrom)]
}

export interface UseSessionLifecycleOptions {
  colsRef: { current: number }
  composerActions: ComposerActions
  gw: GatewayClient
  panel: (title: string, sections: PanelSection[]) => void
  rpc: GatewayRpc
  scrollRef: RefObject<null | ScrollBoxHandle>
  setHistoryItems: StateSetter<Msg[]>
  setLastUserMsg: StateSetter<string>
  setSessionStartedAt: StateSetter<number>
  setStickyPrompt: StateSetter<string>
  setTurnStartedAt?: StateSetter<null | number>
  setVoiceProcessing: StateSetter<boolean>
  setVoiceRecording: StateSetter<boolean>
  sys: (text: string) => void
}

export function useSessionLifecycle(opts: UseSessionLifecycleOptions) {
  const {
    colsRef,
    composerActions,
    gw,
    panel,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    setTurnStartedAt,
    setVoiceProcessing,
    setVoiceRecording,
    sys
  } = opts
  // Session switches may overlap when a user selects twice before the first
  // RPC settles. Only the newest request is allowed to replace visible state.
  const switchGenerationRef = useRef(0)
  // Gateway create/resume/activate calls commit the client's active session key.
  // Serialize those calls so a stale response cannot commit its key after the
  // newest request has already made the visible UI point at another session.
  const clientSwitchTailRef = useRef<Promise<void>>(Promise.resolve())
  const runClientSwitch = useCallback(<T,>(generation: number, request: () => Promise<T>): Promise<null | T> => {
    const queued = clientSwitchTailRef.current.then(() =>
      generation === switchGenerationRef.current ? request() : null
    )
    clientSwitchTailRef.current = queued.then(
      () => undefined,
      () => undefined
    )

    return queued
  }, [])
  // `sid` names the session being left until a switch commits. Hold composer
  // submissions for that whole window; only the newest generation releases.
  const beginSwitch = useCallback(() => {
    patchUiState({ switching: true })

    return ++switchGenerationRef.current
  }, [])
  const settleSwitch = useCallback((generation: number) => {
    if (generation !== switchGenerationRef.current) return
    // A committed switch already carried its held prompts over; one that
    // failed leaves them queued on the session the user is still in.
    composerActions.releaseSwitchHold()
    if (getUiState().switching) patchUiState({ switching: false })
  }, [composerActions])

  // Native sessions are durable records rather than closeable daemon handles.
  // Keep callers' lifecycle sequencing intact without sending the retired
  // `session.close` compatibility RPC.
  const closeSession = useCallback(async (_targetSid?: null | string) => null, [])

  const resetSession = useCallback(() => {
    turnController.fullReset()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    // Background tasks belong to other live sessions too; switching tabs must
    // not forget them before their completion event arrives.
    patchUiState({ disconnected: false, info: null, sid: null, usage: ZERO })
    setHistoryItems([])
    setLastUserMsg('')
    setStickyPrompt('')
    composerActions.setPasteSnips([])
    // Half-prune: new session has new keys, but keep a warm pool in case
    // the user resumes back to the prior session.
    releaseTerminalCaches('half')
  }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt, setVoiceProcessing, setVoiceRecording])

  const resetVisibleHistory = useCallback(
    (info: null | SessionInfo = null) => {
      turnController.idle()
      turnController.clearReasoning()
      turnController.turnTools = []
      turnController.persistedToolLabels.clear()

      setHistoryItems(info ? [introMsg(info)] : [])
      setStickyPrompt('')
      setLastUserMsg('')
      composerActions.setPasteSnips([])
      patchTurnState({ activity: [] })
      patchUiState({ info, usage: usageFrom(info) })
    },
    [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt]
  )

  const createSessionFor = useCallback(
    async (generation: number, msg?: string, title?: string, keepCurrent = false, agentPreset?: string) => {
      const setup = await rpc<SetupStatusResponse>('setup.status', {})
      if (generation !== switchGenerationRef.current) return null

      if (setup?.provider_configured === false) {
        panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
        patchUiState({ status: 'setup required' })

        return null
      }

      if (!keepCurrent) {
        await closeSession(getUiState().sid)
        if (generation !== switchGenerationRef.current) return null
      }

      const r = await runClientSwitch(generation, () =>
        rpc<SessionCreateResponse>('session.create', {
          cols: colsRef.current,
          ...(agentPreset?.trim() ? { agent_id: agentPreset.trim() } : {})
        })
      )
      if (generation !== switchGenerationRef.current) return null

      if (!r) {
        patchUiState({ status: 'ready' })

        return null
      }

      const info = r.info ?? null
      const requestedTitle = title?.trim() ?? ''

      composerActions.activateSessionQueue(r.session_id)
      resetSession()
      setSessionStartedAt(Date.now())

      writeActiveSessionFile(r.session_id)
      patchUiState({
        info,
        sid: r.session_id,
        status: info?.version ? 'ready' : 'starting agent…',
        switching: false,
        usage: usageFrom(info)
      })

      if (info) {
        setHistoryItems([introMsg(info)])
      }

      if (info?.credential_warning) {
        sys(`warning: ${info.credential_warning}`)
      }

      if (info?.config_warning) {
        sys(`warning: ${info.config_warning}`)
      }

      if (msg) {
        sys(msg)
      }

      if (requestedTitle) {
        rpc<SessionTitleResponse>('session.title', {
          session_id: r.session_id,
          title: requestedTitle
        })
          .then(result => {
            if (!result || getUiState().sid !== r.session_id) {
              return
            }

            const nextTitle = (result.title ?? requestedTitle).trim()
            const suffix = result.pending ? ' (queued while session initializes)' : ''
            sys(`session title set: ${nextTitle}${suffix}`)
          })
          .catch((err: unknown) => {
            if (getUiState().sid !== r.session_id) {
              return
            }

            const message = err instanceof Error ? err.message : String(err)
            sys(`warning: failed to set session title: ${message}`)
          })
      }

      return r.session_id
    },
    [closeSession, colsRef, panel, resetSession, rpc, runClientSwitch, setHistoryItems, setSessionStartedAt, sys]
  )

  const startNewSession = useCallback(
    async (msg?: string, title?: string, keepCurrent = false, agentPreset?: string) => {
      const generation = beginSwitch()
      try {
        return await createSessionFor(generation, msg, title, keepCurrent, agentPreset)
      } finally {
        settleSwitch(generation)
      }
    },
    [beginSwitch, createSessionFor, settleSwitch]
  )

  const newSession = useCallback(
    (msg?: string, title?: string, agentPreset?: string) => startNewSession(msg, title, false, agentPreset),
    [startNewSession]
  )

  const newLiveSession = useCallback(
    (msg = 'new live session started', title?: string) => {
      patchOverlayState({ sessions: false })

      return startNewSession(msg, title, true)
    },
    [startNewSession]
  )

  const activateLiveSession = useCallback(
    (id: string) => {
      patchOverlayState({ sessions: false })
      // Agent View includes the attached chat. Entering that row is a return to
      // the screen already in memory, not a reattach. Reinitializing it here
      // discarded the optimistic user/skill row, live subagents, and original
      // turn clock before session.open could provide a weaker snapshot.
      if (id === getUiState().sid) {
        return
      }

      const generation = beginSwitch()
      patchUiState({ status: 'switching session…' })
      // Set while this switch holds the target's stream at the transport.
      let held: null | string = null

      runClientSwitch(generation, () =>
        gw.request<SessionActivateResponse>('session.activate', { session_id: id })
      )
        .then(raw => {
          const r = asRpcResult<SessionActivateResponse>(raw)
          if (generation !== switchGenerationRef.current) {
            // Superseded: this view will never be shown, and its held events
            // must not stay armed and swallow the next session's stream.
            if (r?.recovery_pending) gw.discardSessionRecovery(r.session_id)
            return
          }

          if (!r) {
            // No session id to match, but this is the newest switch, so any
            // hold is its own; left armed it would buffer every later event.
            gw.discardSessionRecovery()
            sys('error: invalid response: session.activate')

            return patchUiState({ status: 'ready' })
          }

          if (r.recovery_pending) held = r.session_id

          const info = r.info ?? null
          const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

          composerActions.activateSessionQueue(r.session_id)
          resetSession()
          setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())
          seedTurnClock(r.inflight, running, setTurnStartedAt)
          const subagentTrail = subagentTrailFromSnapshots(r.subagent_snapshots, r.session_id, running)
          const transcript = [
            ...toTranscriptMessages(r.messages),
            ...(subagentTrail ? [subagentTrail] : []),
            ...liveSessionInflightMessages(r.inflight)
          ]
          setHistoryItems(capTranscriptHistory(info ? [introMsg(info), ...transcript] : transcript))
          writeActiveSessionFile(r.session_key ?? r.session_id)
          patchUiState({
            busy: running,
            info,
            sid: r.session_id,
            status: statusFromLiveSession(r.status, running),
            switching: false,
            usage: usageFrom(info)
          })
          hydrateLiveSessionInflight(r.inflight)
          turnController.recordTodos(r.todos)
          // Only now may the stream that arrived during the switch (deltas,
          // turn_end, drained notices, pending approvals) land on this view;
          // applied earlier, resetSession() above wiped it.
          held = null
          if (r.recovery_pending) gw.finishSessionRecovery(r.session_id)
          setTimeout(() => scrollRef.current?.scrollToBottom(), 0)
        })
        .catch((e: Error) => {
          // The view never committed, so there is nothing to replay onto; an
          // armed hold would otherwise swallow every later event.
          if (held) gw.discardSessionRecovery(held)
          if (generation !== switchGenerationRef.current) return
          sys(`error: ${e.message}`)
          patchUiState({ status: 'ready' })
        })
        .finally(() => settleSwitch(generation))
    },
    [beginSwitch, gw, resetSession, runClientSwitch, scrollRef, setHistoryItems, setSessionStartedAt, setTurnStartedAt, settleSwitch, sys]
  )

  const resumeById = useCallback(
    (id: string, options: { keepCurrent?: boolean; preserveView?: boolean } = {}) => {
      const generation = beginSwitch()
      patchOverlayState({ sessions: false })
      patchUiState({ status: 'resuming…' })

      rpc<SetupStatusResponse>('setup.status', {}).then(setup => {
        if (generation !== switchGenerationRef.current) return
        if (setup?.provider_configured === false) {
          panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
          patchUiState({ status: 'setup required' })

          return
        }

        const previousSid = getUiState().sid
        let held: null | string = null

        return runClientSwitch(generation, () =>
          gw.request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: id, ...(options.preserveView ? { preserve_view: true } : {}) })
        )
          .then(raw => {
            const r = asRpcResult<SessionResumeResponse>(raw)
            if (generation !== switchGenerationRef.current) {
              if (r?.recovery_pending) gw.discardSessionRecovery(r.session_id)
              return
            }

            if (!r) {
              gw.discardSessionRecovery()
              sys('error: invalid response: session.resume')

              return patchUiState({ status: 'ready' })
            }

            if (r.recovery_pending) held = r.session_id

            const info = r.info ?? null
            const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')
            const recovering = options.preserveView === true && (previousSid ?? getUiState().info?.session_id) === r.session_id
            const replaying = recovering && r.reconnected === true
            const scrollPosition = recovering ? scrollRef.current?.getScrollTop() : undefined
            const followedBottom = recovering ? scrollRef.current?.isSticky() : true

            composerActions.activateSessionQueue(r.session_id)
            if (!replaying) {
              if (recovering) turnController.fullReset(true)
              else resetSession()
            }
            setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())
            seedTurnClock(r.inflight, running, setTurnStartedAt)

            const subagentTrail = subagentTrailFromSnapshots(r.subagent_snapshots, r.session_id, running)
            const resumed = [
              ...toTranscriptMessages(r.messages),
              ...(subagentTrail ? [subagentTrail] : []),
              ...liveSessionInflightMessages(r.inflight)
            ]

            if (!replaying) setHistoryItems(capTranscriptHistory(info ? [introMsg(info), ...resumed] : resumed))
            writeActiveSessionFile(r.resumed ?? r.session_id)
            patchUiState({
              disconnected: false,
              busy: running,
              info,
              sid: r.session_id,
              status: statusFromLiveSession(r.status, running),
              switching: false,
              usage: usageFrom(info)
            })
            if (!replaying) {
              hydrateLiveSessionInflight(r.inflight)
              turnController.recordTodos(r.todos)
            }
            held = null
            if (r.recovery_pending) gw.finishSessionRecovery(r.session_id)
            if (recovering) patchTurnState(state => ({ ...state,
              activity: state.activity.filter(item => item.text !== 'gateway connection lost · recovering session…')
            }))

            // Agent View "attach" is non-destructive: the chat you came from
            // remains live (and may keep working) while the saved chat becomes
            // the foreground tab. Explicit /resume keeps replacement semantics
            // and closes the previous idle session as before.
            if (!options.keepCurrent && previousSid && previousSid !== r.session_id) {
              void closeSession(previousSid)
            }

            setTimeout(() => {
              if (generation !== switchGenerationRef.current) return
              if (recovering && !followedBottom && scrollPosition !== undefined) scrollRef.current?.scrollTo(scrollPosition)
              else scrollRef.current?.scrollToBottom()
            }, 0)
          })
          .catch((e: Error) => {
            if (held) gw.discardSessionRecovery(held)
            if (generation !== switchGenerationRef.current) return
            sys(`error: ${e.message}`)
            patchUiState({ status: 'ready' })
          })
      }).finally(() => settleSwitch(generation))
    },
    [beginSwitch, closeSession, colsRef, gw, panel, resetSession, rpc, runClientSwitch, scrollRef, setHistoryItems, setSessionStartedAt, setTurnStartedAt, settleSwitch, sys]
  )

  /** Agent View attach: promote saved history without replacing live work. */
  const attachSavedSession = useCallback(
    (id: string) => resumeById(id, { keepCurrent: true }),
    [resumeById]
  )

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      if (!getUiState().busy) {
        return false
      }

      sys(`interrupt the current turn before trying to ${what}`)

      return true
    },
    [sys]
  )

  return {
    activateLiveSession,
    attachSavedSession,
    closeSession,
    guardBusySessionSwitch,
    newLiveSession,
    newSession,
    resetSession,
    resetVisibleHistory,
    resumeById,
    trimLastExchange
  }
}
