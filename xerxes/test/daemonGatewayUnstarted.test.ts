// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import { createGatewayEventHandler } from '../src/ui/app/createGatewayEventHandler.js'
import { turnController } from '../src/ui/app/turnController.js'
import { resetOverlayState } from '../src/ui/app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../src/ui/app/uiStore.js'
import { adaptDaemonEvent } from '../src/ui/gatewayAdapter.js'
import type { GatewayClient } from '../src/ui/gatewayClient.js'
import type { GatewayEventHandlerContext } from '../src/ui/app/interfaces.js'
import type { GatewayEvent } from '../src/ui/gatewayTypes.js'
import type { Msg } from '../src/ui/types.js'

const buildHarness = () => {
  const appended: Msg[] = []
  const ctx: GatewayEventHandlerContext = {
    composer: { setInput: () => undefined },
    gateway: {
      gw: {} as GatewayClient,
      rpc: async () => null,
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: () => undefined,
      recoverSidRef: { current: null },
      resetSession: () => undefined,
      resumeById: async () => undefined,
      setCatalog: () => undefined,
    },
    submission: { submitRef: { current: () => undefined } },
    system: {
      bellOnComplete: false,
      stdout: { isTTY: false } as NodeJS.WriteStream,
      sys: () => undefined,
    },
    transcript: {
      appendMessage: message => appended.push(message),
      panel: () => undefined,
      setHistoryItems: () => undefined,
    },
    voice: {
      setProcessing: () => undefined,
      setRecording: () => undefined,
      setVoiceEnabled: () => undefined,
      setVoiceTts: () => undefined,
    },
  }
  return { appended, handle: createGatewayEventHandler(ctx) }
}

const asEvent = (event: Record<string, unknown>): GatewayEvent => event as unknown as GatewayEvent

test("an unstarted cancel settles the turn without synthesizing an empty assistant row", () => {
  const { appended, handle } = buildHarness()

  // The daemon emits this edge when a submit was cancelled during admission
  // setup or suppressed after its owner vanished: no turn_begin, no content.
  handle(asEvent({
    type: 'message.complete',
    payload: { interrupted: true, unstarted: true },
  }))

  expect(appended).toHaveLength(0)
  // The optimistic busy state still settles — only artifacts are suppressed.
  expect(getUiState().status).toBe('ready')

  // Control: a completion carrying real text still records its row.
  handle(asEvent({ type: 'message.complete', payload: { text: 'Done.' } }))
  expect(appended).toEqual([{ role: 'assistant', text: 'Done.' }])
})

test("a content-less natural complete no longer appends a phantom assistant row", () => {
  const { appended, handle } = buildHarness()
  handle(asEvent({ type: 'message.complete', payload: {} }))
  expect(appended).toHaveLength(0)
  expect(getUiState().status).toBe('ready')
})

test("the adapter forwards unstarted only for setup-abort style cancels", () => {
  const unstarted = adaptDaemonEvent('turn_end', { cancelled: true, unstarted: true })
  expect(unstarted).toHaveLength(1)
  expect(unstarted[0]).toMatchObject({
    type: 'message.complete',
    payload: { interrupted: true, unstarted: true },
  })

  // A cancel that landed mid-turn is real content-bearing interruption.
  const midTurn = adaptDaemonEvent('turn_end', { cancelled: true })
  expect(midTurn[0]).toMatchObject({
    type: 'message.complete',
    payload: { interrupted: true },
  })
  expect((midTurn[0] as { payload?: Record<string, unknown> }).payload?.unstarted).toBeUndefined()

  // A natural end carries neither flag.
  const natural = adaptDaemonEvent('turn_end', { cancelled: false })
  expect((natural[0] as { payload?: Record<string, unknown> }).payload).toEqual({})
})

afterEach(() => {
  turnController.fullReset()
  resetOverlayState()
  resetUiState()
  patchUiState({ status: 'ready' })
})
