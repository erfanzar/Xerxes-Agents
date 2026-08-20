// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { addAttachment, clearAttachments, getAttachments, type PendingAttachment } from '../app/attachmentsStore.js'
import type { ComposerActions, ComposerRefs, ComposerState } from '../app/interfaces.js'
import { turnController } from '../app/turnController.js'
import { steerWasAccepted, useSubmission } from '../app/useSubmission.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { queuedMessage } from '../domain/queuedMessage.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { Msg } from '../types.js'

const deferred = <T,>() => Promise.withResolvers<T>()

const attachment: PendingAttachment = {
  data: 'iVBORw0KGgo=',
  mediaType: 'image/png',
  name: 'retry.png',
  path: '/tmp/retry.png',
  size: 8
}

async function mountSubmission(request: ReturnType<typeof vi.fn>) {
  const messages: Msg[] = []
  const queueRef = { current: [] as ReturnType<typeof queuedMessage>[] }
  const submitRef = { current: (_value: string) => undefined }
  const slashRef = { current: (_value: string) => true }
  const sys = vi.fn()
  const setLastUserMsg = vi.fn()
  const composerActions = {
    clearIn: vi.fn(),
    enqueue: vi.fn((submitText: string, displayText = submitText) => {
      queueRef.current.push(queuedMessage(displayText, submitText))
    }),
    pushHistory: vi.fn(),
    setInput: vi.fn(),
    setInputBuf: vi.fn(),
    setQueueEdit: vi.fn(),
    syncQueue: vi.fn()
  } as unknown as ComposerActions
  const composerRefs = {
    historyDraftRef: { current: '' },
    historyRef: { current: [] },
    queueEditRef: { current: null },
    queueRef,
    submitRef
  } satisfies ComposerRefs
  const composerState = {
    compIdx: 0,
    compReplace: 0,
    completions: [],
    historyIdx: null,
    input: '',
    inputBuf: [],
    pasteSnips: [],
    queueEditIdx: null,
    queuedDisplay: []
  } satisfies ComposerState
  let submission: ReturnType<typeof useSubmission> | undefined

  const Probe = () => {
    submission = useSubmission({
      appendMessage: msg => messages.push(msg),
      composerActions,
      composerRefs,
      composerState,
      gw: { request } as unknown as GatewayClient,
      maybeGoodVibes: vi.fn(),
      removeMessage: msg => {
        const index = messages.indexOf(msg)
        if (index >= 0) messages.splice(index, 1)
      },
      setLastUserMsg,
      slashRef,
      submitRef,
      sys
    })

    return null
  }

  const rendered = await testRender(createElement(Probe), { height: 6, width: 40 })
  await rendered.flush()
  if (!submission) throw new Error('submission hook did not mount')

  return { composerActions, messages, queueRef, rendered, setLastUserMsg, submission, sys }
}

afterEach(() => {
  clearAttachments()
  turnController.fullReset()
  resetUiState()
})

describe('steer submission acknowledgement', () => {
  it('accepts the native daemon ok response produced when Enter sends a steer', () => {
    expect(steerWasAccepted({ ok: true })).toBe(true)
  })

  it('accepts the legacy queued response and rejects explicit failures', () => {
    expect(steerWasAccepted({ status: 'queued' })).toBe(true)
    expect(steerWasAccepted({ ok: false, status: 'rejected' })).toBe(false)
    expect(steerWasAccepted(null)).toBe(false)
  })

  it('ignores a late steer failure after the user switches live sessions', async () => {
    const response = deferred<never>()
    const request = vi.fn(() => response.promise)
    patchUiState({ busy: true, busyInputMode: 'steer', sid: 'session-a' })
    const fixture = await mountSubmission(request)

    try {
      act(() => fixture.submission.dispatchSubmission('keep working'))
      patchUiState({ busy: true, sid: 'session-b' })
      response.reject(new Error('old session rejected steer'))
      await fixture.rendered.flush()

      expect(fixture.queueRef.current).toEqual([])
      expect(fixture.sys).not.toHaveBeenCalled()
      expect(getUiState()).toMatchObject({ busy: true, sid: 'session-b' })
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

  it('sends a rejected steer as the next turn when its original turn already settled', async () => {
    const steer = deferred<{ ok: false; status: 'rejected' }>()
    const request = vi.fn((method: string) =>
      method === 'session.steer' ? steer.promise : Promise.resolve({ ok: true })
    )
    patchUiState({ busy: true, busyInputMode: 'steer', sid: 'session-a' })
    const fixture = await mountSubmission(request)

    try {
      act(() => fixture.submission.dispatchSubmission('next step'))
      patchUiState({ busy: false })
      steer.resolve({ ok: false, status: 'rejected' })
      await fixture.rendered.flush()

      expect(request).toHaveBeenLastCalledWith(
        'prompt.submit',
        expect.objectContaining({ session_id: 'session-a', text: 'next step' })
      )
      expect(fixture.queueRef.current).toEqual([])
      expect(fixture.sys).toHaveBeenCalledWith('steer rejected after the turn settled — sending as next turn')
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

  it('rolls back a rejected optimistic prompt and restores its attachments', async () => {
    const response = deferred<never>()
    const request = vi.fn(() => response.promise)
    addAttachment(attachment)
    patchUiState({ busy: false, sid: 'session-a' })
    const fixture = await mountSubmission(request)

    try {
      act(() => fixture.submission.dispatchSubmission('inspect this image'))
      expect(fixture.messages).toEqual([{ role: 'user', text: 'inspect this image' }])
      expect(getAttachments()).toEqual([])

      response.reject(new Error('transport unavailable'))
      await fixture.rendered.flush()

      expect(fixture.messages).toEqual([])
      expect(getAttachments()).toEqual([attachment])
      expect(fixture.sys).toHaveBeenCalledWith('error: transport unavailable')
      expect(getUiState()).toMatchObject({ busy: false, status: 'ready' })
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

  it('does not submit an interpolated prompt into a session selected while expansion was pending', async () => {
    const shell = deferred<{ code: number; stderr: string; stdout: string }>()
    const request = vi.fn((method: string) =>
      method === 'shell.exec' ? shell.promise : Promise.resolve({ ok: true })
    )
    patchUiState({ busy: false, sid: 'session-a' })
    const fixture = await mountSubmission(request)

    try {
      act(() => fixture.submission.dispatchSubmission('working tree: {!pwd}'))
      patchUiState({ busy: true, sid: 'session-b', status: 'running…' })
      shell.resolve({ code: 0, stderr: '', stdout: '/work/a' })
      await fixture.rendered.flush()

      expect(request).toHaveBeenCalledOnce()
      expect(request).toHaveBeenCalledWith('shell.exec', { command: 'pwd' })
      expect(getUiState()).toMatchObject({ busy: true, sid: 'session-b', status: 'running…' })
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

  it('does not publish late shell output or clear busy state in a newly selected session', async () => {
    const shell = deferred<{ code: number; stderr: string; stdout: string }>()
    const request = vi.fn(() => shell.promise)
    patchUiState({ busy: false, sid: 'session-a' })
    const fixture = await mountSubmission(request)

    try {
      act(() => fixture.submission.dispatchSubmission('!sleep 1'))
      patchUiState({ busy: true, sid: 'session-b', status: 'running…' })
      shell.resolve({ code: 0, stderr: '', stdout: 'finished in A' })
      await fixture.rendered.flush()

      expect(fixture.sys).not.toHaveBeenCalled()
      expect(getUiState()).toMatchObject({ busy: true, sid: 'session-b', status: 'running…' })
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })
})
