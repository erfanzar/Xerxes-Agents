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
  it('keeps rejected busy submissions paired with their images for retry', async () => {
    const request = vi.fn().mockRejectedValueOnce(new Error('session busy')).mockResolvedValue({ ok: true })
    patchUiState({ busy: false, sid: 'session-a' })
    const fixture = await mountSubmission(request)
    try {
      addAttachment(attachment)
      act(() => fixture.submission.dispatchSubmission('see'))
      await fixture.rendered.flush()
      expect(getAttachments()).toEqual([])
      expect(fixture.queueRef.current[0]?.images).toEqual([attachment])
      const id = fixture.queueRef.current[0]!.submissionId
      patchUiState({ busy: false })
      act(() => fixture.submission.sendQueued(fixture.queueRef.current.shift()!))
      await fixture.rendered.flush()
      expect(request).toHaveBeenLastCalledWith('prompt.submit', expect.objectContaining({ submission_id: id, text: 'see', images: [{ data: attachment.data, media_type: attachment.mediaType }] }))
    } finally { act(() => fixture.rendered.renderer.destroy()) }
  })

  // The fixture above uses "session busy", which no daemon emits. These are
  // the strings the Bun daemon actually produces, plus its structured code.
  // The old regex matched none of them, so a real refusal fell through to the
  // hard-error path: the user's bubble was deleted and the text was neither
  // queued nor returned to the composer.
  for (const [label, rejection] of [
    ['server.ts prose', new Error('a turn is already active for this session')],
    ['runtime.ts prose', new Error('A turn is already active for this session')],
    ['structured code', Object.assign(new Error('rejected'), { code: 'turn-active' })]
  ] as const) {
    it(`re-queues a turn-active refusal rather than discarding it (${label})`, async () => {
      const request = vi.fn().mockRejectedValueOnce(rejection).mockResolvedValue({ ok: true })
      patchUiState({ busy: false, sid: 'session-a' })
      const fixture = await mountSubmission(request)
      try {
        act(() => fixture.submission.dispatchSubmission('keep this prompt'))
        await fixture.rendered.flush()

        expect(fixture.queueRef.current[0]?.submitText).toBe('keep this prompt')
        expect(getUiState().status).toBe('queued for next turn')
        expect(fixture.sys).not.toHaveBeenCalledWith(expect.stringContaining('error'))
      } finally { act(() => fixture.rendered.renderer.destroy()) }
    })
  }

  it('queues a busy image with its text and never lends it to another message', async () => {
    const request = vi.fn(() => Promise.resolve({ ok: true }))
    patchUiState({ busy: true, busyInputMode: 'steer', sid: 'session-a' })
    const fixture = await mountSubmission(request)
    try {
      addAttachment(attachment)
      act(() => fixture.submission.dispatchSubmission('see'))
      expect(request).not.toHaveBeenCalled()
      expect(getAttachments()).toEqual([])
      expect(fixture.queueRef.current[0]).toMatchObject({ displayText: 'see', images: [attachment] })
      expect(fixture.sys).toHaveBeenCalledWith(expect.stringContaining('Image and message queued together'))
      const later = { ...attachment, name: 'later.png', path: '/tmp/later.png' }
      addAttachment(later)
      patchUiState({ busy: false })
      act(() => fixture.submission.sendQueued(fixture.queueRef.current.shift()!))
      await fixture.rendered.flush()
      expect(request).toHaveBeenLastCalledWith('prompt.submit', expect.objectContaining({ text: 'see', images: [{ data: attachment.data, media_type: attachment.mediaType }] }))
      expect(getAttachments()).toEqual([later])
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

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

it.each([0, 2])('shows shell output and starts one model follow-up for exit %s', async code => {
  const request = vi.fn(async (method: string) => method === 'shell.exec' ? { code, stdout: 'src\nREADME.md', stderr: code ? 'partial failure' : '' } : { ok: true })
  patchUiState({ busy: false, sid: 'session-a' })
  const fixture = await mountSubmission(request)
  try {
    await act(async () => { fixture.submission.dispatchSubmission('!ls'); await Bun.sleep(0) })
    await fixture.rendered.flush()
    expect(fixture.sys).toHaveBeenCalledWith(code ? 'src\nREADME.md\npartial failure' : 'src\nREADME.md')
    expect(request).toHaveBeenCalledTimes(2)
    expect(request).toHaveBeenLastCalledWith('prompt.submit', expect.objectContaining({ session_id: 'session-a', text: expect.stringContaining('stdout:\nsrc\nREADME.md'), display_text: expect.stringContaining('README.md') }))
    expect(getUiState().busy).toBe(true)
  } finally { act(() => fixture.rendered.renderer.destroy()) }
})

it('keeps visible output and releases busy state if shell follow-up is rejected', async () => {
  const request = vi.fn(async (method: string) => { if (method === 'shell.exec') return { code: 0, stdout: 'files', stderr: '' }; throw new Error('provider unavailable') })
  patchUiState({ busy: false, sid: 'session-a' })
  const fixture = await mountSubmission(request)
  try {
    await act(async () => { fixture.submission.dispatchSubmission('!ls'); await Bun.sleep(0) }); await fixture.rendered.flush()
    expect(fixture.sys).toHaveBeenCalledWith('files')
    expect(fixture.sys).toHaveBeenCalledWith('error: provider unavailable')
    expect(getUiState().busy).toBe(false)
  } finally { act(() => fixture.rendered.renderer.destroy()) }
})

it('queues a bang command while the model is working instead of racing its follow-up', async () => {
  const request = vi.fn()
  patchUiState({ busy: true, sid: 'session-a' })
  const fixture = await mountSubmission(request)
  try {
    act(() => fixture.submission.dispatchSubmission('!ls'))
    expect(request).not.toHaveBeenCalled()
    expect(fixture.queueRef.current[0]?.submitText).toBe('!ls')
    expect(getUiState().busy).toBe(true)
  } finally { act(() => fixture.rendered.renderer.destroy()) }
})
