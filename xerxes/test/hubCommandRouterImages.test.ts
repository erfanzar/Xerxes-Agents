// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { join } from 'node:path'

import { HubCommandRouter } from '../src/hub/commandRouter.js'
import { verifyHubAuth } from '../src/security/hubAuthEnvelope.js'
import { FakeBackend, FakeIdentity, disposeAll, registerCleanup, standardUsers, tempDir, testSigner } from './hubHarness.js'

afterEach(disposeAll)

// 1x1 transparent PNG.
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='

async function setup() {
  const dir = await tempDir('xh-img-')
  const identity = standardUsers(new FakeIdentity())
  const backend = new FakeBackend()
  const signer = testSigner()
  const router = new HubCommandRouter({ databasePath: join(dir, 'hub.sqlite'), backend, identity, signer })
  registerCleanup(() => router.close())
  const mutations = () => backend.calls.filter(call => call.method !== 'session.active_list')
  return { identity, backend, signer, router, mutations }
}

const command = (action: 'turn.submit' | 'turn.steer', id: string, args: Record<string, unknown>) => ({
  workspace_id: 'w1', session_id: 's1', command_id: id, action, args,
})

describe('Hub command router images', () => {
  test('submit and steer forward { media_type, data } only, and the signed hash covers them', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const image = { media_type: 'image/png', data: PNG, width: 1, height: 1 }
    for (const action of ['turn.submit', 'turn.steer'] as const) {
      t.backend.calls.length = 0
      const result = await t.router.execute(alice, command(action, `img-${action}`, { text: 'look', images: [image] }))
      expect(result.state).toBe('accepted')
      const [call] = t.mutations()
      expect(call?.method).toBe(action)
      expect(call?.params.images).toEqual([{ media_type: 'image/png', data: PNG }])
      const verified = verifyHubAuth(call?.params.hub_auth, { publicKey: t.signer.publicKeyPem, method: action, params: call!.params })
      expect(verified.ok).toBe(true)
    }
  })

  test('an image alone is a valid turn; no text and no images is refused', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const result = await t.router.execute(alice, command('turn.submit', 'only-image', { images: [{ media_type: 'image/png', data: PNG }] }))
    expect(result.state).toBe('accepted')
    expect(t.mutations()[0]?.params).toMatchObject({ text: '', images: [{ media_type: 'image/png', data: PNG }] })
    await expect(t.router.execute(alice, command('turn.submit', 'empty', {}))).rejects.toMatchObject({ code: 'invalid_request' })
    await expect(t.router.execute(alice, command('turn.submit', 'empty-list', { images: [] }))).rejects.toMatchObject({ code: 'invalid_request' })
  })

  test('malformed or excessive attachments are refused before any daemon call', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const bad: unknown[] = [
      'not-an-array',
      [{ media_type: 'text/html', data: PNG }],
      [{ media_type: 'image/png' }],
      [{ media_type: 'image/png', data: '' }],
      ['nope'],
      Array.from({ length: 17 }, () => ({ media_type: 'image/png', data: PNG })),
    ]
    for (const [index, images] of bad.entries()) {
      await expect(t.router.execute(alice, command('turn.submit', `bad-${index}`, { text: 'x', images }))).rejects.toMatchObject({ code: 'invalid_request' })
    }
    expect(t.mutations()).toEqual([])
  })
})
