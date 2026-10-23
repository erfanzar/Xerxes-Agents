// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { createDraftStore } from '../../client/drafts.js'
import { IMAGE_LIMITS, bytesToBase64, checkImageFile, fitWithin, fitsWireBudget, prepareImage, type ImageEngine } from '../../client/imagePrep.js'
import { statusView, type StatusEvidence } from '../../client/orbMapping.js'
import { NO_ACTIVITY } from '../../client/taskTypes.js'
import { deviceLabelFromUserAgent, initialPairState, parsePairFragment, reducePair, takePairingFragment } from '../pairing.js'

const evidence = (overrides: Partial<StatusEvidence>): StatusEvidence => ({
  connection: 'live', status: 'idle', activity: NO_ACTIVITY, pendingCount: 0, stale: false, ...overrides,
})

describe('orb mapping', () => {
  it('maps connection states to a connecting orb', () => {
    for (const connection of ['connecting', 'replaying'] as const) {
      expect(statusView(evidence({ connection, status: 'working' }))).toMatchObject({ kind: 'orb', state: 'connecting', label: 'Connecting' })
    }
  })

  it('maps reasoning to solving "Thinking"', () => {
    expect(statusView(evidence({ status: 'working', activity: { kind: 'thinking' } }))).toMatchObject({ state: 'solving', label: 'Thinking' })
  })

  it('maps tools to the existing activity vocabulary', () => {
    expect(statusView(evidence({ status: 'working', activity: { kind: 'tool', name: 'exec_command' } }))).toMatchObject({ state: 'working', label: 'Running a command' })
    expect(statusView(evidence({ status: 'working', activity: { kind: 'tool', name: 'GrepTool' } }))).toMatchObject({ state: 'searching' })
    expect(statusView(evidence({ status: 'working', activity: { kind: 'tool', name: 'FileEditTool' } }))).toMatchObject({ state: 'shaping' })
    expect(statusView(evidence({ status: 'working', activity: { kind: 'tool', name: 'TodoWrite' } }))).toMatchObject({ state: 'solving' })
  })

  it('maps an agent wait to the constellation with a count', () => {
    expect(statusView(evidence({ status: 'working', activity: { kind: 'agents', waiting: 3 } }))).toMatchObject({ state: 'connecting', label: 'Waiting on 3 agents' })
    expect(statusView(evidence({ status: 'working', activity: { kind: 'agents', waiting: 1 } })).label).toBe('Waiting on 1 agent')
  })

  it('maps a pending approval or question to a breathing "Needs you"', () => {
    expect(statusView(evidence({ status: 'working', pendingCount: 1 }))).toMatchObject({ state: 'breathing', live: true, label: 'Needs you', tone: 'attention' })
    expect(statusView(evidence({ status: 'needs_you' }))).toMatchObject({ label: 'Needs you' })
  })

  it('maps confirmed idle to a still breathing frame and a confirmed failure to a static mark', () => {
    expect(statusView(evidence({ status: 'idle' }))).toMatchObject({ kind: 'orb', state: 'breathing', live: false, label: 'Idle' })
    expect(statusView(evidence({ status: 'failed' }))).toMatchObject({ kind: 'mark', mark: 'failed', label: 'Failed' })
  })

  it('says "Working" with no evidence, and "Waiting for update" when the data is stale — never invented thinking', () => {
    const working = statusView(evidence({ status: 'working' }))
    expect(working).toMatchObject({ state: 'working', label: 'Working' })
    const stale = statusView(evidence({ status: 'working', stale: true }))
    expect(stale).toMatchObject({ label: 'Waiting for update', outdated: true })
    for (const view of [working, stale]) expect(JSON.stringify(view)).not.toMatch(/Thinking|%/)
  })

  it('keeps connection health apart from task activity: a dropped link marks outdated instead of changing the activity', () => {
    const view = statusView(evidence({ connection: 'reconnecting', status: 'working', activity: { kind: 'thinking' } }))
    expect(view).toMatchObject({ label: 'Thinking', outdated: true })
  })

  it('shows unavailable workspaces as unavailable, not idle or failed', () => {
    expect(statusView(evidence({ workspaceHealth: 'offline', status: 'idle' }))).toMatchObject({ kind: 'mark', mark: 'offline', label: 'Unavailable' })
    expect(statusView(evidence({ workspaceHealth: 'needs_operator', status: 'failed' }))).toMatchObject({ label: 'Unavailable' })
    expect(statusView(evidence({ interrupted: true, status: 'unknown' }))).toMatchObject({ label: 'Interrupted' })
  })
})

describe('image attachments', () => {
  const file = (over: Partial<{ name: string; type: string; size: number }> = {}) => ({ name: 'p.jpg', type: 'image/jpeg', size: 1000, ...over })

  it('accepts supported formats and refuses the rest with an actionable message', () => {
    expect(checkImageFile(file(), 0)).toEqual({ ok: true })
    for (const type of ['image/png', 'image/webp', 'image/gif']) expect(checkImageFile(file({ type }), 0).ok).toBe(true)
    const heic = checkImageFile(file({ name: 'IMG.HEIC', type: 'image/heic' }), 0)
    expect(heic.ok).toBe(false)
    expect(heic.ok ? '' : heic.message).toContain('Most Compatible')
    const heicNoType = checkImageFile(file({ name: 'IMG_1.heif', type: '' }), 0)
    expect(heicNoType.ok ? '' : heicNoType.message).toContain('HEIC')
    const pdf = checkImageFile(file({ name: 'a.pdf', type: 'application/pdf' }), 0)
    expect(pdf.ok ? '' : pdf.message).toContain('Choose a JPEG, PNG, WebP or GIF')
  })

  it('enforces count and size limits', () => {
    const tooMany = checkImageFile(file(), IMAGE_LIMITS.maxImages)
    expect(tooMany.ok ? '' : tooMany.message).toContain(`up to ${IMAGE_LIMITS.maxImages}`)
    expect(checkImageFile(file({ size: IMAGE_LIMITS.maxSourceBytes + 1 }), 0).ok).toBe(false)
    expect(checkImageFile(file({ size: 0 }), 0).ok).toBe(false)
  })

  it('resizes to at most 1600px on the long edge and never upscales', () => {
    expect(fitWithin(4000, 3000)).toEqual({ width: 1600, height: 1200 })
    expect(fitWithin(3000, 4000)).toEqual({ width: 1200, height: 1600 })
    expect(fitWithin(800, 600)).toEqual({ width: 800, height: 600 })
  })

  it('re-encodes as JPEG, stepping quality down until it fits, and reports progress', async () => {
    const qualities: number[] = []
    const engine: ImageEngine = {
      measure: async () => ({ width: 4000, height: 3000 }),
      encode: async (_file, target, quality) => {
        qualities.push(quality)
        expect(Math.max(target.width, target.height)).toBeLessThanOrEqual(IMAGE_LIMITS.maxEdge)
        const size = quality > 0.7 ? IMAGE_LIMITS.maxEncodedBytes + 1 : 1000
        return { bytes: new Uint8Array(size).fill(1), previewUrl: 'data:image/jpeg;base64,AA==' }
      },
    }
    const progress: number[] = []
    const image = await prepareImage(new Blob(['x']), engine, { id: 'i1', onProgress: p => progress.push(p) })
    expect(image).toMatchObject({ id: 'i1', media_type: 'image/jpeg', width: 1600, height: 1200, bytes: 1000 })
    expect(qualities.slice(0, 2)).toEqual([0.85, 0.72])
    expect(image.previewUrl.startsWith('data:')).toBe(true)
    expect(progress.at(-1)).toBe(1)
  })

  it('gives up with a readable message when an image cannot be made small enough', async () => {
    const engine: ImageEngine = {
      measure: async () => ({ width: 2000, height: 2000 }),
      encode: async () => ({ bytes: new Uint8Array(IMAGE_LIMITS.maxEncodedBytes + 1), previewUrl: '' }),
    }
    await expect(prepareImage(new Blob(['x']), engine)).rejects.toThrow('too detailed')
  })

  it('keeps a set of attachments inside the wire budget', () => {
    expect(fitsWireBudget([{ bytes: 200_000 }, { bytes: 200_000 }, { bytes: 200_000 }])).toBe(true)
    expect(fitsWireBudget([{ bytes: 400_000 }, { bytes: 400_000 }])).toBe(false)
    expect(bytesToBase64(new Uint8Array([1, 2, 3]))).toBe('AQID')
  })
})

describe('pairing', () => {
  function fakeWindow(hash: string) {
    const replaced: (string | null | undefined)[] = []
    return {
      replaced,
      win: { location: { hash, pathname: '/pair', search: '' }, history: { replaceState: (_d: unknown, _u: string, url?: string | null) => { replaced.push(url) } } },
    }
  }

  it('reads the fragment and removes it from the address bar before anything else happens', () => {
    const { win, replaced } = fakeWindow('#pairing_id=p1&code=sekret&cc=4829')
    const read = takePairingFragment(win)
    expect(read).toEqual({ kind: 'found', fragment: { pairingId: 'p1', code: 'sekret', comparison: '4829' } })
    expect(replaced).toEqual(['/pair'])
    expect(replaced.join()).not.toContain('sekret')
  })

  it('clears a malformed fragment too, and does nothing when there is none', () => {
    const bad = fakeWindow('#garbage')
    expect(takePairingFragment(bad.win)).toEqual({ kind: 'invalid' })
    expect(bad.replaced).toEqual(['/pair'])
    const none = fakeWindow('')
    expect(takePairingFragment(none.win)).toEqual({ kind: 'missing' })
    expect(none.replaced).toEqual([])
    expect(parsePairFragment('#pairing_id=only')).toBeUndefined()
    // The Hub's ticket form: base64url JSON {hub_id, pairing_id, code}.
    const ticket = btoa(JSON.stringify({ hub_id: 'h', pairing_id: 'pr_1', code: 'sekret' })).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
    expect(parsePairFragment(`#${ticket}`)).toEqual({ pairingId: 'pr_1', code: 'sekret' })
  })

  it('walks submitting, waiting for the desktop, then paired; errors and timeouts are explicit', () => {
    let state = initialPairState({ kind: 'found', fragment: { pairingId: 'p', code: 'c', comparison: '1111' } })
    expect(state).toEqual({ phase: 'submitting', comparison: '1111' })
    state = reducePair(state, { type: 'accepted', comparison: '2222' })
    expect(state).toEqual({ phase: 'waiting', comparison: '2222' })
    expect(reducePair(state, { type: 'confirmed' })).toEqual({ phase: 'paired' })
    expect(reducePair(state, { type: 'expired' })).toEqual({ phase: 'timeout', comparison: '2222' })
    expect(reducePair({ phase: 'submitting' }, { type: 'confirmed' })).toEqual({ phase: 'submitting' })
    const down = reducePair({ phase: 'submitting', comparison: '1' }, { type: 'unavailable', message: 'down' })
    expect(reducePair(down, { type: 'retry' })).toEqual({ phase: 'submitting', comparison: '1' })
    expect(reducePair(state, { type: 'rejected', message: 'no' })).toEqual({ phase: 'rejected', message: 'no' })
  })

  it('labels the device from the user agent', () => {
    expect(deviceLabelFromUserAgent('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)')).toBe('iPhone')
    expect(deviceLabelFromUserAgent('Mozilla/5.0 (Linux; Android 14; Pixel) Mobile')).toBe('Android phone')
    expect(deviceLabelFromUserAgent('Mozilla/5.0 (X11; Linux)')).toBe('Browser')
  })
})

describe('drafts', () => {
  it('keeps what was typed per task, forgets cleared ones, and stays bounded', () => {
    const drafts = createDraftStore(2)
    drafts.set('a', 'one')
    drafts.set('b', 'two')
    expect(drafts.get('a')).toBe('one')
    drafts.set('c', 'three')
    expect(drafts.get('a')).toBe('')
    expect(drafts.get('c')).toBe('three')
    drafts.clear('c')
    expect(drafts.get('c')).toBe('')
  })
})
