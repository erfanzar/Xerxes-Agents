// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { randomBytes } from 'node:crypto'
import { CaptureTurnSink, type CaptureDisclosure } from '../src/hub/supervisor/abilities/captureSink.js'
import { captureEnvelope, png } from './support/captureImage.js'

const source = { workspace: 'B', machine: 'source-machine', policyRevision: 'p1', viewId: 'a'.repeat(32) }
const destination = { machine: 'model-machine', profile: 'vision', model: 'fixture-model', revision: 'm1' }
function rig() {
  let live = true; let admitted = 0
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => live, maxImages: 2,
    disclose: async () => { admitted += 1 } })
  return { sink, revoke: () => { live = false }, admitted: () => admitted }
}

test('turn images remain outside textual results and are drained once as real untrusted image parts', async () => {
  const r = rig(); const raw = captureEnvelope()
  const metadata = await r.sink.add(source, raw)
  expect(JSON.stringify(metadata)).not.toContain(raw.png_base64)
  expect(JSON.stringify(r.sink)).not.toContain(raw.png_base64)
  expect(r.admitted()).toBe(1)
  const messages = r.sink.drain(destination)
  expect(messages).toHaveLength(1)
  expect(messages[0]).toMatchObject({ role: 'user', content: [
    { type: 'text', text: expect.stringContaining('UNTRUSTED') },
    { type: 'image_url', image_url: { url: `data:image/png;base64,${raw.png_base64}` } },
  ] })
  expect(r.sink.drain(destination)).toEqual([])
})

test('one bounded turn can inspect all four supported viewports without a reset or budget bypass', async () => {
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true, disclose: async () => {} })
  for (const width of [390, 768, 1280, 1440]) {
    await sink.add(source, { ...captureEnvelope(png(width)), width })
  }
  expect(sink.drain(destination)).toHaveLength(4)
  await expect(sink.add(source, captureEnvelope())).rejects.toThrow('budget')
})

test('aggregate image bytes are bounded even below the four-image count limit', async () => {
  const pixels = Buffer.alloc((390 * 4 + 1) * 900); randomBytes(200_000).copy(pixels)
  for (let row = 0; row < 900; row += 1) pixels[row * (390 * 4 + 1)] = 0
  const raw = captureEnvelope(png(390, 900, [], pixels))
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true, disclose: async () => {} })
  await sink.add(source, raw); await sink.add(source, raw)
  await expect(sink.add(source, raw)).rejects.toThrow('byte budget')
  expect(sink.drain(destination)).toHaveLength(2)
})

test('source/destination snapshots carry only canonical routing fields, never inherited grants or credentials', async () => {
  let seen: CaptureDisclosure | undefined
  const configured = { ...destination, apiKey: 'fixture-secret' }; const observed = { ...source, approved: true }
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination: configured, current: () => true,
    disclose: async request => { seen = request } })
  await sink.add(observed, captureEnvelope())
  expect(seen?.source).toEqual(source)
  expect(seen?.destination).toEqual(destination)
})

test('missing/rejected disclosure and source capture mismatch cannot enqueue an image', async () => {
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true,
    disclose: async () => { throw new Error('source private policy') } })
  await expect(sink.add(source, captureEnvelope())).rejects.toThrow('disclosure')
  expect(sink.drain(destination)).toEqual([])
  const r = rig()
  await expect(r.sink.add({ ...source, viewId: 'b'.repeat(32) }, captureEnvelope())).rejects.toThrow()
  expect(r.admitted()).toBe(0)
  expect(r.sink.drain(destination)).toEqual([])
})

test('disclosure host errors never echo image bytes or credential-bearing diagnostics', async () => {
  const raw = captureEnvelope()
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true,
    disclose: async () => { throw new Error(`fixture-secret ${raw.png_base64}`) } })
  const error = await sink.add(source, raw).then(() => '', failure => String(failure))
  expect(error).not.toContain('fixture-secret')
  expect(error).not.toContain(raw.png_base64)
  expect(sink.drain(destination)).toEqual([])
})

test('live-authority host failure is a sanitized terminal refusal', async () => {
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination,
    current: () => { throw new Error('fixture-secret') }, disclose: async () => {} })
  const error = await sink.add(source, captureEnvelope()).then(() => '', failure => String(failure))
  expect(error).not.toContain('fixture-secret')
  expect(() => sink.drain(destination)).toThrow('closed')
})

test('cancellation or owner/config revocation during awaited disclosure discards the image', async () => {
  for (const mode of ['abort', 'revoke']) {
    let finish = (): void => {}; let ready = (): void => {}; let live = true
    const held = new Promise<void>(resolve => { finish = resolve }); const started = new Promise<void>(resolve => { ready = resolve })
    const controller = new AbortController()
    const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => live,
      disclose: async () => { ready(); await held } })
    const work = sink.add(source, captureEnvelope(), controller.signal)
    try {
      await started
      if (mode === 'abort') controller.abort(); else live = false
      finish()
      await expect(work).rejects.toThrow()
      live = true
      expect(() => sink.drain(destination)).toThrow('closed')
    } finally { finish(); sink.close(); await work.catch(() => {}) }
  }
})

test('aborting an ignored disclosure returns promptly without releasing pixels later', async () => {
  let finish = (): void => {}; let ready = (): void => {}
  const held = new Promise<void>(resolve => { finish = resolve }); const started = new Promise<void>(resolve => { ready = resolve })
  const controller = new AbortController()
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true,
    disclose: async () => { ready(); await held } })
  const work = sink.add(source, captureEnvelope(), controller.signal)
  try {
    await started; controller.abort()
    expect(await Promise.race([work.then(() => 'resolved', () => 'refused'), Bun.sleep(30).then(() => 'hung')])).toBe('refused')
  } finally { finish(); sink.close(); await work.catch(() => {}) }
})

test('timed-out disclosure keeps its slots until the ignored host actually settles', async () => {
  let finish = (): void => {}; const held = new Promise<void>(resolve => { finish = resolve })
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true,
    disclosureTimeoutMs: 5, maxImages: 2, disclose: async () => held })
  const one = sink.add(source, captureEnvelope()).catch(() => 'refused'); const two = sink.add(source, captureEnvelope()).catch(() => 'refused')
  try {
    expect(await Promise.race([Promise.all([one, two]).then(() => 'refused'), Bun.sleep(50).then(() => 'hung')])).toBe('refused')
    await expect(sink.add(source, captureEnvelope())).rejects.toThrow('budget')
  } finally { finish(); sink.close(); await Promise.all([one, two]) }
})

test('changed destination and closed turns never release queued pixels', async () => {
  const r = rig(); await r.sink.add(source, captureEnvelope())
  expect(() => r.sink.drain({ ...destination, revision: 'm2' })).toThrow()
  r.sink.close()
  expect(() => r.sink.drain(destination)).toThrow()
  await expect(r.sink.add(source, captureEnvelope())).rejects.toThrow()
})

test('an exhausted sink refuses before touching an image payload or running its codec', async () => {
  const r = rig(); await r.sink.add(source, captureEnvelope()); await r.sink.add(source, captureEnvelope())
  let inspected = 0
  const raw = new Proxy(captureEnvelope(), { get(target, key, receiver) { inspected += 1; return Reflect.get(target, key, receiver) },
    ownKeys(target) { inspected += 1; return Reflect.ownKeys(target) } })
  await expect(r.sink.add(source, raw)).rejects.toThrow('budget')
  expect(inspected).toBe(0)
})

test('the cumulative two-image budget remains charged after draining and concurrent admission is atomic', async () => {
  let finish = (): void => {}; const held = new Promise<void>(resolve => { finish = resolve })
  const sink = new CaptureTurnSink({ owner: 'owner', thread: 'thread', destination, current: () => true, maxImages: 2, disclose: async () => held })
  const one = sink.add(source, captureEnvelope()); const two = sink.add(source, captureEnvelope())
  try {
    await expect(sink.add(source, captureEnvelope())).rejects.toThrow('budget')
    finish(); await Promise.all([one, two]); expect(sink.drain(destination)).toHaveLength(2)
    await expect(sink.add(source, captureEnvelope())).rejects.toThrow('budget')
  } finally { finish(); sink.close(); await Promise.allSettled([one, two]) }
})
