// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { validateCaptureImage } from '../src/hub/supervisor/abilities/captureImage.js'
import { captureEnvelope as envelope, chunk, png } from './support/captureImage.js'

test('capture validation returns actual inline PNG image content and immutable metadata, not a descriptive substitute', () => {
  const source = envelope(); const image = validateCaptureImage(source)
  expect(image).toMatchObject({ width: 390, height: 900, mime: 'image/png', viewId: source.view_id, sha256: source.sha256 })
  expect(image.imageUrl).toBe(`data:image/png;base64,${source.png_base64}`)
  source.png_base64 = 'changed'
  expect(image.imageUrl).not.toContain('changed')
  expect(Object.isFrozen(image)).toBe(true)
})

test.each([390, 768, 1280, 1440])('each approved viewport admits a real fixed-height image: %i', width => {
  const source = { ...envelope(png(width)), width }
  const image = validateCaptureImage(source)
  expect(image.width).toBe(width)
  expect(image.height).toBe(900)
})

test('non-interlaced eight-bit RGB captures are supported without an alpha channel', () => {
  const image = validateCaptureImage(envelope(png(390, 900, [], undefined, undefined, 2)))
  expect(image.mime).toBe('image/png')
  expect(image.width).toBe(390)
})

test('PNG dimensions, fixed viewports, digest and base64 are validated before image admission', () => {
  const valid = envelope()
  for (const value of [
    { ...valid, width: 999 }, { ...valid, height: 100_000 }, { ...valid, sha256: 'b'.repeat(64) },
    { ...valid, png_base64: `${valid.png_base64}\n` }, { ...valid, png_base64: 'https://elsewhere.invalid/image.png' },
    envelope(png(768)), { ...valid, authorization: 'forged' }, { ...valid, mime: 'image/svg+xml' },
  ]) expect(() => validateCaptureImage(value)).toThrow()
})

test('unknown PNG metadata, CRC corruption and trailing bytes never reach an image consumer', () => {
  const corrupt = png(); corrupt[corrupt.length - 1] = 0
  for (const bytes of [png(390, 900, [chunk('tEXt', Buffer.from('credential=fixture-token'))]), corrupt, Buffer.concat([png(), Buffer.from('hidden')])]) {
    expect(() => validateCaptureImage(envelope(bytes))).toThrow()
  }
})

test('PNG chunk names are exact bytes, not ASCII aliases of high-bit bytes', () => {
  const bytes = png(); bytes[12] = 0xc9
  bytes.writeUInt32BE(Bun.hash.crc32(bytes.subarray(12, 29)), 29)
  expect(() => validateCaptureImage(envelope(bytes))).toThrow()
})

test('admitted PNG output is re-encoded from validated raster, excluding unused compressed-stream bytes', () => {
  const hidden = Buffer.from('fixture-token-not-a-pixel')
  const image = validateCaptureImage(envelope(png(390, 900, [], undefined, hidden)))
  const output = Buffer.from(image.imageUrl.split(',')[1] ?? '', 'base64')
  expect(output.includes(hidden)).toBe(false)
  expect(image.sourceSha256).toBe(envelope(png(390, 900, [], undefined, hidden)).sha256)
  expect(image.normalization).toBe('png-raster-v1')
  expect(image.sha256).toBe(new Bun.CryptoHasher('sha256').update(output).digest('hex'))
  expect(validateCaptureImage(envelope(output)).imageUrl).toBe(image.imageUrl)
})

test('bounded PNG inflation rejects compressed pixel bombs and malformed raster length', () => {
  const expected = (390 * 4 + 1) * 900
  for (const pixels of [Buffer.alloc(expected - 1), Buffer.alloc(expected * 4)]) {
    expect(() => validateCaptureImage(envelope(png(390, 900, [], pixels)))).toThrow()
  }
  expect(() => validateCaptureImage({ ...envelope(), png_base64: 'A'.repeat(400_000) })).toThrow()
})
