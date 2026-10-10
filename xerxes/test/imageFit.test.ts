// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { encode as encodePng } from 'fast-png'
import { fitImage, imageSize, MAX_IMAGE_EDGE } from '../src/core/imageFit.js'

const png = (width: number, height: number, channels: 3 | 4 = 4, alpha = 255): string => {
  const data = new Uint8Array(width * height * channels)
  for (let index = 0; index < width * height; index += 1) {
    data[index * channels] = (index * 7) % 256
    data[index * channels + 1] = (index * 13) % 256
    data[index * channels + 2] = 200
    if (channels === 4) data[index * channels + 3] = alpha
  }
  return Buffer.from(encodePng({ width, height, data, channels, depth: 8 })).toString('base64')
}

test('an oversized screenshot is scaled to the edge Claude Code passes through, the same bytes every time', () => {
  // A retina screenshot Claude Code would resize itself, which stops the prompt cache extending.
  const input = { mediaType: 'image/png', data: png(3600, 2338) }
  const first = fitImage(input)
  expect(first.mediaType).toBe('image/jpeg')
  expect(imageSize(Buffer.from(first.data, 'base64'))).toEqual({ width: MAX_IMAGE_EDGE, height: 1299, format: 'jpeg' })
  // A fresh copy of the same image (no shared string) gives identical bytes.
  expect(fitImage({ mediaType: 'image/png', data: input.data.slice() }).data).toBe(first.data)
})

test('an image that already fits is passed through untouched', () => {
  const input = { mediaType: 'image/png', data: png(800, 600) }
  expect(fitImage(input)).toBe(input)
})

test('transparency is flattened onto white, and an image this cannot read is left alone', () => {
  const clear = fitImage({ mediaType: 'image/png', data: png(2400, 100, 4, 0) })
  expect(clear.mediaType).toBe('image/jpeg')
  const garbage = { mediaType: 'image/png', data: Buffer.from('not an image at all').toString('base64') }
  expect(fitImage(garbage)).toBe(garbage)
})
