// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { coerceImage, serializeImage } from '../src/core/multimodal.js'

const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

test('multimodal helpers preserve bytes and decode raw or data-url base64 image payloads', () => {
  const rawBase64 = Buffer.from(PNG_SIGNATURE).toString('base64')
  const fromBytes = coerceImage(PNG_SIGNATURE)
  const fromBase64 = coerceImage(rawBase64)
  const fromDataUrl = coerceImage('data:image/png;base64,' + rawBase64)

  expect(fromBytes).toEqual({ bytes: PNG_SIGNATURE, mediaType: 'image/png' })
  expect(fromBytes.bytes).not.toBe(PNG_SIGNATURE)
  expect(fromBase64).toEqual({ bytes: PNG_SIGNATURE, mediaType: 'image/png' })
  expect(fromDataUrl).toEqual({ bytes: PNG_SIGNATURE, mediaType: 'image/png' })
})

test('multimodal serialization honors bounded payloads and rejects invalid native image inputs', () => {
  const encoded = Buffer.from(PNG_SIGNATURE).toString('base64')
  expect(serializeImage(PNG_SIGNATURE, { maxImageB64Length: 4 })).toBe(encoded.slice(0, 4) + '...')
  expect(serializeImage(PNG_SIGNATURE, { addFormatPrefix: true }))
    .toBe('data:image/png;base64,' + encoded)

  expect(() => coerceImage('not-valid-base64!!!')).toThrow(ValidationError)
  expect(() => coerceImage(PNG_SIGNATURE, 'image/tiff')).toThrow(ValidationError)
  expect(() => serializeImage(PNG_SIGNATURE, { maxImageB64Length: 0 })).toThrow(ValidationError)
})
