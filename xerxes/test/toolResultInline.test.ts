// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { keepsToolResultInline } from '../src/daemon/turnRunner.js'

test('large screenshots reach the model whole; other large results are spilled', () => {
  const screenshot = JSON.stringify({ _multimodal: true, content: [{ type: 'image_url', image_url: { url: 'data:image/jpeg;base64,' + 'A'.repeat(300_000) } }], text_summary: 'Screen capture: 1568x1018' })
  expect(keepsToolResultInline(screenshot)).toBe(true)
  expect(keepsToolResultInline('x'.repeat(300_000))).toBe(false)
  expect(keepsToolResultInline('short')).toBe(true)
  // Only a result that starts as a multimodal payload counts as a screenshot.
  expect(keepsToolResultInline('{"text":"' + 'x'.repeat(20_000) + '","_multimodal":true}')).toBe(false)
})
