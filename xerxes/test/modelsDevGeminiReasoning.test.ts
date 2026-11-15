// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { findInModelsDev, parseModelsDev } from '../src/llms/modelsDev.js'

// models.dev files Gemini under `google`; Xerxes calls the provider `gemini`.
const document = {
  google: {
    id: 'google', api: 'https://generativelanguage.googleapis.com/v1beta',
    models: {
      'gemini-3.1-pro-preview': { id: 'gemini-3.1-pro-preview', reasoning: true, reasoning_options: [{ type: 'effort', values: ['low', 'medium', 'high'] }] },
    },
  },
  other: {
    id: 'other', api: 'https://other.example/v1',
    models: { 'gemini-3.1-pro-preview': { id: 'gemini-3.1-pro-preview', reasoning: false } },
  },
}

test('a gemini profile finds its model under models.dev google, so effort levels are offered', () => {
  const found = findInModelsDev(parseModelsDev(document), { model: 'gemini-3.1-pro-preview', provider: 'gemini' })
  expect(found?.reasoning).toMatchObject({ supported: true, efforts: ['low', 'medium', 'high'] })
})
