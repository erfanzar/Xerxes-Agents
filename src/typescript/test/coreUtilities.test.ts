// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  PromptSection,
  PromptTemplate,
  agentsSubdirFor,
  coerceImage,
  downloadImage,
  serializeImage,
  xerxesSubdirFor,
} from '../src/index.js'

test('prompt templates preserve canonical ordering while rendering only populated sections', () => {
  const template = new PromptTemplate()
  expect(template.render({
    [PromptSection.SYSTEM]: 'Be concise.',
    [PromptSection.RULES]: 'Use tools safely.',
    [PromptSection.PROMPT]: 'Inspect this file.',
  })).toBe([
    'SYSTEM:\nBe concise.',
    'RULES:\nUse tools safely.',
    'PROMPT:\nInspect this file.',
  ].join('\n\n'))

  const custom = new PromptTemplate({ sectionOrder: [PromptSection.PROMPT] })
  expect(custom.render({ [PromptSection.PROMPT]: 'hello' })).toBe('PROMPT:\nhello')
})

test('core path helpers retain explicit home/environment routing without creating paths', () => {
  expect(xerxesSubdirFor({ XERXES_HOME: '/tmp/xerxes-home' }, 'sessions', 'one.json'))
    .toBe('/tmp/xerxes-home/sessions/one.json')
  expect(agentsSubdirFor('/tmp/home', 'skills', 'demo')).toBe('/tmp/home/.agents/skills/demo')
})

test('multimodal helpers serialize binary images and bound HTTP downloads', async () => {
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const encoded = serializeImage({ bytes: png, mediaType: 'image/png' }, { addFormatPrefix: true })
  expect(encoded).toStartWith('data:image/png;base64,')
  expect(coerceImage(encoded)).toEqual({ bytes: png, mediaType: 'image/png' })

  const downloaded = await downloadImage('https://example.test/image', {
    fetchImplementation: async () => new Response(png, { headers: { 'content-type': 'image/png' } }),
  })
  expect(downloaded).toEqual({ bytes: png, mediaType: 'image/png' })

  await expect(downloadImage('https://example.test/large', {
    maxBytes: 4,
    fetchImplementation: async () => new Response(png, { headers: { 'content-type': 'image/png' } }),
  })).rejects.toThrow('Image exceeds maximum size')
})
