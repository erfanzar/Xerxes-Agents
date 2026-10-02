// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderTokenCounter } from '../src/context/tokenCounter.js'

function estimate(text: string, provider = 'openai'): number {
  return ProviderTokenCounter.countTokensForProvider(text, provider)
}

test('pure-Latin inputs keep their exact legacy estimates', () => {
  // Golden values captured from the pre-CJK lexical estimator.
  expect(estimate('the quick brown fox jumps over the lazy dog')).toBe(11)
  expect(estimate('{}'.repeat(500))).toBe(250)
  expect(estimate('hello')).toBe(2)
  expect(estimate('one two three four')).toBe(5)
  expect(estimate('!!! ??? ...')).toBe(3)
  expect(estimate('12345 678')).toBe(3)
  expect(estimate('x'.repeat(2000))).toBe(500)

  // Floors: empty input stays at zero and non-empty input never estimates below one.
  expect(estimate('')).toBe(0)
  expect(estimate('   ')).toBe(1)
})

test('CJK text is estimated near 0.75 tokens per code point instead of one token per run', () => {
  const han = '漢'.repeat(720)
  // 720 Han characters previously estimated as a single lexical run (180 via len/4);
  // real BPE emits roughly 0.6-1.0 tokens per character.
  expect(estimate(han)).toBe(Math.ceil(720 * 0.75))
  expect(estimate(han)).toBeGreaterThanOrEqual(486)
  expect(estimate(han)).toBeLessThanOrEqual(594)

  // Every covered script block counts code points individually rather than
  // collapsing to one lexical token: Hiragana, Katakana, Hangul syllables,
  // CJK punctuation, fullwidth forms, Han Extension A, and compatibility ideographs.
  for (const sample of [
    'あ'.repeat(100),
    'ア'.repeat(100),
    '한'.repeat(100),
    '、'.repeat(100),
    'Ａ'.repeat(100),
    '㐀'.repeat(100),
    '豈'.repeat(100),
  ]) {
    expect(estimate(sample)).toBe(75)
  }
})

test('mixed-script text combines the CJK contribution with the Latin remainder', () => {
  const mixed = `Hello ${'世'.repeat(100)} world`
  // 75 for the Han run plus the "Hello world" remainder (lexical 2, ceil(12/4) = 3).
  // The whole-text baseline (ceil(112/4) = 28) must not mask the correction.
  expect(estimate(mixed)).toBe(78)

  // Short mixed input keeps the historical floor instead of over-counting.
  expect(estimate('你')).toBe(1)
  expect(estimate('a 你 b')).toBe(3)
})

test('provider adjustment and message serialization stay consistent for CJK payloads', () => {
  const han = '漢'.repeat(720)
  expect(ProviderTokenCounter.countTokensForProvider(han, 'google')).toBe(Math.ceil(540 * 1.1))

  const messages = [{ role: 'user', content: han }]
  // Serialization adds only its small role/newline wrapper on top of the CJK
  // correction instead of collapsing the whole Han payload into one run.
  const wrapped = ProviderTokenCounter.countTokensForProvider(messages, 'openai')
  expect(wrapped).toBeGreaterThanOrEqual(540)
  expect(wrapped).toBeLessThanOrEqual(545)
})

test('an image part is priced as one image, not as its base64 payload', () => {
  // Regression: a 300 KB screenshot's data URL went through the char/4
  // heuristic and estimated near 100K tokens, so the in-turn compaction check
  // failed every turn that carried it.
  const data = 'iVBORw0KGgo'.repeat(30_000)
  const openai = [{ role: 'user', content: [
    { type: 'text', text: 'what is on screen?' },
    { type: 'image_url', image_url: { url: `data:image/png;base64,${data}` } },
  ] }]
  const anthropic = [{ role: 'user', content: [
    { type: 'image', source: { type: 'base64', media_type: 'image/png', data } },
  ] }]
  const nested = [{ role: 'tool', tool_call_id: 'shot', content: [
    { type: 'image_url', image_url: { url: `data:image/png;base64,${data}` } },
  ] }, { role: 'assistant', content: 'done', meta: { parts: [{ type: 'image', data, mimeType: 'image/png' }] } }]
  const textOnly = ProviderTokenCounter.countTokensForProvider([{ role: 'user', content: 'what is on screen?' }], 'openai')

  const withImage = ProviderTokenCounter.countTokensForProvider(openai, 'openai')
  expect(withImage).toBeGreaterThan(textOnly + 1_000)
  expect(withImage).toBeLessThan(textOnly + 2_000)
  expect(ProviderTokenCounter.countTokensForProvider(anthropic, 'anthropic')).toBeLessThan(2_000)
  // Nested in a structured field the payload is not counted a second time.
  expect(ProviderTokenCounter.countTokensForProvider(nested, 'openai')).toBeLessThan(4_000)
  expect(ProviderTokenCounter.messagesToText(openai)).not.toContain(data.slice(0, 64))
})
