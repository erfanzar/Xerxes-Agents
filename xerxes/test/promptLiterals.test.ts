// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { maskPromptLiterals } from '../src/runtime/promptLiterals.js'
import { detectThinkingDirective } from '../src/runtime/thinkingLevels.js'

test('masking preserves length, newlines and every index', () => {
  const text = 'a "bc" d\n`ef` g'
  const masked = maskPromptLiterals(text)
  expect(masked).toHaveLength(text.length)
  expect(masked.split('\n')).toHaveLength(2)
  // Delimiters survive; only their contents are blanked.
  expect(masked).toBe('a "  " d\n`  ` g')
})

test('a keyword the user is quoting does not escalate the turn', () => {
  for (const prompt of [
    'fix the parser in `ultrathink.ts`',
    'the docs say "think harder" is the strongest rung',
    "grep for 'megathink' in the logs",
    'paste:\n```\nWARN think hard budget exceeded\n```\nwhat does that mean?',
    '~~~\nultrathink\n~~~',
  ]) {
    expect(detectThinkingDirective(prompt), prompt).toBeUndefined()
  }
})

test('a keyword the user is actually issuing still escalates', () => {
  expect(detectThinkingDirective('ultrathink about this')?.level).toBe('ultrathink')
  expect(detectThinkingDirective('think harder please')?.level).toBe('think_harder')
  // Outside the fence is still live text.
  expect(detectThinkingDirective('```\nsome code\n```\nnow ultrathink')?.level).toBe('ultrathink')
  // A quoted mention plus a real instruction: the instruction wins.
  expect(detectThinkingDirective('unlike "think hard", please ultrathink')?.level).toBe('ultrathink')
})

test('an apostrophe inside a word opens nothing', () => {
  // "don't ... ultrathink" would otherwise mask the rest of the line.
  expect(detectThinkingDirective("don't hold back, ultrathink this")?.level).toBe('ultrathink')
  expect(maskPromptLiterals("the user's think budget")).toBe("the user's think budget")
})

test('an unterminated span masks conservatively and stops at its line', () => {
  expect(detectThinkingDirective('he said "ultrathink')).toBeUndefined()
  // The next line is outside the unterminated quote, so it stays live.
  expect(detectThinkingDirective('he said "blah\nultrathink now')?.level).toBe('ultrathink')
})

test('an escaped delimiter does not open a span', () => {
  expect(detectThinkingDirective('a \\" ultrathink')?.level).toBe('ultrathink')
})
