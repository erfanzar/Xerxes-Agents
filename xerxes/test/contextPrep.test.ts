// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import {
  detectSensitive,
  parseRedactConfig,
  prepareOutboundContext,
  restorePseudonyms,
  type OutboundSegment,
  type SensitiveKind,
} from '../src/security/contextPrep.js'

const CANARY = 'CANARY-7f3a91-do-not-leak'

function kindsOf(text: string): SensitiveKind[] {
  return detectSensitive(text).map(span => span.kind)
}

function slice(text: string, index = 0): string {
  const span = detectSensitive(text)[index]
  if (span === undefined) throw new Error(`no span ${index} in ${JSON.stringify(text)}`)
  return text.slice(span.start, span.end)
}

function validationField(run: () => unknown): string {
  try {
    run()
  } catch (error) {
    if (error instanceof ValidationError) return error.field
    throw error
  }
  throw new Error('expected a ValidationError')
}

describe('parseRedactConfig', () => {
  test('defaults to off', () => {
    expect(parseRedactConfig({}, {})).toBe('off')
    expect(parseRedactConfig({ boundary: {} }, {})).toBe('off')
  })

  test('reads runtime.boundary.redact and lets the env win', () => {
    expect(parseRedactConfig({ boundary: { redact: 'suggest' } }, {})).toBe('suggest')
    expect(parseRedactConfig({ boundary: { redact: 'suggest' } }, { XERXES_BOUNDARY_REDACT: 'auto' })).toBe('auto')
    expect(parseRedactConfig({ boundary: { redact: 'auto' } }, { XERXES_BOUNDARY_REDACT: '  ' })).toBe('auto')
  })

  test('bad values throw a ValidationError naming the key', () => {
    expect(validationField(() => parseRedactConfig({ boundary: { redact: 'sometimes' } }, {}))).toBe('boundary.redact')
    expect(validationField(() => parseRedactConfig({ boundary: { redact: 3 } }, {}))).toBe('boundary.redact')
    expect(validationField(() => parseRedactConfig({}, { XERXES_BOUNDARY_REDACT: 'yes' }))).toBe('XERXES_BOUNDARY_REDACT')
    expect(validationField(() => parseRedactConfig({ boundary: 'auto' }, {}))).toBe('boundary')
  })
})

describe('detectSensitive', () => {
  test('email', () => {
    const text = 'write to alice@example.com today'
    expect(kindsOf(text)).toEqual(['email'])
    expect(slice(text)).toBe('alice@example.com')
  })

  test('phone numbers', () => {
    expect(slice('call (415) 555-0132 now')).toBe('(415) 555-0132')
    expect(slice('call +1 415-555-0132 now')).toBe('+1 415-555-0132')
    expect(slice('call +44 20 7946 0958 now')).toBe('+44 20 7946 0958')
    const bare = detectSensitive('number 4155550132 end')
    expect(bare).toHaveLength(1)
    expect(bare[0]?.kind).toBe('phone')
    expect(bare[0]?.confidence).toBe('medium')
  })

  test('ssn', () => {
    const text = 'ssn 123-45-6789 on file'
    expect(kindsOf(text)).toEqual(['ssn'])
    expect(slice(text)).toBe('123-45-6789')
  })

  test('credit card passes Luhn, a near miss is rejected', () => {
    expect(kindsOf('card 4111 1111 1111 1111 end')).toEqual(['credit_card'])
    expect(slice('card 4111-1111-1111-1111 end')).toBe('4111-1111-1111-1111')
    expect(kindsOf('card 4111 1111 1111 1112 end')).toEqual([])
    expect(kindsOf('order 1234567890123456 shipped')).toEqual([])
  })

  test('iban is checksum-validated', () => {
    const text = 'pay GB82 WEST 1234 5698 7654 32 please'
    expect(kindsOf(text)).toEqual(['iban'])
    expect(slice(text)).toBe('GB82 WEST 1234 5698 7654 32')
    expect(kindsOf('pay GB82 WEST 1234 5698 7654 33 please')).toEqual([])
  })

  test('ip addresses', () => {
    expect(slice('host 192.168.1.10 up')).toBe('192.168.1.10')
    expect(kindsOf('host 2001:0db8:85a3:0000:0000:8a2e:0370:7334 up')).toEqual(['ip_address'])
    expect(kindsOf('version 999.1.1.1 shipped')).toEqual([])
  })

  test('api keys reuse the credential rules and cover only the secret', () => {
    const text = 'api_key=sk-abcdefghijklmnopqrstuv'
    const spans = detectSensitive(text)
    expect(spans).toHaveLength(1)
    expect(spans[0]?.kind).toBe('api_key')
    expect(slice(text)).toBe('sk-abcdefghijklmnopqrstuv')
    expect(slice('key AKIAABCDEFGHIJKLMNOP here')).toBe('AKIAABCDEFGHIJKLMNOP')
    expect(slice('Authorization: Bearer abcdef0123456789abcdef')).toBe('abcdef0123456789abcdef')
  })

  test('person name hints are medium confidence', () => {
    const titled = detectSensitive('ask Dr. Alice Smith about it')
    expect(titled[0]?.kind).toBe('person_name_hint')
    expect(titled[0]?.confidence).toBe('medium')
    expect(slice('hello, My name is Bob Jones and I')).toBe('Bob Jones')
  })

  test('clean text yields nothing', () => {
    expect(detectSensitive('refactor the parser and add a test for empty input')).toEqual([])
  })

  test('spans are sorted and overlaps merged', () => {
    const text = 'b@example.com then 123-45-6789 then a@example.com'
    const spans = detectSensitive(text)
    expect(spans.map(span => span.kind)).toEqual(['email', 'ssn', 'email'])
    for (let i = 1; i < spans.length; i++) {
      expect(spans[i]!.start).toBeGreaterThanOrEqual(spans[i - 1]!.end)
    }
  })

  test('custom patterns are detected as custom', () => {
    const text = 'see PROJ-1234 and PROJ-99'
    const spans = detectSensitive(text, [{ name: 'ticket', pattern: 'PROJ-\\d{4}' }])
    expect(spans).toHaveLength(1)
    expect(spans[0]?.kind).toBe('custom')
    const span = spans[0]
    expect(text.slice(span?.start, span?.end)).toBe('PROJ-1234')
  })

  test('invalid custom patterns throw a ValidationError', () => {
    expect(() => detectSensitive('x', [{ name: 'bad', pattern: '(' }])).toThrow(ValidationError)
    expect(() => detectSensitive('x', [{ name: 'empty', pattern: 'x*' }])).toThrow(ValidationError)
    expect(() => detectSensitive('x', [{ name: 'long', pattern: 'a'.repeat(257) }])).toThrow(ValidationError)
    const many = Array.from({ length: 65 }, (_, i) => ({ name: `p${i}`, pattern: `z${i}` }))
    expect(() => detectSensitive('x', many)).toThrow(ValidationError)
    expect(() => detectSensitive('x', many.slice(0, 64))).not.toThrow()
  })
})

describe('prepareOutboundContext', () => {
  const shareable = (id: string, text: string): OutboundSegment => ({ id, text, sensitivity: 'shareable' })

  test('off mode is byte-identical and does no detection', () => {
    const segments: OutboundSegment[] = [
      shareable('a', 'mail alice@example.com ssn 123-45-6789'),
      { id: 'b', text: `private ${CANARY}`, sensitivity: 'private' },
    ]
    const prepared = prepareOutboundContext({ segments, mode: 'off' })
    expect(prepared.text).toBe(segments.map(segment => segment.text).join('\n\n'))
    expect(prepared.findings).toBe(0)
    expect(prepared.withheld).toEqual([])
    expect(prepared.needsApproval).toBe(false)
    expect(prepared.pseudonyms.size).toBe(0)
    expect(prepared.segments.map(segment => segment.action)).toEqual(['kept', 'kept'])
  })

  test('a private segment is withheld even when it is clean', () => {
    const prepared = prepareOutboundContext({
      segments: [{ id: 'notes', text: `nothing sensitive here ${CANARY}`, sensitivity: 'private', source: `/private/${CANARY}.txt` }],
      mode: 'auto',
    })
    expect(prepared.text).toBe('[withheld: private segment notes]')
    expect(prepared.withheld).toEqual(['notes'])
    expect(prepared.segments).toEqual([{ id: 'notes', action: 'withheld', findings: 0 }])
    expect(prepared.text).not.toContain(CANARY)
    expect(prepared.summary).not.toContain(CANARY)
  })

  test('a canary inside a private segment never reaches text, summary or pseudonyms', () => {
    const prepared = prepareOutboundContext({
      segments: [
        { id: 'secret', text: `${CANARY} carol@example.com 123-45-6789`, sensitivity: 'private' },
        shareable('pub', 'public bob@example.com'),
      ],
      mode: 'suggest',
    })
    expect(prepared.text).not.toContain(CANARY)
    expect(prepared.summary).not.toContain(CANARY)
    expect(prepared.text).not.toContain('carol')
    expect([...prepared.pseudonyms.values()]).toEqual(['bob@example.com'])
    expect(prepared.findings).toBe(1)
    expect(JSON.stringify(prepared.segments)).not.toContain(CANARY)
  })

  test('default rules mask high-confidence kinds and pseudonymize email', () => {
    const prepared = prepareOutboundContext({
      segments: [shareable('a', 'ssn 123-45-6789 mail alice@example.com ip 10.0.0.1')],
      mode: 'auto',
    })
    expect(prepared.text).toBe('ssn [redacted:ssn] mail <EMAIL_1> ip [redacted:ip_address]')
    expect(prepared.findings).toBe(3)
    expect(prepared.segments).toEqual([{ id: 'a', action: 'redacted', findings: 3 }])
    expect(prepared.pseudonyms.get('<EMAIL_1>')).toBe('alice@example.com')
  })

  test('person name hints are pseudonymized by default', () => {
    const prepared = prepareOutboundContext({ segments: [shareable('a', 'ask Dr. Alice Smith')], mode: 'auto' })
    expect(prepared.text).toBe('ask <PERSON_1>')
    expect(prepared.pseudonyms.get('<PERSON_1>')).toBe('Dr. Alice Smith')
  })

  test('suggest requires approval when something changed, auto does not', () => {
    const segments = [shareable('a', 'mail alice@example.com')]
    expect(prepareOutboundContext({ segments, mode: 'suggest' }).needsApproval).toBe(true)
    expect(prepareOutboundContext({ segments, mode: 'auto' }).needsApproval).toBe(false)
    const clean = [shareable('a', 'plain text')]
    expect(prepareOutboundContext({ segments: clean, mode: 'suggest' }).needsApproval).toBe(false)
    const withPrivate: OutboundSegment[] = [{ id: 'p', text: 'x', sensitivity: 'private' }]
    expect(prepareOutboundContext({ segments: withPrivate, mode: 'suggest' }).needsApproval).toBe(true)
    expect(prepareOutboundContext({ segments: withPrivate, mode: 'auto' }).needsApproval).toBe(false)
  })

  test('pseudonyms are stable within a call, across segments, and restorable', () => {
    const original = [
      shareable('a', 'alice@example.com wrote to bob@example.com'),
      shareable('b', 'then alice@example.com replied'),
    ]
    const prepared = prepareOutboundContext({ segments: original, mode: 'auto' })
    expect(prepared.text).toBe('<EMAIL_1> wrote to <EMAIL_2>\n\nthen <EMAIL_1> replied')
    expect(prepared.pseudonyms.size).toBe(2)
    expect(restorePseudonyms(prepared.text, prepared.pseudonyms)).toBe(original.map(s => s.text).join('\n\n'))
    expect(restorePseudonyms('Tell <EMAIL_2> and <EMAIL_9>', prepared.pseudonyms)).toBe('Tell bob@example.com and <EMAIL_9>')
  })

  test('a placeholder already present in the source is never reused', () => {
    const prepared = prepareOutboundContext({
      segments: [shareable('a', 'literal <EMAIL_1> and bob@example.com')],
      mode: 'auto',
    })
    expect(prepared.text).toBe('literal <EMAIL_1> and <EMAIL_2>')
    expect(restorePseudonyms(prepared.text, prepared.pseudonyms)).toBe('literal <EMAIL_1> and bob@example.com')
  })

  test('release rules can drop, allow and override the defaults', () => {
    const segments = [shareable('a', 'mail alice@example.com ssn 123-45-6789 ip 10.0.0.1')]
    const prepared = prepareOutboundContext({
      segments,
      mode: 'auto',
      rules: [
        { kind: 'email', action: 'allow' },
        { kind: 'ssn', action: 'drop' },
        { kind: 'ip_address', action: 'pseudonymize' },
      ],
    })
    expect(prepared.text).toBe('mail alice@example.com ssn  ip <IP_1>')
    expect(prepared.pseudonyms.get('<IP_1>')).toBe('10.0.0.1')
  })

  test('credentials can never be allowed through and bad rules are rejected', () => {
    const segments = [shareable('a', 'x')]
    expect(() => prepareOutboundContext({ segments, mode: 'auto', rules: [{ kind: 'api_key', action: 'allow' }] })).toThrow(
      ValidationError,
    )
    expect(() =>
      prepareOutboundContext({ segments, mode: 'auto', rules: [{ kind: 'email', action: 'delete' as never }] }),
    ).toThrow(ValidationError)
  })

  test('api keys are masked and the secret value is gone', () => {
    const prepared = prepareOutboundContext({
      segments: [shareable('a', 'use api_key=sk-abcdefghijklmnopqrstuv for it')],
      mode: 'auto',
    })
    expect(prepared.text).toBe('use api_key=[redacted:api_key] for it')
    expect(prepared.text).not.toContain('sk-abcdefghijkl')
  })

  test('custom patterns are applied with the release rules', () => {
    const segments = [shareable('a', 'ticket PROJ-1234 is open')]
    const customPatterns = [{ name: 'ticket', pattern: 'PROJ-\\d{4}' }]
    expect(prepareOutboundContext({ segments, mode: 'auto', customPatterns }).text).toBe('ticket [redacted:custom] is open')
    const abstracted = prepareOutboundContext({
      segments,
      mode: 'auto',
      customPatterns,
      rules: [{ kind: 'custom', action: 'pseudonymize' }],
    })
    expect(abstracted.text).toBe('ticket <CUSTOM_1> is open')
    expect(abstracted.pseudonyms.get('<CUSTOM_1>')).toBe('PROJ-1234')
  })

  test('an invalid custom pattern throws before anything is prepared', () => {
    expect(() =>
      prepareOutboundContext({ segments: [shareable('a', 'x')], mode: 'auto', customPatterns: [{ name: 'bad', pattern: '[' }] }),
    ).toThrow(ValidationError)
  })

  test('maxChars drops later shareable segments first and says so', () => {
    const prepared = prepareOutboundContext({
      segments: [shareable('a', 'A'.repeat(10)), shareable('b', 'B'.repeat(10)), shareable('c', 'C'.repeat(10))],
      mode: 'auto',
      maxChars: 22,
    })
    expect(prepared.text).toBe(`${'A'.repeat(10)}\n\n${'B'.repeat(10)}`)
    expect(prepared.withheld).toEqual(['c'])
    expect(prepared.segments.map(segment => segment.action)).toEqual(['kept', 'kept', 'withheld'])
    expect(prepared.summary).toContain('maxChars=22')
    expect(prepared.needsApproval).toBe(false)
  })

  test('maxChars never drops a pseudonym the kept text still uses', () => {
    const prepared = prepareOutboundContext({
      segments: [shareable('a', 'hi alice@example.com'), shareable('b', 'bob@example.com '.repeat(5))],
      mode: 'auto',
      maxChars: 25,
    })
    expect(prepared.withheld).toEqual(['b'])
    expect([...prepared.pseudonyms.keys()]).toEqual(['<EMAIL_1>'])
  })

  test('an invalid maxChars is rejected', () => {
    expect(() => prepareOutboundContext({ segments: [], mode: 'auto', maxChars: 0 })).toThrow(ValidationError)
  })
})
