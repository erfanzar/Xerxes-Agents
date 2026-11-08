// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Context preparation for outbound cloud requests (M2.2).
 *
 * Detects sensitive spans with deterministic rules (no ML, no network) and builds
 * a request that is permitted by selection (private segments are withheld),
 * redaction (mask / drop) or abstraction (stable pseudonyms the local side can map
 * back). Detection supports policy; it never grants permission: a segment marked
 * private is withheld whether or not anything was detected in it, and is never
 * scanned or copied into the prepared text, the summary or the pseudonym map.
 *
 * Everything here is OFF by default (`boundary.redact`: off | suggest | auto).
 */

import { ValidationError } from '../core/errors.js'
import type { Sensitivity } from '../routing/types.js'
import { DEFAULT_REDACTION_RULES } from './redact.js'
import { CREDENTIAL_REDACTION_RULES } from './promptScanner.js'

export type RedactMode = 'off' | 'suggest' | 'auto'

export type SensitiveKind =
  | 'email'
  | 'phone'
  | 'ssn'
  | 'credit_card'
  | 'iban'
  | 'ip_address'
  | 'api_key'
  | 'person_name_hint'
  | 'custom'

export interface SensitiveSpan {
  /** Inclusive start offset into the scanned text. */
  readonly start: number
  /** Exclusive end offset into the scanned text. */
  readonly end: number
  readonly kind: SensitiveKind
  readonly confidence: 'high' | 'medium'
}

export type ReleaseAction = 'drop' | 'mask' | 'pseudonymize' | 'allow'

export interface ReleaseRule {
  readonly kind: SensitiveKind
  readonly action: ReleaseAction
}

export interface CustomPattern {
  readonly name: string
  readonly pattern: string
}

export interface OutboundSegment {
  readonly id: string
  readonly text: string
  readonly sensitivity: Sensitivity
  readonly source?: string
}

export interface PreparedSegment {
  readonly id: string
  readonly action: 'kept' | 'redacted' | 'withheld'
  readonly findings: number
}

export interface PreparedContext {
  readonly text: string
  readonly segments: readonly PreparedSegment[]
  readonly findings: number
  /** Ids of segments that were withheld (private) or omitted to fit `maxChars`. */
  readonly withheld: readonly string[]
  /** Placeholder -> original value. Local use only: never send this map anywhere. */
  readonly pseudonyms: ReadonlyMap<string, string>
  readonly needsApproval: boolean
  readonly summary: string
}

export interface PrepareInput {
  readonly segments: readonly OutboundSegment[]
  readonly mode: RedactMode
  readonly rules?: readonly ReleaseRule[]
  readonly customPatterns?: readonly CustomPattern[]
  readonly maxChars?: number
}

export const MAX_CUSTOM_PATTERNS = 64
export const MAX_CUSTOM_PATTERN_CHARS = 256
const MAX_CUSTOM_NAME_CHARS = 64
const SEGMENT_SEPARATOR = '\n\n'

const KINDS: readonly SensitiveKind[] = [
  'email',
  'phone',
  'ssn',
  'credit_card',
  'iban',
  'ip_address',
  'api_key',
  'person_name_hint',
  'custom',
]
const ACTIONS: readonly ReleaseAction[] = ['drop', 'mask', 'pseudonymize', 'allow']

/** Mask every kind; pseudonymize email and name hints so meaning survives. */
export const DEFAULT_RELEASE_RULES: readonly ReleaseRule[] = Object.freeze(
  KINDS.map(kind =>
    Object.freeze({
      kind,
      action: (kind === 'email' || kind === 'person_name_hint' ? 'pseudonymize' : 'mask') satisfies ReleaseAction,
    }),
  ),
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// No setting turns redaction on yet: the engine below has no production
// caller until the M4 data boundary wires it (see the M1 cut list).

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/** When spans overlap they merge; the merged span takes the kind with the lowest number here. */
const KIND_PRIORITY: Readonly<Record<SensitiveKind, number>> = {
  api_key: 0,
  credit_card: 1,
  ssn: 2,
  iban: 3,
  email: 4,
  phone: 5,
  ip_address: 6,
  custom: 7,
  person_name_hint: 8,
}

const EMAIL = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g
const PHONE_US = /(?<![\w-])(?:\+1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])/g
const PHONE_INTL = /(?<![\w])\+\d{1,3}(?:[\s.-]?\d{2,5}){2,4}(?!\w)/g
const PHONE_BARE = /(?<![\w.-])\d{10}(?![\w.-])/g
const SSN = /(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])/g
const CARD = /(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)/g
const IBAN = /\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b/g
const IPV4 = /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g
const IPV6 = /\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b/g

/** Credential patterns reused from the existing redaction rules; the secret is the last capture group. */
const CREDENTIAL_SCANNERS: readonly RegExp[] = [
  ...CREDENTIAL_REDACTION_RULES,
  ...DEFAULT_REDACTION_RULES.filter(rule => !/^(?:phone|email)/.test(rule.name)),
].map(rule => new RegExp(rule.pattern.source, rule.pattern.flags.includes('d') ? rule.pattern.flags : `${rule.pattern.flags}d`))

const PERSON_SCANNERS: readonly RegExp[] = [
  /\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?/gd,
  /(?:\b[Mm]y name is|\b[Nn]ame is|\b[Nn]ame:|\bPatient:|\bCustomer:|\bClient:|\bDear)\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})/gd,
]

function luhnValid(digits: string): boolean {
  let sum = 0
  let double = false
  for (let i = digits.length - 1; i >= 0; i--) {
    let digit = digits.charCodeAt(i) - 48
    if (double) {
      digit *= 2
      if (digit > 9) digit -= 9
    }
    sum += digit
    double = !double
  }
  return sum % 10 === 0
}

function ibanValid(raw: string): boolean {
  const compact = raw.replace(/ /g, '')
  if (compact.length < 15 || compact.length > 34) return false
  const rearranged = compact.slice(4) + compact.slice(0, 4)
  let remainder = 0
  for (const ch of rearranged) {
    const value = ch >= 'A' && ch <= 'Z' ? ch.charCodeAt(0) - 55 : ch.charCodeAt(0) - 48
    remainder = value >= 10 ? (remainder * 100 + value) % 97 : (remainder * 10 + value) % 97
  }
  return remainder === 1
}

interface CompiledCustomPattern {
  readonly name: string
  readonly regex: RegExp
}

/** Validate and compile user patterns: at most 64, each at most 256 chars, non-empty match required. */
export function validateCustomPatterns(patterns: readonly CustomPattern[] | undefined): readonly CompiledCustomPattern[] {
  if (patterns === undefined) return []
  if (!Array.isArray(patterns)) throw new ValidationError('customPatterns', 'must be an array', patterns)
  if (patterns.length > MAX_CUSTOM_PATTERNS) {
    throw new ValidationError('customPatterns', `at most ${MAX_CUSTOM_PATTERNS} patterns are allowed`, patterns.length)
  }
  return patterns.map((entry, index): CompiledCustomPattern => {
    const field = `customPatterns[${index}]`
    if (typeof entry !== 'object' || entry === null) throw new ValidationError(field, 'must be an object', entry)
    const { name, pattern } = entry
    if (typeof name !== 'string' || name.trim() === '' || name.length > MAX_CUSTOM_NAME_CHARS) {
      throw new ValidationError(`${field}.name`, `must be a non-empty string of at most ${MAX_CUSTOM_NAME_CHARS} chars`, name)
    }
    if (typeof pattern !== 'string' || pattern === '') {
      throw new ValidationError(`${field}.pattern`, 'must be a non-empty string', pattern)
    }
    if (pattern.length > MAX_CUSTOM_PATTERN_CHARS) {
      throw new ValidationError(`${field}.pattern`, `must be at most ${MAX_CUSTOM_PATTERN_CHARS} chars`, pattern.length)
    }
    let regex: RegExp
    try {
      regex = new RegExp(pattern, 'g')
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error)
      throw new ValidationError(`${field}.pattern`, `invalid regular expression: ${reason}`, pattern)
    }
    if (new RegExp(pattern).test('')) {
      throw new ValidationError(`${field}.pattern`, 'must not match the empty string', pattern)
    }
    return { name, regex }
  })
}

type RawSpan = SensitiveSpan

function collect(
  out: RawSpan[],
  text: string,
  regex: RegExp,
  kind: SensitiveKind,
  confidence: RawSpan['confidence'],
  accept?: (matched: string) => boolean,
): void {
  for (const match of text.matchAll(regex)) {
    const matched = match[0]
    if (matched.length === 0 || (accept !== undefined && !accept(matched))) continue
    const start = match.index
    out.push({ start, end: start + matched.length, kind, confidence })
  }
}

/** Collect matches whose span is the last capture group (or the whole match when there is none). */
function collectGroup(out: RawSpan[], text: string, regex: RegExp, kind: SensitiveKind, confidence: RawSpan['confidence']): void {
  for (const match of text.matchAll(regex)) {
    const indices = match.indices
    if (indices === undefined) continue
    const range = indices.length > 1 ? indices[indices.length - 1] : indices[0]
    if (range === undefined || range[1] <= range[0]) continue
    out.push({ start: range[0], end: range[1], kind, confidence })
  }
}

function mergeSpans(spans: readonly RawSpan[]): SensitiveSpan[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start || b.end - a.end)
  const merged: SensitiveSpan[] = []
  for (const span of sorted) {
    const last = merged[merged.length - 1]
    if (last !== undefined && span.start < last.end) {
      merged[merged.length - 1] = {
        start: last.start,
        end: Math.max(last.end, span.end),
        kind: KIND_PRIORITY[span.kind] < KIND_PRIORITY[last.kind] ? span.kind : last.kind,
        confidence: last.confidence === 'high' || span.confidence === 'high' ? 'high' : 'medium',
      }
    } else {
      merged.push({ ...span })
    }
  }
  return merged
}

function scan(text: string, custom: readonly CompiledCustomPattern[]): SensitiveSpan[] {
  const spans: RawSpan[] = []
  collect(spans, text, EMAIL, 'email', 'high')
  collect(spans, text, PHONE_US, 'phone', 'high')
  collect(spans, text, PHONE_INTL, 'phone', 'high', matched => {
    const digits = matched.replace(/\D/g, '').length
    return digits >= 8 && digits <= 15
  })
  collect(spans, text, PHONE_BARE, 'phone', 'medium')
  collect(spans, text, SSN, 'ssn', 'high')
  collect(spans, text, CARD, 'credit_card', 'high', matched => {
    const digits = matched.replace(/\D/g, '')
    return digits.length >= 13 && digits.length <= 19 && luhnValid(digits)
  })
  collect(spans, text, IBAN, 'iban', 'high', ibanValid)
  collect(spans, text, IPV4, 'ip_address', 'high')
  collect(spans, text, IPV6, 'ip_address', 'high')
  for (const scanner of CREDENTIAL_SCANNERS) collectGroup(spans, text, scanner, 'api_key', 'high')
  for (const scanner of PERSON_SCANNERS) collectGroup(spans, text, scanner, 'person_name_hint', 'medium')
  for (const entry of custom) collect(spans, text, entry.regex, 'custom', 'high')
  return mergeSpans(spans)
}

/**
 * Find sensitive spans in `text`, sorted by start with overlaps merged.
 * Credit cards must pass the Luhn check and IBANs the mod-97 check.
 */
export function detectSensitive(text: string, customPatterns?: readonly CustomPattern[]): SensitiveSpan[] {
  return scan(text, validateCustomPatterns(customPatterns))
}

// ---------------------------------------------------------------------------
// Release rules
// ---------------------------------------------------------------------------

function resolveRules(rules: readonly ReleaseRule[] | undefined): ReadonlyMap<SensitiveKind, ReleaseAction> {
  const byKind = new Map<SensitiveKind, ReleaseAction>()
  for (const rule of DEFAULT_RELEASE_RULES) byKind.set(rule.kind, rule.action)
  if (rules === undefined) return byKind
  if (!Array.isArray(rules)) throw new ValidationError('rules', 'must be an array', rules)
  rules.forEach((rule, index) => {
    const field = `rules[${index}]`
    if (typeof rule !== 'object' || rule === null) throw new ValidationError(field, 'must be an object', rule)
    if (!KINDS.includes(rule.kind)) throw new ValidationError(`${field}.kind`, `unknown kind ${JSON.stringify(rule.kind)}`, rule.kind)
    if (!ACTIONS.includes(rule.action)) {
      throw new ValidationError(`${field}.action`, `must be one of ${ACTIONS.join(', ')}`, rule.action)
    }
    // A credential is never released verbatim, whatever the rule set says.
    if (rule.kind === 'api_key' && rule.action === 'allow') {
      throw new ValidationError(`${field}.action`, 'credentials (api_key) cannot be allowed through', rule.action)
    }
    byKind.set(rule.kind, rule.action)
  })
  return byKind
}

const PLACEHOLDER_LABEL: Readonly<Record<SensitiveKind, string>> = {
  email: 'EMAIL',
  phone: 'PHONE',
  ssn: 'SSN',
  credit_card: 'CARD',
  iban: 'IBAN',
  ip_address: 'IP',
  api_key: 'SECRET',
  person_name_hint: 'PERSON',
  custom: 'CUSTOM',
}

/** Allocates stable placeholders within one call and remembers what each stands for. */
class PseudonymTable {
  readonly reverse = new Map<string, string>()
  private readonly forward = new Map<string, string>()
  private readonly counters = new Map<string, number>()

  constructor(private readonly sourceText: string) {}

  placeholder(kind: SensitiveKind, original: string): string {
    const key = `${kind}\u0000${original}`
    const existing = this.forward.get(key)
    if (existing !== undefined) return existing
    const label = PLACEHOLDER_LABEL[kind]
    let next = this.counters.get(label) ?? 0
    let candidate: string
    do {
      next += 1
      candidate = `<${label}_${next}>`
      // Never reuse a token the source text already contains: restore would corrupt it.
    } while (this.sourceText.includes(candidate))
    this.counters.set(label, next)
    this.forward.set(key, candidate)
    this.reverse.set(candidate, original)
    return candidate
  }
}

interface ProcessedSegment {
  readonly id: string
  text: string
  action: PreparedSegment['action']
  findings: number
  readonly used: readonly string[]
  readonly shareable: boolean
}

function applyRules(
  text: string,
  spans: readonly SensitiveSpan[],
  rules: ReadonlyMap<SensitiveKind, ReleaseAction>,
  table: PseudonymTable,
  used: string[],
): { text: string; altered: boolean } {
  let out = ''
  let cursor = 0
  let altered = false
  for (const span of spans) {
    const action = rules.get(span.kind) ?? 'mask'
    if (action === 'allow') continue
    out += text.slice(cursor, span.start)
    cursor = span.end
    altered = true
    if (action === 'mask') {
      out += `[redacted:${span.kind}]`
    } else if (action === 'pseudonymize') {
      const token = table.placeholder(span.kind, text.slice(span.start, span.end))
      used.push(token)
      out += token
    }
  }
  return { text: out + text.slice(cursor), altered }
}

function renderLength(items: readonly ProcessedSegment[]): number {
  let total = 0
  let count = 0
  for (const item of items) {
    if (item.action === 'withheld' && item.shareable) continue
    total += item.text.length
    count += 1
  }
  return total + Math.max(0, count - 1) * SEGMENT_SEPARATOR.length
}

function render(items: readonly ProcessedSegment[]): string {
  return items
    .filter(item => !(item.action === 'withheld' && item.shareable))
    .map(item => item.text)
    .join(SEGMENT_SEPARATOR)
}

/**
 * Build the request text that may leave the machine.
 *
 * `off` returns the joined segments untouched (today's behavior). Otherwise private
 * segments are withheld outright, shareable segments get the release rules, and
 * `maxChars` drops whole shareable segments from the end until the text fits.
 */
export function prepareOutboundContext(input: PrepareInput): PreparedContext {
  const { segments, mode } = input
  if (mode !== 'off' && mode !== 'suggest' && mode !== 'auto') {
    throw new ValidationError('mode', 'must be one of off, suggest, auto', mode)
  }
  if (mode === 'off') {
    return {
      text: segments.map(segment => segment.text).join(SEGMENT_SEPARATOR),
      segments: segments.map(segment => ({ id: segment.id, action: 'kept', findings: 0 })),
      findings: 0,
      withheld: [],
      pseudonyms: new Map(),
      needsApproval: false,
      summary: 'Context preparation is off; the request is sent unchanged.',
    }
  }

  const { maxChars } = input
  if (maxChars !== undefined && (!Number.isInteger(maxChars) || maxChars <= 0)) {
    throw new ValidationError('maxChars', 'must be a positive integer', maxChars)
  }
  const rules = resolveRules(input.rules)
  const custom = validateCustomPatterns(input.customPatterns)
  const table = new PseudonymTable(segments.filter(s => s.sensitivity === 'shareable').map(s => s.text).join('\n'))
  const kindCounts = new Map<SensitiveKind, number>()

  const items: ProcessedSegment[] = segments.map(segment => {
    // Anything not explicitly shareable is treated as private (fail closed).
    if (segment.sensitivity !== 'shareable') {
      return {
        id: segment.id,
        text: `[withheld: private segment ${segment.id}]`,
        action: 'withheld',
        findings: 0,
        used: [],
        shareable: false,
      }
    }
    const spans = scan(segment.text, custom)
    for (const span of spans) kindCounts.set(span.kind, (kindCounts.get(span.kind) ?? 0) + 1)
    const used: string[] = []
    const result = applyRules(segment.text, spans, rules, table, used)
    return {
      id: segment.id,
      text: result.text,
      action: result.altered ? 'redacted' : 'kept',
      findings: spans.length,
      used,
      shareable: true,
    }
  })

  let omitted = 0
  if (maxChars !== undefined) {
    while (renderLength(items) > maxChars) {
      let index = -1
      for (let i = items.length - 1; i >= 0; i--) {
        const item = items[i]
        if (item !== undefined && item.shareable && item.action !== 'withheld' && item.text.length > 0) {
          index = i
          break
        }
      }
      if (index < 0) break
      const victim = items[index]
      if (victim === undefined) break
      victim.action = 'withheld'
      omitted += 1
    }
  }

  const pseudonyms = new Map<string, string>()
  for (const item of items) {
    if (item.action === 'withheld') continue
    for (const token of item.used) {
      const original = table.reverse.get(token)
      if (original !== undefined) pseudonyms.set(token, original)
    }
  }

  const withheld = items.filter(item => item.action === 'withheld').map(item => item.id)
  const privateCount = items.filter(item => !item.shareable).length
  const findings = items.reduce((sum, item) => sum + item.findings, 0)
  const anyRedacted = items.some(item => item.action === 'redacted')
  const needsApproval = mode === 'suggest' && (anyRedacted || withheld.length > 0)

  const parts: string[] = []
  if (findings > 0) {
    const byKind = [...kindCounts.entries()].map(([kind, count]) => `${count} ${kind}`).join(', ')
    parts.push(`${findings} sensitive finding(s) (${byKind})`)
  } else {
    parts.push('no sensitive content found')
  }
  if (privateCount > 0) parts.push(`${privateCount} private segment(s) withheld`)
  if (omitted > 0) parts.push(`${omitted} segment(s) omitted to fit maxChars=${maxChars}`)
  if (needsApproval) parts.push('approval required before sending')
  const summary = `Context preparation (${mode}): ${parts.join('; ')}.`

  return {
    text: render(items),
    segments: items.map(item => ({ id: item.id, action: item.action, findings: item.findings })),
    findings,
    withheld,
    pseudonyms,
    needsApproval,
    summary,
  }
}

/** Map placeholders in a model answer back to the original values. Local use only. */
export function restorePseudonyms(answer: string, pseudonyms: ReadonlyMap<string, string>): string {
  if (pseudonyms.size === 0) return answer
  return answer.replace(/<[A-Z]+_\d+>/g, token => pseudonyms.get(token) ?? token)
}
