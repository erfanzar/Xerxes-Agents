// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  NormalizedUntrustedText,
  ObfuscationIndicator,
  PromptInjectionAssessment,
  PromptInjectionSignal,
  SafetyRiskLevel,
} from './types.js'

const CONTROL_CHARACTERS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/gu
const FORMAT_CHARACTERS = /[\u00AD\u034F\u061C\u180E\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]/gu
const OPAQUE_TOKEN = /(?:^|\s)[A-Za-z0-9+/_=-]{48,}(?=\s|$)/u
const SPACE_LIKE_CHARACTERS = /[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/gu
const WHITESPACE = /\s+/gu

const SCRIPT_PATTERNS = [/\p{Script=Latin}/u, /\p{Script=Cyrillic}/u, /\p{Script=Greek}/u] as const

const INSTRUCTION_OVERRIDE =
  /\b(?:instruction|policy|rule)\s+(?:override|replacement)\b|\b(?:ignore|disregard|override)\b.{0,32}\b(?:instruction|policy|rule)s?\b/iu
const AUTHORITY_IMPERSONATION = /\b(?:system|developer|administrator)\s+(?:message|instruction|role)\b/iu
const PROTECTED_INFORMATION_REQUEST = /\b(?:hidden|internal|private)\s+(?:instruction|message|configuration|data)\b/iu
const PRIVILEGED_ACTION_REQUEST =
  /\b(?:credential|secret|token|account)\b.{0,32}\b(?:send|share|export|delete|upload)\b|\b(?:send|share|export|delete|upload)\b.{0,32}\b(?:credential|secret|token|account)\b/iu

/** Normalize untrusted text for defensive inspection without interpreting or decoding it. */
export function normalizeUntrustedText(input: string): NormalizedUntrustedText {
  if (typeof input !== 'string') throw new TypeError('untrusted text must be a string')

  const indicators: ObfuscationIndicator[] = []
  if (FORMAT_CHARACTERS.test(input)) indicators.push('format-characters')
  if (CONTROL_CHARACTERS.test(input)) indicators.push('control-characters')
  if (SPACE_LIKE_CHARACTERS.test(input)) indicators.push('unusual-whitespace')

  const text = input
    .normalize('NFKC')
    .replace(FORMAT_CHARACTERS, '')
    .replace(CONTROL_CHARACTERS, ' ')
    .replace(SPACE_LIKE_CHARACTERS, ' ')
    .replace(WHITESPACE, ' ')
    .trim()

  if (containsMixedScriptToken(text)) indicators.push('mixed-script-token')
  if (OPAQUE_TOKEN.test(text)) indicators.push('opaque-token')
  return { indicators, text }
}

/** Assess whether untrusted text contains generic prompt-injection signals without acting on it. */
export function inspectPromptInjection(input: string): PromptInjectionAssessment {
  const normalized = normalizeUntrustedText(input)
  const signals: PromptInjectionSignal[] = []

  if (INSTRUCTION_OVERRIDE.test(normalized.text)) signals.push('instruction-override')
  if (AUTHORITY_IMPERSONATION.test(normalized.text)) signals.push('authority-impersonation')
  if (PROTECTED_INFORMATION_REQUEST.test(normalized.text)) signals.push('protected-information-request')
  if (PRIVILEGED_ACTION_REQUEST.test(normalized.text)) signals.push('privileged-action-request')

  return {
    ...normalized,
    risk: riskFor(normalized.indicators, signals),
    signals,
  }
}

function containsMixedScriptToken(text: string): boolean {
  return text.split(' ').some(token => SCRIPT_PATTERNS.filter(pattern => pattern.test(token)).length > 1)
}

function riskFor(
  indicators: readonly ObfuscationIndicator[],
  signals: readonly PromptInjectionSignal[],
): SafetyRiskLevel {
  if (
    signals.length >= 3 ||
    (signals.includes('instruction-override') && signals.includes('protected-information-request'))
  ) {
    return 'high'
  }
  if (signals.length > 0) return 'medium'
  if (indicators.length > 0) return 'low'
  return 'none'
}
