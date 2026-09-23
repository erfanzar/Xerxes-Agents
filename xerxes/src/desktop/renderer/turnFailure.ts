// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Classify a failed turn well enough to offer the right way out.
 *
 * Provider failures reach the shell as the provider's own words, which for
 * the most common recoverable case — a usage or rate limit — is a line of
 * JSON with a unix timestamp in it. The card used to render that verbatim
 * and offer only "Retry", which for a rate limit can do nothing but fail
 * again.
 *
 * Only the summary and the choice of action change here. The raw text is
 * always kept and shown, because it is the thing worth pasting into a bug
 * report.
 */

export type FailureKind = 'compaction' | 'rate-limit' | 'credentials' | 'context-length' | 'cancelled' | 'unknown'

export interface FailureView {
  readonly kind: FailureKind
  /** One plain sentence, or '' when the raw error is already the best text. */
  readonly summary: string
  /** Retry cannot succeed until something changes. */
  readonly retryIsFutile: boolean
  /** Offer the provider/model settings. */
  readonly offerProviderSettings: boolean
}

/** `resets_at`/`reset_at` unix seconds (or ms) → a local wall-clock time. */
export function resetTimeOf(error: string, now: number): string {
  const match = /"?resets?_at"?\s*[:=]\s*"?(\d{9,13})/i.exec(error)
  if (!match) return ''
  const raw = Number(match[1])
  if (!Number.isFinite(raw)) return ''
  const stamp = raw > 1e11 ? raw : raw * 1000
  // A reset far outside a plausible window is a field we misread.
  if (stamp < now - 86_400_000 || stamp > now + 45 * 86_400_000) return ''
  try {
    return new Date(stamp).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
  } catch {
    return ''
  }
}

export function failureView(error: string, now: number): FailureView {
  if (error.startsWith('Automatic context compaction failed:')) {
    return { kind: 'compaction', summary: 'Couldn’t compact the conversation. Original history preserved; this turn has stopped.', retryIsFutile: false, offerProviderSettings: false }
  }
  if (/\bcancell?ed\b|aborted by (the )?(user|operator)/i.test(error)) {
    return { kind: 'cancelled', summary: 'This turn was stopped.', retryIsFutile: false, offerProviderSettings: false }
  }
  if (/usage[ _-]?limit|rate[ _-]?limit|quota|too many requests|\b429\b|insufficient_quota/i.test(error)) {
    const at = resetTimeOf(error, now)
    return {
      kind: 'rate-limit',
      summary: `Your provider is out of capacity for now${at ? `. It resets around ${at}` : ''}. Switching model or provider starts working again immediately.`,
      retryIsFutile: true,
      offerProviderSettings: true,
    }
  }
  if (/context[ _-]?length|maximum context|too many tokens|context window/i.test(error)) {
    return { kind: 'context-length', summary: 'This conversation is longer than the model can hold. Compact it, or switch to a model with a larger context.', retryIsFutile: true, offerProviderSettings: true }
  }
  if (/authenticat|credential|api[ _-]?key|certificate|self.signed|provider.*config|\b401\b|\b403\b|unauthorized/i.test(error)) {
    return { kind: 'credentials', summary: 'The provider rejected this app’s credentials.', retryIsFutile: true, offerProviderSettings: true }
  }
  return { kind: 'unknown', summary: '', retryIsFutile: false, offerProviderSettings: false }
}
