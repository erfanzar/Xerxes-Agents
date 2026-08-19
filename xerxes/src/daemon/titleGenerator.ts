// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Background session-title generation.
 *
 * A session's title starts life as the first user message (`titleFromMessages`
 * in the runtime), which reads as a truncated fragment rather than a name.
 * Once the first exchange completes, this module asks a cheap provider for a
 * short title and replaces the fallback — in the background, never blocking or
 * failing the turn it follows. Any failure leaves the fallback in place; a
 * title the user set explicitly is never touched.
 */

import { closeLlmClient, completeLlm, createLlmClient, type LlmClient } from '../llms/client.js'
import type { ProviderProfile } from '../bridge/profiles.js'

/** Titles longer than this are noise in pickers and tab strips. */
export const TITLE_MAX_CHARS = 60

/** Raw conversation text fed to the title prompt is capped so a paste of a file cannot bloat the call. */
const TITLE_SOURCE_MAX_CHARS = 2_000

/** One title attempt per session; a failed attempt is not retried on every later turn. */
const attempted = new Set<string>()

/**
 * Cheap, fast models per provider, in preference order. Title generation is a
 * background nicety; it must never burn the frontier model the session runs
 * on, so the cheapest model of a provider the user already has credentials
 * for is chosen first.
 */
const CHEAP_MODEL_BY_PROVIDER: Record<string, readonly string[]> = {
  anthropic: ['claude-haiku-4-5-20251001', 'claude-3-5-haiku-20241022'],
  openai: ['gpt-4o-mini', 'gpt-4.1-mini'],
  gemini: ['gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-1.5-flash'],
}

/** Factory used to build the title-generation client; injectable for tests. */
export type TitleClientFactory = (
  model: string,
  profile: ProviderProfile | undefined,
) => LlmClient

export interface TitleGenerationOptions {
  /** First user message of the session. */
  readonly userText: string
  /** First assistant reply of the session. */
  readonly assistantText: string
  /** Model the session itself runs on; last-resort title model. */
  readonly sessionModel: string
  /** Active provider profile, carrying credentials and provider identity. */
  readonly profile: ProviderProfile | undefined
  /** Abort tied to the daemon/session lifetime, not the finished turn. */
  readonly signal?: AbortSignal
  /** Test seam: bypass client construction. */
  readonly clientFactory?: TitleClientFactory
}

/**
 * Pick the model that generates the title.
 *
 * The cheap tier of the *session's own* provider is preferred, because the
 * profile's credentials are known to work for it. A session on a provider with
 * no known cheap tier (or a plugin/provider we do not track) falls back to the
 * session's own model — a working expensive title beats a failing cheap one.
 */
export function titleModelFor(sessionModel: string, profile: ProviderProfile | undefined): string {
  const provider = profile?.provider?.trim()
  const cheap = provider ? CHEAP_MODEL_BY_PROVIDER[provider] : undefined
  return cheap?.[0] ?? sessionModel
}

/** Render the title prompt from the opening exchange. */
export function titlePrompt(userText: string, assistantText: string): string {
  const clip = (text: string) => text.slice(0, TITLE_SOURCE_MAX_CHARS)
  return [
    'Write a very short title (5 words or fewer, plain text, no quotes, no trailing punctuation)',
    'for a chat that begins with this exchange. Reply with the title and nothing else.',
    '',
    `User: ${clip(userText)}`,
    `Assistant: ${clip(assistantText)}`,
  ].join('\n')
}

/** Normalize a model's reply into a safe one-line title, or undefined when unusable. */
export function sanitizeTitle(raw: string): string | undefined {
  const firstLine = raw.split('\n').map(line => line.trim()).find(Boolean) ?? ''
  const unquoted = firstLine.replace(/^["'`]+|["'`]+$/g, '').replace(/[.!?]+$/, '').trim()
  if (!unquoted) return undefined
  return unquoted.length > TITLE_MAX_CHARS ? `${unquoted.slice(0, TITLE_MAX_CHARS - 1)}…` : unquoted
}

/**
 * Generate a title for a session's opening exchange.
 *
 * Returns undefined instead of throwing on any provider or shape failure — the
 * caller falls back to the first-message title and the turn that triggered
 * this is long finished, so there is nowhere useful to surface an error.
 */
export async function generateSessionTitle(options: TitleGenerationOptions): Promise<string | undefined> {
  const model = titleModelFor(options.sessionModel, options.profile)
  if (!model.trim()) return undefined
  const factory = options.clientFactory ?? ((m: string, profile: ProviderProfile | undefined) =>
    createLlmClient(m, {
      ...(profile?.api_key ? { api_key: profile.api_key } : {}),
      ...(profile?.base_url ? { base_url: profile.base_url } : {}),
      ...(profile?.provider ? { provider: profile.provider } : {}),
    }))
  let client: LlmClient
  try {
    // Construction itself can throw (unconfigured model, unknown provider) —
    // same "never surface in a finished turn" rule as a failed request.
    client = factory(model, options.profile)
  } catch {
    return undefined
  }
  try {
    const result = await completeLlm(client, {
      model,
      messages: [{ role: 'user', content: titlePrompt(options.userText, options.assistantText) }],
      maxTokens: 40,
      temperature: 0.2,
    }, options.signal)
    return sanitizeTitle(result.content)
  } catch {
    return undefined
  } finally {
    await closeLlmClient(client).catch(() => undefined)
  }
}

/**
 * Attempt a title exactly once per session id.
 *
 * The "once" rule is what makes background generation cheap to trigger on
 * every turn-end: after the first attempt — success or failure — later turns
 * short-circuit without any provider call.
 */
export function attemptSessionTitleOnce(
  sessionId: string,
  run: () => Promise<string | undefined>,
): Promise<string | undefined> | undefined {
  if (attempted.has(sessionId)) return undefined
  attempted.add(sessionId)
  return run()
}

/** Test hook: forget prior attempts. */
export function resetTitleAttempts(): void {
  attempted.clear()
}
