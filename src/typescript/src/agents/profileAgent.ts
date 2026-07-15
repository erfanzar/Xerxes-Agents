// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfidentValue, type UserProfile, type UserProfileStore } from '../memory/userProfile.js'

export const TECH_DOMAINS = Object.freeze([
  ['python', ['python', 'pytest', 'django', 'flask', 'fastapi', 'uv', 'poetry', 'pip']],
  ['rust', ['rust', 'cargo', 'tokio', 'serde', 'axum']],
  ['javascript', ['javascript', 'typescript', 'node', 'react', 'next', 'vite', 'npm', 'yarn', 'pnpm']],
  ['go', ['golang', ' go ']],
  ['devops', ['docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'ci/cd', 'github actions', 'gitlab ci']],
  ['ml', ['pytorch', 'tensorflow', 'huggingface', 'embedding', 'fine-tune', 'lora']],
  ['databases', ['postgres', 'postgresql', 'mysql', 'sqlite', 'redis', 'mongodb']],
  ['security', ['owasp', 'csrf', 'xss', 'sql injection', 'auth', 'oauth', 'jwt', 'tls']],
] as const)

export interface UserProfileStorePort {
  getOrCreate(userId: string): UserProfile
  save(profile: UserProfile): void
}

export type ProfileLlmSummarizer = (
  conversation: string,
  profile: UserProfile,
) => Promise<string> | string

export type NamedEntityExtractor = (text: string) => Readonly<Record<string, readonly string[]>>

export interface ProfileAgentOptions {
  readonly llmSummarizer?: ProfileLlmSummarizer
  readonly nerExtractor?: NamedEntityExtractor
  readonly now?: () => Date
  readonly onSummarizerError?: (error: unknown) => void
  readonly store: UserProfileStorePort | UserProfileStore
}

export interface ProfileUpdate {
  readonly confidenceChanges: Readonly<Record<string, number>>
  readonly domainsAdded: readonly string[]
  readonly prefsAdded: readonly string[]
  readonly userId: string
}

/** Heuristic profile updater with optional caller-owned NER and LLM ports. */
export class ProfileAgent {
  readonly store: UserProfileStorePort
  private readonly llmSummarizer: ProfileLlmSummarizer | undefined
  private readonly nerExtractor: NamedEntityExtractor | undefined
  private readonly now: () => Date
  private readonly onSummarizerError: (error: unknown) => void

  constructor(options: ProfileAgentOptions) {
    this.store = options.store
    this.llmSummarizer = options.llmSummarizer
    this.nerExtractor = options.nerExtractor
    this.now = options.now ?? (() => new Date())
    this.onSummarizerError = options.onSummarizerError ?? (() => undefined)
  }

  /** Fold one interaction into a durable profile through the supplied store. */
  async update(
    userId: string,
    options: {
      readonly agentResponse?: string
      readonly signals?: Iterable<string>
      readonly userPrompt?: string
    } = {},
  ): Promise<ProfileUpdate> {
    const userPrompt = options.userPrompt ?? ''
    const agentResponse = options.agentResponse ?? ''
    const profile = this.store.getOrCreate(userId)
    profile.lastSeen = this.now()
    const domainsAdded: string[] = []
    const prefsAdded: string[] = []
    const confidenceChanges: Record<string, number> = {}

    if (userPrompt) {
      for (const domain of this.inferDomains(userPrompt)) {
        if (!profile.domains.includes(domain)) {
          profile.domains.push(domain)
          domainsAdded.push(domain)
        }
      }
      const tone = this.inferTone(userPrompt)
      if (tone) {
        if (profile.tone === undefined) {
          profile.tone = new ConfidentValue(tone, { confidence: 0.2 })
        } else if (profile.tone.value === tone) {
          profile.tone.reinforce(0.1)
        } else {
          profile.tone.demote(0.1)
          if (profile.tone.confidence < 0.05) profile.tone = new ConfidentValue(tone, { confidence: 0.2 })
        }
        confidenceChanges.tone = profile.tone.confidence
      }
      for (const phrase of this.extractPreferencePhrases(userPrompt)) {
        const key = phrase.toLowerCase().slice(0, 60)
        const preference = profile.explicitPreferences.get(key)
        if (preference === undefined) {
          profile.explicitPreferences.set(key, new ConfidentValue(phrase, { confidence: 0.6 }))
          prefsAdded.push(phrase)
        } else {
          preference.reinforce(0.2)
        }
      }
    }

    for (const signal of options.signals ?? []) {
      profile.recordFeedback(signal)
      if ((signal === 'correction' || signal === 'revert' || signal === 'retry') && profile.tone !== undefined) {
        profile.tone.demote(0.1)
      }
    }

    if (this.llmSummarizer !== undefined && (userPrompt || agentResponse)) {
      try {
        const note = await this.llmSummarizer(`USER: ${userPrompt}\nAGENT: ${agentResponse}`, profile)
        this.appendNote(profile, note)
      } catch (error) {
        this.onSummarizerError(error)
      }
    }

    this.store.save(profile)
    return Object.freeze({
      userId,
      domainsAdded: Object.freeze(domainsAdded),
      prefsAdded: Object.freeze(prefsAdded),
      confidenceChanges: Object.freeze(confidenceChanges),
    })
  }

  /** Expose optional NER extraction without inventing a default network/tool integration. */
  extractEntities(text: string): Readonly<Record<string, readonly string[]>> {
    return this.nerExtractor?.(text) ?? {}
  }

  inferDomains(text: string): string[] {
    const lower = text.toLowerCase()
    return TECH_DOMAINS
      .filter(([, keywords]) => keywords.some(keyword => lower.includes(keyword)))
      .map(([domain]) => domain)
  }

  inferTone(text: string): '' | 'balanced' | 'casual' | 'terse' | 'verbose' {
    const words = text.trim() ? text.trim().split(/\s+/u) : []
    if (!words.length) return ''
    const exclamations = [...text].filter(character => character === '!').length
    if (words.length <= 6) return 'terse'
    if (exclamations / words.length > 0.05) return 'casual'
    if (words.length > 80) return 'verbose'
    return 'balanced'
  }

  extractPreferencePhrases(text: string): string[] {
    const patterns = [
      /\bi (?:prefer|want|like|wish|need)(?: to)?\s+.{3,80}/giu,
      /\b(?:please|always|make sure to)\s+.{3,80}/giu,
      /\b(?:don'?t|do not|never)\s+.{3,80}/giu,
    ]
    const phrases: string[] = []
    for (const pattern of patterns) {
      for (const match of text.matchAll(pattern)) {
        const phrase = match[0].trim().replace(/[.!?,]+$/u, '')
        if (phrase.length >= 3 && phrase.length <= 200) phrases.push(phrase)
      }
    }
    return phrases
  }

  private appendNote(profile: UserProfile, note: string): void {
    const normalized = note.trim()
    if (!normalized || profile.notes.includes(normalized)) return
    profile.notes.push(normalized.slice(0, 500))
    if (profile.notes.length > 50) profile.notes.splice(0, profile.notes.length - 50)
  }
}
