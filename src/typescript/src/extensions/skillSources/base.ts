// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** JSON-like metadata retained with a source bundle without interpreting source-specific fields. */
export type SkillSourceMetadata = Readonly<Record<string, unknown>>

/** A single SKILL.md payload ready for installation by a caller-owned skill store. */
export interface SkillBundle {
  readonly bodyMarkdown: string
  readonly metadata: SkillSourceMetadata
  readonly name: string
  readonly sourceName: string
  readonly version: string
}

/** One source-owned result returned from a skill search. */
export interface SkillSearchHit {
  readonly description: string
  readonly name: string
  readonly sourceName: string
  readonly tags: readonly string[]
  readonly version: string
}

/** A native skill catalogue. Sources never install bundles or choose a network transport themselves. */
export interface SkillSource {
  readonly name: string
  fetch(identifier: string): Promise<SkillBundle>
  search(query: string, limit?: number): Promise<readonly SkillSearchHit[]>
}

/** An error caused by invalid source configuration or an invalid source request. */
export class SkillSourceError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'SkillSourceError'
  }
}

/** Raised when a source does not contain a requested skill. */
export class SkillSourceNotFoundError extends SkillSourceError {
  constructor(sourceName: string, identifier: string) {
    super(`skill not found in ${sourceName}: ${identifier}`)
    this.name = 'SkillSourceNotFoundError'
  }
}

/** Raised when a source needs a host-owned port that was not supplied. */
export class SkillSourceConfigurationError extends SkillSourceError {
  constructor(message: string) {
    super(message)
    this.name = 'SkillSourceConfigurationError'
  }
}

/** Normalise the optional search limit before a source opens files or sends a request. */
export function normalizeSkillSearchLimit(limit = 20): number {
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new SkillSourceError('skill search limit must be a non-negative safe integer')
  }
  return limit
}

/** Require a non-empty identifier before giving it to a host callback or URL builder. */
export function requireSkillIdentifier(identifier: string): string {
  if (typeof identifier !== 'string' || !identifier.trim()) {
    throw new SkillSourceError('skill identifier must be a non-empty string')
  }
  return identifier
}

/** Require a string query rather than silently coercing a caller-owned search value. */
export function requireSkillSearchQuery(query: string): string {
  if (typeof query !== 'string') {
    throw new SkillSourceError('skill search query must be a string')
  }
  return query
}
