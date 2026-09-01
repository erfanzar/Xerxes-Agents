// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const SKILL_SUGGESTIONS_METADATA_KEY = 'xerxes_skill_suggestions_v1'
const MAX_SKILL_SUGGESTIONS = 32

export interface SkillSuggestionRecord {
  readonly description: string
  readonly skillName: string
  readonly sourcePath: string
  readonly toolCount: number
  readonly uniqueTools: readonly string[]
  readonly version: string
}

/** Keep the newest version of each suggested skill in bounded session metadata. */
export function appendSkillSuggestion(
  metadata: Record<string, unknown>,
  suggestion: SkillSuggestionRecord,
): void {
  const rows = skillSuggestionValues(metadata).filter(row => row.skillName !== suggestion.skillName)
  rows.push({
    description: bounded(suggestion.description, 1_000),
    skillName: bounded(suggestion.skillName, 128),
    sourcePath: bounded(suggestion.sourcePath, 1_000),
    toolCount: nonNegativeInteger(suggestion.toolCount),
    uniqueTools: suggestion.uniqueTools.filter(value => typeof value === 'string' && value).slice(0, 64),
    version: bounded(suggestion.version, 64),
  })
  metadata[SKILL_SUGGESTIONS_METADATA_KEY] = rows.slice(-MAX_SKILL_SUGGESTIONS)
}

/** Read validated suggestion telemetry from untrusted persisted metadata. */
export function skillSuggestionValues(
  metadata: Readonly<Record<string, unknown>>,
): SkillSuggestionRecord[] {
  const value = metadata[SKILL_SUGGESTIONS_METADATA_KEY]
  if (!Array.isArray(value)) return []
  return value.slice(-MAX_SKILL_SUGGESTIONS).flatMap(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const row = item as Readonly<Record<string, unknown>>
    const skillName = text(row.skillName ?? row.skill_name)
    if (!skillName) return []
    const rawTools = row.uniqueTools ?? row.unique_tools
    return [{
      skillName,
      description: text(row.description),
      version: text(row.version),
      sourcePath: text(row.sourcePath ?? row.source_path),
      toolCount: nonNegativeInteger(row.toolCount ?? row.tool_count),
      uniqueTools: Array.isArray(rawTools)
        ? rawTools.filter((tool): tool is string => typeof tool === 'string' && tool.length > 0).slice(0, 64)
        : [],
    } satisfies SkillSuggestionRecord]
  })
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function bounded(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}

function nonNegativeInteger(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0
}
