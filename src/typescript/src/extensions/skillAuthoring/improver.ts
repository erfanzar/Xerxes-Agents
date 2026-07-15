// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate } from './model.js'
import { SkillProposalDrafter, type SkillDraftRefinerPort, type SkillProposal } from './proposal.js'

/** Host-controlled read/write/backup boundary for an existing skill document. */
export interface SkillDocumentStorePort {
  backup?(input: {
    readonly location: string
    readonly markdown: string
    readonly maxBackups: number
    readonly version: string
  }): string | undefined | Promise<string | undefined>
  read(location: string): string | undefined | Promise<string | undefined>
  write(input: { readonly location: string; readonly markdown: string }): void | Promise<void>
}

export interface ImprovementResult {
  readonly backupLocation?: string
  readonly improved: boolean
  readonly newVersion: string
  readonly oldVersion: string
  readonly reason: string
  readonly skillLocation?: string
}

export interface SkillImproverOptions {
  readonly documents: SkillDocumentStorePort
  /** Optional model-backed refinement boundary owned by the caller. */
  readonly refiner?: SkillDraftRefinerPort
  /** Reuse a configured proposal drafter when a host owns its refinement policy. */
  readonly proposalDrafter?: SkillProposalDrafter
}

export interface ImproveSkillOptions {
  /** Maximum backups the document store should retain. Defaults to five. */
  readonly maxBackups?: number
}

/**
 * Rewrites an existing SKILL.md from a newer observation through injected storage.
 *
 * It owns the deterministic version bump and frontmatter extraction but never
 * reads, writes, or deletes host files directly.
 */
export class SkillImprover {
  private readonly documents: SkillDocumentStorePort
  private readonly proposalDrafter: SkillProposalDrafter

  constructor(options: SkillImproverOptions) {
    if (options.proposalDrafter && options.refiner) {
      throw new TypeError('provide either proposalDrafter or refiner, not both')
    }
    this.documents = options.documents
    this.proposalDrafter = options.proposalDrafter ?? new SkillProposalDrafter(
      options.refiner === undefined ? {} : { refiner: options.refiner },
    )
  }

  /** Apply a newer candidate to an existing document while preserving its name and bumping its patch version. */
  async improve(
    skillLocation: string,
    candidate: SkillCandidate,
    options: ImproveSkillOptions = {},
  ): Promise<ImprovementResult> {
    const location = nonEmptyLocation(skillLocation)
    const maxBackups = options.maxBackups ?? 5
    if (!Number.isInteger(maxBackups) || maxBackups < 0) {
      throw new RangeError('maxBackups must be a non-negative integer')
    }

    let oldMarkdown: string | undefined
    try {
      oldMarkdown = await this.documents.read(location)
    } catch {
      return failed('failed to read existing SKILL.md')
    }
    if (oldMarkdown === undefined) {
      return failed('missing skill at ' + location)
    }

    const oldVersion = extractSkillVersion(oldMarkdown) ?? '0.1.0'
    const newVersion = bumpPatchVersion(oldVersion)
    const name = extractSkillName(oldMarkdown) ?? fallbackSkillName(location)
    let proposal: SkillProposal
    try {
      const initial = this.proposalDrafter.create(candidate, { name, version: newVersion })
      proposal = await this.proposalDrafter.refine(initial)
    } catch {
      return failed('skill draft preparation failed', oldVersion, newVersion)
    }

    let backupLocation: string | undefined
    try {
      backupLocation = await this.documents.backup?.({
        location,
        markdown: oldMarkdown,
        maxBackups,
        version: oldVersion,
      })
    } catch {
      return failed('failed to create SKILL.md backup', oldVersion, newVersion)
    }
    try {
      await this.documents.write({ location, markdown: proposal.markdown })
    } catch {
      return failed('failed to write SKILL.md', oldVersion, newVersion)
    }
    return {
      improved: true,
      oldVersion,
      newVersion,
      reason: '',
      skillLocation: location,
      ...(backupLocation === undefined ? {} : { backupLocation }),
    }
  }
}

/** Increment the patch part of a semver-like version, falling back to 0.1.1 when malformed. */
export function bumpPatchVersion(version: string): string {
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(version.trim())
  if (!match) {
    return '0.1.1'
  }
  const major = match[1]
  const minor = match[2]
  const patch = match[3]
  if (major === undefined || minor === undefined || patch === undefined) {
    return '0.1.1'
  }
  return major + '.' + minor + '.' + (BigInt(patch) + 1n).toString()
}

/** Extract the first frontmatter-style name field, if available. */
export function extractSkillName(markdown: string): string | undefined {
  return extractFrontmatterValue(markdown, 'name')
}

/** Extract the first frontmatter-style version field, if available. */
export function extractSkillVersion(markdown: string): string | undefined {
  return extractFrontmatterValue(markdown, 'version')
}

function extractFrontmatterValue(markdown: string, key: string): string | undefined {
  const match = new RegExp('^' + key + ':\\s*(.+?)\\s*$', 'm').exec(markdown)
  const value = match?.[1]?.trim().replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, '$1$2').trim()
  return value || undefined
}

function failed(reason: string, oldVersion = '', newVersion = ''): ImprovementResult {
  return { improved: false, oldVersion, newVersion, reason }
}

function fallbackSkillName(location: string): string {
  const segments = location.replace(/\\/g, '/').split('/').filter(Boolean)
  return segments.at(-2) ?? segments.at(-1)?.replace(/\.[^.]+$/, '') ?? 'skill'
}

function nonEmptyLocation(location: string): string {
  const normalized = location.trim()
  if (!normalized) {
    throw new TypeError('skillLocation must not be empty')
  }
  return normalized
}
