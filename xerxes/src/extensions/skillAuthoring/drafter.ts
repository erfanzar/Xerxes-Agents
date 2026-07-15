// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate } from './model.js'
import {
  DEFAULT_SKILL_VERSION,
  SkillProposalDrafter,
  type SkillDraftRefinerPort,
  type SkillProposal,
  type SkillProposalDraftOptions,
} from './proposal.js'

/** Version assigned to a newly drafted skill unless its caller overrides it. */
export const DEFAULT_SKILL_DRAFT_VERSION = DEFAULT_SKILL_VERSION

/** Host-owned result of persisting a newly drafted skill document. */
export interface SkillDraftPersistence {
  readonly location: string
}

/**
 * Explicit artifact boundary for a drafted skill.
 *
 * The authoring subsystem never chooses a filesystem root or writes a file by
 * itself. A host can persist to disk, a database, or a review queue through
 * this port.
 */
export interface SkillDraftStorePort {
  persist(input: { readonly proposal: SkillProposal }): SkillDraftPersistence | Promise<SkillDraftPersistence>
}

export interface SkillDrafterOptions {
  /** Optional model-backed refinement boundary owned by the caller. */
  readonly refiner?: SkillDraftRefinerPort
  /** Reuse a configured proposal drafter when a host owns its refinement policy. */
  readonly proposalDrafter?: SkillProposalDrafter
  /** Optional destination for a completed draft. No persistence occurs without one. */
  readonly store?: SkillDraftStorePort
}

export interface SkillDraftOptions extends SkillProposalDraftOptions {
  /** Persist through the injected store when present. Defaults to true. */
  readonly persist?: boolean
}

/** Result of rendering a candidate and, when requested, persisting it through a host port. */
export interface SkillDraftResult {
  readonly markdown: string
  readonly persisted: boolean
  readonly persistence?: SkillDraftPersistence
  readonly proposal: SkillProposal
}

/**
 * Renders observed tool sequences as skill proposals and delegates all effects
 * to caller-supplied ports.
 */
export class SkillDrafter {
  private readonly proposalDrafter: SkillProposalDrafter
  private readonly store: SkillDraftStorePort | undefined

  constructor(options: SkillDrafterOptions = {}) {
    if (options.proposalDrafter && options.refiner) {
      throw new TypeError('provide either proposalDrafter or refiner, not both')
    }
    this.proposalDrafter = options.proposalDrafter ?? new SkillProposalDrafter(
      options.refiner === undefined ? {} : { refiner: options.refiner },
    )
    this.store = options.store
  }

  /** Render a candidate, safely fall back from a failed refiner, then optionally persist it. */
  async draft(candidate: SkillCandidate, options: SkillDraftOptions = {}): Promise<SkillDraftResult> {
    const proposal = await this.createProposal(candidate, options)
    if (options.persist === false || !this.store) {
      return { proposal, markdown: proposal.markdown, persisted: false }
    }

    const persistence = await this.store.persist({ proposal })
    if (!persistence.location.trim()) {
      throw new TypeError('skill draft store returned an empty location')
    }
    return { proposal, markdown: proposal.markdown, persisted: true, persistence }
  }

  private async createProposal(candidate: SkillCandidate, options: SkillDraftOptions): Promise<SkillProposal> {
    const initial = this.proposalDrafter.create(candidate, proposalOptions(options))
    try {
      return await this.proposalDrafter.refine(initial)
    } catch {
      // A supplied model is advisory; the deterministic observed procedure remains usable.
      return { ...initial, refinement: 'rejected' }
    }
  }
}

/** Render the canonical SKILL.md shape without invoking a model or writing an artifact. */
export function renderSkillTemplate(candidate: SkillCandidate, options: SkillProposalDraftOptions = {}): string {
  return new SkillProposalDrafter().create(candidate, options).markdown
}

function proposalOptions(options: SkillDraftOptions): SkillProposalDraftOptions {
  return {
    ...(options.description === undefined ? {} : { description: options.description }),
    ...(options.name === undefined ? {} : { name: options.name }),
    ...(options.tags === undefined ? {} : { tags: options.tags }),
    ...(options.version === undefined ? {} : { version: options.version }),
  }
}
