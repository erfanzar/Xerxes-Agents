// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate, ToolCallEvent, ToolCallInput } from './model.js'
import { ToolSequenceTracker } from './model.js'
import { SkillProposalDrafter, type SkillProposal } from './proposal.js'
import { SkillTelemetry } from './lifecycle.js'
import {
  SkillAuthoringConfig,
  SkillAuthoringTrigger,
  type SkillAuthoringConfigOptions,
  type SkillAuthoringDecision,
  type SkillCatalogPort,
} from './trigger.js'
import { SkillVerifier, type VerificationResult, type VerificationStep } from './verifier.js'

/** Explicit persistence boundary for turning an in-memory proposal into a host-owned skill artifact. */
export interface SkillProposalStorePort {
  persist(input: {
    readonly proposal: SkillProposal
    readonly verificationSteps: readonly VerificationStep[]
  }): Promise<SkillProposalPersistence>
}

export interface SkillProposalPersistence {
  readonly id: string
  readonly location?: string
}

/** Optional observer boundary; failures here never change an already-persisted result. */
export interface SkillAuthoringObserverPort {
  proposalPersisted(input: {
    readonly candidate: SkillCandidate
    readonly persistence: SkillProposalPersistence
    readonly proposal: SkillProposal
  }): void | Promise<void>
}

export type AuthoringStatus = 'failed' | 'persisted' | 'proposed' | 'skipped'

/** Outcome of finishing one tracked turn. Authored means an explicit store persisted the proposal. */
export interface AuthoringResult {
  readonly authored: boolean
  readonly candidate: SkillCandidate
  readonly decision: SkillAuthoringDecision
  readonly persistence?: SkillProposalPersistence
  readonly proposal?: SkillProposal
  readonly reason: string
  readonly status: AuthoringStatus
  readonly verification?: VerificationResult
  readonly verificationSteps?: readonly VerificationStep[]
}

export interface SkillAuthoringPipelineOptions {
  readonly catalog?: SkillCatalogPort
  readonly config?: SkillAuthoringConfig | SkillAuthoringConfigOptions
  readonly drafter?: SkillProposalDrafter
  readonly observer?: SkillAuthoringObserverPort
  readonly proposalStore?: SkillProposalStorePort
  readonly telemetry?: SkillTelemetry
  readonly tracker?: ToolSequenceTracker
  readonly verifier?: SkillVerifier
}

/**
 * Coordinates observation, eligibility, drafting, verification, and explicit persistence.
 *
 * The default result is a proposal only. A host must inject a proposal store
 * before this pipeline can report a skill as authored.
 */
export class SkillAuthoringPipeline {
  readonly drafter: SkillProposalDrafter
  readonly tracker: ToolSequenceTracker
  readonly trigger: SkillAuthoringTrigger
  readonly verifier: SkillVerifier
  private readonly observer: SkillAuthoringObserverPort | undefined
  private readonly proposalStore: SkillProposalStorePort | undefined
  private readonly telemetry: SkillTelemetry | undefined

  constructor(options: SkillAuthoringPipelineOptions = {}) {
    this.tracker = options.tracker ?? new ToolSequenceTracker()
    this.trigger = new SkillAuthoringTrigger({
      ...(options.catalog === undefined ? {} : { catalog: options.catalog }),
      ...(options.config === undefined ? {} : { config: options.config }),
    })
    this.drafter = options.drafter ?? new SkillProposalDrafter()
    this.verifier = options.verifier ?? new SkillVerifier()
    this.proposalStore = options.proposalStore
    this.observer = options.observer
    this.telemetry = options.telemetry
  }

  beginTurn(options: { readonly agentId?: string; readonly turnId?: string; readonly userPrompt?: string } = {}): void {
    this.tracker.beginTurn(options)
  }

  /** Forward one observed tool call to the turn-scoped tracker. */
  recordCall(input: ToolCallInput): ToolCallEvent {
    return this.tracker.recordCall(input)
  }

  /** Alias kept for event-loop integrations that describe this boundary as turn end. */
  onTurnEnd(finalResponse = ''): Promise<AuthoringResult> {
    return this.endTurn(finalResponse)
  }

  async endTurn(finalResponse = ''): Promise<AuthoringResult> {
    const candidate = this.tracker.endTurn(finalResponse)
    const decision = this.trigger.evaluate(candidate)
    if (!decision.eligible) {
      return {
        candidate,
        decision,
        authored: false,
        status: 'skipped',
        reason: decision.reason,
      }
    }

    let proposal: SkillProposal
    let verificationSteps: VerificationStep[]
    let verification: VerificationResult
    try {
      proposal = await this.drafter.refine(this.drafter.create(candidate))
      verificationSteps = this.verifier.generate(candidate)
      verification = this.verifier.verify(verificationSteps, candidate)
    } catch (error) {
      return {
        candidate,
        decision,
        authored: false,
        status: 'failed',
        reason: 'proposal preparation failed: ' + errorKind(error),
      }
    }
    if (!verification.passed) {
      return {
        candidate,
        decision,
        proposal,
        verificationSteps,
        verification,
        authored: false,
        status: 'failed',
        reason: 'generated verification did not validate the observed candidate',
      }
    }
    if (!this.proposalStore) {
      return {
        candidate,
        decision,
        proposal,
        verificationSteps,
        verification,
        authored: false,
        status: 'proposed',
        reason: '',
      }
    }

    let persistence: SkillProposalPersistence
    try {
      persistence = await this.proposalStore.persist({ proposal, verificationSteps })
      if (!persistence.id.trim()) {
        throw new TypeError('proposal store returned an empty persistence id')
      }
    } catch (error) {
      return {
        candidate,
        decision,
        proposal,
        verificationSteps,
        verification,
        authored: false,
        status: 'failed',
        reason: 'proposal persistence failed: ' + errorKind(error),
      }
    }
    try {
      this.telemetry?.recordAuthored({ skillName: proposal.name, version: proposal.version })
    } catch {
      // Telemetry cannot turn a persisted artifact into a failed authoring result.
    }
    try {
      await this.observer?.proposalPersisted({ candidate, proposal, persistence })
    } catch {
      // Observers cannot turn a persisted artifact into a failed authoring result.
    }
    return {
      candidate,
      decision,
      proposal,
      verificationSteps,
      verification,
      persistence,
      authored: true,
      status: 'persisted',
      reason: '',
    }
  }
}

function errorKind(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'unknown error'
}
