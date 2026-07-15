// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate, ToolArguments } from './model.js'

export const DEFAULT_SKILL_VERSION = '0.1.0'

export type SkillRefinementStatus = 'applied' | 'none' | 'rejected'

/** A deterministic proposal that has not necessarily been persisted by a host. */
export interface SkillProposal {
  readonly candidate: SkillCandidate
  readonly description: string
  readonly markdown: string
  readonly name: string
  readonly refinement: SkillRefinementStatus
  readonly requiredTools: readonly string[]
  readonly tags: readonly string[]
  readonly version: string
}

export interface SkillProposalDraftOptions {
  readonly description?: string
  readonly name?: string
  readonly tags?: readonly string[]
  readonly version?: string
}

/** Host-provided optional refinement boundary. There is intentionally no default LLM client. */
export interface SkillDraftRefinerPort {
  refine(input: { readonly proposal: SkillProposal }): string | undefined | Promise<string | undefined>
}

/** Produces deterministic drafts and applies only explicitly supplied safe refinements. */
export class SkillProposalDrafter {
  private readonly refiner: SkillDraftRefinerPort | undefined

  constructor(options: { readonly refiner?: SkillDraftRefinerPort } = {}) {
    this.refiner = options.refiner
  }

  create(candidate: SkillCandidate, options: SkillProposalDraftOptions = {}): SkillProposal {
    const name = slugify(options.name ?? (candidate.userPrompt || candidate.signature()))
    const description = describe(candidate, options.description)
    const tags = Object.freeze([...(options.tags ?? candidate.uniqueTools)])
    const requiredTools = Object.freeze([...candidate.uniqueTools])
    const version = options.version ?? DEFAULT_SKILL_VERSION
    return {
      candidate,
      name,
      description,
      version,
      tags,
      requiredTools,
      markdown: renderSkillProposalMarkdown(candidate, { name, description, version, tags, requiredTools }),
      refinement: 'none',
    }
  }

  /**
   * Call a host-supplied refiner only when one is configured.
   *
   * An invalid response is rejected in favor of the deterministic proposal;
   * no model call or synthetic rewritten text occurs otherwise.
   */
  async refine(proposal: SkillProposal): Promise<SkillProposal> {
    if (!this.refiner) {
      return proposal
    }
    const markdown = await this.refiner.refine({ proposal })
    if (typeof markdown !== 'string' || !validRefinement(markdown, proposal)) {
      return { ...proposal, refinement: 'rejected' }
    }
    return { ...proposal, markdown: normalizeMarkdown(markdown), refinement: 'applied' }
  }
}

export function renderSkillProposalMarkdown(
  candidate: SkillCandidate,
  metadata: {
    readonly description: string
    readonly name: string
    readonly requiredTools: readonly string[]
    readonly tags: readonly string[]
    readonly version: string
  },
): string {
  const frontmatter = [
    '---',
    'name: ' + metadata.name,
    'description: ' + JSON.stringify(metadata.description),
    'version: ' + metadata.version,
    'tags: ' + yamlList(metadata.tags),
    'required_tools: ' + yamlList(metadata.requiredTools),
    '---',
  ]
  const when = candidate.userPrompt.trim()
    ? 'Apply this skill for tasks similar to: ' + candidate.userPrompt.trim().slice(0, 240)
    : 'Apply this skill when the observed tool sequence matches the current task.'
  const procedure = candidate.successfulEvents.map((event, index) => (
    String(index + 1) + '. **' + event.toolName + '** — ' + summarizeArguments(event.arguments)
  ))
  const failures = candidate.events.filter(event => event.status !== 'success')
  const pitfalls = [
    ...failures.map(event => {
      const failure = safeFailure(event.errorMessage ?? event.errorType ?? event.status)
      return '- ' + event.toolName + ' may fail with ' + failure
    }),
    ...candidate.events
      .filter(event => event.retryOf !== undefined)
      .map(event => '- ' + event.toolName + ' was retried in this run; expect transient failures.'),
  ]
  const verification = [
    'After running the procedure, invoke these tools in order: ' + candidate.signature(),
    'Total successful calls expected: ' + candidate.successfulEvents.length + '.',
    ...(candidate.finalResponse.trim()
      ? ['Reference final response: ' + candidate.finalResponse.trim().replace(/\s+/g, ' ').slice(0, 160)]
      : []),
  ]
  return [
    frontmatter.join('\n'),
    '# When to use\n\n' + when,
    '# Procedure\n\n' + (procedure.length ? procedure.join('\n') : 'No successful calls were observed.'),
    ...(pitfalls.length ? ['# Pitfalls\n\n' + pitfalls.join('\n')] : []),
    '# Verification\n\n' + verification.join('\n'),
  ].join('\n\n') + '\n'
}

export function slugify(text: string): string {
  const normalized = text.trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/g, '')
  return normalized || 'skill'
}

function describe(candidate: SkillCandidate, override: string | undefined): string {
  const text = (override ?? (candidate.userPrompt.trim() || 'Auto-authored skill from tool sequence.')).trim()
  return text.length <= 200 ? text : text.slice(0, 197) + '...'
}

function yamlList(values: readonly string[]): string {
  return '[' + values.map(value => JSON.stringify(value)).join(', ') + ']'
}

function summarizeArguments(arguments_: ToolArguments): string {
  const entries = Object.entries(arguments_)
  if (!entries.length) {
    return '(no args)'
  }
  return entries.map(([key, value]) => key + '=' + safeValue(key, value)).join(', ')
}

function safeFailure(value: string): string {
  return value
    .replace(/\b(api[_-]?key|token|password)\b\s*(?::|=|\s)\s*['"]?([A-Za-z0-9._-]{8,})/gi, '$1=[redacted]')
    .replace(/\b(authorization\s*:\s*bearer)\s+([A-Za-z0-9._-]+)/gi, '$1=[redacted]')
    .replace(/\bsk-[A-Za-z0-9_-]{16,}\b/g, '[redacted]')
    .replace(/\s+/g, ' ')
    .slice(0, 120)
}

function safeValue(key: string, value: unknown): string {
  if (/(api.?key|authorization|credential|password|secret|token)/i.test(key)) {
    return '[redacted]'
  }
  let text: string | undefined
  try {
    text = typeof value === 'string' ? value : JSON.stringify(value)
  } catch {
    text = undefined
  }
  const normalized = (text ?? String(value)).replace(/\s+/g, ' ')
  return normalized.length <= 30 ? normalized : normalized.slice(0, 27) + '...'
}

function validRefinement(markdown: string, proposal: SkillProposal): boolean {
  const normalized = markdown.trim()
  if (!normalized.startsWith('---')) {
    return false
  }
  if (!['# When to use', '# Procedure', '# Verification'].every(section => normalized.includes(section))) {
    return false
  }
  return proposal.candidate.successfulEvents.every(event => normalized.includes(event.toolName))
}

function normalizeMarkdown(markdown: string): string {
  return markdown.trimEnd() + '\n'
}
