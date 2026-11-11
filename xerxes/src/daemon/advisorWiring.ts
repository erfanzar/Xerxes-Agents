// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ProfileStore } from '../bridge/profiles.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import { createLlmClient } from '../llms/client.js'
import { recordModelUsage } from '../runtime/modelUsageLedger.js'
import { AdvisorBudget, parseAdvisorConfig } from '../runtime/advisorBudget.js'
import type { OutboundGate, OutboundRequestSummary } from '../security/boundaryTypes.js'
import { DataBoundary, deriveDataTrust, parseDataBoundaryConfig } from '../security/dataBoundary.js'
import { DisclosureLog } from '../security/disclosureLog.js'
import { ADVISOR_MAX_OUTPUT_TOKENS, createAdvisorClientFromLlm, registerAdvisorTool } from '../tools/advisorTool.js'

export interface AdvisorWiringOptions {
  readonly runtime: Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  readonly profiles: Pick<ProfileStore, 'get'>
  readonly workspaceRoot: string
  /**
   * Ask the user to approve one outbound request. Absent: every request that
   * needs approval is refused (fail closed) with a reason the model sees.
   */
  readonly approveOutbound?: (summary: OutboundRequestSummary, signal?: AbortSignal) => Promise<boolean>
  readonly log?: DisclosureLog
  readonly reportError?: (message: string) => void
}

/**
 * Register `AskAdvisor` when `advisor.profile` (or XERXES_ADVISOR_PROFILE) is
 * set. Off by default: with no profile nothing is registered and the model
 * never sees the tool. An unknown profile registers nothing and reports why.
 */
export function registerConfiguredAdvisor(tools: ToolRegistry, options: AdvisorWiringOptions): boolean {
  const env = options.env ?? process.env
  const config = parseAdvisorConfig(options.runtime, env)
  if (!config.profile) return false
  const profile = options.profiles.get(config.profile)
  if (!profile) {
    options.reportError?.(`advisor: provider profile "${config.profile}" not found; AskAdvisor is not available`)
    return false
  }
  const model = config.model ?? profile.model
  const llm = createLlmClient(model, {
    provider: profile.provider,
    ...(profile.api_key ? { api_key: profile.api_key } : {}),
    ...(profile.base_url ? { base_url: profile.base_url } : {}),
  })
  const budgets = new Map<string, AdvisorBudget>()
  const log = options.log
  const gate: OutboundGate = {
    async check(request, signal) {
      const boundaryConfig = parseDataBoundaryConfig(options.runtime, env, { root: options.workspaceRoot })
      if (boundaryConfig.mode === 'off') return { allow: true, decision: 'sent' }
      const boundary = new DataBoundary(boundaryConfig, options.workspaceRoot)
      return boundary.outboundGate({
        approve: options.approveOutbound ?? (async () => false),
        log: log ?? new DisclosureLog(),
      }).check({ ...request, provider: profile.name, model }, signal)
    },
  }
  return registerAdvisorTool(tools, {
    config: { ...config, model },
    client: createAdvisorClientFromLlm(llm, model, ADVISOR_MAX_OUTPUT_TOKENS),
    gate,
    trust: deriveDataTrust(profile),
    // Each consult is charged to its task in the usage record, so Hybrid can
    // show how much of a task ran on the cloud advisor and what it sent.
    onConsult: record => recordModelUsage({
      model: record.model,
      profile: profile.name,
      ...(record.sessionId ? { session: record.sessionId } : {}),
      role: 'advisor',
      inputTokens: record.inputTokens,
      outputTokens: record.outputTokens,
      bytesSent: record.bytes,
    }),
    budgetFor: sessionId => {
      let budget = budgets.get(sessionId)
      if (!budget) {
        budget = new AdvisorBudget(config)
        budgets.set(sessionId, budget)
      }
      return budget
    },
  })
}
