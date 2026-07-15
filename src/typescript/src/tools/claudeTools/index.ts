// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry } from '../../executors/toolRegistry.js'
import { WorkspacePathResolver } from '../pathSafety.js'
import { registerClaudeAgentTools, type ClaudeAgentToolsOptions } from './agentOps.js'
import { registerClaudeMcpTools, type ClaudeMcpToolsOptions } from './mcpOps.js'
import { registerClaudeNotebookTools } from './notebook.js'
import { registerClaudeRemoteTools, type ClaudeRemoteToolsOptions } from './remote.js'
import { registerClaudeSearchTools, type ClaudeSearchToolsOptions } from './search.js'
import { registerClaudeWorkflowTools, type ClaudeWorkflowToolsOptions } from './workflow.js'

export * from './agentOps.js'
export * from './mcpOps.js'
export * from './notebook.js'
export * from './remote.js'
export * from './search.js'
export * from './workflow.js'

/** Optional host ports for one complete Claude-compatible tool installation. */
export interface ClaudeCompatibilityToolsOptions {
  readonly agentId?: string
  readonly agents?: ClaudeAgentToolsOptions
  readonly mcp?: ClaudeMcpToolsOptions
  readonly notebookPaths?: WorkspacePathResolver
  readonly remote?: ClaudeRemoteToolsOptions
  readonly search?: ClaudeSearchToolsOptions
  readonly workflow?: ClaudeWorkflowToolsOptions
}

/**
 * Install every available non-duplicative Claude-code tool family.
 *
 * `FileEditTool`, `GlobTool`, and `GrepTool` deliberately stay owned by the
 * existing workspace-safe core tools, which use the same public names.
 */
export function registerClaudeCompatibilityTools(
  registry: ToolRegistry,
  options: ClaudeCompatibilityToolsOptions = {},
): void {
  const agentId = options.agentId ?? 'default'
  if (options.agents !== undefined) registerClaudeAgentTools(registry, options.agents, agentId)
  if (options.mcp !== undefined) registerClaudeMcpTools(registry, options.mcp, agentId)
  if (options.remote !== undefined) registerClaudeRemoteTools(registry, options.remote, agentId)
  if (options.search !== undefined) registerClaudeSearchTools(registry, options.search, agentId)
  if (options.workflow !== undefined) registerClaudeWorkflowTools(registry, options.workflow, agentId)
  if (options.notebookPaths !== undefined) registerClaudeNotebookTools(registry, options.notebookPaths, agentId)
}

/**
 * Explicit non-parity list for hosts deciding whether this Bun layer can replace
 * the legacy Python tool family for their configured runtime.
 */
export function claudeCompatibilityGaps(options: ClaudeCompatibilityToolsOptions = {}): readonly string[] {
  const gaps = [
    'AgentTool isolation/worktree execution depends on a host-provided isolated subagent runner.',
    'AwaitAgents supports terminal-agent status, timeout, and AbortSignal cancellation, but not Python session steer/user-input wakeups.',
    'CheckAgentMessages observes managed snapshot lifecycle/output changes; it does not receive Python-only per-tool streamed mailbox events.',
    'PeekAgent exposes the managed snapshot but not Python-only current-tool, tool-call-count, or recent-stream diagnostics.',
    'ResetAgent restarts the existing managed handle rather than constructing a new Python SubAgentManager task object.',
    'Plan and interaction mode state do not independently enforce a mutation gate; the host permission layer must apply that policy.',
    'Existing FileEditTool has workspace-safe exact replacement but does not yet reproduce Python fuzzy-whitespace replacement or unified-diff output.',
    'Existing GlobTool and GrepTool intentionally use Bun workspace-safe search semantics rather than Python pathlib/ripgrep absolute-path output.',
    'Native worktree removal is limited to worktrees created by the current Bun runtime process.',
    'Cron registration persists jobs, but a daemon must separately attach a CronScheduler runner and delivery target.',
  ]
  if (options.search?.lspAdapter === undefined) gaps.push('LSPTool requires a host-provided language-server adapter.')
  if (options.workflow?.userPromptManager === undefined) gaps.push('AskUserQuestionTool requires a host-provided UserPromptManager.')
  if (options.workflow?.planGenerator === undefined) gaps.push('PlanTool requires a host-provided WorkflowPlanGenerator backed by an LLM planner.')
  if (options.agents === undefined) gaps.push('Claude subagent tools require an attached SpawnedAgentManager.')
  if (options.mcp === undefined) gaps.push('Claude MCP tools require connected MCP clients supplied by the host.')
  if (options.remote?.remoteTriggers === undefined) gaps.push('RemoteTriggerTool requires explicitly configured remote endpoints.')
  if (options.remote?.cronStore === undefined) gaps.push('ScheduleCronTool requires an attached persistent JobStore.')
  return Object.freeze(gaps)
}
