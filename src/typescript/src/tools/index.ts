// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { registerAgentMemoryTools, type AgentMemoryToolsOptions } from './agentMemoryTools.js'
import { registerAgentMetaTools, type AgentMetaToolsOptions } from './agentMetaTools.js'
import { registerAiTools, type AiToolProviders } from './aiTools.js'
import { registerBrowserTools, type BrowserToolsOptions } from './browserTools.js'
import { registerClaudeCompatibilityTools, type ClaudeCompatibilityToolsOptions } from './claudeTools/index.js'
import { registerClarifyTool, type ClarifyToolOptions } from './clarify.js'
import { registerComputerUseTool, type ComputerUseToolsOptions } from './computerUse/index.js'
import { registerCodingTools } from './codingTools.js'
import { registerDataTools } from './dataTools.js'
import { registerFileTools } from './fileTools.js'
import { registerHomeAssistantTools, type HomeAssistantToolsOptions } from './homeAssistantTools.js'
import { registerSearchHistoryTool, type SearchHistoryTool } from './history.js'
import { registerMathTools } from './mathTools.js'
import { registerMediaTools, type MediaToolPorts } from './mediaTools.js'
import { registerMemoryTools, type MemoryToolsOptions } from './memoryTools.js'
import { WorkspacePathResolver } from './pathSafety.js'
import { registerProcessTools } from './processTools.js'
import { registerRlTools, type RLToolsOptions } from './rlTools.js'
import { registerSendMessageTool, type SendMessageToolOptions } from './sendMessage.js'
import { registerSkillManageTool, type SkillManageOptions } from './skillManage.js'
import { registerSystemTools } from './systemTools.js'
import { PublicWebClient, registerWebTools } from './webTools.js'
import { registerWorkspaceMemoryTools, type WorkspaceMemoryToolsOptions } from './workspaceMemory.js'

export * from './agentMemoryTools.js'
export * from './agentMetaTools.js'
export * from './aiTools.js'
export * from './browserTools.js'
export * from './dataTools.js'
export * from './claudeTools/index.js'
export * from './clarify.js'
export * from './computerUse/index.js'
export * as codingTools from './codingTools.js'
export {
  CODING_TOOL_DEFINITIONS,
  createUnifiedDiff,
  detectLanguage,
  registerCodingTools,
} from './codingTools.js'
export * from './duckduckgoEngine.js'
export * from './fileTools.js'
export * from './googleSearch.js'
export * from './history.js'
export * from './homeAssistantTools.js'
export * from './imageGeneration.js'
export * from './mathTools.js'
export * from './mediaHttp.js'
export * from './mediaTools.js'
export * from './memoryTools.js'
export * from './pathSafety.js'
export * from './processTools.js'
export * from './rlTools.js'
export * from './sendMessage.js'
export * from './skillManage.js'
export * from './systemTools.js'
export * from './transcription.js'
export * from './tts.js'
export * from './vision.js'
export * from './voiceMode.js'
export * from './webTools.js'
export * from './workspaceMemory.js'
export * from './workspaceTools.js'

export interface CoreToolsOptions {
  /** Persistent memory is host/session scoped and is registered only when supplied. */
  readonly agentMemoryTools?: AgentMemoryToolsOptions
  /** Agent orchestration, session search, and skill ports are host-owned and opt-in. */
  readonly agentMetaTools?: AgentMetaToolsOptions
  /** Optional explicit provider hooks for AI text tools; local deterministic methods need none. */
  readonly aiTools?: AiToolProviders
  /** Browser tools are registered only when a host injects a real browser session or resolver. */
  readonly browserTools?: BrowserToolsOptions
  /** Clarification tooling requires a host-owned UI adapter. */
  readonly clarifyTool?: ClarifyToolOptions
  /** Desktop computer use is registered only with a host-injected privileged session or resolver. */
  readonly computerUseTool?: ComputerUseToolsOptions
  /**
   * Host-bound Claude-compatible adapters. Omit this until the caller has
   * supplied real subagent, MCP, prompt, remote, or notebook ports.
   */
  readonly claudeCompatibilityTools?: ClaudeCompatibilityToolsOptions
  readonly includeDataTools?: boolean
  readonly includeAiTools?: boolean
  /** Lower-case compatibility coding tools include destructive Git/file actions and stay opt-in. */
  readonly includeCodingTools?: boolean
  readonly includeMathTools?: boolean
  readonly includeProcessTools?: boolean
  readonly includeSystemTools?: boolean
  /** Safe public HTTP tools; enabled by default and independently permission-gated. */
  readonly includeWebTools?: boolean
  /** Home Assistant is registered only after a host supplies an explicit client or resolver. */
  readonly homeAssistantTools?: HomeAssistantToolsOptions
  /** History search requires a host-owned index or session store wrapper. */
  readonly historyTool?: SearchHistoryTool
  /** Media tools are registered only for explicitly configured provider ports. */
  readonly mediaTools?: MediaToolPorts
  /** Legacy long-term memory management is registered only with a host-owned memory resolver. */
  readonly memoryTools?: MemoryToolsOptions
  /** Outbound messages are registered only with a configured channel dispatcher. */
  readonly sendMessageTool?: SendMessageToolOptions
  /** RL controls are opt-in because training and inference require a host-selected backend. */
  readonly rlTools?: RLToolsOptions
  /** Skill authoring writes are opt-in because they mutate persistent user storage. */
  readonly skillManageTools?: SkillManageOptions
  /** Optional injectable web transport, useful for hosts with DNS-pinning policy. */
  readonly webClient?: PublicWebClient
  /** Workspace MEMORY.md/USER.md CRUD is opt-in for hosts that own a markdown workspace. */
  readonly workspaceMemoryTools?: WorkspaceMemoryToolsOptions
  readonly workspaceRoot?: string
}

/**
 * Register the Bun-native baseline tool surface for one workspace.
 *
 * Filesystem, data, mathematical, and inspection tools are registered by
 * default. Process execution is direct argv-only and can be disabled while a
 * caller has not yet connected its permission and sandbox layers.
 */
export function registerCoreTools(registry: ToolRegistry, options: CoreToolsOptions = {}): WorkspacePathResolver {
  if (options.agentMetaTools !== undefined && options.skillManageTools !== undefined) {
    throw new ConfigurationError(
      'coreTools',
      'agentMetaTools and skillManageTools both register skill_manage; configure one skill-management surface',
    )
  }
  const paths = new WorkspacePathResolver(options.workspaceRoot ?? process.cwd())
  registerFileTools(registry, paths)
  if (options.includeAiTools ?? true) {
    registerAiTools(registry, options.aiTools ?? {})
  }
  if (options.includeDataTools ?? true) {
    registerDataTools(registry, paths)
  }
  if (options.includeCodingTools ?? false) {
    registerCodingTools(registry, paths)
  }
  if (options.includeMathTools ?? true) {
    registerMathTools(registry)
  }
  if (options.includeProcessTools ?? true) {
    registerProcessTools(registry, paths)
  }
  if (options.includeSystemTools ?? true) {
    registerSystemTools(registry)
  }
  if (options.includeWebTools ?? true) {
    registerWebTools(registry, options.webClient)
  }
  if (options.agentMemoryTools !== undefined) {
    registerAgentMemoryTools(registry, options.agentMemoryTools)
  }
  if (options.agentMetaTools !== undefined) {
    registerAgentMetaTools(registry, options.agentMetaTools)
  }
  if (options.browserTools !== undefined) {
    registerBrowserTools(registry, options.browserTools)
  }
  if (options.clarifyTool !== undefined) {
    registerClarifyTool(registry, options.clarifyTool)
  }
  if (options.computerUseTool !== undefined) {
    registerComputerUseTool(registry, options.computerUseTool)
  }
  if (options.mediaTools !== undefined) {
    registerMediaTools(registry, options.mediaTools)
  }
  if (options.memoryTools !== undefined) {
    registerMemoryTools(registry, options.memoryTools)
  }
  if (options.homeAssistantTools !== undefined) {
    registerHomeAssistantTools(registry, options.homeAssistantTools)
  }
  if (options.historyTool !== undefined) {
    registerSearchHistoryTool(registry, options.historyTool)
  }
  if (options.sendMessageTool !== undefined) {
    registerSendMessageTool(registry, options.sendMessageTool)
  }
  if (options.rlTools !== undefined) {
    registerRlTools(registry, options.rlTools)
  }
  if (options.skillManageTools !== undefined) {
    registerSkillManageTool(registry, options.skillManageTools)
  }
  if (options.workspaceMemoryTools !== undefined) {
    registerWorkspaceMemoryTools(registry, options.workspaceMemoryTools)
  }
  if (options.claudeCompatibilityTools !== undefined) {
    registerClaudeCompatibilityTools(registry, options.claudeCompatibilityTools)
  }
  return paths
}
