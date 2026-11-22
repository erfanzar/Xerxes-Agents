// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry } from '../executors/toolRegistry.js'
import type { LlmClient } from '../llms/client.js'
import { createAgentState, type StreamEvent } from '../streaming/events.js'
import { runTurn, type TurnDependencies, type TurnRequest } from '../streaming/loop.js'
import type { PermissionMode } from '../streaming/permissions.js'
import type { ChatMessage } from '../types/messages.js'
import { registerSwarmTools, type SwarmToolHost } from '../tools/swarmTools.js'
import {
  isSwarmKillSignal,
  isSwarmParkSignal,
  type SwarmMemberSnapshot,
  type SwarmWorkerWakeReason,
} from './types.js'

export interface SwarmWorkerTurnInput {
  readonly swarm: SwarmToolHost
  readonly sessionId: string
  readonly member: SwarmMemberSnapshot
  readonly workspacePath: string
  readonly reason: SwarmWorkerWakeReason
  readonly prompt: string
  readonly systemPromptOverlay: string
  readonly messages: ChatMessage[]
  readonly signal: AbortSignal
}

export interface SwarmWorkerTurnResult {
  readonly messages: readonly ChatMessage[]
  readonly parked?: boolean
  readonly error?: string
}

export type SwarmWorkerRunnerPort = (
  input: SwarmWorkerTurnInput,
) => Promise<SwarmWorkerTurnResult | void>

export const MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS = 32_768

export interface NormalizeWorkerHistoryOptions {
  /** Deep-clone messages (needed when forking across two workers). Defaults to true. */
  readonly clone?: boolean
  /** Cap oversized string tool outputs retained in memory and persisted worker histories. */
  readonly maxToolContentChars?: number
}

function capOversizedToolMessage(message: ChatMessage, maxToolContentChars: number | undefined): ChatMessage {
  if (
    maxToolContentChars === undefined
    || message.role !== 'tool'
    || typeof message.content !== 'string'
    || message.content.length <= maxToolContentChars
  ) {
    return message
  }
  const omitted = message.content.length - maxToolContentChars
  return {
    ...message,
    content: `${message.content.slice(0, maxToolContentChars)}… [truncated ${omitted} chars for memory]`,
  }
}

/**
 * Strip system prompts, pair any in-flight assistant `tool_calls`, and
 * optionally cap oversized tool outputs without unnecessary full deep-cloning
 * on single-owner worker turns.
 */
export function normalizeWorkerHistory(
  source: readonly ChatMessage[],
  options: NormalizeWorkerHistoryOptions = {},
): ChatMessage[] {
  const shouldClone = options.clone !== false
  const maxToolChars = options.maxToolContentChars
  const filtered: ChatMessage[] = []
  for (const message of source) {
    if (message.role === 'system') continue
    const bounded = capOversizedToolMessage(message, maxToolChars)
    filtered.push(shouldClone ? structuredClone(bounded) : bounded)
  }

  const repaired: ChatMessage[] = []
  let index = 0
  while (index < filtered.length) {
    const current = filtered[index]
    if (!current) {
      index += 1
      continue
    }
    repaired.push(current)
    if (current.role === 'assistant' && current.tool_calls && current.tool_calls.length > 0) {
      const answeredIds = new Set<string>()
      let cursor = index + 1
      while (cursor < filtered.length) {
        const nextMsg = filtered[cursor]
        if (!nextMsg || nextMsg.role !== 'tool') break
        answeredIds.add(nextMsg.tool_call_id)
        repaired.push(nextMsg)
        cursor += 1
      }
      for (const call of current.tool_calls) {
        if (!answeredIds.has(call.id)) {
          repaired.push({
            role: 'tool',
            tool_call_id: call.id,
            name: call.function.name,
            content: '[forked while parent tool call was still in progress]',
          })
        }
      }
      index = cursor
      continue
    }
    index += 1
  }

  return repaired
}

/**
 * Clone a conversation transcript for `fork` while stripping system prompts
 * and pairing any in-flight assistant `tool_calls` so the forked history is
 * strictly valid for OpenAI and Anthropic providers on the new worker's first turn.
 */
export function sanitizeForkedMessages(source: readonly ChatMessage[]): ChatMessage[] {
  return normalizeWorkerHistory(source, { clone: true })
}

export interface CreateRunTurnWorkerRunnerOptions {
  readonly llm: LlmClient | ((member: SwarmMemberSnapshot) => LlmClient)
  readonly configureWorkerRegistry?: (params: {
    readonly registry: ToolRegistry
    readonly member: SwarmMemberSnapshot
    readonly workspacePath: string
    readonly swarm: SwarmToolHost
  }) => Promise<void> | void
  readonly baseSystemPrompt?: string | ((member: SwarmMemberSnapshot) => string)
  readonly permissionMode?: PermissionMode
  readonly maxToolTurns?: number
  readonly turnDependencies?: Omit<Partial<TurnDependencies>, 'llm' | 'toolExecutor' | 'capabilities'>
  readonly onStreamEvent?: (memberName: string, event: StreamEvent) => void
}

/**
 * Create a production `SwarmWorkerRunnerPort` backed directly by Xerxes's
 * `runTurn` streaming loop and `ToolRegistry`.
 */
export function createRunTurnWorkerRunner(
  options: CreateRunTurnWorkerRunnerOptions,
): SwarmWorkerRunnerPort {
  return async (input: SwarmWorkerTurnInput): Promise<SwarmWorkerTurnResult> => {
    const registry = new ToolRegistry({ onDuplicateRegistration: () => undefined })
    if (options.configureWorkerRegistry) {
      await options.configureWorkerRegistry({
        registry,
        member: input.member,
        workspacePath: input.workspacePath,
        swarm: input.swarm,
      })
    }
    const registration = registerSwarmTools(registry, {
      swarm: input.swarm,
      caller: input.member.name,
      agentId: input.member.name,
    })

    try {
      const llm = typeof options.llm === 'function' ? options.llm(input.member) : options.llm
      const baseSystem = typeof options.baseSystemPrompt === 'function'
        ? options.baseSystemPrompt(input.member)
        : options.baseSystemPrompt
      const systemPrompt = baseSystem
        ? `${baseSystem.trim()}\n\n${input.systemPromptOverlay}`
        : input.systemPromptOverlay

      const state = createAgentState([...input.messages])
      state.metadata.swarmMember = input.member.name
      state.metadata.swarmSessionId = input.sessionId
      state.metadata.workspaceDir = input.workspacePath

      const tools = registry.definitions(input.member.name)
      const request: TurnRequest = {
        agentId: input.member.name,
        sessionId: `${input.sessionId}:swarm:${input.member.name}`,
        model: input.member.model,
        state,
        systemPrompt,
        systemPromptRequestOnly: true,
        tools,
        userMessage: input.prompt,
        userOrigin: 'agent',
        permissionMode: options.permissionMode ?? 'accept-all',
        ...(options.maxToolTurns === undefined ? {} : { maxToolTurns: options.maxToolTurns }),
      }

      const dependencies: TurnDependencies = {
        ...options.turnDependencies,
        llm,
        toolExecutor: registry,
        capabilities: (toolName, agentId, args) => registry.capabilities(toolName, agentId, args),
      }

      let finalProviderError: string | undefined
      let failedStopReason: string | undefined
      for await (const event of runTurn(request, dependencies, input.signal)) {
        if (event.type === 'provider_retry' && event.final && event.error) {
          finalProviderError = event.error
        } else if (
          event.type === 'turn_done'
          && (event.reason === 'provider_failed' || event.reason === 'turn_failed' || event.reason === 'context_overflow')
        ) {
          failedStopReason = event.reason
        }
        options.onStreamEvent?.(input.member.name, event)
      }

      const reason: unknown = input.signal.aborted ? input.signal.reason : undefined
      if (isSwarmKillSignal(reason)) {
        return {
          messages: state.messages,
          parked: false,
        }
      }
      if (isSwarmParkSignal(reason)) {
        return {
          messages: state.messages,
          parked: true,
        }
      }
      if (failedStopReason !== undefined || finalProviderError !== undefined) {
        return {
          messages: state.messages,
          parked: false,
          error: finalProviderError ?? `Worker turn failed (${failedStopReason})`,
        }
      }

      return {
        messages: state.messages,
        parked: false,
      }
    } finally {
      registration.dispose()
    }
  }
}
