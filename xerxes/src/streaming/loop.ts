// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  isScreenshotToolResult,
  supersedeScreenshotToolResults,
} from '../context/screenshotSuperseder.js'
import {
  errorMessage,
  toolSearchLoadedNames,
  type ToolExecutor,
  type ToolExecutionContext,
} from '../executors/toolRegistry.js'
import type { ToolLoopBlockAuditInput } from '../audit/emitter.js'
import { resolveToolPermission, type HookPoint, type HookRunner } from '../extensions/hooks.js'
import type { LlmClient, LlmDelta, QuerySource, ThinkingRequest, TokenUsage } from '../llms/client.js'
import {
  DENIAL_LOOP_PATTERN,
  DenialBudget,
  denialBudgetStopText,
} from '../runtime/denialBudget.js'
import {
  renderContextOverflowStopGuard,
  renderIntervention,
  renderOutputLimitResumeDirective,
} from '../runtime/interventions.js'
import { classifyError, ErrorKind } from '../runtime/errorClassifier.js'
import {
  inspectObjectiveResponse,
  objectiveGuardRetryLimit,
  type ObjectiveToolExecutionEvidence,
} from '../runtime/objectiveGuard.js'
import { getGoal } from '../runtime/goalDomain.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import { isJsonObject, type ToolCall, type ToolDefinition } from '../types/toolCalls.js'
import { appendInjection } from './attachments.js'
import type {
  AgentState,
  PermissionRequest,
  StreamEvent,
  ToolResult,
  TurnStopReason,
} from './events.js'
import {
  DEFAULT_PERMISSION_MODE,
  deniedResult,
  permissionDisposition,
  permissionDescription,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from './permissions.js'
import type { SystemPromptSegment } from './promptCaching.js'
import { ThinkingParser, type ThinkingStreamParser } from './thinkingParser.js'
import { neutralizeSystemReminders } from './toolMarkers.js'

/** Distinct productive tool rounds are unbounded unless the caller opts into a budget. */
export const DEFAULT_MAX_TOOL_TURNS = Number.POSITIVE_INFINITY
export const DEFAULT_RETRY_DELAYS = [1_000, 2_000] as const
/** Generous default chunk-arrival budget before a provider stream is treated as stalled. */
export const DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS = 120_000
/** Cap provider-suggested Retry-After waits so a bad hint cannot park a turn for hours. */
export const MAX_SUGGESTED_RETRY_DELAY_MS = 60_000
/**
 * Consecutive rounds in which the model requests only unconfigured tools
 * before the turn stops with an explicit error. Without a cap that pattern
 * loops one provider call per round forever (maxToolTurns is unbounded).
 */
export const MAX_UNCONFIGURED_ONLY_ROUNDS = 3
/**
 * Output ceiling used after a first `finish_reason: length`, when the caller
 * pinned no maxTokens of its own. A truncation means the model wanted more
 * room than the provider default, so the retry gives it a window large enough
 * that a second truncation is a genuinely long answer rather than a bad
 * default.
 */
export const OUTPUT_LIMIT_RETRY_MAX_TOKENS = 64_000
/**
 * Consecutive output-token truncations tolerated before the turn stops. Without
 * a cap a model that keeps filling the window resumes forever, one full
 * generation per round.
 */
export const MAX_OUTPUT_LIMIT_ESCALATIONS = 3
/**
 * Resume directive pushed after a truncation that a larger window did not fix.
 * The model has already spent a whole window, so an apology or a recap spends
 * the next one restating text the user has already read.
 */
export const OUTPUT_LIMIT_RESUME_REMINDER = renderOutputLimitResumeDirective()
/**
 * Terminal wording for a context overflow no reducer could relieve. The
 * provider's own string names a token count the user cannot act on, so echoing
 * it leaves the session repeating an identical failure; these three commands
 * are the actual remedies.
 */
export const CONTEXT_OVERFLOW_STOP_TEXT = renderContextOverflowStopGuard()

/** Upper bound on tools executed at once, so a wide round cannot exhaust file handles or sockets. */
export const MAX_CONCURRENT_TOOL_CALLS = 8

/** A per-call verdict from the sequential permission phase, executed later in order. */
type ToolDecision =
  | { readonly call: ToolCall; readonly effectiveCall: ToolCall; readonly kind: 'allowed' }
  | { readonly call: ToolCall; readonly kind: 'cancelled' }
  | {
    readonly call: ToolCall
    readonly detail?: string
    readonly kind: 'denied'
    readonly reason: 'permission_rejected' | 'policy_denied'
  }

/**
 * Partition decisions into maximal runs of consecutive concurrency-safe calls.
 *
 * Consecutive is the load-bearing word: the model emits tool calls in an order
 * it may have reasoned about, so reordering across an unsafe call could run a
 * read before the write it was meant to observe. A run therefore never spans an
 * unsafe call, a denial, or a cancellation — each of those is its own group of
 * one, and the model's sequence is preserved exactly.
 */
export function groupToolDecisions<T extends { readonly call: ToolCall; readonly kind: string }>(
  decisions: readonly T[],
  capabilitiesOf: (call: ToolCall) => { readonly concurrencySafe: boolean },
  maxConcurrent: number = MAX_CONCURRENT_TOOL_CALLS,
): T[][] {
  const groups: T[][] = []
  let run: T[] = []
  const flush = (): void => {
    if (run.length) groups.push(run)
    run = []
  }
  for (const decision of decisions) {
    const parallelizable = decision.kind === 'allowed'
      && capabilitiesOf(decision.call).concurrencySafe
    if (!parallelizable) {
      flush()
      groups.push([decision])
      continue
    }
    run.push(decision)
    if (run.length >= maxConcurrent) flush()
  }
  flush()
  return groups
}

/** Outcome of one context-reduction pass over the live turn history. */
export interface ContextReduction {
  readonly messages: readonly ChatMessage[]
  readonly tokensFreed: number
}

/**
 * Compact the turn history in place of a failed oversized request. The loop
 * never constructs one: the daemon owns compaction policy and injects it.
 */
export type ContextReducer = (
  messages: readonly ChatMessage[],
  signal?: AbortSignal,
) => Promise<ContextReduction>

/**
 * True when a tool result carries nothing a provider will accept as content.
 *
 * `[]` and `{}` are deliberately not empty: several tools return a truthful
 * empty collection, and the model must be able to tell that apart from silence.
 */
export function isEffectivelyEmpty(content: string): boolean {
  return content.trim().length === 0
}

/** Stand-in body for a tool that succeeded without producing any output. */
export function emptyToolResult(name: string): string {
  return `[${name} produced no output.]`
}

/**
 * Make one tool's output safe to place in the transcript.
 *
 * Two failure modes are prevented here. A blank body reaches the Anthropic
 * adapter as an empty content block and the API rejects the whole request, so
 * reading a zero-byte file or calling a silent MCP tool would end the turn with
 * a 400. And `<system-reminder>` is the tag our own system prompt declares
 * authoritative, so any file, page, or command output containing it would
 * otherwise be an instruction-injection channel.
 */
export function normalizeToolOutput(name: string, output: string): string {
  const neutralized = neutralizeSystemReminders(output)
  return isEffectivelyEmpty(neutralized) ? emptyToolResult(name) : neutralized
}

export interface TurnRequest {
  readonly agentId?: string
  /** Session interaction mode; objective mode rejects unsupported narrative stops. */
  readonly interactionMode?: string
  /**
   * Consecutive refused tool calls tolerated before the turn stops. Omitted
   * takes the DenialBudget default; a non-positive value opts out entirely.
   */
  readonly maxConsecutiveDenials?: number
  readonly maxToolTurns?: number
  readonly maxTokens?: number
  readonly model: string
  /** Optional maximum retries for objective-mode text-only stopping attempts. */
  readonly objectiveGuardMaxRetries?: number
  readonly permissionMode?: PermissionMode
  /** Why this completion is being made; drives cost attribution and retry policy. */
  readonly querySource?: QuerySource
  readonly sessionId?: string
  /** OpenAI processing tier request (`auto`/`default`/`flex`/`priority`); Responses-family only. */
  readonly serviceTier?: string
  readonly state: AgentState
  readonly systemPrompt?: string
  /** Send the system prompt in a request copy without mutating durable AgentState messages. */
  readonly systemPromptRequestOnly?: boolean
  /** Same prompt as `systemPrompt`, kept as named sources for prefix caching. */
  readonly systemSegments?: readonly SystemPromptSegment[]
  readonly temperature?: number
  /**
   * Per-turn extended-thinking directive; adapters map it to provider wire
   * fields. WHY a neutral ThinkingRequest type instead of extraBody: thinking
   * is expressed per provider dialect (reasoning_effort / thinking_budget on
   * OpenAI-compatible transports, thinking.budget_tokens on Anthropic), and
   * extraBody only merges into OpenAI-style payloads. A single typed,
   * provider-neutral shape keeps the resolution in runtime/thinkingLevels.ts
   * decoupled from wire formats, stays type-checked, and lets every adapter
   * translate the same directive into its own dialect.
   */
  readonly thinking?: ThinkingRequest
  readonly topK?: number
  readonly topP?: number
  readonly tools?: readonly ToolDefinition[]
  /** Plain text, or a structured part list when the turn carries image attachments. */
  readonly userMessage: MessageContent
}

export interface TurnDependencies {
  /**
   * Record that the turn stopped a runaway refusal loop.
   *
   * A function rather than an AuditEmitter so the loop keeps no dependency on
   * audit construction, sinks, or redaction; hosts pass
   * `input => emitter.emitToolLoopBlock(input)`. Absent it the guard still
   * stops the turn — the audit trail is the only thing lost.
   */
  readonly auditToolLoopBlock?: (input: ToolLoopBlockAuditInput) => void
  /** Waits for explicitly backgrounded subagents before a text-only stop. */
  readonly awaitAgentEvents?: (signal?: AbortSignal) => Promise<readonly string[]>
  /** Marks the last returned subagent event batch delivered after transcript injection succeeds. */
  readonly acknowledgeAgentEvents?: () => void
  readonly delay?: (milliseconds: number, signal?: AbortSignal) => Promise<void>
  /** Supplies passive sub-agent status lines at safe provider/tool boundaries. */
  readonly drainAgentEvents?: () => readonly string[]
  /** Supplies steering text at safe provider/tool boundaries for daemon turns. */
  readonly drainSteer?: () => readonly string[]
  /** Optional plugin hook dispatch surface; when absent the turn dispatches no hooks. */
  readonly hookRunner?: HookRunner
  readonly llm: LlmClient
  /** Monotonic millisecond clock for provider telemetry; Date.now by default. */
  readonly now?: () => number
  /**
   * Observes provider-requested tools that were not included in the model-visible
   * surface. Returning `stop` ends the current turn without executing or retrying
   * those calls.
   */
  readonly onUnconfiguredToolCalls?: (calls: readonly ToolCall[]) => 'continue' | 'stop'
  /**
   * Per-tool execution axes. Absent it every tool is treated as unsafe to run
   * concurrently, so the loop stays strictly sequential — the previous behavior.
   */
  readonly capabilities?: (
    toolName: string,
    agentId?: string,
    /**
     * The call's arguments, so a tool whose safety depends on them can answer
     * per invocation. `exec_command` is the case that matters: deciding by name
     * alone makes every shell call a concurrency barrier, which splits a round
     * of five reads plus one `git status` into three serial groups.
     */
    args?: Readonly<Record<string, unknown>>,
  ) => {
    readonly concurrencySafe: boolean
    readonly interruptBehavior: 'block' | 'cancel'
  }
  readonly permissionBroker?: PermissionBroker
  /**
   * Replace an oversized tool result with a bounded, provider-facing stand-in.
   *
   * The loop owns no storage policy, so this is injected: the host decides
   * where the full bytes go and what the model is told about recovering them.
   * Returning the input unchanged is always valid. The raw text is still
   * recorded in `state.toolExecutions` before this runs, so the objective
   * guard's evidence and the audit trail stay lossless — only what the
   * provider sees is reduced.
   */
  readonly persistToolResult?: (toolName: string, content: string) => string
  readonly policy?: ToolPolicy
  /**
   * Relieve a context overflow once per turn. Absent it the loop can only
   * report the overflow, because it has no compaction policy of its own.
   */
  readonly reduceContext?: ContextReducer
  readonly retryDelays?: readonly number[]
  /**
   * Ceiling for provider-suggested Retry-After waits (ms). Route-owned via the
   * provider registry; defaults to {@link MAX_SUGGESTED_RETRY_DELAY_MS}.
   */
  readonly maxSuggestedRetryDelayMs?: number
  /**
   * Abort a provider attempt that yields no chunk within this budget (ms), so a
   * socket held open without data stalls one attempt instead of the whole turn.
   * Defaults to {@link DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS}; non-positive or
   * non-finite values disable the watchdog.
   */
  readonly streamInactivityTimeoutMs?: number
  /** Override parser behavior for an isolated diagnostic streaming surface. */
  readonly thinkingParserFactory?: () => ThinkingStreamParser
  readonly toolExecutor?: ToolExecutor
}

/**
 * Drive a complete agent turn. Events are fully serializable so daemon, HTTP,
 * MCP, channels, and the OpenTUI client share one internal vocabulary.
 */
export async function* runTurn(
  request: TurnRequest,
  dependencies: TurnDependencies,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const state = request.state
  const permissionMode = request.permissionMode ?? DEFAULT_PERMISSION_MODE
  const maxToolTurns = request.maxToolTurns ?? DEFAULT_MAX_TOOL_TURNS
  if (maxToolTurns !== Number.POSITIVE_INFINITY && (!Number.isInteger(maxToolTurns) || maxToolTurns < 1)) {
    throw new TypeError('maxToolTurns must be a positive integer or Infinity')
  }
  const retryDelays = dependencies.retryDelays ?? DEFAULT_RETRY_DELAYS
  const maxSuggestedRetryDelayMs =
    dependencies.maxSuggestedRetryDelayMs ?? MAX_SUGGESTED_RETRY_DELAY_MS
  const streamInactivityTimeoutMs =
    dependencies.streamInactivityTimeoutMs ?? DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS
  const now = dependencies.now ?? (() => performance.now())
  const hookRunner = dependencies.hookRunner
  // One instance per turn, so a subagent running under a stricter policy burns
  // its own budget instead of the parent's.
  const denialBudget = new DenialBudget(request.maxConsecutiveDenials)
  const toolContext: ToolExecutionContext = {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    metadata: state.metadata,
  }
  if (!request.systemPromptRequestOnly) ensureSystemPrompt(state.messages, request.systemPrompt)
  state.messages.push({ role: 'user', content: request.userMessage })
  state.metadata.model = request.model
  state.turnCount += 1
  await dispatchHook(hookRunner, 'on_turn_start', {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    model: request.model,
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    turnCount: state.turnCount,
    userMessage: request.userMessage,
  })

  let inputTokens = 0
  let outputTokens = 0
  let cacheReadTokens = 0
  let cacheCreationTokens = 0
  let reasoningTokens = 0
  let reasoningUsageComplete = true
  let usageComplete = true
  let turnServiceTier: string | undefined
  let apiCallsCount = 0
  let objectiveGuardRetries = 0
  const objectiveToolExecutions: ObjectiveToolExecutionEvidence[] = []
  let toolCallsCount = 0
  let forceToolFreeSummary = false
  let latestToolRoundText: string | undefined
  let turnLimit = maxToolTurns
  const objectiveGuardLimit = objectiveGuardRetryLimit(
    request.objectiveGuardMaxRetries === undefined
      ? {}
      : { objective_guard_max_retries: request.objectiveGuardMaxRetries },
  )
  /** Record one tool result, letting tool_result_persist hooks rewrite it first. */
  const recordToolResult = async (result: ToolResult, call: ToolCall): Promise<ToolResult> => {
    const mutated = hookRunner === undefined
      ? undefined
      : hookMutation(await dispatchHook(hookRunner, 'tool_result_persist', {
        name: result.name,
        permitted: result.permitted,
        result: result.result,
        toolCallId: result.toolCallId,
      }), result.result)
    const hooked = typeof mutated === 'string' && mutated !== result.result
      ? { ...result, result: mutated }
      : result
    // Single normalization point for everything the provider will see, placed
    // after the hooks so a hook that blanks or injects into a result is
    // covered too. Persisting the same string the tool_end event carries keeps
    // the transcript and the rendered result byte-identical.
    const normalized = normalizeToolOutput(hooked.name, hooked.result)
    const recorded = normalized === hooked.result ? hooked : { ...hooked, result: normalized }
    // The provider's copy may be reduced while the human's is not. The TUI
    // renders `tool_end`, and a user watching a build scroll past is not helped
    // by a preview of it, so the returned result stays whole and only the
    // transcript message carries the stand-in.
    let providerContent: string | undefined
    try {
      providerContent = dependencies.persistToolResult?.(recorded.name, recorded.result)
    } catch {
      // Spill storage is an optimization, never part of tool correctness. If
      // persistence fails, keep the complete matched result inline rather than
      // letting the exception strand the assistant tool call without a reply.
      providerContent = undefined
    }
    appendToolResult(
      state,
      recorded,
      call,
      objectiveToolExecutions,
      providerContent === recorded.result ? undefined : providerContent,
    )
    return recorded
  }

  let consecutiveUnconfiguredOnlyRounds = 0
  /** Set when a round ends without post-processing: a terminal provider failure or a caller abort. */
  let terminalProviderFailure = false
  let stopReason: TurnStopReason = signal?.aborted ? 'aborted' : 'tool_budget_exhausted'
  /** One-shot per turn: a reducer that already ran cannot free the same tokens twice. */
  let contextReductionAttempted = false
  let outputLimitEscalations = 0
  let outputTokenOverride: number | undefined
  try {
    for (let toolTurn = 0; !signal?.aborted && toolTurn < turnLimit; toolTurn += 1) {
      appendAgentEventMessage(state, dependencies.drainAgentEvents?.())
      for (const steer of dependencies.drainSteer?.() ?? []) {
        const content = steer.trim()
        if (content) {
          state.messages.push({
            role: 'user',
            content: `[steer from user]\n${content}`,
          })
        }
      }
      // Per-attempt accumulators sit at round scope so the surviving attempt is
      // readable after the retry loop, but every attempt starts from a clean
      // slate: partial text, thinking, usage, and tool calls from a failed
      // attempt must never leak into the persisted assistant message.
      let parser = dependencies.thinkingParserFactory?.() ?? new ThinkingParser()
      let textParts: string[] = []
      let thinkingParts: string[] = []
      let thinkingSignature: string | undefined
      let roundToolCalls: readonly ToolCall[] = []
      let lastUsage: TokenUsage | undefined
      let finishReason: string | undefined
      let streamCompleted = false
      let roundStartedAt = 0
      let roundFirstOutputAt: number | undefined
      let roundCompletedAt = 0
      let textDeduper = new ToolRoundTextDeduper(latestToolRoundText)

      for (let attempt = 0; attempt <= retryDelays.length; attempt += 1) {
        parser = dependencies.thinkingParserFactory?.() ?? new ThinkingParser()
        textParts = []
        thinkingParts = []
        thinkingSignature = undefined
        roundToolCalls = []
        lastUsage = undefined
        finishReason = undefined
        textDeduper = new ToolRoundTextDeduper(latestToolRoundText)
        const attemptSignal = linkAttemptSignal(signal)
        try {
          apiCallsCount += 1
          roundStartedAt = now()
          roundFirstOutputAt = undefined
          roundCompletedAt = 0
          for await (const delta of watchProviderStream(
            dependencies.llm.stream(
              completionRequest(
                request,
                state.messages,
                forceToolFreeSummary ? [] : request.tools,
                outputTokenOverride,
              ),
              attemptSignal.controller.signal,
            ),
            streamInactivityTimeoutMs,
            attemptSignal,
          )) {
            if (roundFirstOutputAt === undefined && hasModelOutput(delta)) {
              roundFirstOutputAt = now()
            }
            const parts = processDelta(delta, parser, textParts, thinkingParts)
            for (const part of parts) {
              // Live incremental emission: each deduped part is yielded the
              // moment it arrives, so consumers render text while the provider
              // is still streaming. 95c53d6 buffered whole rounds here, which
              // silenced live output end to end; this restores the inline
              // yield. The deduper still withholds exactly one thing — the
              // not-yet-diverged replay prefix it must hold back — so retry
              // replay suppression is unchanged.
              for (const visible of textDeduper.push(part)) yield visible
            }
            if (delta.toolCalls) {
              const merged = [...roundToolCalls]
              for (const toolCall of delta.toolCalls) {
                const existing = merged.findIndex(candidate => candidate.id === toolCall.id)
                if (existing === -1) {
                  merged.push(toolCall)
                } else {
                  merged[existing] = toolCall
                }
              }
              roundToolCalls = merged
            }
            if (delta.thinkingSignature) {
              thinkingSignature = delta.thinkingSignature
            }
            if (delta.usage) {
              lastUsage = mergeUsage(lastUsage, delta.usage)
            }
            // Every adapter normalizes its own truncation token onto 'length',
            // so one assignment beside the usage merge is the whole detection.
            if (delta.finishReason) {
              finishReason = delta.finishReason
            }
          }
          for (const flushed of parser.process('')) {
            if (flushed.type === 'text') {
              textParts.push(flushed.text)
              for (const visible of textDeduper.push({ type: 'text', text: flushed.text })) yield visible
            } else {
              thinkingParts.push(flushed.text)
              for (const visible of textDeduper.push({ type: 'thinking', text: flushed.text })) yield visible
            }
          }
          roundCompletedAt = now()
          streamCompleted = true
          break
        } catch (error) {
          // A failed provider attempt may have consumed tokens without returning
          // usage. Keep the exact API-call count, but do not present later
          // successful-round usage as a complete total for the turn.
          usageComplete = false
          reasoningUsageComplete = false
          const classified = classifyError(error)
          await dispatchHook(hookRunner, 'on_error', {
            ...(request.agentId ? { agentId: request.agentId } : {}),
            attempt: attempt + 1,
            error: errorMessage(error),
            kind: classified.kind,
            ...(request.sessionId ? { sessionId: request.sessionId } : {}),
          })
          // A context overflow is not a failed try at the same request, it is a
          // request that will never fit. Reducing the history and reissuing the
          // round is a different request, so it must not spend a retry slot —
          // otherwise a single overflow burns the budget the next genuinely
          // transient failure needs.
          if (
            classified.kind === ErrorKind.CONTEXT_OVERFLOW
            && !contextReductionAttempted
            && dependencies.reduceContext !== undefined
            && signal?.aborted !== true
          ) {
            contextReductionAttempted = true
            const reduction = await reduceContextSafely(dependencies.reduceContext, state.messages, signal)
            if (reduction !== undefined && reduction.tokensFreed > 0) {
              state.messages.splice(0, state.messages.length, ...reduction.messages)
              yield {
                type: 'provider_retry',
                error: errorMessage(error),
                attempt: attempt + 1,
                maxAttempts: retryDelays.length + 1,
                delay: 0,
                final: false,
              }
              attempt -= 1
              continue
            }
          }
          // Only transient failures earn another attempt. Auth, validation,
          // configuration, and other terminal errors fail the round at once.
          const final = attempt === retryDelays.length
            || !classified.retryable
            || signal?.aborted === true
          const suggestedDelay = classified.suggestedBackoffSeconds === undefined
            ? 0
            : Math.min(maxSuggestedRetryDelayMs, classified.suggestedBackoffSeconds * 1_000)
          const delay = final ? 0 : Math.max(retryDelays[attempt] ?? 0, suggestedDelay)
          yield {
            type: 'provider_retry',
            error: errorMessage(error),
            attempt: attempt + 1,
            maxAttempts: retryDelays.length + 1,
            delay,
            final,
          }
          if (final) {
            // Terminal provider failure: surface the error as an explicit event
            // and stop the turn. Persisting it as ordinary assistant text
            // polluted durable history with `[Error: ...]` messages, and the
            // objective guard could then re-call a terminally failed provider
            // (auth/config) until its retry limit.
            //
            // Cancellation is not a provider failure (bugfix): when the caller's
            // signal aborted mid-stream the attempt error is just the abort
            // surfacing, so emitting a synthetic `[Error: ...]` text event would
            // fabricate model output and reporting `provider_failed` would map
            // a user escape to a hard subagent failure downstream. The abort
            // wins even if the raced error classified as a context overflow —
            // that classification is preserved for genuine overflows below.
            if (signal?.aborted) {
              terminalProviderFailure = true
              stopReason = 'aborted'
              break
            }
            const overflow = classified.kind === ErrorKind.CONTEXT_OVERFLOW
            yield {
              type: 'text',
              text: overflow ? CONTEXT_OVERFLOW_STOP_TEXT : `[Error: ${errorMessage(error)}]`,
            }
            terminalProviderFailure = true
            stopReason = overflow ? 'context_overflow' : 'provider_failed'
            break
          }
          await (dependencies.delay ?? defaultDelay)(delay, signal)
        } finally {
          attemptSignal.release()
        }
      }

      if (terminalProviderFailure) {
        break
      }
      if (!streamCompleted) {
        throw new Error('LLM stream exited without completion or error')
      }

      if (lastUsage === undefined) {
        usageComplete = false
        reasoningUsageComplete = false
      } else if (lastUsage.reasoningTokens === undefined) {
        reasoningUsageComplete = false
      }
      accumulateUsage(lastUsage, state, (usage) => {
        inputTokens += usage.inputTokens
        outputTokens += usage.outputTokens
        cacheReadTokens += usage.cacheReadTokens ?? 0
        cacheCreationTokens += usage.cacheCreationTokens ?? 0
        reasoningTokens += usage.reasoningTokens ?? 0
      })
      if (lastUsage?.serviceTier) {
        turnServiceTier = lastUsage.serviceTier
      }
      if (lastUsage) {
        // Emitted per provider round rather than only at turn_done: a turn that
        // runs for minutes across many rounds would otherwise report nothing
        // until it ended.
        const decodeMilliseconds = roundFirstOutputAt === undefined
          ? 0
          : Math.max(0, roundCompletedAt - roundFirstOutputAt)
        const cacheInput = lastUsage.inputTokens + (lastUsage.cacheReadTokens ?? 0)
        yield {
          type: 'usage_update',
          model: request.model,
          usage: lastUsage,
          cumulative: {
            inputTokens,
            outputTokens,
            ...(cacheReadTokens ? { cacheReadTokens } : {}),
            ...(cacheCreationTokens ? { cacheCreationTokens } : {}),
            ...(reasoningTokens ? { reasoningTokens } : {}),
          },
          durationMs: Math.max(0, roundCompletedAt - roundStartedAt),
          ...(roundFirstOutputAt === undefined
            ? {}
            : { ttftMs: Math.max(0, roundFirstOutputAt - roundStartedAt) }),
          ...(decodeMilliseconds > 0 && lastUsage.outputTokens > 0
            ? { tokensPerSecond: lastUsage.outputTokens * 1_000 / decodeMilliseconds }
            : {}),
          ...(lastUsage.cacheReadTokens !== undefined && cacheInput > 0
            ? { cacheHitRate: lastUsage.cacheReadTokens / cacheInput }
            : {}),
        }
      }

      const rawAssistantText = textParts.join('')
      // Flush whatever the deduper was still holding at round end: a replay
      // prefix that never diverged is emitted here, stripped of the overlap it
      // suppressed. Everything else already went out inline in the delta loop.
      const deduplication = textDeduper.finish()
      for (const visible of deduplication.events) yield visible
      const assistantText = rawAssistantText.slice(deduplication.suppressedPrefix)
      const providerToolCalls = roundToolCalls
      const visibleTools = forceToolFreeSummary ? [] : request.tools
      const { exposed, unconfigured } = partitionToolCalls(providerToolCalls, visibleTools)
      roundToolCalls = exposed
      const assistant: ChatMessage = {
        role: 'assistant',
        content: assistantText,
        ...(thinkingParts.length ? { thinking: thinkingParts.join('') } : {}),
        ...(thinkingSignature ? { thinking_signature: thinkingSignature } : {}),
        ...(providerToolCalls.length ? { tool_calls: providerToolCalls } : {}),
      }
      // Several providers reject an assistant message with no content at all.
      // A round that produced no text, no thinking, and no tool calls leaves
      // nothing worth persisting, so skip the empty assistant message and its
      // placeholder thinking entry.
      const hasAssistantContent =
        assistantText !== '' || thinkingParts.length > 0 || providerToolCalls.length > 0
      if (hasAssistantContent) {
        state.messages.push(assistant)
        if (thinkingParts.length) {
          state.thinkingContent.push(thinkingParts.join(''))
        } else {
          state.thinkingContent.push('')
        }
      }
      if (providerToolCalls.length && assistantText) {
        latestToolRoundText = assistantText
      }

      // A truncated round is a half-finished thought. Persisted unexamined it
      // reads to the objective guard, and to the next turn, as a completed
      // answer, so the loop resumes it before anything downstream sees it.
      //
      // A truncated round that still asked for tools keeps its ordinary path:
      // the model continues on the next round regardless, and diverting here
      // would leave the persisted tool_use blocks without the tool_result
      // blocks Anthropic requires them to be paired with.
      if (finishReason === 'length' && providerToolCalls.length === 0) {
        if (outputLimitEscalations >= MAX_OUTPUT_LIMIT_ESCALATIONS) {
          yield {
            type: 'text',
            text: renderIntervention({
              kind: 'stop-guard',
              attempts: MAX_OUTPUT_LIMIT_ESCALATIONS,
              variant: 'output-limit-escalated',
            }),
          }
          stopReason = 'output_limit'
          break
        }
        outputLimitEscalations += 1
        // A truncation still consumed a provider call, but it produced no tool
        // work, so charging it to the tool budget would shorten the turn the
        // model is trying to finish. Same bookkeeping as an appended agent event.
        if (toolTurn + 1 >= turnLimit) turnLimit += 1
        if (outputLimitEscalations === 1 && request.maxTokens === undefined) {
          // First truncation with no caller-pinned ceiling: the provider default
          // was simply too small. Regenerate the round with a real window rather
          // than asking the model to continue from a sentence it cut in half.
          if (hasAssistantContent) {
            state.messages.pop()
            state.thinkingContent.pop()
          }
          // Deliberately not fed to the cross-round text deduper: the popped
          // text is gone from history, so suppressing the regenerated prefix
          // would drop it from the transcript as well as from the stream.
          //
          // Live-emission note (bugfix reverting the 95c53d6 round buffering):
          // because deduped parts now stream inline, this discarded round's
          // truncated text was already yielded while the provider was still
          // streaming, before `finishReason: 'length'` revealed it as
          // regenerable. Text events have no supersession mechanism — unlike
          // screenshot tool results, which supersedeScreenshotToolResults
          // collapses in history — so an emitted prefix cannot be retracted,
          // and buffering every round until its finish reason arrives (the
          // 95c53d6 approach) would silence all live streaming to keep only
          // this rare window clean. The accepted tradeoff: consumers briefly
          // see the severed prefix before the wider-window regeneration below;
          // the transcript of record stays correct because the assistant
          // message was popped here.
          outputTokenOverride = OUTPUT_LIMIT_RETRY_MAX_TOKENS
          continue
        }
        state.messages.push({ role: 'user', content: OUTPUT_LIMIT_RESUME_REMINDER })
        yield { type: 'text', text: '\n[Output limit reached. Resuming.]' }
        continue
      }

      if (unconfigured.length) {
        toolCallsCount += unconfigured.length
        for (const call of unconfigured) {
          const result = await recordToolResult(unconfiguredToolResult(call), call)
          yield { type: 'tool_end', result }
        }
        if (dependencies.onUnconfiguredToolCalls?.(unconfigured) === 'stop') {
          stopReason = 'unconfigured_tools'
          break
        }
        if (forceToolFreeSummary) {
          stopReason = 'unconfigured_tools'
          break
        }
        if (!roundToolCalls.length) {
          consecutiveUnconfiguredOnlyRounds += 1
          if (consecutiveUnconfiguredOnlyRounds >= MAX_UNCONFIGURED_ONLY_ROUNDS) {
            // The model keeps requesting tools outside the configured surface
            // and no stop hook intervened. Without a cap this loops one
            // provider call per round forever (maxToolTurns is unbounded).
            yield {
              type: 'text',
              text: renderIntervention({
                kind: 'stop-guard',
                attempts: consecutiveUnconfiguredOnlyRounds,
                variant: 'unconfigured-tools-loop',
              }),
            }
            stopReason = 'unconfigured_tools'
            break
          }
          continue
        }
      }
      if (roundToolCalls.length > 0 || unconfigured.length === 0) {
        consecutiveUnconfiguredOnlyRounds = 0
      }

      if (!roundToolCalls.length) {
        const agentEvents = await dependencies.awaitAgentEvents?.(signal) ?? []
        const appendedAgentEvents = appendAgentEventMessage(state, agentEvents)
        // A coordinator can acknowledge returned snapshots before cancellation
        // becomes observable here. Persist the delivered results first so an
        // interrupted parent either synthesizes them now or receives them from
        // its durable history on the next turn.
        if (signal?.aborted) {
          stopReason = 'aborted'
          break
        }
        if (appendedAgentEvents) {
          dependencies.acknowledgeAgentEvents?.()
          if (toolTurn + 1 >= turnLimit) turnLimit += 1
          continue
        }
        for (const steer of dependencies.drainSteer?.() ?? []) {
          const content = steer.trim()
          if (!content) continue
          state.messages.push({
            role: 'user',
            content: `[steer from user saved for next turn]\n${content}`,
          })
          yield {
            type: 'text',
            text: renderIntervention({ kind: 'steer-note', content }),
          }
        }
        // A session holding a live goal is driven by that goal, not by reading
        // its own prose: the turn ends here and the round driver decides at
        // idle whether the durable goal wants another round. Phrase matching
        // remains only for objective mode without a goal, where there is no
        // typed state to consult.
        const liveGoal = request.sessionId
          ? getGoal(state.metadata, request.sessionId)
          : undefined
        const objectiveDecision = liveGoal
          ? { shouldContinue: false, reason: '', reminder: '' }
          : inspectObjectiveResponse(assistantText, {
              evidence: { toolExecutions: objectiveToolExecutions },
              mode: currentInteractionMode(state, request.interactionMode),
            })
        if (!objectiveDecision.shouldContinue) {
          // The guard states its own grounds when it has any; today an accepted
          // answer carries an empty reason, so plain completion is the fallback.
          stopReason = objectiveDecision.reason.trim() ? 'objective_verified' : 'completed'
          break
        }
        objectiveGuardRetries += 1
        if (objectiveGuardRetries > objectiveGuardLimit) {
          yield {
            type: 'text',
            text: renderIntervention({
              kind: 'stop-guard',
              attempts: objectiveGuardLimit,
              reason: objectiveDecision.reason,
              variant: 'objective-guard-exhausted',
            }),
          }
          stopReason = 'objective_guard_exhausted'
          break
        }
        state.messages.push({ role: 'user', content: objectiveDecision.reminder })
        yield {
          type: 'text',
          text:
            '\n[Objective gate: ' + objectiveDecision.reason + '. Continuing.]',
        }
        continue
      }

      toolCallsCount += roundToolCalls.length

      // PHASE A — decide, sequentially, inside the generator.
      //
      // Permission has to stay here: it yields a `permission_request` and awaits
      // the broker's answer, and a generator cannot yield from inside a
      // Promise.all callback. Keeping the whole interactive half single-threaded
      // also preserves the abort races handled below, which are the subtle part.
      // Nothing is recorded yet — the transcript is written in phase B, in
      // model-emitted order, so a parallel round replays byte-identically.
      const decisions: ToolDecision[] = []
      for (const call of roundToolCalls) {
        if (!call) continue
        if (signal?.aborted) {
          decisions.push({ call, kind: 'cancelled' })
          continue
        }
        const beforeResult = await dispatchHook(hookRunner, 'before_tool_call', {
          ...(request.agentId ? { agentId: request.agentId } : {}),
          arguments: call.function.arguments,
          name: call.function.name,
          ...(request.sessionId ? { sessionId: request.sessionId } : {}),
          toolCallId: call.id,
        })
        const effectiveCall = applyToolArgumentsMutation(
          call,
          hookMutation(beforeResult, call.function.arguments),
        )
        const permission = permissionDisposition(
          effectiveCall,
          permissionMode,
          dependencies.policy,
          request.agentId,
        )
        if (permission === 'deny') {
          decisions.push({ call, kind: 'denied', reason: 'policy_denied' })
          continue
        }
        if (permission === 'prompt') {
          const permissionRequest = createPermissionRequest(effectiveCall)
          yield { type: 'permission_request', request: permissionRequest }
          const decision = (await dependencies.permissionBroker?.request(permissionRequest, signal)) ?? 'reject'
          // Injected brokers are allowed to resolve asynchronously. Cancellation
          // may land while a prompt is open, so an approval that races the abort
          // must not start a privileged tool with an already-aborted signal.
          if (signal?.aborted) {
            decisions.push({ call, kind: 'cancelled' })
            continue
          }
          if (decision === 'reject') {
            decisions.push({ call, kind: 'denied', reason: 'permission_rejected' })
            continue
          }
        }
        if (hookRunner !== undefined) {
          const extensionPermission = await resolveToolPermission(hookRunner, {
            arguments: effectiveCall.function.arguments,
            toolName: effectiveCall.function.name,
          })
          if (!extensionPermission.allowed) {
            decisions.push({
              call,
              detail: extensionPermission.reason,
              kind: 'denied',
              reason: 'policy_denied',
            })
            continue
          }
        }
        decisions.push({ call, effectiveCall, kind: 'allowed' })
      }

      // PHASE B — execute in maximal runs of consecutive concurrency-safe calls,
      // then emit strictly in model-emitted order regardless of completion order.
      for (const group of groupToolDecisions(
        decisions.map(decision => decision.kind === 'allowed'
          ? { ...decision, call: decision.effectiveCall }
          : decision),
        call => capabilitiesFor(dependencies, request, call),
      )) {
        if (signal?.aborted && group.some(decision => decision.kind === 'allowed')) {
          for (const decision of group) {
            denialBudget.record('cancelled', decision.call.function.name)
            const result = await recordToolResult(cancelledToolResult(decision.call), decision.call)
            yield { type: 'tool_end', result }
          }
          continue
        }
        // Starts are announced before any work begins, in model-emitted order.
        // Consumers rely on this window: the daemon interleaves a subagent's
        // live events between a tool's start and its result, and the child
        // checkpointer commits a tool call while the tool is still running.
        // Emitting starts after Promise.all would collapse that window to zero.
        for (const decision of group) {
          if (decision.kind === 'allowed') yield { type: 'tool_start', call: decision.effectiveCall }
        }
        const outcomes = await Promise.all(group.map(async (decision, member) => {
          if (decision.kind !== 'allowed') return undefined
          const effectiveCall = decision.effectiveCall
          // A parallel member gets its own metadata object and its writes are
          // replayed in block order afterwards. `metadata` is one shared mutable
          // record handed to every handler, safe until now only because
          // execution was serial. A lone call keeps the shared object so a
          // single-tool round behaves exactly as it always has.
          const memberContext = group.length === 1
            ? toolContext
            : { ...toolContext, metadata: { ...toolContext.metadata } }
          const memberSignal = capabilitiesFor(dependencies, request, effectiveCall)
            .interruptBehavior === 'block'
            ? undefined
            : signal
          const startedAt = performance.now()
          try {
            const output = dependencies.toolExecutor
              ? await dependencies.toolExecutor.execute(effectiveCall, memberContext, memberSignal)
              : `Tool ${effectiveCall.function.name} is unavailable.`
            return { context: memberContext, member, output, startedAt }
          } catch (error) {
            return { context: memberContext, error, member, startedAt }
          }
        }))
        if (group.length > 1) {
          // Ordered merge, so two members writing the same key resolve the way
          // a serial round would have: the later block wins.
          for (const outcome of outcomes) {
            if (outcome) Object.assign(toolContext.metadata, outcome.context.metadata)
          }
        }
        for (const [member, decision] of group.entries()) {
          if (decision.kind === 'cancelled') {
            denialBudget.record('cancelled', decision.call.function.name)
            const result = await recordToolResult(cancelledToolResult(decision.call), decision.call)
            yield { type: 'tool_end', result }
            continue
          }
          if (decision.kind === 'denied') {
            denialBudget.record(decision.reason, decision.call.function.name)
            const denied = deniedToolResult(decision.call)
            const result = await recordToolResult(
              decision.detail === undefined
                ? denied
                : { ...denied, result: `${denied.result} ${decision.detail}` },
              decision.call,
            )
            yield { type: 'tool_end', result }
            continue
          }
          const effectiveCall = decision.effectiveCall
          const outcome = outcomes[member]
          if (!outcome || outcome.error !== undefined) {
            const result = await recordToolResult(
              failedToolResult(effectiveCall, outcome?.error, performance.now() - (outcome?.startedAt ?? 0)),
              effectiveCall,
            )
            yield { type: 'tool_end', result }
            continue
          }
          // A tool that actually ran means the model found a permitted route,
          // so any refusals before it were search rather than a denial loop.
          if (dependencies.toolExecutor) denialBudget.reset()
          let result: ToolResult = {
            name: effectiveCall.function.name,
            // Guarded here as well as on persist so after_tool_call hooks and the
            // tool_end event never see the bare '' that serializeToolResult
            // returns for a tool with nothing to report.
            result: isEffectivelyEmpty(outcome.output ?? '')
              ? emptyToolResult(effectiveCall.function.name)
              : outcome.output ?? '',
            permitted: true,
            toolCallId: effectiveCall.id,
            durationMs: performance.now() - outcome.startedAt,
          }
          const afterResult = await dispatchHook(hookRunner, 'after_tool_call', {
            ...(request.agentId ? { agentId: request.agentId } : {}),
            arguments: effectiveCall.function.arguments,
            name: effectiveCall.function.name,
            result: result.result,
            ...(request.sessionId ? { sessionId: request.sessionId } : {}),
            toolCallId: effectiveCall.id,
          })
          const mutatedOutput = hookMutation(afterResult, result.result)
          if (typeof mutatedOutput === 'string' && mutatedOutput !== result.result) {
            result = { ...result, result: mutatedOutput }
          }
          const recorded = await recordToolResult(result, effectiveCall)
          yield { type: 'tool_end', result: recorded }
        }
      }
      if (signal?.aborted) {
        stopReason = 'aborted'
        break
      }
      // Checked after the whole round rather than at the refusal site: every
      // tool_use block the assistant message carries must still get its
      // tool_result, or the next turn replays a history Anthropic rejects.
      // Waiting also lets a later permitted call in the same round clear the
      // streak, which is the correct reading of "consecutive".
      if (denialBudget.exhausted) {
        yield { type: 'text', text: denialBudgetStopText(denialBudget) }
        reportDenialLoop(dependencies, request, denialBudget)
        stopReason = 'tool_budget_exhausted'
        break
      }
      let needsFinalization = false
      if (toolTurn + 1 >= turnLimit) {
        const agentEvents = await dependencies.awaitAgentEvents?.(signal) ?? []
        const appendedAgentEvents = appendAgentEventMessage(state, agentEvents)
        if (signal?.aborted) {
          stopReason = 'aborted'
          break
        }
        if (appendedAgentEvents) {
          dependencies.acknowledgeAgentEvents?.()
          forceToolFreeSummary = true
          needsFinalization = true
        }
        if (needsFinalization) turnLimit += 1
      }
    }
  } catch (error) {
    // Rejections outside the provider-attempt handler — an abort during
    // retry backoff, a permission broker failure, a subagent join failure —
    // must not skip the turn epilogue. Surface the failure as an explicit
    // final error event so the turn still ends with exactly one turn_done.
    const classified = classifyError(error)
    await dispatchHook(hookRunner, 'on_error', {
      ...(request.agentId ? { agentId: request.agentId } : {}),
      attempt: retryDelays.length + 1,
      error: errorMessage(error),
      kind: classified.kind,
      ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    })
    // An abort that lands here — typically the injected delay rejecting with
    // the signal's reason during retry backoff — is a user cancellation, not a
    // turn failure (bugfix): report `aborted` and skip the synthetic
    // `[Error: ...]` text event, mirroring the in-stream abort branch. The
    // final provider_retry event still carries the abort reason for
    // observability.
    const cancelled = signal?.aborted === true || isAbortLikeError(error)
    yield {
      type: 'provider_retry',
      error: errorMessage(error),
      attempt: retryDelays.length + 1,
      maxAttempts: retryDelays.length + 1,
      delay: 0,
      final: true,
    }
    if (cancelled) {
      stopReason = 'aborted'
    } else {
      yield { type: 'text', text: `[Error: ${errorMessage(error)}]` }
      stopReason = 'turn_failed'
    }
  } finally {
    state.totalApiCalls += apiCallsCount
    state.usageComplete &&= usageComplete
  }
  await dispatchHook(hookRunner, 'on_turn_end', {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    apiCallsCount,
    model: request.model,
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    toolCallsCount,
    turnCount: state.turnCount,
    usageComplete,
  })
  // Codex's proxy echoes `default` even when it served the requested flex or
  // priority tier, so the requested tier wins for pricing whenever the served
  // report is absent or the meaningless default (pi-ai's reconciliation).
  const pricedServiceTier = (turnServiceTier === undefined || turnServiceTier === 'default')
    && (request.serviceTier === 'flex' || request.serviceTier === 'priority')
    ? request.serviceTier
    : turnServiceTier
  yield {
    type: 'turn_done',
    apiCallsCount,
    model: request.model,
    reason: stopReason,
    toolCallsCount,
    usageComplete,
    usage: {
      inputTokens,
      outputTokens,
      ...(cacheReadTokens ? { cacheReadTokens } : {}),
      ...(cacheCreationTokens ? { cacheCreationTokens } : {}),
      ...(usageComplete && reasoningUsageComplete ? { reasoningTokens } : {}),
      ...(pricedServiceTier === undefined ? {} : { serviceTier: pricedServiceTier }),
    },
  }
}

function currentInteractionMode(state: AgentState, initialMode: string | undefined): string {
  const liveMode = state.metadata.interaction_mode
  return typeof liveMode === 'string' && liveMode.trim() ? liveMode : initialMode ?? 'code'
}

type IncrementalTextEvent = Extract<StreamEvent, { readonly type: 'text' | 'thinking' }>

/**
 * Hold only the prefix that could be an exact replay of the latest assistant
 * text attached to a tool round. As soon as it differs, normal streaming
 * resumes. Exact repeats stay suppressed so live output and saved history show
 * the same assistant text once.
 */
class ToolRoundTextDeduper {
  private candidate = ''
  private diverged = false
  private readonly pending: IncrementalTextEvent[] = []
  private suppressedPrefix = 0
  /** Tail of the previous tool-round text, capped to the overlap window. */
  private readonly tail: string
  /** True when the previous text is shorter than the partial-overlap floor. */
  private readonly shortPrevious: boolean

  constructor(private readonly previous: string | undefined) {
    this.tail = previous === undefined ? '' : previous.slice(-MAX_TEXT_OVERLAP_WINDOW)
    this.shortPrevious = previous !== undefined && previous.length < MIN_PARTIAL_TEXT_OVERLAP
  }

  push(event: IncrementalTextEvent): readonly IncrementalTextEvent[] {
    if (this.previous === undefined || this.diverged) {
      return [event]
    }
    // Thinking shares the pending buffer with text so thinking that arrives
    // after held-back text is never emitted before it.
    this.pending.push(event)
    if (event.type === 'text') {
      this.candidate += event.text
    }
    if (this.isPossibleReplay()) {
      return []
    }
    this.diverged = true
    this.suppressedPrefix = this.computeOverlap()
    return stripTextEventPrefix(this.pending.splice(0), this.suppressedPrefix)
  }

  finish(): { readonly events: readonly IncrementalTextEvent[]; readonly suppressedPrefix: number } {
    if (!this.diverged && this.previous !== undefined) {
      this.suppressedPrefix = this.computeOverlap()
    }
    return {
      events: stripTextEventPrefix(this.pending.splice(0), this.suppressedPrefix),
      suppressedPrefix: this.suppressedPrefix,
    }
  }

  /** True while the streamed text can still be an exact replay of an eligible previous-text suffix. */
  private isPossibleReplay(): boolean {
    if (this.candidate.length === 0) {
      return true
    }
    if (this.shortPrevious) {
      // Only the full previous text is eligible below the partial-overlap floor.
      return this.candidate.length <= this.tail.length && this.tail.startsWith(this.candidate)
    }
    // The candidate must be a prefix of some previous-text suffix whose length
    // is at least max(MIN_PARTIAL_TEXT_OVERLAP, candidate.length). In tail
    // coordinates that is an occurrence starting at or before
    // tail.length - max(MIN_PARTIAL_TEXT_OVERLAP, candidate.length).
    const start = this.tail.indexOf(this.candidate)
    return start !== -1
      && start + Math.max(MIN_PARTIAL_TEXT_OVERLAP, this.candidate.length) <= this.tail.length
  }

  /** Longest eligible previous-text suffix that prefixes the streamed candidate. */
  private computeOverlap(): number {
    if (this.candidate.length === 0) {
      return 0
    }
    if (this.shortPrevious) {
      return this.candidate.startsWith(this.tail) ? this.tail.length : 0
    }
    const match = longestPrefixSuffixMatch(this.candidate, this.tail)
    return match >= MIN_PARTIAL_TEXT_OVERLAP ? match : 0
  }
}

const MIN_PARTIAL_TEXT_OVERLAP = 12
/**
 * Upper bound on the cross-round overlap window. The deduper inspects only
 * the tail of the previous tool-round text, so a push is a bounded suffix
 * check and the final longest-suffix-prefix scan is a single KMP failure
 * computation over this window — never per-character overlap entries or
 * variadic Math.max over huge arrays.
 */
const MAX_TEXT_OVERLAP_WINDOW = 16_384

/**
 * Longest prefix of `prefix` that is also a suffix of `text`, computed with a
 * KMP failure function in O(prefix.length + text.length) without per-length
 * slicing.
 */
function longestPrefixSuffixMatch(prefix: string, text: string): number {
  if (prefix.length === 0 || text.length === 0) {
    return 0
  }
  // Only the last prefix.length characters of text can participate in a
  // suffix match, so the scan window stays bounded by the candidate length.
  const window = text.length > prefix.length ? text.slice(-prefix.length) : text
  const combined = `${prefix}\u0000${window}`
  const failure = new Int32Array(combined.length)
  for (let index = 1; index < combined.length; index += 1) {
    let length = failure[index - 1] ?? 0
    while (length > 0 && combined[index] !== combined[length]) {
      length = failure[length - 1] ?? 0
    }
    if (combined[index] === combined[length]) {
      length += 1
    }
    failure[index] = length
  }
  return Math.min(failure[combined.length - 1] ?? 0, prefix.length, window.length)
}

function stripTextEventPrefix(
  events: readonly IncrementalTextEvent[],
  count: number,
): IncrementalTextEvent[] {
  let remaining = count
  const visible: IncrementalTextEvent[] = []
  for (const event of events) {
    if (event.type === 'thinking' || remaining === 0) {
      visible.push(event)
      continue
    }
    if (event.text.length <= remaining) {
      remaining -= event.text.length
      continue
    }
    visible.push({ type: 'text', text: event.text.slice(remaining) })
    remaining = 0
  }
  return visible
}

/**
 * Push drained sub-agent status through the shared injection seam.
 *
 * First consumer of {@link appendInjection}: the visible text is unchanged, but
 * the block is now counted, character-capped and deduplicated against the rest
 * of the turn. A skipped injection reports false, which is the same answer the
 * loop already handles for "nothing to say" — the round finalizes instead of
 * spending another provider call on a reminder the model has already seen.
 */
function appendAgentEventMessage(
  state: AgentState,
  events: readonly string[] | undefined,
): boolean {
  if (!events?.length) return false
  return appendInjection(state.messages, { events, kind: 'agent_events' }).status === 'ready'
}

/** Audit an exhausted denial budget without letting a failing sink break the turn. */
function reportDenialLoop(
  dependencies: TurnDependencies,
  request: TurnRequest,
  budget: DenialBudget,
): void {
  if (dependencies.auditToolLoopBlock === undefined) return
  try {
    dependencies.auditToolLoopBlock({
      ...(request.agentId ? { agentId: request.agentId } : {}),
      count: budget.used,
      pattern: DENIAL_LOOP_PATTERN,
      ...(request.sessionId ? { sessionId: request.sessionId } : {}),
      toolName: budget.lastDenial?.toolName ?? '',
    })
  } catch {
    // An audit sink is observability, never a turn-ending dependency.
  }
}

function completionRequest(
  request: TurnRequest,
  messages: readonly ChatMessage[],
  tools: readonly ToolDefinition[] | undefined,
  maxTokensOverride?: number,
) {
  // The override only exists when the caller pinned nothing, so a user-chosen
  // ceiling is never silently widened by truncation recovery.
  const maxTokens = request.maxTokens ?? maxTokensOverride
  return {
    model: request.model,
    messages: request.systemPromptRequestOnly && request.systemPrompt
      ? [{ role: 'system' as const, content: request.systemPrompt }, ...messages]
      : [...messages],
    ...(tools?.length ? { tools } : {}),
    ...(maxTokens !== undefined
      ? { maxTokens }
      : {}),
    ...(request.temperature !== undefined
      ? { temperature: request.temperature }
      : {}),
    ...(request.topK !== undefined ? { topK: request.topK } : {}),
    ...(request.systemSegments?.length ? { systemSegments: request.systemSegments } : {}),
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    ...(request.serviceTier !== undefined ? { serviceTier: request.serviceTier } : {}),
    ...(request.querySource ? { querySource: request.querySource } : {}),
    ...(request.topP !== undefined ? { topP: request.topP } : {}),
    // Passthrough, not translation: the resolved per-turn directive travels
    // untouched from the TurnRequest to the CompletionRequest so the owning
    // provider adapter (client.ts addSampling, anthropic.ts) is the single
    // place that maps it onto wire-specific fields.
    ...(request.thinking !== undefined ? { thinking: request.thinking } : {}),
  }
}

/**
 * Run an injected reducer without letting it replace the failure it was called
 * about: a reducer that throws must still leave the turn reporting the overflow
 * and its remedy, not the reducer's own internal error.
 */
async function reduceContextSafely(
  reduce: ContextReducer,
  messages: readonly ChatMessage[],
  signal: AbortSignal | undefined,
): Promise<ContextReduction | undefined> {
  try {
    return await reduce(messages, signal)
  } catch {
    return undefined
  }
}

function ensureSystemPrompt(
  messages: ChatMessage[],
  systemPrompt: string | undefined,
): void {
  if (!systemPrompt) return
  const index = messages.findIndex((message) => message.role === 'system')
  if (index < 0) {
    messages.unshift({ role: 'system', content: systemPrompt })
    return
  }
  messages[index] = { role: 'system', content: systemPrompt }
}

/** Separate model-visible calls from provider calls outside the configured surface. */
function partitionToolCalls(
  calls: readonly ToolCall[],
  tools: readonly ToolDefinition[] | undefined,
): { readonly exposed: readonly ToolCall[]; readonly unconfigured: readonly ToolCall[] } {
  if (!calls.length) return { exposed: calls, unconfigured: calls }
  const exposedNames = new Set((tools ?? []).map((tool) => tool.function.name))
  const exposed: ToolCall[] = []
  const unconfigured: ToolCall[] = []
  for (const call of calls) {
    if (exposedNames.has(call.function.name)) {
      exposed.push(call)
    } else {
      unconfigured.push(call)
    }
  }
  return { exposed, unconfigured }
}

function hasModelOutput(delta: LlmDelta): boolean {
  return Boolean(delta.content || delta.thinking || delta.toolCalls?.length)
}

function processDelta(
  delta: LlmDelta,
  parser: ThinkingStreamParser,
  textParts: string[],
  thinkingParts: string[],
): IncrementalTextEvent[] {
  const events: IncrementalTextEvent[] = []
  if (delta.thinking) {
    thinkingParts.push(delta.thinking)
    events.push({ type: 'thinking', text: delta.thinking })
  }
  if (delta.content) {
    for (const part of parser.process(delta.content)) {
      if (part.type === 'text') {
        textParts.push(part.text)
        events.push({ type: 'text', text: part.text })
      } else {
        thinkingParts.push(part.text)
        events.push({ type: 'thinking', text: part.text })
      }
    }
  }
  return events
}

function accumulateUsage(
  usage: TokenUsage | undefined,
  state: AgentState,
  receive: (usage: TokenUsage) => void,
): void {
  if (!usage) {
    return
  }
  state.totalInputTokens += usage.inputTokens
  state.totalOutputTokens += usage.outputTokens
  state.totalCacheReadTokens += usage.cacheReadTokens ?? 0
  state.totalCacheCreationTokens += usage.cacheCreationTokens ?? 0
  receive(usage)
}

function mergeUsage(
  existing: TokenUsage | undefined,
  incoming: TokenUsage,
): TokenUsage {
  if (!existing) {
    return incoming
  }
  // Nullish coalescing, not `||`: a legitimate 0 reading (for example a fully
  // cached round) must not be masked by the earlier reading.
  const cacheReadTokens = incoming.cacheReadTokens ?? existing.cacheReadTokens
  const cacheCreationTokens = incoming.cacheCreationTokens ?? existing.cacheCreationTokens
  const reasoningTokens = incoming.reasoningTokens ?? existing.reasoningTokens
  const serviceTier = incoming.serviceTier ?? existing.serviceTier
  return {
    inputTokens: incoming.inputTokens ?? existing.inputTokens,
    outputTokens: incoming.outputTokens ?? existing.outputTokens,
    ...(cacheReadTokens !== undefined ? { cacheReadTokens } : {}),
    ...(cacheCreationTokens !== undefined ? { cacheCreationTokens } : {}),
    ...(reasoningTokens !== undefined ? { reasoningTokens } : {}),
    ...(serviceTier !== undefined ? { serviceTier } : {}),
  }
}

function createPermissionRequest(call: ToolCall): PermissionRequest {
  return {
    requestId: `${call.id}:${crypto.randomUUID()}`,
    toolCall: call,
    inputs: call.function.arguments,
    description: permissionDescription(call),
  }
}

function appendToolResult(
  state: AgentState,
  result: ToolResult,
  call: ToolCall,
  objectiveToolExecutions: ObjectiveToolExecutionEvidence[],
  /**
   * Bounded stand-in for the transcript when the full result is too large to
   * carry in context. The execution records below deliberately keep the whole
   * text: the objective guard reasons over that evidence, and reducing it there
   * would let a verified claim look unverified purely because its output was big.
   */
  providerContent?: string,
): void {
  state.messages.push({
    role: 'tool',
    content: providerContent ?? result.result,
    name: result.name,
    tool_call_id: result.toolCallId,
    // Structured load point for provider-native deferred tool protocols
    // (pi-ai addedToolNames); derived from the same marker the regex scan in
    // revealedToolNames recovers, so compaction stays consistent. Errored
    // searches never serialize `loaded_tool` keys, so no error guard is needed.
    ...(toolSearchLoadedNames(result.result).length
      ? { added_tool_names: toolSearchLoadedNames(result.result) }
      : {}),
  })
  // A fresh screenshot supersedes every earlier one: screenshots are the
  // largest payloads that ever enter the transcript, so only the latest
  // capture stays inline and the rest collapse to compact markers.
  if (isScreenshotToolResult(result.result)) {
    supersedeScreenshotToolResults(state.messages)
  }
  const execution = {
    name: result.name,
    inputs: call.function.arguments,
    result: result.result,
    permitted: result.permitted,
    toolCallId: result.toolCallId,
    durationMs: result.durationMs,
  }
  state.toolExecutions.push(execution)
  objectiveToolExecutions.push(execution)
  // Recomputed per result rather than stored as a list: the goal tools need to
  // know whether this turn has proved anything, and putting the executions
  // themselves on metadata would write every tool result into the session file.
}

function deniedToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: deniedResult(call),
    permitted: false,
    toolCallId: call.id,
    durationMs: 0,
  }
}

function cancelledToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: 'Cancelled before execution.',
    permitted: false,
    toolCallId: call.id,
    durationMs: 0,
  }
}

function failedToolResult(
  call: ToolCall,
  error: unknown,
  durationMs = 0,
): ToolResult {
  return {
    name: call.function.name,
    result: `Tool execution failed: ${errorMessage(error)}`,
    permitted: true,
    toolCallId: call.id,
    durationMs,
  }
}

function unconfiguredToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: `Tool execution failed: ${call.function.name} was not configured for this turn.`,
    permitted: true,
    toolCallId: call.id,
    durationMs: 0,
  }
}

/**
 * Recognize cancellation-shaped rejections by name: a DOMException AbortError
 * from an aborted fetch/stream, and InterruptToken's InterruptRequestedError.
 * Used only where the caller's signal may not have flipped yet (or IS the
 * rejection reason), so genuine provider failures never match.
 */
function isAbortLikeError(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  return error.name === 'AbortError'
    || error.name === 'InterruptedError'
    || error.name === 'InterruptRequestedError'
}

function defaultDelay(
  milliseconds: number,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const timer = setTimeout(resolve, milliseconds)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(signal.reason)
      },
      { once: true },
    )
  })
}

/** Raised when a provider stream yields no chunk inside the inactivity budget. */
export class StreamInactivityError extends Error {
  readonly timeoutMs: number

  constructor(timeoutMs: number) {
    super(`Provider stream stalled: no chunk received within ${timeoutMs}ms (stream inactivity timeout)`)
    this.name = 'StreamInactivityError'
    this.timeoutMs = timeoutMs
  }
}

interface AttemptSignal {
  readonly controller: AbortController
  readonly release: () => void
}

/** Chain a per-attempt controller to the caller's signal so the watchdog can cancel one attempt only. */
function linkAttemptSignal(signal: AbortSignal | undefined): AttemptSignal {
  const controller = new AbortController()
  if (signal === undefined) {
    return { controller, release: () => undefined }
  }
  if (signal.aborted) {
    controller.abort(signal.reason)
    return { controller, release: () => undefined }
  }
  const forward = () => controller.abort(signal.reason)
  signal.addEventListener('abort', forward, { once: true })
  return { controller, release: () => signal.removeEventListener('abort', forward) }
}

/**
 * Wrap provider iteration with an inactivity watchdog. The timer only exists
 * while waiting for the next chunk, so it cannot fire during tool execution,
 * and it is cleared on completion, error, and consumer-close paths.
 */
async function* watchProviderStream(
  stream: AsyncIterable<LlmDelta>,
  timeoutMs: number,
  attempt: AttemptSignal,
): AsyncGenerator<LlmDelta> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    yield* stream
    return
  }
  const iterator = stream[Symbol.asyncIterator]()
  try {
    while (true) {
      const result = await nextDeltaWithTimeout(iterator, timeoutMs, attempt)
      if (result.done) {
        return
      }
      yield result.value
    }
  } finally {
    // Best-effort release of the abandoned provider stream. A stalled provider
    // may never settle its pending read, so cleanup must not block retry.
    void iterator.return?.()?.catch(() => undefined)
  }
}

async function nextDeltaWithTimeout(
  iterator: AsyncIterator<LlmDelta>,
  timeoutMs: number,
  attempt: AttemptSignal,
): Promise<IteratorResult<LlmDelta>> {
  const signal = attempt.controller.signal
  let timer: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined
  const pending = iterator.next()
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const error = new StreamInactivityError(timeoutMs)
          attempt.controller.abort(error)
          reject(error)
        }, timeoutMs)
      }),
      new Promise<never>((_, reject) => {
        if (signal.aborted) {
          reject(signal.reason)
          return
        }
        onAbort = () => reject(signal.reason)
        signal.addEventListener('abort', onAbort, { once: true })
      }),
    ])
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
    if (onAbort !== undefined) {
      signal.removeEventListener('abort', onAbort)
    }
    // A timed-out or aborted attempt abandons the pending provider read; keep
    // its late rejection from surfacing as an unhandled rejection.
    void pending.catch(() => undefined)
  }
}

/**
 * Dispatch one plugin hook point without letting plugin code break the turn.
 * HookRunner.run resolves with every hook result (sync and async); a
 * runner-level rejection is tolerated the way per-callback failures are
 * isolated inside the runner.
 */
async function dispatchHook(
  hookRunner: HookRunner | undefined,
  point: HookPoint,
  payload: Record<string, unknown>,
): Promise<unknown> {
  if (hookRunner === undefined) {
    return undefined
  }
  try {
    return await hookRunner.run(point, payload)
  } catch {
    return undefined
  }
}

/** Latest non-empty hook return wins, matching sequential mutation application. */
function hookMutation(result: unknown, current: unknown): unknown {
  if (Array.isArray(result)) {
    for (let index = result.length - 1; index >= 0; index -= 1) {
      if (result[index] !== undefined && result[index] !== null) {
        return result[index]
      }
    }
    return current
  }
  return result ?? current
}

/** Apply a before_tool_call argument mutation only when it is a real JSON object. */
function applyToolArgumentsMutation(call: ToolCall, mutated: unknown): ToolCall {
  if (!isJsonObject(mutated) || mutated === call.function.arguments) {
    return call
  }
  return { ...call, function: { ...call.function, arguments: mutated } }
}

/** Fail-closed capability lookup: no accessor means sequential, interruptible execution. */
function capabilitiesFor(
  dependencies: TurnDependencies,
  request: TurnRequest,
  call: ToolCall,
): { readonly concurrencySafe: boolean; readonly interruptBehavior: 'block' | 'cancel' } {
  return dependencies.capabilities?.(call.function.name, request.agentId, call.function.arguments)
    ?? { concurrencySafe: false, interruptBehavior: 'cancel' }
}
