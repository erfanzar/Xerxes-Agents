// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext, type ToolHandler } from '../executors/toolRegistry.js'
import type { UserMessage } from '../types/messages.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { BrowserManager } from './browser.js'
import { HIGH_POWER_OPERATOR_TOOLS, createOperatorRuntimeConfig, type OperatorRuntimeConfig } from './config.js'
import {
  offsetTime,
  requireImageInspector,
  requirePatchApplier,
  requireWebPort,
  runParallelReadonlyCalls,
  validateUnifiedPatch,
  type OperatorClock,
  type OperatorImageInspection,
  type OperatorImageInspector,
  type OperatorPatchApplier,
  type OperatorPatchResult,
  type OperatorWebPort,
  type ParallelReadonlyToolPort,
} from './operatorPorts.js'
import { PlanStateManager } from './plans.js'
import { PtySessionManager } from './pty.js'
import { SpawnedAgentManager, type SpawnedAgentDescriptor, type SpawnedAgentSnapshot } from './subagents.js'
import { UserPromptManager } from './userPrompt.js'

export interface OperatorStateOptions {
  readonly agentResolver?: (agentId: string | undefined) => SpawnedAgentDescriptor | undefined
  readonly browserManager?: BrowserManager
  readonly clock?: OperatorClock
  readonly config?: OperatorRuntimeConfig
  readonly imageInspector?: OperatorImageInspector
  readonly parallelReadonlyToolPort?: ParallelReadonlyToolPort
  readonly patchApplier?: OperatorPatchApplier
  readonly planManager?: PlanStateManager
  readonly ptyManager?: PtySessionManager
  readonly subagentManager?: SpawnedAgentManager
  readonly userPromptManager?: UserPromptManager
  readonly webPort?: OperatorWebPort
}

export interface OperatorToolRegistrationOptions {
  readonly agentId?: string
}

/**
 * Session attachment point for persistent terminals, browser state, plans,
 * human prompts, and spawned-agent handles.
 */
export class OperatorState {
  readonly browserManager: BrowserManager
  readonly config: OperatorRuntimeConfig
  readonly planManager: PlanStateManager
  readonly ptyManager: PtySessionManager
  readonly userPromptManager: UserPromptManager
  private readonly agentResolver: (agentId: string | undefined) => SpawnedAgentDescriptor | undefined
  private readonly clock: OperatorClock
  private readonly imageInspector: OperatorImageInspector | undefined
  private readonly parallelReadonlyToolPort: ParallelReadonlyToolPort | undefined
  private readonly patchApplier: OperatorPatchApplier | undefined
  private powerToolsEnabled: boolean
  private readonly registrations: OperatorToolRegistration[] = []
  private subagentManager: SpawnedAgentManager | undefined
  private readonly webPort: OperatorWebPort | undefined

  constructor(options: OperatorStateOptions = {}) {
    this.config = options.config ?? createOperatorRuntimeConfig()
    this.ptyManager = options.ptyManager ?? new PtySessionManager({
      ...(this.config.shellDefaultWorkdir === undefined ? {} : { workspaceRoot: this.config.shellDefaultWorkdir }),
    })
    this.browserManager = options.browserManager ?? new BrowserManager()
    this.planManager = options.planManager ?? new PlanStateManager()
    this.userPromptManager = options.userPromptManager ?? new UserPromptManager()
    this.subagentManager = options.subagentManager
    this.agentResolver = options.agentResolver ?? (() => undefined)
    this.clock = options.clock ?? { now: () => new Date() }
    this.imageInspector = options.imageInspector
    this.parallelReadonlyToolPort = options.parallelReadonlyToolPort
    this.patchApplier = options.patchApplier
    this.webPort = options.webPort
    this.powerToolsEnabled = this.config.powerToolsEnabled
  }

  setSubagentManager(manager: SpawnedAgentManager | undefined): void {
    this.subagentManager = manager
    this.refreshRegistrations()
  }

  setPowerToolsEnabled(enabled: boolean): void {
    if (this.powerToolsEnabled === enabled) return
    this.powerToolsEnabled = enabled
    this.refreshRegistrations()
  }

  listOperatorState(): Record<string, unknown> {
    return Object.freeze({
      power_tools_enabled: this.powerToolsEnabled,
      pty_sessions: this.ptyManager.listSessions().map(ptyWire),
      browser_pages: this.browserManager.listPages().map(page => ({
        ref_id: page.refId,
        url: page.url,
        title: page.title,
      })),
      spawned_agents: this.subagentManager?.listHandles().map(spawnedAgentWire) ?? [],
      plan: planWire(this.planManager.state),
      pending_user_prompt: this.userPromptManager.getPending() === undefined
        ? null
        : pendingPromptWire(this.userPromptManager.getPending()!),
    })
  }

  /**
   * Build a follow-up multimodal message from a host-owned image inspection.
   *
   * The normal tool wire intentionally strips the data URL, so embedding
   * runtimes that support image reinvocation retain the original host result
   * and call this method before sending the next model request.
   */
  createReinvokeMessage(result: unknown): UserMessage | undefined {
    if (!isImageInspection(result) || !result.imageDataUrl?.startsWith('data:')) return undefined
    const message: UserMessage = {
      role: 'user',
      content: [
        { type: 'text', text: `[TOOL IMAGE RESULT] ${imageSummary(result)}` },
        {
          type: 'image_url',
          image_url: {
            url: result.imageDataUrl,
            detail: result.detail === 'original' ? 'high' : result.detail,
          },
        },
      ],
    }
    return Object.freeze(message)
  }

  /** Install the available operator tools, replacing same-agent temporary implementations. */
  registerTools(registry: ToolRegistry, options: OperatorToolRegistrationOptions = {}): ToolDefinition[] {
    const agentId = options.agentId ?? 'default'
    const tools = this.availableToolEntries()
    this.synchronizeRegistration(registry, agentId, tools)
    return tools.map(tool => tool.definition)
  }

  toolDefinitions(): ToolDefinition[] {
    return this.availableToolEntries()
      .map(entry => entry.definition)
  }

  private availableToolEntries(): OperatorToolEntry[] {
    return this.toolEntries()
      .filter(entry => this.config.enabled)
      .filter(entry => this.config.allowedToolNames.has(entry.definition.function.name))
      .filter(entry => this.powerToolsEnabled || !HIGH_POWER_OPERATOR_TOOLS.has(entry.definition.function.name))
  }

  async close(): Promise<void> {
    this.userPromptManager.cancel('Operator session closed')
    await Promise.all([this.ptyManager.closeAll(), this.browserManager.close()])
  }

  private toolEntries(): OperatorToolEntry[] {
    const entries: OperatorToolEntry[] = [
      {
        definition: definition(
          'exec_command',
          'Start a persistent PTY-backed shell session: an interactive terminal session that stays alive across calls.',
          {
            cmd: stringSchema('Shell command to launch in a persistent terminal.'),
            workdir: stringSchema('Workspace-relative directory for the terminal.'),
            yield_time_ms: integerSchema('Soft output deadline in milliseconds.'),
            max_output_chars: integerSchema('Maximum output characters returned.'),
            login: booleanSchema('Use login shell semantics where available.'),
          },
          ['cmd'],
        ),
        handler: async inputs => {
          const workdir = optionalString(inputs, 'workdir') ?? this.config.shellDefaultWorkdir
          return ptyWire(await this.ptyManager.createSession(requiredString(inputs, 'cmd'), {
            ...(workdir === undefined ? {} : { workdir }),
            yieldTimeMs: optionalInteger(inputs, 'yield_time_ms') ?? this.config.shellDefaultYieldMs,
            maxOutputChars: optionalInteger(inputs, 'max_output_chars') ?? this.config.shellDefaultMaxOutputChars,
            login: optionalBoolean(inputs, 'login') ?? true,
          }))
        },
      },
      {
        definition: definition('write_stdin', 'Send text, EOF, or an interrupt to a persistent terminal session.', {
          session_id: stringSchema('Session id returned by exec_command.'),
          chars: stringSchema('Characters to send; leave empty to poll.'),
          yield_time_ms: integerSchema('Soft output deadline in milliseconds.'),
          max_output_chars: integerSchema('Maximum output characters returned.'),
          close_stdin: booleanSchema('Send EOF after writing.'),
          interrupt: booleanSchema('Send SIGINT before writing.'),
        }, ['session_id']),
        handler: async inputs => {
          const chars = optionalString(inputs, 'chars')
          return ptyWire(await this.ptyManager.write(requiredString(inputs, 'session_id'), {
            ...(chars === undefined ? {} : { chars }),
            yieldTimeMs: optionalInteger(inputs, 'yield_time_ms') ?? this.config.shellDefaultYieldMs,
            maxOutputChars: optionalInteger(inputs, 'max_output_chars') ?? this.config.shellDefaultMaxOutputChars,
            closeStdin: optionalBoolean(inputs, 'close_stdin') ?? false,
            interrupt: optionalBoolean(inputs, 'interrupt') ?? false,
          }))
        },
      },
      {
        definition: definition('list_terminal_sessions', 'List persistent terminal sessions for this Xerxes session.', {}),
        handler: () => this.ptyManager.listSessions().map(ptyWire),
      },
      {
        definition: definition('close_terminal_session', 'Terminate and forget a persistent terminal session.', {
          session_id: stringSchema('Session id returned by exec_command.'),
        }, ['session_id']),
        handler: async inputs => ptyClosedWire(await this.ptyManager.close(requiredString(inputs, 'session_id'))),
      },
      {
        definition: definition('apply_patch', 'Apply a validated unified diff through an explicitly configured workspace patch host.', {
          patch: stringSchema('Unified diff with ---/+++ headers and @@ hunks.'),
          check: booleanSchema('Validate the patch without modifying the workspace.'),
          workdir: stringSchema('Host-approved workspace directory for the patch.'),
        }, ['patch']),
        handler: async (inputs, _context, signal) => {
          const patch = validateUnifiedPatch(requiredString(inputs, 'patch'))
          const workdir = optionalString(inputs, 'workdir')
          const result = await requirePatchApplier(this.patchApplier).applyPatch({
            patch,
            check: optionalBoolean(inputs, 'check') ?? false,
            ...(workdir === undefined ? {} : { workdir }),
          }, signal)
          return patchWire(result)
        },
      },
      {
        definition: definition('parallel_tools', 'Run independent read-only Xerxes tool calls concurrently through an injected safe dispatcher.', {
          calls: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: stringSchema('Registered read-only tool name.'),
                input: { type: 'object', description: 'JSON object passed to the named tool.' },
                arguments: { type: 'object', description: 'Alias for input.' },
              },
            },
          },
          max_workers: integerSchema('Maximum concurrent calls, clamped from 1 through 16.'),
        }, ['calls']),
        handler: async (inputs, _context, signal) => {
          const calls = inputs.calls
          if (!Array.isArray(calls)) throw new ValidationError('calls', 'must be an array of call objects', calls)
          const result = await runParallelReadonlyCalls(
            calls,
            optionalInteger(inputs, 'max_workers') ?? 4,
            this.parallelReadonlyToolPort,
            signal,
          )
          return Object.freeze({ max_workers: result.maxWorkers, results: result.results })
        },
      },
      {
        definition: definition('ask_user', 'Pause the run and ask the user a clarification question.', {
          question: stringSchema('Question shown to the user.'),
          options: stringArraySchema('Optional selectable choices.'),
          allow_freeform: booleanSchema('Whether custom text answers are allowed.'),
          placeholder: stringSchema('Optional input placeholder.'),
        }, ['question']),
        handler: async (inputs, _context, signal) => {
          const options = optionalStringArray(inputs, 'options')
          const placeholder = optionalString(inputs, 'placeholder')
          return userPromptAnswerWire(await this.userPromptManager.request({
            question: requiredString(inputs, 'question'),
            ...(options === undefined ? {} : { options }),
            allowFreeform: optionalBoolean(inputs, 'allow_freeform') ?? true,
            ...(placeholder === undefined ? {} : { placeholder }),
          }, signal))
        },
      },
      {
        definition: definition('view_image', 'Inspect a local image through an explicitly configured image host.', {
          path: stringSchema('Host-approved local image path.'),
          detail: stringSchema('Requested inspection detail: auto, low, high, or original.'),
        }, ['path']),
        handler: async (inputs, _context, signal) => {
          const path = requiredString(inputs, 'path')
          const detail = imageDetail(optionalString(inputs, 'detail'))
          return imageInspectionWire(await requireImageInspector(this.imageInspector).inspectImage({ path, detail }, signal))
        },
      },
      {
        definition: definition('update_plan', 'Replace this session’s structured execution plan.', {
          explanation: stringSchema('Optional plan preamble.'),
          plan: {
            type: 'array',
            items: {
              type: 'object',
              properties: { step: stringSchema('Imperative plan step.'), status: stringSchema('Step status.') },
              required: ['step'],
            },
          },
        }),
        handler: inputs => planWire(this.planManager.update(optionalString(inputs, 'explanation'), planSteps(inputs.plan))),
      },
      {
        definition: definition('web.open', 'Open or revisit a browser page through the configured browser adapter.', {
          url: stringSchema('Public http(s) URL.'),
          ref_id: stringSchema('Existing browser page id.'),
          wait_ms: integerSchema('Wait after navigation in milliseconds.'),
        }),
        handler: async inputs => {
          const url = optionalString(inputs, 'url')
          const refId = optionalString(inputs, 'ref_id')
          const waitMs = optionalInteger(inputs, 'wait_ms')
          return browserOpenWire(await this.browserManager.open({
            ...(url === undefined ? {} : { url }),
            ...(refId === undefined ? {} : { refId }),
            ...(waitMs === undefined ? {} : { waitMs }),
          }))
        },
      },
      {
        definition: definition('web.click', 'Click a discovered browser link, selector, or text target.', {
          ref_id: stringSchema('Browser page id.'),
          link_id: integerSchema('Numeric link id from web.open.'),
          selector: stringSchema('CSS selector.'),
          text: stringSchema('Visible text target.'),
          wait_ms: integerSchema('Wait after interaction in milliseconds.'),
        }, ['ref_id']),
        handler: async inputs => {
          const linkId = optionalInteger(inputs, 'link_id')
          const selector = optionalString(inputs, 'selector')
          const text = optionalString(inputs, 'text')
          const waitMs = optionalInteger(inputs, 'wait_ms')
          return browserOpenWire(await this.browserManager.click(requiredString(inputs, 'ref_id'), {
            ...(linkId === undefined ? {} : { linkId }),
            ...(selector === undefined ? {} : { selector }),
            ...(text === undefined ? {} : { text }),
            ...(waitMs === undefined ? {} : { waitMs }),
          }))
        },
      },
      {
        definition: definition('web.find', 'Find a regex pattern in a browser page.', {
          ref_id: stringSchema('Browser page id.'),
          pattern: stringSchema('Case-insensitive regular expression.'),
        }, ['ref_id', 'pattern']),
        handler: async inputs => browserFindWire(await this.browserManager.find(
          requiredString(inputs, 'ref_id'),
          requiredString(inputs, 'pattern'),
        )),
      },
      {
        definition: definition('web.screenshot', 'Save a browser-page screenshot through the configured browser adapter.', {
          ref_id: stringSchema('Browser page id.'),
          path: stringSchema('Optional output path.'),
          full_page: booleanSchema('Capture the entire page.'),
        }, ['ref_id']),
        handler: async inputs => {
          const path = optionalString(inputs, 'path')
          const fullPage = optionalBoolean(inputs, 'full_page')
          return browserScreenshotWire(await this.browserManager.screenshot(requiredString(inputs, 'ref_id'), {
            ...(path === undefined ? {} : { path }),
            ...(fullPage === undefined ? {} : { fullPage }),
          }))
        },
      },
      {
        definition: definition('web.search_query', 'Search the public web through an explicitly configured search host.', {
          q: stringSchema('Public-web search query.'),
          search_type: stringSchema('Search mode: text or news.'),
          n_results: integerSchema('Maximum number of result records.'),
          domains: stringArraySchema('Optional public-domain restrictions.'),
        }, ['q']),
        handler: async (inputs, _context, signal) => {
          const query = requiredString(inputs, 'q')
          const searchType = webSearchType(optionalString(inputs, 'search_type'))
          const response = await requireWebPort(this.webPort).search({
            kind: 'text',
            query,
            maxResults: webResultCount(optionalInteger(inputs, 'n_results')),
            domains: normalizedDomains(optionalStringArray(inputs, 'domains')),
            recency: searchType === 'news' ? 'day' : undefined,
          }, signal)
          return Object.freeze({
            query,
            search_type: searchType,
            results: response.results,
            engine: response.engine ?? 'host',
          })
        },
      },
      {
        definition: definition('web.image_query', 'Search public image references through an explicitly configured search host.', {
          q: stringSchema('Public image-search query.'),
          n_results: integerSchema('Maximum number of result records.'),
          domains: stringArraySchema('Optional public-domain restrictions.'),
        }, ['q']),
        handler: async (inputs, _context, signal) => {
          const query = requiredString(inputs, 'q')
          const response = await requireWebPort(this.webPort).search({
            kind: 'image',
            query,
            maxResults: webResultCount(optionalInteger(inputs, 'n_results')),
            domains: normalizedDomains(optionalStringArray(inputs, 'domains')),
            recency: undefined,
          }, signal)
          return Object.freeze({ query, results: response.results, engine: response.engine ?? 'host' })
        },
      },
      {
        definition: definition('web.weather', 'Fetch a weather report through an explicitly configured public-data host.', {
          location: stringSchema('Location name or host-supported location identifier.'),
        }, ['location']),
        handler: async (inputs, _context, signal) => requireWebPort(this.webPort).weather({
          location: requiredString(inputs, 'location'),
        }, signal),
      },
      {
        definition: definition('web.finance', 'Fetch a current market quote through an explicitly configured public-data host.', {
          ticker: stringSchema('Ticker or symbol.'),
          market: stringSchema('Optional market suffix.'),
          kind: stringSchema('Asset kind, such as equity, fund, crypto, or index.'),
        }, ['ticker']),
        handler: async (inputs, _context, signal) => {
          const market = optionalString(inputs, 'market')
          return requireWebPort(this.webPort).finance({
            ticker: requiredString(inputs, 'ticker'),
            kind: optionalString(inputs, 'kind')?.trim() || 'equity',
            ...(market?.trim() ? { market: market.trim() } : {}),
          }, signal)
        },
      },
      {
        definition: definition('web.sports', 'Fetch a sports schedule or standings report through an explicitly configured public-data host.', {
          league: stringSchema('League identifier accepted by the configured host.'),
          fn: stringSchema('Requested operation: schedule or standings.'),
          team: stringSchema('Optional team identifier.'),
          opponent: stringSchema('Optional opponent identifier.'),
          date_from: stringSchema('Optional ISO date lower bound.'),
          date_to: stringSchema('Optional ISO date upper bound.'),
          num_games: integerSchema('Optional maximum number of games.'),
        }, ['league']),
        handler: async (inputs, _context, signal) => {
          const team = optionalString(inputs, 'team')
          const opponent = optionalString(inputs, 'opponent')
          const dateFrom = optionalString(inputs, 'date_from')
          const dateTo = optionalString(inputs, 'date_to')
          const numGames = optionalInteger(inputs, 'num_games')
          return requireWebPort(this.webPort).sports({
            league: requiredString(inputs, 'league'),
            fn: sportsFunction(optionalString(inputs, 'fn')),
            ...(team?.trim() ? { team: team.trim() } : {}),
            ...(opponent?.trim() ? { opponent: opponent.trim() } : {}),
            ...(dateFrom?.trim() ? { dateFrom: dateFrom.trim() } : {}),
            ...(dateTo?.trim() ? { dateTo: dateTo.trim() } : {}),
            ...(numGames === undefined ? {} : { numGames: positiveInteger('num_games', numGames) }),
          }, signal)
        },
      },
      {
        definition: definition('web.time', 'Return the current wall time for a UTC offset without using the network.', {
          utc_offset: stringSchema('UTC offset in +HH:MM or -HH:MM form.'),
        }, ['utc_offset']),
        handler: inputs => offsetTime(requiredString(inputs, 'utc_offset'), this.clock),
      },
    ]
    if (this.subagentManager !== undefined) entries.push(...this.subagentEntries(this.subagentManager))
    return entries
  }

  private subagentEntries(manager: SpawnedAgentManager): OperatorToolEntry[] {
    return [
      {
        definition: definition('spawn_agent', 'Create a managed background subagent and optionally give it work.', {
          message: stringSchema('Initial task for the subagent.'),
          task_description: stringSchema('Legacy alias for message.'),
          agent_id: stringSchema('Parent agent to clone.'),
          prompt_profile: stringSchema('Prompt profile for the subagent.'),
          nickname: stringSchema('Stable subagent handle.'),
          permission_mode: stringSchema('Permission mode for the subagent.'),
        }),
        handler: async inputs => {
          const message = optionalString(inputs, 'message')
          const taskDescription = optionalString(inputs, 'task_description')
          const agentId = optionalString(inputs, 'agent_id')
          const agent = this.agentResolver(agentId)
          const promptProfile = optionalString(inputs, 'prompt_profile')
          const nickname = optionalString(inputs, 'nickname')
          const permissionMode = optionalString(inputs, 'permission_mode')
          return spawnedAgentWire(await manager.spawn({
            ...(message === undefined ? {} : { message }),
            ...(taskDescription === undefined ? {} : { taskDescription }),
            ...(agentId === undefined ? {} : { agentId }),
            ...(agent === undefined ? {} : { agent }),
            ...(promptProfile === undefined ? {} : { promptProfile }),
            ...(nickname === undefined ? {} : { nickname }),
            ...(permissionMode === undefined ? {} : { permissionMode }),
          }))
        },
      },
      {
        definition: definition('resume_agent', 'Reopen a closed spawned-agent handle.', {
          agent_id: stringSchema('Spawned-agent handle id.'),
        }, ['agent_id']),
        handler: inputs => spawnedAgentWire(manager.resume(requiredString(inputs, 'agent_id'))),
      },
      {
        definition: definition('send_input', 'Send queued or interrupting work to a spawned agent.', {
          target: stringSchema('Target spawned-agent handle.'),
          id: stringSchema('Alias for target.'),
          agent_id: stringSchema('Alias for target.'),
          handle_id: stringSchema('Alias for target.'),
          message: stringSchema('Input sent to the subagent.'),
          task_description: stringSchema('Legacy alias for message.'),
          interrupt: booleanSchema('Interrupt active work before sending input.'),
        }),
        handler: async inputs => {
          const message = optionalString(inputs, 'message')
          const taskDescription = optionalString(inputs, 'task_description')
          return spawnedAgentWire(await manager.sendInput(firstString(inputs, ['target', 'id', 'agent_id', 'handle_id']), {
            ...(message === undefined ? {} : { message }),
            ...(taskDescription === undefined ? {} : { taskDescription }),
            interrupt: optionalBoolean(inputs, 'interrupt') ?? false,
          }))
        },
      },
      {
        definition: definition('wait_agent', 'Wait for spawned agents to settle or a timeout to elapse.', {
          targets: stringArraySchema('Spawned-agent handle ids.'),
          timeout_ms: integerSchema('Maximum wait time in milliseconds.'),
        }, ['targets']),
        handler: async inputs => {
          const result = await manager.wait(requiredStringArray(inputs, 'targets'), optionalInteger(inputs, 'timeout_ms') ?? 30_000)
          return {
            completed: result.completed.map(spawnedAgentWire),
            pending: result.pending.map(spawnedAgentWire),
          }
        },
      },
      {
        definition: definition('close_agent', 'Cancel and close a spawned-agent handle.', {
          target: stringSchema('Spawned-agent handle id.'),
        }, ['target']),
        handler: inputs => spawnedAgentWire(manager.close(requiredString(inputs, 'target'))),
      },
    ]
  }

  private refreshRegistrations(): void {
    const entries = this.availableToolEntries()
    for (const registration of this.registrations) {
      this.synchronizeRegistration(registration.registry, registration.agentId, entries)
    }
  }

  private synchronizeRegistration(
    registry: ToolRegistry,
    agentId: string,
    entries: readonly OperatorToolEntry[],
  ): void {
    const registration = this.registrations.find(candidate => candidate.registry === registry && candidate.agentId === agentId)
    const previousNames = registration?.names ?? new Set<string>()
    const nextNames = new Set(entries.map(entry => entry.definition.function.name))
    for (const name of previousNames) {
      if (!nextNames.has(name)) registry.unregister(name, agentId)
    }
    for (const entry of entries) registry.replace(entry.definition, entry.handler, agentId)
    if (registration === undefined) this.registrations.push({ registry, agentId, names: nextNames })
    else registration.names = nextNames
  }
}

interface OperatorToolEntry {
  readonly definition: ToolDefinition
  readonly handler: ToolHandler
}

interface OperatorToolRegistration {
  readonly agentId: string
  readonly registry: ToolRegistry
  names: Set<string>
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return Object.freeze({
    type: 'function',
    function: Object.freeze({
      name,
      description,
      parameters: Object.freeze({ type: 'object', additionalProperties: false, properties, ...(required.length ? { required } : {}) }),
    }),
  })
}

function stringSchema(description: string): Record<string, unknown> {
  return { type: 'string', description }
}

function integerSchema(description: string): Record<string, unknown> {
  return { type: 'integer', description }
}

function booleanSchema(description: string): Record<string, unknown> {
  return { type: 'boolean', description }
}

function stringArraySchema(description: string): Record<string, unknown> {
  return { type: 'array', description, items: { type: 'string' } }
}

function requiredString(inputs: JsonObject, field: string): string {
  const value = inputs[field]
  if (typeof value !== 'string' || !value.trim()) throw new ValidationError(field, 'must be a non-empty string', value)
  return value
}

function optionalString(inputs: JsonObject, field: string): string | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string') throw new ValidationError(field, 'must be a string', value)
  return value
}

function optionalInteger(inputs: JsonObject, field: string): number | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'number' || !Number.isInteger(value)) throw new ValidationError(field, 'must be an integer', value)
  return value
}

function optionalBoolean(inputs: JsonObject, field: string): boolean | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'boolean') throw new ValidationError(field, 'must be a boolean', value)
  return value
}

function optionalStringArray(inputs: JsonObject, field: string): string[] | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new ValidationError(field, 'must be an array of strings', value)
  }
  return value.map(item => String(item))
}

function requiredStringArray(inputs: JsonObject, field: string): string[] {
  const value = optionalStringArray(inputs, field)
  if (value === undefined || !value.length || value.some(item => !item.trim())) {
    throw new ValidationError(field, 'must be a non-empty array of non-empty strings', inputs[field])
  }
  return value
}

function firstString(inputs: JsonObject, fields: readonly string[]): string | undefined {
  for (const field of fields) {
    const value = optionalString(inputs, field)
    if (value?.trim()) return value
  }
  return undefined
}

function planSteps(value: JsonValue | undefined): Array<{ readonly status: string; readonly step: string }> {
  if (value === undefined || value === null) return []
  if (!Array.isArray(value)) throw new ValidationError('plan', 'must be an array', value)
  return value.map((item, index) => {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
      throw new ValidationError(`plan[${index}]`, 'must be an object', item)
    }
    const record = item as Record<string, JsonValue>
    const step = record.step
    const status = record.status
    if (typeof step !== 'string' || !step.trim()) throw new ValidationError(`plan[${index}].step`, 'must be a non-empty string', step)
    if (status !== undefined && status !== null && typeof status !== 'string') {
      throw new ValidationError(`plan[${index}].status`, 'must be a string', status)
    }
    return { step, status: typeof status === 'string' ? status : 'pending' }
  })
}

function ptyWire(value: { readonly command: string; readonly exitCode: number | null; readonly maxOutputChars?: number; readonly note?: string; readonly outputTruncated?: boolean; readonly running: boolean; readonly sessionId: string; readonly stdout?: string; readonly workdir: string; readonly yieldTimeMs?: number }): Record<string, unknown> {
  return Object.freeze({
    session_id: value.sessionId,
    command: value.command,
    workdir: value.workdir,
    ...(value.stdout === undefined ? {} : { stdout: value.stdout }),
    ...(value.outputTruncated === undefined ? {} : { output_truncated: value.outputTruncated }),
    ...(value.yieldTimeMs === undefined ? {} : { yield_time_ms: value.yieldTimeMs }),
    ...(value.maxOutputChars === undefined ? {} : { max_output_chars: value.maxOutputChars }),
    ...(value.note === undefined ? {} : { note: value.note }),
    running: value.running,
    exit_code: value.exitCode,
  })
}

function ptyClosedWire(value: { readonly closed: true; readonly exitCode: number | null; readonly sessionId: string }): Record<string, unknown> {
  return Object.freeze({ session_id: value.sessionId, closed: value.closed, exit_code: value.exitCode })
}

function planWire(value: { readonly explanation?: string; readonly revision: number; readonly steps: readonly { readonly status: string; readonly step: string }[]; readonly updatedAt: string }): Record<string, unknown> {
  return Object.freeze({
    explanation: value.explanation ?? null,
    revision: value.revision,
    updated_at: value.updatedAt,
    steps: value.steps.map(step => ({ step: step.step, status: step.status })),
  })
}

function patchWire(value: OperatorPatchResult): Record<string, unknown> {
  return Object.freeze({
    applied: value.applied,
    checked: value.checked,
    ...(value.workdir === undefined ? {} : { workdir: value.workdir }),
    ...(value.stdout === undefined ? {} : { stdout: value.stdout }),
  })
}

function imageInspectionWire(value: OperatorImageInspection): Record<string, unknown> {
  return Object.freeze({
    path: value.path,
    detail: value.detail,
    width: value.width,
    height: value.height,
    mode: value.mode,
    ...(value.format === undefined ? {} : { format: value.format }),
  })
}

function isImageInspection(value: unknown): value is OperatorImageInspection {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return typeof record.path === 'string'
    && typeof record.width === 'number'
    && typeof record.height === 'number'
    && typeof record.mode === 'string'
    && (record.detail === 'auto' || record.detail === 'high' || record.detail === 'low' || record.detail === 'original')
    && (record.imageDataUrl === undefined || typeof record.imageDataUrl === 'string')
}

function imageSummary(value: OperatorImageInspection): string {
  const format = value.format ? `${value.format} ` : ''
  return `${format}${value.width}x${value.height} ${value.mode} image at ${value.path}`
}

function imageDetail(value: string | undefined): 'auto' | 'high' | 'low' | 'original' {
  const detail = value?.trim() || 'auto'
  if (detail === 'auto' || detail === 'high' || detail === 'low' || detail === 'original') return detail
  throw new ValidationError('detail', 'must be auto, low, high, or original', value)
}

function webSearchType(value: string | undefined): 'news' | 'text' {
  const type = value?.trim() || 'text'
  if (type === 'text' || type === 'news') return type
  throw new ValidationError('search_type', 'must be text or news', value)
}

function webResultCount(value: number | undefined): number {
  return positiveInteger('n_results', value ?? 5, 50)
}

function normalizedDomains(value: readonly string[] | undefined): readonly string[] {
  if (value === undefined) return Object.freeze([])
  const domains = value.map(domain => domain.trim())
  if (domains.some(domain => !domain)) throw new ValidationError('domains', 'must not include empty values', value)
  return Object.freeze(domains)
}

function sportsFunction(value: string | undefined): 'schedule' | 'standings' {
  const fn = value?.trim() || 'schedule'
  if (fn === 'schedule' || fn === 'standings') return fn
  throw new ValidationError('fn', 'must be schedule or standings', value)
}

function positiveInteger(field: string, value: number, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isInteger(value) || value < 1 || value > maximum) {
    throw new ValidationError(field, `must be an integer from 1 through ${maximum}`, value)
  }
  return value
}

function pendingPromptWire(value: NonNullable<ReturnType<UserPromptManager['getPending']>>): Record<string, unknown> {
  return Object.freeze({
    request_id: value.requestId,
    question: value.question,
    options: value.options.map(option => ({ label: option.label, value: option.value })),
    allow_freeform: value.allowFreeform,
    placeholder: value.placeholder ?? null,
    created_at: value.createdAt,
  })
}

function userPromptAnswerWire(value: Awaited<ReturnType<UserPromptManager['request']>>): Record<string, unknown> {
  return Object.freeze({
    request_id: value.requestId,
    question: value.question,
    answer: value.answer,
    raw_input: value.rawInput,
    selected_option: value.selectedOption === undefined ? null : { label: value.selectedOption.label, value: value.selectedOption.value },
    used_freeform: value.usedFreeform,
  })
}

function spawnedAgentWire(value: SpawnedAgentSnapshot & { readonly previousStatus?: string }): Record<string, unknown> {
  return Object.freeze({
    id: value.id,
    name: value.name,
    agent_id: value.agentId,
    source_agent_id: value.sourceAgentId ?? null,
    status: value.status,
    created_at: value.createdAt,
    updated_at: value.updatedAt,
    prompt_profile: value.promptProfile,
    last_input: value.lastInput ?? null,
    last_output: value.lastOutput ?? null,
    error: value.error ?? null,
    queue_size: value.queueSize,
    queued_preview: value.queuedPreview ?? null,
    closed: value.closed,
    ...(value.previousStatus === undefined ? {} : { previous_status: value.previousStatus }),
  })
}

function browserOpenWire(value: { readonly contentPreview?: string; readonly links: readonly { readonly id?: number; readonly url: string }[]; readonly refId: string; readonly title: string; readonly url: string }): Record<string, unknown> {
  return Object.freeze({
    ref_id: value.refId,
    url: value.url,
    title: value.title,
    content_preview: value.contentPreview ?? '',
    links: value.links.map(link => ({ id: link.id, url: link.url })),
  })
}

function browserFindWire(value: { readonly matchCount: number; readonly matches: readonly string[]; readonly pattern: string; readonly refId: string }): Record<string, unknown> {
  return Object.freeze({ ref_id: value.refId, pattern: value.pattern, match_count: value.matchCount, matches: [...value.matches] })
}

function browserScreenshotWire(value: { readonly fullPage: boolean; readonly path: string; readonly refId: string }): Record<string, unknown> {
  return Object.freeze({ ref_id: value.refId, path: value.path, full_page: value.fullPage })
}
