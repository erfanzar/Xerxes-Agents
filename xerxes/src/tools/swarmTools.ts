// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolCapabilities, type ToolExecutionContext } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { isNarratorName } from '../swarm/roster.js'
import {
  COORDINATOR_NAME,
  type SwarmAddInput,
  type SwarmAddResult,
  type SwarmChannelActionInput,
  type SwarmChannelActionResult,
  type SwarmChannelOp,
  type SwarmKillInput,
  type SwarmKillResult,
  type SwarmLifecycleEvent,
  type SwarmModelInfo,
  type SwarmOverviewUpdateInput,
  type SwarmOverviewUpdateResult,
  type SwarmPostMessageInput,
  type SwarmPostMessageResult,
  type SwarmReadBoardInput,
  type SwarmReadBoardResult,
  type SwarmStatusSnapshot,
  type SwarmSwitchModelInput,
  type SwarmSwitchModelResult,
  type SwarmTitleInput,
  type SwarmTitleResult,
  type SwarmWaitInput,
  type SwarmWaitResult,
  type SwarmWorkspaceMode,
} from '../swarm/types.js'
import { optionalBoolean, optionalInteger, optionalString, optionalStringArray, requiredString } from './inputs.js'

export interface SwarmToolHost {
  readonly sessionId: string
  isActive(): boolean
  status(): SwarmStatusSnapshot
  getMember(name: string): unknown
  listModels(options?: { readonly all?: boolean }): Promise<{ readonly models: SwarmModelInfo[]; readonly default_model: string }>
  setTitle(input: SwarmTitleInput): Promise<SwarmTitleResult>
  add(input: SwarmAddInput): Promise<SwarmAddResult>
  switchModel(input: SwarmSwitchModelInput): Promise<SwarmSwitchModelResult>
  kill(input: SwarmKillInput): Promise<SwarmKillResult>
  channelAction(input: SwarmChannelActionInput): Promise<SwarmChannelActionResult>
  postMessage(input: SwarmPostMessageInput): Promise<SwarmPostMessageResult>
  readBoard(input?: SwarmReadBoardInput): Promise<SwarmReadBoardResult>
  waitForBoard(input: SwarmWaitInput): Promise<SwarmWaitResult>
  updateOverview(input: SwarmOverviewUpdateInput): Promise<SwarmOverviewUpdateResult>
  onEvent(listener: (event: SwarmLifecycleEvent) => void): () => void
}

export const SWARM_TOOL_NAME = 'swarm'
export const POST_MESSAGE_TOOL_NAME = 'post_message'
export const READ_BOARD_TOOL_NAME = 'read_board'
export const SWARM_WAIT_TOOL_NAME = 'swarm_wait'
export const SWARM_OVERVIEW_TOOL_NAME = 'swarm_overview'

export const SWARM_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: SWARM_TOOL_NAME,
    description: [
      'Manage the multi-agent Xerxes Immortals force collaborating on the shared message board.',
      'Actions: `status` inspects the roster, channels, and overview; `models` lists selectable models;',
      '`add` spawns one worker or a sharded group (`n` homogeneous copies or `tasks` per-slice briefs) and posts their public assignment;',
      '`model` switches a waiting/stopped member (or yourself) to another model while keeping conversation history;',
      '`kill` permanently stops a worker (or group) you added and removes any isolated `share`/`branch` workspace;',
      '`title` sets the Immortals title (coordinator only); `channel` creates, lists, subscribes, unsubscribes, or archives board channels.',
    ].join(' '),
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        action: {
          type: 'string',
          enum: ['status', 'models', 'add', 'model', 'kill', 'title', 'channel'],
          description: 'Immortals management operation to perform.',
        },
        name: {
          type: 'string',
          description: 'Worker or group base name (`^[a-z0-9][a-z0-9_-]{0,23}$`, excluding `all` and `coordinator*`).',
        },
        brief: {
          type: 'string',
          description: 'Public assignment posted on the board when `action` is `add`.',
        },
        model: {
          type: 'string',
          description: 'Optional model identifier or display name from `models` for `add` or `model`.',
        },
        reason: {
          type: 'string',
          description: 'For `model`: optional concise reason for switching models (for example, quota exhaustion or user request); recorded and shown to the resumed worker.',
        },
        all: {
          type: 'boolean',
          description: 'For `models`: include every selectable model rather than only suggested defaults.',
        },
        n: {
          type: 'integer',
          minimum: 1,
          description: 'For `add`: spawn `<name>-1`..`<name>-<n>` homogeneous copies when greater than 1.',
        },
        tasks: {
          type: 'array',
          items: { type: 'string' },
          description: 'For `add`: per-worker slice descriptions spawning `<name>-1`..`<name>-<len(tasks)>`.',
        },
        fork: {
          type: 'string',
          description: 'For `add`: `"coordinator"` or an existing member name whose conversation history to clone.',
        },
        workspace: {
          type: 'string',
          enum: ['inherit', 'share', 'branch'],
          description: 'For `add`: `"inherit"` shares the caller workspace, `"share"` allocates a Git worktree, `"branch"` clones an isolated copy.',
        },
        channels: {
          type: 'array',
          items: { type: 'string' },
          description: 'For `add`: initial channels for the new worker(s); brief posts in `channels[0] ?? "main"`.',
        },
        title: {
          type: 'string',
          description: 'Short Immortals title when `action` is `title` (coordinator only).',
        },
        op: {
          type: 'string',
          enum: ['create', 'subscribe', 'unsubscribe', 'list', 'archive'],
          description: 'Channel sub-operation when `action` is `channel`.',
        },
        channel: {
          type: 'string',
          description: 'Target channel name for `channel` operations.',
        },
        topic: {
          type: 'string',
          description: 'Optional topic description when creating a channel.',
        },
      },
      required: ['action'],
    },
  },
}

export const POST_MESSAGE_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: POST_MESSAGE_TOOL_NAME,
    description: 'Post a message to a shared Immortals board channel, tagged with your member name. @mentions wake named members.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: {
          type: 'string',
          description: 'Message body to post on the board. Prefix `[done]` after `@<recipient>` when reporting final completion.',
        },
        channel: {
          type: 'string',
          description: 'Target board channel (defaults to `"main"`).',
        },
      },
      required: ['text'],
    },
  },
}

export const READ_BOARD_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: READ_BOARD_TOOL_NAME,
    description: 'Read messages from the shared Immortals board in chronological order, or inspect live member status when `status` is true.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        since: {
          type: 'integer',
          minimum: 0,
          description: 'Return oldest visible messages strictly after this sequence number.',
        },
        limit: {
          type: 'integer',
          minimum: 1,
          maximum: 500,
          description: 'Maximum number of messages to return (default 20).',
        },
        seqs: {
          type: 'array',
          items: { type: 'integer', minimum: 1 },
          description: 'Fetch exact messages by sequence number, ignoring `since` and `limit`.',
        },
        channel: {
          type: 'string',
          description: 'Optional channel name filter (or `"*"` for all channels visible to you).',
        },
        status: {
          type: 'boolean',
          description: 'Return the Immortals member roster and parked cursors instead of board messages.',
        },
      },
    },
  },
}

export const SWARM_WAIT_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: SWARM_WAIT_TOOL_NAME,
    description: 'Park until a new board message after `since` arrives from another Immortals member, ending worker turns cleanly with zero polling.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        since: {
          type: 'integer',
          minimum: 0,
          description: 'Highest board sequence number you have already read; wakes on matching posts with `seq > since`.',
        },
        mentions_only: {
          type: 'boolean',
          description: 'When true (default), wake only for `@<your-name>`, `@<your-group>`, `@all`, or `@#<subscribed-channel>`. Pass false to wake on any visible post.',
        },
      },
      required: ['since'],
    },
  },
}

export const SWARM_OVERVIEW_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: SWARM_OVERVIEW_TOOL_NAME,
    description: 'Update the live Markdown overview card displayed at the top of the Immortals Board (narrator role only).',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        markdown: {
          type: 'string',
          description: 'Structured Markdown overview of current Immortals goals, active slices, and completed work.',
        },
        summary: {
          type: 'string',
          description: 'Optional concise one-line status headline.',
        },
      },
      required: ['markdown'],
    },
  },
}

const SWARM_CAPABILITIES: Partial<ToolCapabilities> = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: false,
  interruptBehavior: 'block',
  openWorld: true,
  readOnly: false,
})

const POST_MESSAGE_CAPABILITIES: Partial<ToolCapabilities> = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: false,
  interruptBehavior: 'block',
  openWorld: true,
  readOnly: false,
})

const READ_BOARD_CAPABILITIES: Partial<ToolCapabilities> = Object.freeze({
  concurrencySafe: true,
  defer: false,
  destructive: false,
  interruptBehavior: 'cancel',
  openWorld: false,
  readOnly: true,
})

const SWARM_WAIT_CAPABILITIES: Partial<ToolCapabilities> = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: false,
  interruptBehavior: 'cancel',
  openWorld: true,
  readOnly: true,
})

const SWARM_OVERVIEW_CAPABILITIES: Partial<ToolCapabilities> = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: false,
  interruptBehavior: 'block',
  openWorld: false,
  readOnly: false,
})

export interface SwarmBoundTool {
  readonly definition: ToolDefinition
  readonly capabilities: Partial<ToolCapabilities>
  readonly handler: (
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ) => Promise<unknown>
}

export interface CreateSwarmToolsOptions {
  readonly swarm: SwarmToolHost
  readonly caller?: string
}

export interface SwarmToolSuite {
  readonly allTools: readonly SwarmBoundTool[]
  visibleDefinitions(options?: { readonly caller?: string; readonly turnStartedActive?: boolean }): ToolDefinition[]
}

function resolveCallerName(
  explicitCaller: string | undefined,
  context: ToolExecutionContext,
  swarm: SwarmToolHost,
): string {
  const fromMeta = context.metadata.swarmMember
  if (typeof fromMeta === 'string' && fromMeta.trim()) {
    return fromMeta.trim()
  }
  if (explicitCaller && explicitCaller.trim()) {
    return explicitCaller.trim()
  }
  if (context.agentId && context.agentId !== 'default' && swarm.getMember(context.agentId) !== undefined) {
    return context.agentId
  }
  return COORDINATOR_NAME
}

function parseOptionalIntArray(inputs: JsonObject, field: string): number[] | undefined {
  const raw = inputs[field]
  if (raw === undefined) return undefined
  if (!Array.isArray(raw)) {
    throw new ValidationError(field, 'must be an array of positive integers', raw)
  }
  const values: number[] = []
  for (const item of raw) {
    const num = typeof item === 'string' && /^\d+$/.test(item) ? Number(item) : item
    if (typeof num !== 'number' || !Number.isInteger(num) || num < 1) {
      throw new ValidationError(field, 'must contain positive integers', raw)
    }
    values.push(num)
  }
  return values
}

function parseWorkspaceMode(raw: string | undefined): SwarmWorkspaceMode | undefined {
  if (raw === undefined) return undefined
  if (raw === 'inherit' || raw === 'share' || raw === 'branch') return raw
  throw new ValidationError('workspace', 'must be one of "inherit", "share", or "branch"', raw)
}

function parseChannelOp(raw: string | undefined): SwarmChannelOp {
  if (raw === 'create' || raw === 'subscribe' || raw === 'unsubscribe' || raw === 'list' || raw === 'archive') {
    return raw
  }
  throw new ValidationError('op', 'must be one of "create", "subscribe", "unsubscribe", "list", or "archive"', raw)
}

export function visibleSwarmToolDefinitions(
  swarm: Pick<SwarmToolHost, 'isActive'>,
  caller = COORDINATOR_NAME,
  options: { readonly turnStartedActive?: boolean } = {},
): ToolDefinition[] {
  if (caller === COORDINATOR_NAME) {
    const unlocked = options.turnStartedActive ?? swarm.isActive()
    return unlocked
      ? [
          SWARM_TOOL_DEFINITION,
          POST_MESSAGE_TOOL_DEFINITION,
          READ_BOARD_TOOL_DEFINITION,
          SWARM_WAIT_TOOL_DEFINITION,
        ]
      : [SWARM_TOOL_DEFINITION]
  }

  const workerDefs: ToolDefinition[] = [
    SWARM_TOOL_DEFINITION,
    POST_MESSAGE_TOOL_DEFINITION,
    READ_BOARD_TOOL_DEFINITION,
    SWARM_WAIT_TOOL_DEFINITION,
  ]
  if (isNarratorName(caller)) {
    workerDefs.push(SWARM_OVERVIEW_TOOL_DEFINITION)
  }
  return workerDefs
}

export function createSwarmTools(options: CreateSwarmToolsOptions): SwarmToolSuite {
  const { swarm, caller: boundCaller } = options

  const swarmTool: SwarmBoundTool = {
    definition: SWARM_TOOL_DEFINITION,
    capabilities: SWARM_CAPABILITIES,
    handler: async (inputs, context) => {
      const caller = resolveCallerName(boundCaller, context, swarm)
      const action = requiredString(inputs, 'action')
      switch (action) {
        case 'status':
          return swarm.status()
        case 'models': {
          const all = optionalBoolean(inputs, 'all', false)
          return swarm.listModels({ all })
        }
        case 'title': {
          const title = requiredString(inputs, 'title')
          return swarm.setTitle({ caller, title })
        }
        case 'add': {
          const name = requiredString(inputs, 'name')
          const brief = requiredString(inputs, 'brief')
          const model = optionalString(inputs, 'model')
          const nRaw = inputs.n === undefined ? undefined : optionalInteger(inputs, 'n', 1)
          const tasks = inputs.tasks === undefined ? undefined : optionalStringArray(inputs, 'tasks')
          const fork = optionalString(inputs, 'fork')
          const workspace = parseWorkspaceMode(optionalString(inputs, 'workspace'))
          const channels = inputs.channels === undefined ? undefined : optionalStringArray(inputs, 'channels')
          return swarm.add({
            caller,
            name,
            brief,
            ...(model === undefined ? {} : { model }),
            ...(nRaw === undefined ? {} : { n: nRaw }),
            ...(tasks === undefined ? {} : { tasks }),
            ...(fork === undefined ? {} : { fork }),
            ...(workspace === undefined ? {} : { workspace }),
            ...(channels === undefined ? {} : { channels }),
          })
        }
        case 'model': {
          const name = optionalString(inputs, 'name')
          const model = optionalString(inputs, 'model')
          const reason = optionalString(inputs, 'reason')
          return swarm.switchModel({
            caller,
            ...(name === undefined ? {} : { name }),
            ...(model === undefined ? {} : { model }),
            ...(reason === undefined ? {} : { reason }),
          })
        }
        case 'kill': {
          const name = requiredString(inputs, 'name')
          return swarm.kill({ caller, name })
        }
        case 'channel': {
          const op = parseChannelOp(optionalString(inputs, 'op'))
          const channel = optionalString(inputs, 'channel')
          const topic = optionalString(inputs, 'topic')
          const name = optionalString(inputs, 'name')
          return swarm.channelAction({
            caller,
            op,
            ...(channel === undefined ? {} : { channel }),
            ...(topic === undefined ? {} : { topic }),
            ...(name === undefined ? {} : { name }),
          })
        }
        default:
          throw new ValidationError('action', `unsupported swarm action "${action}"`, action)
      }
    },
  }

  const postMessageTool: SwarmBoundTool = {
    definition: POST_MESSAGE_TOOL_DEFINITION,
    capabilities: POST_MESSAGE_CAPABILITIES,
    handler: async (inputs, context) => {
      const caller = resolveCallerName(boundCaller, context, swarm)
      const text = requiredString(inputs, 'text')
      const channel = optionalString(inputs, 'channel')
      const posted = await swarm.postMessage({
        caller,
        text,
        ...(channel === undefined ? {} : { channel }),
      })
      return {
        status: posted.status,
        seq: posted.seq,
        channel: posted.channel,
        mentions: posted.mentions,
        channel_mentions: posted.channel_mentions,
        broadcast: posted.broadcast,
        woken: posted.woken,
      }
    },
  }

  const readBoardTool: SwarmBoundTool = {
    definition: READ_BOARD_TOOL_DEFINITION,
    capabilities: READ_BOARD_CAPABILITIES,
    handler: async (inputs, context) => {
      const caller = resolveCallerName(boundCaller, context, swarm)
      const since = inputs.since === undefined ? undefined : optionalInteger(inputs, 'since', 0)
      if (since !== undefined && since < 0) {
        throw new ValidationError('since', 'must be >= 0', since)
      }
      const limit = inputs.limit === undefined ? undefined : optionalInteger(inputs, 'limit', 20)
      if (limit !== undefined && (limit < 1 || limit > 500)) {
        throw new ValidationError('limit', 'must be between 1 and 500', limit)
      }
      const seqs = parseOptionalIntArray(inputs, 'seqs')
      const channel = optionalString(inputs, 'channel')
      const status = optionalBoolean(inputs, 'status', false)
      return swarm.readBoard({
        caller,
        ...(since === undefined ? {} : { since }),
        ...(limit === undefined ? {} : { limit }),
        ...(seqs === undefined ? {} : { seqs }),
        ...(channel === undefined ? {} : { channel }),
        ...(status ? { status: true } : {}),
      })
    },
  }

  const swarmWaitTool: SwarmBoundTool = {
    definition: SWARM_WAIT_TOOL_DEFINITION,
    capabilities: SWARM_WAIT_CAPABILITIES,
    handler: async (inputs, context, signal) => {
      const caller = resolveCallerName(boundCaller, context, swarm)
      const since = optionalInteger(inputs, 'since', -1)
      if (since < 0 || inputs.since === undefined) {
        throw new ValidationError('since', 'must be a non-negative integer', inputs.since)
      }
      const mentionsOnly = optionalBoolean(inputs, 'mentions_only', true)
      return swarm.waitForBoard({
        caller,
        since,
        mentions_only: mentionsOnly,
        ...(signal === undefined ? {} : { signal }),
        ...(context.wakeSignal === undefined ? {} : { wakeSignal: context.wakeSignal }),
      })
    },
  }

  const swarmOverviewTool: SwarmBoundTool = {
    definition: SWARM_OVERVIEW_TOOL_DEFINITION,
    capabilities: SWARM_OVERVIEW_CAPABILITIES,
    handler: async (inputs, context) => {
      const caller = resolveCallerName(boundCaller, context, swarm)
      const markdown = requiredString(inputs, 'markdown')
      const summary = optionalString(inputs, 'summary')
      return swarm.updateOverview({
        caller,
        markdown,
        ...(summary === undefined ? {} : { summary }),
      })
    },
  }

  return {
    allTools: [swarmTool, postMessageTool, readBoardTool, swarmWaitTool, swarmOverviewTool],
    visibleDefinitions(viewOptions = {}) {
      const effectiveCaller = viewOptions.caller ?? boundCaller ?? COORDINATOR_NAME
      return visibleSwarmToolDefinitions(swarm, effectiveCaller, {
        ...(viewOptions.turnStartedActive === undefined
          ? {}
          : { turnStartedActive: viewOptions.turnStartedActive }),
      })
    },
  }
}

export interface RegisterSwarmToolsOptions extends CreateSwarmToolsOptions {
  readonly agentId?: string
}

export interface SwarmToolRegistration {
  readonly suite: SwarmToolSuite
  visibleDefinitions(options?: { readonly turnStartedActive?: boolean }): ToolDefinition[]
  sync(): void
  dispose(): void
}

/**
 * Register Swarm tools on a Xerxes `ToolRegistry` with dynamic coordinator
 * unlocking: before `swarm.isActive()` becomes true, `coordinator` sees only
 * `swarm`; once the first `add` runs, `post_message`, `read_board`, and
 * `swarm_wait` are registered so the next coordinator turn receives all 4 tools.
 * Workers always receive `swarm`, `post_message`, `read_board`, and `swarm_wait`
 * (plus `swarm_overview` when named `narrator` or `narrator-<N>`).
 */
export function registerSwarmTools(
  registry: ToolRegistry,
  options: RegisterSwarmToolsOptions,
): SwarmToolRegistration {
  const caller = options.caller ?? COORDINATOR_NAME
  const agentId = options.agentId ?? (caller === COORDINATOR_NAME ? 'default' : caller)
  const suite = createSwarmTools({ swarm: options.swarm, caller })
  const byName = new Map(suite.allTools.map(tool => [tool.definition.function.name, tool]))
  const registeredNames = new Set<string>()

  const ensureRegistered = (name: string): void => {
    if (registeredNames.has(name)) return
    const entry = byName.get(name)
    if (!entry) return
    registry.replace(entry.definition, entry.handler, agentId, entry.capabilities)
    registeredNames.add(name)
  }

  const sync = (): void => {
    const visible = visibleSwarmToolDefinitions(options.swarm, caller)
    for (const def of visible) {
      ensureRegistered(def.function.name)
    }
  }

  sync()
  const unsubscribe = caller === COORDINATOR_NAME
    ? options.swarm.onEvent(() => {
        if (options.swarm.isActive()) {
          sync()
        }
      })
    : () => undefined

  return {
    suite,
    visibleDefinitions(turnOptions = {}) {
      return suite.visibleDefinitions({
        caller,
        ...(turnOptions.turnStartedActive === undefined
          ? {}
          : { turnStartedActive: turnOptions.turnStartedActive }),
      })
    },
    sync,
    dispose() {
      unsubscribe()
      for (const name of registeredNames) {
        registry.unregister(name, agentId)
      }
      registeredNames.clear()
    },
  }
}
