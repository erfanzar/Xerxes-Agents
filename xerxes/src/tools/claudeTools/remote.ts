// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { CronJob, JobStore, nextFireAt } from '../../cron/index.js'
import { ValidationError } from '../../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import { checkUrl, type UrlSafetyOptions } from '../../security/urlSafety.js'
import type { JsonObject, ToolDefinition } from '../../types/toolCalls.js'
import { optionalString, requiredString } from '../inputs.js'

const MAX_REMOTE_RESPONSE_CHARS = 8_000

/**
 * Cap on a model-supplied cron prompt. The prompt is persisted and
 * re-executed on every schedule tick, so an unbounded body would bloat the
 * durable job store; 4k characters is generous for a scheduled instruction.
 */
const MAX_CRON_PROMPT_CHARS = 4_000

/** Attempts to mint a collision-free server-side cron job id before failing. */
const CRON_ID_GENERATION_ATTEMPTS = 3

export interface RemoteTriggerEndpoint {
  readonly headers?: Readonly<Record<string, string>>
  readonly method?: 'POST' | 'PUT'
  readonly name: string
  readonly url: string
}

export interface RemoteTriggerResponse {
  readonly ok: boolean
  readonly status: number
  text(): Promise<string>
}

export type RemoteTriggerFetch = (url: string, init: RequestInit) => Promise<RemoteTriggerResponse>

export interface RemoteTriggerRegistryOptions {
  readonly fetcher?: RemoteTriggerFetch
  readonly urlSafety?: UrlSafetyOptions
}

/** Explicitly configured webhook target registry; tool callers cannot supply arbitrary URLs. */
export class RemoteTriggerRegistry {
  private readonly endpoints = new Map<string, RemoteTriggerEndpoint>()
  private readonly fetcher: RemoteTriggerFetch
  private readonly urlSafety: UrlSafetyOptions

  constructor(options: RemoteTriggerRegistryOptions = {}) {
    this.fetcher = options.fetcher ?? nativeFetch
    this.urlSafety = options.urlSafety ?? {}
  }

  register(endpoint: RemoteTriggerEndpoint): void {
    const name = endpoint.name.trim()
    if (!name) throw new ValidationError('trigger_name', 'must not be empty')
    const safety = checkUrl(endpoint.url, this.urlSafety)
    if (!safety.allowed) throw new ValidationError('url', `is not allowed: ${safety.reason}`, endpoint.url)
    this.endpoints.set(name, Object.freeze({
      name,
      url: endpoint.url,
      method: endpoint.method ?? 'POST',
      ...(endpoint.headers === undefined ? {} : { headers: Object.freeze({ ...endpoint.headers }) }),
    }))
  }

  names(): readonly string[] {
    return Object.freeze([...this.endpoints.keys()].sort())
  }

  async trigger(name: string, payload: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const endpoint = this.endpoints.get(name)
    if (endpoint === undefined) {
      const known = this.names()
      throw new ValidationError(
        'trigger_name',
        known.length ? `is not configured; available triggers: ${known.join(', ')}` : 'is not configured and no triggers are attached',
        name,
      )
    }
    const response = await this.fetcher(endpoint.url, {
      method: endpoint.method ?? 'POST',
      headers: { 'content-type': 'text/plain; charset=utf-8', ...endpoint.headers },
      body: payload,
      ...(signal === undefined ? {} : { signal }),
    })
    const body = (await response.text()).slice(0, MAX_REMOTE_RESPONSE_CHARS)
    return {
      name,
      status: response.status,
      ok: response.ok,
      response: body,
    }
  }
}

export interface ClaudeRemoteToolsOptions {
  readonly cronStore?: JobStore
  readonly remoteTriggers?: RemoteTriggerRegistry
}

export const CLAUDE_REMOTE_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  definition('RemoteTriggerTool', 'Send a payload to a named, explicitly configured remote trigger endpoint.', {
    trigger_name: stringSchema('Configured remote trigger name.'),
    payload: stringSchema('Payload sent to the endpoint.'),
  }, ['trigger_name']),
  definition('ScheduleCronTool', 'Persist a cron-triggered agent prompt in the attached scheduler store. Requires explicit user approval; job ids are generated server-side.', {
    schedule: stringSchema('Five-field UTC cron schedule.'),
    prompt: stringSchema(`Agent prompt to run on each occurrence (at most ${MAX_CRON_PROMPT_CHARS} characters).`),
    name: stringSchema('Optional human-readable label stored as job metadata; the durable job id is always generated server-side.'),
  }, ['schedule', 'prompt']),
]

/** Register functional remote-trigger and cron scheduling adapters. */
export function registerClaudeRemoteTools(
  registry: ToolRegistry,
  options: ClaudeRemoteToolsOptions,
  agentId = 'default',
): readonly ToolDefinition[] {
  const adapter = new ClaudeRemoteTools(options)
  for (const tool of CLAUDE_REMOTE_TOOL_DEFINITIONS) {
    registry.replace(tool, (inputs, context, signal) => adapter.execute(tool.function.name, inputs, context, signal), agentId)
  }
  return CLAUDE_REMOTE_TOOL_DEFINITIONS
}

/** Native remote and scheduler tool surface with explicit host configuration. */
export class ClaudeRemoteTools {
  constructor(private readonly options: ClaudeRemoteToolsOptions) {}

  async execute(
    name: string,
    inputs: JsonObject,
    _context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<unknown> {
    switch (name) {
      case 'RemoteTriggerTool': return this.trigger(inputs, signal)
      case 'ScheduleCronTool': return this.schedule(inputs)
      default: throw new ValidationError('tool', 'is not handled by ClaudeRemoteTools', name)
    }
  }

  private async trigger(inputs: JsonObject, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const triggers = this.options.remoteTriggers
    if (triggers === undefined) {
      throw new ValidationError('RemoteTriggerTool', 'requires an attached RemoteTriggerRegistry')
    }
    return triggers.trigger(requiredString(inputs, 'trigger_name'), optionalString(inputs, 'payload') ?? '', signal)
  }

  private schedule(inputs: JsonObject): Record<string, unknown> {
    const store = this.options.cronStore
    if (store === undefined) {
      throw new ValidationError('ScheduleCronTool', 'requires an attached persistent JobStore')
    }
    const schedule = requiredString(inputs, 'schedule')
    const prompt = requiredString(inputs, 'prompt')
    if (prompt.length > MAX_CRON_PROMPT_CHARS) {
      throw new ValidationError('prompt', `must be at most ${MAX_CRON_PROMPT_CHARS} characters`, prompt.slice(0, 80))
    }
    // The caller-supplied `name` is only a human-readable label. The durable
    // job id is always minted server-side and checked against the store, so a
    // model can never overwrite (or squat on) an existing job by choosing its
    // id. This tool also sits in the ALWAYS_APPROVAL_TOOLS permission tier,
    // so persisting a re-executing prompt requires explicit user approval.
    const label = optionalString(inputs, 'name')?.trim()
    const nextRunAt = nextFireAt(schedule).toISOString()
    const job = new CronJob({
      id: mintJobId(store),
      prompt,
      schedule,
      nextRunAt,
      ...(label === undefined ? {} : { metadata: { name: label } }),
    })
    store.add(job)
    return {
      id: job.id,
      name: label ?? null,
      schedule: job.schedule,
      prompt: job.prompt,
      next_run_at: job.nextRunAt ?? null,
      paused: job.paused,
    }
  }
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: { type: 'object', additionalProperties: false, properties, ...(required.length ? { required } : {}) },
    },
  }
}

function stringSchema(description: string): Record<string, unknown> {
  return { type: 'string', description }
}

/** Mint a server-side job id that does not collide with any persisted job. */
function mintJobId(store: JobStore): string {
  for (let attempt = 0; attempt < CRON_ID_GENERATION_ATTEMPTS; attempt += 1) {
    const id = store.newId()
    if (store.get(id) === undefined) return id
  }
  throw new ValidationError('ScheduleCronTool', 'could not mint a collision-free cron job id')
}

async function nativeFetch(url: string, init: RequestInit): Promise<RemoteTriggerResponse> {
  return fetch(url, init)
}
