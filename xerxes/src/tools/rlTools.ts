// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { isJsonObject, type JsonObject, type JsonValue, type ToolDefinition } from '../types/toolCalls.js'
import { optionalString, requiredString } from './inputs.js'

const DEFAULT_RUN_ID_PREFIX = 'run-'
const RUN_ID_WIDTH = 6
const MAX_INFERENCE_COMPLETION_LENGTH = 80
const MAX_INFERENCE_PROMPT_LENGTH = 200

export type MaybePromise<Value> = Promise<Value> | Value
export type RLRunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'stopped'
export type RLConfig = Readonly<Record<string, JsonValue>>

/** A selectable training environment exposed by an RL backend. */
export interface RLEnvironment {
  readonly config: RLConfig
  readonly description: string
  readonly name: string
}

/** A normalized representation of one training run. */
export interface RLRun {
  readonly config: RLConfig
  readonly endedAt: number
  readonly environment: string
  readonly metrics: RLConfig
  readonly results: RLConfig
  readonly runId: string
  readonly startedAt: number
  readonly status: RLRunStatus
}

export interface RLCurrentConfig {
  readonly config: RLConfig
  readonly environment: string | null
}

/**
 * Host boundary for reinforcement-learning services.
 *
 * Remote integrations must be supplied explicitly by the host. The core does
 * not infer endpoints, credentials, or backend state from environment variables.
 */
export interface RLBackend {
  editConfig(updates: JsonObject): MaybePromise<RLCurrentConfig>
  getCurrentConfig(): MaybePromise<RLCurrentConfig>
  listEnvironments(): MaybePromise<readonly RLEnvironment[]>
  listRuns(): MaybePromise<readonly RLRun[]>
  results(runId: string): MaybePromise<JsonObject>
  selectEnvironment(name: string): MaybePromise<RLEnvironment | undefined>
  start(): MaybePromise<RLRun>
  status(runId: string): MaybePromise<RLRun | undefined>
  stop(runId: string): MaybePromise<RLRun | undefined>
  testInference(prompt: string, runId?: string): MaybePromise<JsonObject>
}

export interface InMemoryRLBackendOptions {
  /**
   * Clock for lifecycle timestamps. The default is a per-backend logical clock
   * so development and test responses are deterministic.
   */
  readonly clock?: () => number
  /** Prefix used for deterministic sequential run identifiers. */
  readonly runIdPrefix?: string
}

/**
 * Deterministic local RL backend for development and tests.
 *
 * It never claims to train or serve a real model. Inference responses include
 * explicit simulated/backend markers, and real integrations must implement
 * RLBackend and be injected by the host.
 */
export class InMemoryRLBackend implements RLBackend {
  private readonly clock: () => number
  private readonly environments = new Map<string, RLEnvironment>()
  private readonly runIdPrefix: string
  private readonly runs = new Map<string, RLRun>()
  private config: JsonObject = {}
  private nextRunNumber = 1
  private selected: string | undefined

  constructor(options: InMemoryRLBackendOptions = {}) {
    this.clock = options.clock ?? logicalClock()
    this.runIdPrefix = nonBlank(options.runIdPrefix ?? DEFAULT_RUN_ID_PREFIX, 'runIdPrefix')
  }

  /** Add or replace a local environment available to this backend. */
  register(name: string, config: JsonObject = {}, description = ''): RLEnvironment {
    const normalizedName = nonBlank(name, 'name')
    const environment: RLEnvironment = {
      config: cloneObject(config),
      description,
      name: normalizedName,
    }
    this.environments.set(normalizedName, environment)
    return cloneEnvironment(environment)
  }

  listEnvironments(): readonly RLEnvironment[] {
    return [...this.environments.values()].map(cloneEnvironment)
  }

  selectEnvironment(name: string): RLEnvironment | undefined {
    const environment = this.environments.get(name)
    if (!environment) {
      return undefined
    }
    this.selected = environment.name
    this.config = cloneObject(environment.config)
    return cloneEnvironment(environment)
  }

  getCurrentConfig(): RLCurrentConfig {
    return {
      config: cloneObject(this.config),
      environment: this.selected ?? null,
    }
  }

  editConfig(updates: JsonObject): RLCurrentConfig {
    this.config = { ...this.config, ...cloneObject(updates) }
    return this.getCurrentConfig()
  }

  start(): RLRun {
    if (!this.selected) {
      throw new Error('no environment selected')
    }

    const run: RLRun = {
      config: cloneObject(this.config),
      endedAt: 0,
      environment: this.selected,
      metrics: { reward: 0, step: 0 },
      results: {},
      runId: this.nextRunId(),
      startedAt: this.now(),
      status: 'running',
    }
    this.runs.set(run.runId, run)
    return cloneRun(run)
  }

  stop(runId: string): RLRun | undefined {
    const current = this.runs.get(runId)
    if (!current) {
      return undefined
    }
    if (current.status === 'running') {
      const stopped: RLRun = {
        ...current,
        endedAt: this.now(),
        status: 'stopped',
      }
      this.runs.set(runId, stopped)
      return cloneRun(stopped)
    }
    return cloneRun(current)
  }

  status(runId: string): RLRun | undefined {
    const run = this.runs.get(runId)
    return run ? cloneRun(run) : undefined
  }

  results(runId: string): JsonObject {
    const run = this.runs.get(runId)
    if (!run) {
      return { error: 'not_found' }
    }
    if (run.status === 'running') {
      return { partial_metrics: cloneObject(run.metrics), status: 'running' }
    }
    const endedAt = run.endedAt || this.now()
    return {
      duration_s: Math.max(0, endedAt - run.startedAt),
      metrics: cloneObject(run.metrics),
      results: cloneObject(run.results),
      status: run.status,
    }
  }

  listRuns(): readonly RLRun[] {
    return [...this.runs.values()].map(cloneRun)
  }

  testInference(prompt: string, runId?: string): JsonObject {
    const boundedPrompt = prompt.slice(0, MAX_INFERENCE_PROMPT_LENGTH)
    return {
      backend: 'in_memory',
      completion: '[in-memory simulation] ' + boundedPrompt.slice(0, MAX_INFERENCE_COMPLETION_LENGTH),
      prompt: boundedPrompt,
      run_id: runId ?? this.selected ?? 'ad-hoc',
      simulated: true,
    }
  }

  private nextRunId(): string {
    const suffix = String(this.nextRunNumber).padStart(RUN_ID_WIDTH, '0')
    this.nextRunNumber += 1
    return this.runIdPrefix + suffix
  }

  private now(): number {
    const value = this.clock()
    if (!Number.isFinite(value)) {
      throw new Error('RL backend clock must return a finite number')
    }
    return value
  }
}

let globalBackend: RLBackend = new InMemoryRLBackend()

/** Replace the process-local default backend for single-backend hosts. */
export function setRlBackend(backend: RLBackend): void {
  if (!backend) {
    throw new ValidationError('backend', 'must be supplied', backend)
  }
  globalBackend = backend
}

/** Return the process-local default backend. Multi-session hosts should inject a resolver instead. */
export function getRlBackend(): RLBackend {
  return globalBackend
}

/** Restore the deliberate deterministic in-memory development backend. */
export function resetRlBackend(): void {
  globalBackend = new InMemoryRLBackend()
}

export const RL_LIST_ENVIRONMENTS_DEFINITION: ToolDefinition = definition(
  'rl_list_environments',
  'List training environments exposed by the configured reinforcement-learning backend.',
)

export const RL_SELECT_ENVIRONMENT_DEFINITION: ToolDefinition = definition(
  'rl_select_environment',
  'Select one configured reinforcement-learning environment for subsequent training.',
  { name: { type: 'string', description: 'Registered environment name.' } },
  ['name'],
)

export const RL_GET_CURRENT_CONFIG_DEFINITION: ToolDefinition = definition(
  'rl_get_current_config',
  'Return the currently selected environment and its pending training configuration.',
)

export const RL_EDIT_CONFIG_DEFINITION: ToolDefinition = definition(
  'rl_edit_config',
  'Merge JSON configuration updates into the selected reinforcement-learning run configuration.',
  { updates: { type: 'object', description: 'JSON configuration values to merge.' } },
  ['updates'],
)

export const RL_START_TRAINING_DEFINITION: ToolDefinition = definition(
  'rl_start_training',
  'Start a training run for the currently selected reinforcement-learning environment.',
)

export const RL_STOP_TRAINING_DEFINITION: ToolDefinition = definition(
  'rl_stop_training',
  'Stop a queued or running reinforcement-learning training run.',
  { run_id: { type: 'string', description: 'Training run identifier.' } },
  ['run_id'],
)

export const RL_CHECK_STATUS_DEFINITION: ToolDefinition = definition(
  'rl_check_status',
  'Get a normalized status snapshot for one reinforcement-learning training run.',
  { run_id: { type: 'string', description: 'Training run identifier.' } },
  ['run_id'],
)

export const RL_GET_RESULTS_DEFINITION: ToolDefinition = definition(
  'rl_get_results',
  'Get available metrics and final results for one reinforcement-learning training run.',
  { run_id: { type: 'string', description: 'Training run identifier.' } },
  ['run_id'],
)

export const RL_LIST_RUNS_DEFINITION: ToolDefinition = definition(
  'rl_list_runs',
  'List known reinforcement-learning training runs, newest first.',
)

export const RL_TEST_INFERENCE_DEFINITION: ToolDefinition = definition(
  'rl_test_inference',
  'Run inference through the configured reinforcement-learning backend or selected run.',
  {
    prompt: { type: 'string', description: 'Inference prompt.' },
    run_id: { type: 'string', description: 'Optional training run identifier.' },
  },
  ['prompt'],
)

export const RL_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  RL_LIST_ENVIRONMENTS_DEFINITION,
  RL_SELECT_ENVIRONMENT_DEFINITION,
  RL_GET_CURRENT_CONFIG_DEFINITION,
  RL_EDIT_CONFIG_DEFINITION,
  RL_START_TRAINING_DEFINITION,
  RL_STOP_TRAINING_DEFINITION,
  RL_CHECK_STATUS_DEFINITION,
  RL_GET_RESULTS_DEFINITION,
  RL_LIST_RUNS_DEFINITION,
  RL_TEST_INFERENCE_DEFINITION,
]

/**
 * Per-host RL tool configuration.
 *
 * Supplying a resolver opts into explicit multi-session resolution: when it
 * returns undefined, the tool reports backend_not_configured instead of
 * silently falling back to the global development backend.
 */
export interface RLToolsOptions {
  readonly backend?: RLBackend
  readonly resolveBackend?: (
    context: ToolExecutionContext,
  ) => MaybePromise<RLBackend | undefined>
}

/** Register every rl_* tool against an explicit backend, resolver, or deliberate global development backend. */
export function registerRlTools(registry: ToolRegistry, options: RLToolsOptions = {}, agentId = 'default'): void {
  registry.register(RL_LIST_ENVIRONMENTS_DEFINITION, (inputs, context) => rlListEnvironments(inputs, context, options), agentId)
  registry.register(RL_SELECT_ENVIRONMENT_DEFINITION, (inputs, context) => rlSelectEnvironment(inputs, context, options), agentId)
  registry.register(RL_GET_CURRENT_CONFIG_DEFINITION, (inputs, context) => rlGetCurrentConfig(inputs, context, options), agentId)
  registry.register(RL_EDIT_CONFIG_DEFINITION, (inputs, context) => rlEditConfig(inputs, context, options), agentId)
  registry.register(RL_START_TRAINING_DEFINITION, (inputs, context) => rlStartTraining(inputs, context, options), agentId)
  registry.register(RL_STOP_TRAINING_DEFINITION, (inputs, context) => rlStopTraining(inputs, context, options), agentId)
  registry.register(RL_CHECK_STATUS_DEFINITION, (inputs, context) => rlCheckStatus(inputs, context, options), agentId)
  registry.register(RL_GET_RESULTS_DEFINITION, (inputs, context) => rlGetResults(inputs, context, options), agentId)
  registry.register(RL_LIST_RUNS_DEFINITION, (inputs, context) => rlListRuns(inputs, context, options), agentId)
  registry.register(RL_TEST_INFERENCE_DEFINITION, (inputs, context) => rlTestInference(inputs, context, options), agentId)
}

export async function rlListEnvironments(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  return withBackend(context, options, async backend => {
    const environments = await backend.listEnvironments()
    return {
      count: environments.length,
      environments: environments.map(environmentToObject),
    }
  })
}

export async function rlSelectEnvironment(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const name = requiredString(inputs, 'name')
  return withBackend(context, options, async backend => {
    const environment = await backend.selectEnvironment(name)
    if (!environment) {
      return { error: 'not_found', name }
    }
    return environmentToObject(environment)
  })
}

export async function rlGetCurrentConfig(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  return withBackend(context, options, async backend => {
    const current = await backend.getCurrentConfig()
    return { config: cloneObject(current.config), environment: current.environment }
  })
}

export async function rlEditConfig(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const updates = requiredJsonObject(inputs, 'updates')
  return withBackend(context, options, async backend => {
    const current = await backend.editConfig(updates)
    return { config: cloneObject(current.config), environment: current.environment }
  })
}

export async function rlStartTraining(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  return withBackend(context, options, async backend => runToObject(await backend.start()))
}

export async function rlStopTraining(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const runId = requiredString(inputs, 'run_id')
  return withBackend(context, options, async backend => {
    const run = await backend.stop(runId)
    return run ? runToObject(run) : { error: 'not_found', run_id: runId }
  })
}

export async function rlCheckStatus(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const runId = requiredString(inputs, 'run_id')
  return withBackend(context, options, async backend => {
    const run = await backend.status(runId)
    return run ? runToObject(run) : { error: 'not_found', run_id: runId }
  })
}

export async function rlGetResults(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const runId = requiredString(inputs, 'run_id')
  return withBackend(context, options, async backend => cloneObject(await backend.results(runId)))
}

export async function rlListRuns(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  return withBackend(context, options, async backend => {
    const runs = [...(await backend.listRuns())].sort(newestRunFirst)
    return { count: runs.length, runs: runs.map(runToObject) }
  })
}

export async function rlTestInference(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: RLToolsOptions = {},
): Promise<JsonObject> {
  const prompt = requiredString(inputs, 'prompt')
  const runId = optionalString(inputs, 'run_id')
  return withBackend(context, options, async backend => cloneObject(await backend.testInference(prompt, runId)))
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown> = {},
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length ? { required } : {}),
      },
    },
  }
}

async function withBackend(
  context: ToolExecutionContext,
  options: RLToolsOptions,
  operation: (backend: RLBackend) => Promise<JsonObject>,
): Promise<JsonObject> {
  const backend = await resolveBackend(context, options)
  if (!backend) {
    return {
      error: 'backend_not_configured',
      message: 'No reinforcement-learning backend is configured for this session.',
      ok: false,
    }
  }
  try {
    return await operation(backend)
  } catch (error) {
    return {
      error: 'backend_error',
      message: error instanceof Error ? error.message : String(error),
      ok: false,
    }
  }
}

async function resolveBackend(context: ToolExecutionContext, options: RLToolsOptions): Promise<RLBackend | undefined> {
  if (options.backend) {
    return options.backend
  }
  if (options.resolveBackend) {
    return options.resolveBackend(context)
  }
  return getRlBackend()
}

function environmentToObject(environment: RLEnvironment): JsonObject {
  return {
    config: cloneObject(environment.config),
    description: environment.description,
    name: environment.name,
  }
}

function runToObject(run: RLRun): JsonObject {
  return {
    ended_at: run.endedAt,
    environment: run.environment,
    metrics: cloneObject(run.metrics),
    run_id: run.runId,
    started_at: run.startedAt,
    status: run.status,
  }
}

function cloneEnvironment(environment: RLEnvironment): RLEnvironment {
  return {
    config: cloneObject(environment.config),
    description: environment.description,
    name: environment.name,
  }
}

function cloneRun(run: RLRun): RLRun {
  return {
    config: cloneObject(run.config),
    endedAt: run.endedAt,
    environment: run.environment,
    metrics: cloneObject(run.metrics),
    results: cloneObject(run.results),
    runId: run.runId,
    startedAt: run.startedAt,
    status: run.status,
  }
}

function cloneObject(value: RLConfig | JsonObject): JsonObject {
  const copied: JsonObject = {}
  for (const [key, item] of Object.entries(value)) {
    copied[key] = cloneJson(item)
  }
  return copied
}

function cloneJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    return value.map(cloneJson)
  }
  if (isJsonObject(value)) {
    return cloneObject(value)
  }
  return value
}

function requiredJsonObject(inputs: JsonObject, field: string): JsonObject {
  const value = inputs[field]
  if (!isJsonObject(value)) {
    throw new ValidationError(field, 'must be a JSON object', value)
  }
  return cloneObject(value)
}

function newestRunFirst(left: RLRun, right: RLRun): number {
  if (left.startedAt !== right.startedAt) {
    return right.startedAt - left.startedAt
  }
  return right.runId.localeCompare(left.runId)
}

function logicalClock(): () => number {
  let value = 0
  return () => {
    const current = value
    value += 1
    return current
  }
}

function nonBlank(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new ValidationError(field, 'must be a non-empty string', value)
  }
  return normalized
}
