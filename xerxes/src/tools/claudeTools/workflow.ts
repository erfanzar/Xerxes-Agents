// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ClientError, ValidationError } from '../../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import { skillPromptSection, type SkillRegistry } from '../../extensions/skills.js'
import { SpawnedAgentManager, type SpawnedAgentDescriptor, type SpawnedAgentSnapshot } from '../../operators/subagents.js'
import { UserPromptManager } from '../../operators/userPrompt.js'
import { resolveInteractionMode, type InteractionMode } from '../../runtime/interactionModes.js'
import type { AgentDefinition } from '../../agents/definitions.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../../types/toolCalls.js'
import { optionalBoolean, optionalString, requiredString } from '../inputs.js'

export type { InteractionMode } from '../../runtime/interactionModes.js'

const DEFAULT_PLAN_TIMEOUT_MS = 120_000

export interface WorkflowTodo {
  readonly content: string
  readonly status: string
}

/** Session-scoped state backing the lightweight Claude workflow tools. */
export class WorkflowState {
  private interaction: InteractionMode = 'code'
  private planMode = false
  private todos: WorkflowTodo[] = []

  get interactionMode(): InteractionMode {
    return this.interaction
  }

  get isPlanMode(): boolean {
    return this.planMode
  }

  todoItems(): readonly WorkflowTodo[] {
    return Object.freeze(this.todos.map(todo => Object.freeze({ ...todo })))
  }

  replaceTodos(todos: readonly WorkflowTodo[]): string {
    this.todos = todos.map(todo => Object.freeze({ content: todo.content, status: todo.status }))
    return todoText(this.todos)
  }

  enterPlanMode(): void {
    this.planMode = true
    this.interaction = 'plan'
  }

  exitPlanMode(): void {
    this.planMode = false
    if (this.interaction === 'plan') this.interaction = 'code'
  }

  setInteractionMode(mode: InteractionMode): void {
    this.interaction = mode
    this.planMode = mode === 'plan'
  }
}

export interface WorktreeCreated {
  readonly base: string
  readonly branch: string
  readonly path: string
}

/** Host port for worktree lifecycle. An integration can replace the native implementation. */
export interface WorktreeManager {
  create(branchName?: string): Promise<WorktreeCreated>
  remove(worktreePath: string, force: boolean): Promise<void>
}

/**
 * Native Bun implementation for ephemeral worktrees created during this process.
 * It deliberately only removes paths it created, avoiding arbitrary `git worktree remove`
 * calls from a model tool invocation.
 */
export class NativeWorktreeManager implements WorktreeManager {
  private readonly created = new Set<string>()

  constructor(private readonly cwd = process.cwd()) {}

  async create(branchName?: string): Promise<WorktreeCreated> {
    const base = (await runGit(['rev-parse', '--show-toplevel'], this.cwd)).trim()
    if (!base) throw new ValidationError('worktree', 'could not resolve the current git repository')
    const branch = branchName?.trim() || `xerxes-worktree-${crypto.randomUUID().replaceAll('-', '').slice(0, 8)}`
    const path = await mkdtemp(join(tmpdir(), 'xerxes-wt-'))
    await rm(path, { force: true, recursive: true })
    try {
      await runGit(['worktree', 'add', '-b', branch, path], base)
    } catch (error) {
      await rm(path, { force: true, recursive: true })
      throw error
    }
    this.created.add(path)
    return Object.freeze({ base, branch, path })
  }

  async remove(worktreePath: string, force: boolean): Promise<void> {
    if (!this.created.has(worktreePath)) {
      throw new ValidationError(
        'worktree_path',
        'was not created by this Bun runtime process and cannot be removed through this manager',
        worktreePath,
      )
    }
    await runGit(['worktree', 'remove', ...(force ? ['--force'] : []), worktreePath], this.cwd)
    this.created.delete(worktreePath)
  }
}

export interface WorkflowPlanStep {
  readonly agent: string
  readonly depends: readonly string[]
  readonly description: string
  readonly id: string
}

export interface WorkflowPlanRequest {
  readonly agents: readonly Pick<AgentDefinition, 'description' | 'name'>[]
  readonly objective: string
}

/** Port that turns an objective into a structured plan. The LLM integration owns this policy. */
export interface WorkflowPlanGenerator {
  generate(request: WorkflowPlanRequest, signal?: AbortSignal): Promise<readonly WorkflowPlanStep[] | string>
}

export interface ClaudeWorkflowToolsOptions {
  readonly agentDefinitions?: readonly AgentDefinition[]
  readonly agentResolver?: (name: string) => SpawnedAgentDescriptor | undefined
  readonly planGenerator?: WorkflowPlanGenerator
  readonly skillRegistry?: SkillRegistry
  readonly state?: WorkflowState
  readonly subagentManager?: SpawnedAgentManager
  readonly userPromptManager?: UserPromptManager
  readonly workspaceRoot?: string
  readonly worktreeManager?: WorktreeManager
}

export const SKILL_TOOL_DEFINITION: ToolDefinition = definition(
  'SkillTool',
  'Render instructions for a discovered named skill.',
  {
    skill_name: stringSchema('Skill name.'),
    args: stringSchema('Optional user request appended to the skill instructions.'),
  },
  ['skill_name'],
)

export const CLAUDE_WORKFLOW_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  definition('TodoWriteTool', 'Replace the session-scoped structured todo list.', {
    todos: { description: 'Array or JSON string of {content, status} todo objects.', type: ['array', 'string'] },
  }, ['todos']),
  definition('AskUserQuestionTool', 'Ask the attached user-prompt manager a blocking clarification question.', {
    question: stringSchema('Question shown to the user.'),
  }, ['question']),
  definition('EnterPlanModeTool', 'Enter plan mode; hosts can use this state to gate mutations.', {}),
  definition('ExitPlanModeTool', 'Leave plan mode and resume normal execution.', {}),
  definition('SetInteractionModeTool', 'Set the current code, researcher, plan, or objective interaction mode.', {
    mode: stringSchema('code, researcher, plan, or objective.'),
    reason: stringSchema('Optional reason for the transition.'),
  }, ['mode']),
  definition('EnterWorktreeTool', 'Create an isolated ephemeral git worktree.', {
    branch_name: stringSchema('Optional new branch name.'),
  }),
  definition('ExitWorktreeTool', 'Remove a worktree created by this runtime process.', {
    worktree_path: stringSchema('Worktree path returned by EnterWorktreeTool.'),
    force: booleanSchema('Permit removal with uncommitted changes.'),
  }, ['worktree_path']),
  definition('ToolSearchTool', 'Search currently registered tool names and descriptions.', {
    query: stringSchema('Tool capability query.'),
  }, ['query']),
  SKILL_TOOL_DEFINITION,
  definition('PlanTool', 'Generate and optionally execute a structured multi-agent plan through an attached planner.', {
    objective: stringSchema('High-level objective.'),
    execute: booleanSchema('Run the generated steps through the subagent manager.'),
  }, ['objective']),
]

/** Register Claude-compatible workflow tools without duplicating core file/process tools. */
export function registerClaudeWorkflowTools(
  registry: ToolRegistry,
  options: ClaudeWorkflowToolsOptions = {},
  agentId = 'default',
): readonly ToolDefinition[] {
  const adapter = new ClaudeWorkflowTools(options, registry)
  for (const tool of CLAUDE_WORKFLOW_TOOL_DEFINITIONS) {
    registry.replace(tool, (inputs, context, signal) => adapter.execute(tool.function.name, inputs, context, signal), agentId)
  }
  return CLAUDE_WORKFLOW_TOOL_DEFINITIONS
}

/** Register only the safe, read-only skill activation surface for a live registry. */
export function registerClaudeSkillTool(
  registry: ToolRegistry,
  skillRegistry: SkillRegistry,
  agentId = 'default',
): ToolDefinition {
  registry.replace(SKILL_TOOL_DEFINITION, inputs => renderSkill(skillRegistry, inputs), agentId)
  return SKILL_TOOL_DEFINITION
}

/** Adapter that owns one session's Claude workflow state and host ports. */
export class ClaudeWorkflowTools {
  readonly state: WorkflowState
  private readonly worktrees: WorktreeManager

  constructor(private readonly options: ClaudeWorkflowToolsOptions, private readonly registry: ToolRegistry) {
    this.state = options.state ?? new WorkflowState()
    this.worktrees = options.worktreeManager ?? new NativeWorktreeManager(options.workspaceRoot ?? process.cwd())
  }

  async execute(
    name: string,
    inputs: JsonObject,
    _context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<unknown> {
    switch (name) {
      case 'TodoWriteTool': return this.todoWrite(inputs)
      case 'AskUserQuestionTool': return this.askUser(inputs, signal)
      case 'EnterPlanModeTool': return this.enterPlanMode()
      case 'ExitPlanModeTool': return this.exitPlanMode()
      case 'SetInteractionModeTool': return this.setInteractionMode(inputs)
      case 'EnterWorktreeTool': return this.enterWorktree(inputs)
      case 'ExitWorktreeTool': return this.exitWorktree(inputs)
      case 'ToolSearchTool': return this.searchTools(inputs)
      case 'SkillTool': return this.skill(inputs)
      case 'PlanTool': return this.plan(inputs, signal)
      default: throw new ValidationError('tool', 'is not handled by ClaudeWorkflowTools', name)
    }
  }

  private todoWrite(inputs: JsonObject): string {
    return this.state.replaceTodos(parseTodos(inputs.todos))
  }

  private async askUser(inputs: JsonObject, signal?: AbortSignal): Promise<string> {
    const manager = this.options.userPromptManager
    if (manager === undefined) {
      throw new ClientError('user_prompt', 'no UserPromptManager is attached to this Claude workflow session')
    }
    const answer = await manager.request({ question: requiredString(inputs, 'question') }, signal)
    return answer.answer
  }

  private enterPlanMode(): Record<string, unknown> {
    this.state.enterPlanMode()
    return { mode: this.state.interactionMode, plan_mode: true, message: 'Entered plan mode. Describe the plan without executing actions.' }
  }

  private exitPlanMode(): Record<string, unknown> {
    this.state.exitPlanMode()
    return { mode: this.state.interactionMode, plan_mode: false, message: 'Exited plan mode. Resuming normal execution.' }
  }

  private setInteractionMode(inputs: JsonObject): Record<string, unknown> {
    const mode = requiredInteractionMode(requiredString(inputs, 'mode'))
    const reason = optionalString(inputs, 'reason')?.trim()
    this.state.setInteractionMode(mode)
    return {
      mode,
      plan_mode: this.state.isPlanMode,
      ...(reason ? { reason } : {}),
      message: `Interaction mode switched to ${mode}.${reason ? ` Reason: ${reason}` : ''}`,
    }
  }

  private async enterWorktree(inputs: JsonObject): Promise<Record<string, unknown>> {
    const branchName = optionalString(inputs, 'branch_name')?.trim()
    const created = await this.worktrees.create(branchName)
    return { path: created.path, branch: created.branch, base: created.base }
  }

  private async exitWorktree(inputs: JsonObject): Promise<Record<string, unknown>> {
    const path = requiredString(inputs, 'worktree_path')
    const force = optionalBoolean(inputs, 'force', false)
    await this.worktrees.remove(path, force)
    return { path, removed: true }
  }

  private searchTools(inputs: JsonObject): readonly Record<string, unknown>[] {
    const query = requiredString(inputs, 'query').toLowerCase().trim()
    const terms = query.split(/\s+/).filter(Boolean)
    const matches = this.registry.definitions().map(tool => {
      const name = tool.function.name
      const description = tool.function.description
      const haystack = `${name} ${description}`.toLowerCase()
      const score = terms.reduce((total, term) => total + (name.toLowerCase().includes(term) ? 3 : haystack.includes(term) ? 1 : 0), 0)
      return { name, description, score }
    }).filter(match => match.score > 0)
      .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
      .slice(0, 10)
    return matches
  }

  private skill(inputs: JsonObject): string {
    const registry = this.options.skillRegistry
    if (registry === undefined) {
      throw new ClientError('skills', 'no SkillRegistry is attached to this Claude workflow session')
    }
    return renderSkill(registry, inputs)
  }

  private async plan(inputs: JsonObject, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const generator = this.options.planGenerator
    if (generator === undefined) {
      throw new ClientError(
        'planner',
        'no WorkflowPlanGenerator is attached; connect an LLM planner before invoking PlanTool',
      )
    }
    const objective = requiredString(inputs, 'objective')
    const generated = await generator.generate({
      objective,
      agents: (this.options.agentDefinitions ?? []).map(agent => ({ name: agent.name, description: agent.description })),
    }, signal)
    const steps = typeof generated === 'string' ? parsePlanXml(generated) : normalizePlanSteps(generated)
    if (!steps.length) throw new ValidationError('plan', 'planner generated no usable steps')
    const summary = { objective, steps: steps.map(planStepWire) }
    if (!optionalBoolean(inputs, 'execute', true)) return { ...summary, executed: false }
    const manager = this.options.subagentManager
    if (manager === undefined) {
      throw new ClientError('subagents', 'PlanTool execution requires an attached SpawnedAgentManager')
    }
    return { ...summary, executed: true, results: await this.executePlan(manager, steps, signal) }
  }

  private async executePlan(
    manager: SpawnedAgentManager,
    steps: readonly WorkflowPlanStep[],
    signal?: AbortSignal,
  ): Promise<readonly Record<string, unknown>[]> {
    const remaining = new Map(steps.map(step => [step.id, step]))
    const completed = new Map<string, string>()
    const results: Record<string, unknown>[] = []
    while (remaining.size) {
      if (signal?.aborted) throw signal.reason ?? new Error('Plan execution cancelled')
      const ready = [...remaining.values()].filter(step => step.depends.every(dependency => completed.has(dependency)))
      if (!ready.length) {
        return [...results, { status: 'deadlocked', unresolved_steps: [...remaining.keys()] }]
      }
      const spawned = await Promise.all(ready.map(async step => {
        const agent = this.options.agentResolver?.(step.agent)
        const prompt = planStepPrompt(step, completed)
        const snapshot = await manager.spawn({
          agent: agent ?? { id: step.agent, name: step.agent },
          message: prompt,
          nickname: `plan-${step.id}-${crypto.randomUUID().replaceAll('-', '').slice(0, 6)}`,
        })
        return { step, snapshot }
      }))
      const settled = await manager.wait(spawned.map(entry => entry.snapshot.id), DEFAULT_PLAN_TIMEOUT_MS)
      const snapshots = new Map([...settled.completed, ...settled.pending].map(snapshot => [snapshot.id, snapshot]))
      for (const entry of spawned) {
        const snapshot = snapshots.get(entry.snapshot.id) ?? entry.snapshot
        const result = snapshot.lastOutput ?? snapshot.error ?? ''
        results.push({ ...planStepWire(entry.step), ...planSnapshotWire(snapshot), result: result.slice(0, 2_000) })
        if (isTerminal(snapshot)) {
          completed.set(entry.step.id, result)
          remaining.delete(entry.step.id)
        }
      }
      if (spawned.some(entry => !isTerminal(snapshots.get(entry.snapshot.id) ?? entry.snapshot))) {
        return results
      }
    }
    return results
  }
}

function renderSkill(registry: SkillRegistry, inputs: JsonObject): string {
  const skillName = requiredString(inputs, 'skill_name')
  const skill = registry.get(skillName)
  if (skill === undefined) {
    const names = registry.names
    return names.length
      ? `Skill '${skillName}' not found. Available: ${names.slice(0, 20).join(', ')}`
      : `Skill '${skillName}' not found. No skills discovered.`
  }
  const args = optionalString(inputs, 'args')?.trim()
  return `[Skill: ${skillName}]\n${skillPromptSection(skill)}${args ? `\n\nUser request: ${args}` : ''}`
}

/** Parse the XML shape emitted by the legacy Python planner. */
export function parsePlanXml(xml: string): WorkflowPlanStep[] {
  const steps: WorkflowPlanStep[] = []
  const matcher = /<step\b([^>]*)>\s*<description>([\s\S]*?)<\/description>\s*<\/step>/g
  for (const match of xml.matchAll(matcher)) {
    const attributes = match[1] ?? ''
    const id = xmlAttribute(attributes, 'id')
    const agent = xmlAttribute(attributes, 'agent')
    const depends = xmlAttribute(attributes, 'depends') ?? ''
    const description = (match[2] ?? '').trim()
    if (!id || !agent || !description) continue
    steps.push({ id, agent, description, depends: depends.split(',').map(value => value.trim()).filter(Boolean) })
  }
  return normalizePlanSteps(steps)
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

function booleanSchema(description: string): Record<string, unknown> {
  return { type: 'boolean', description }
}

function parseTodos(value: JsonValue | undefined): WorkflowTodo[] {
  const parsed = arrayValue(value, 'todos')
  return parsed.map((item, index) => {
    if (!isRecord(item)) throw new ValidationError('todos', `entry ${index} must be an object`, item)
    const content = item.content
    const status = item.status
    if (typeof content !== 'string' || !content.trim()) throw new ValidationError('todos', `entry ${index} requires non-empty content`, item)
    if (status !== undefined && typeof status !== 'string') throw new ValidationError('todos', `entry ${index} status must be a string`, item)
    return Object.freeze({ content: content.trim(), status: typeof status === 'string' && status.trim() ? status.trim() : 'pending' })
  })
}

function todoText(todos: readonly WorkflowTodo[]): string {
  const lines = ['# Todo List', '']
  for (const [index, todo] of todos.entries()) {
    const icon = todo.status === 'completed' ? '[x]' : todo.status === 'in_progress' ? '[~]' : '[ ]'
    lines.push(`${index + 1}. ${icon} ${todo.content}`)
  }
  const completed = todos.filter(todo => todo.status === 'completed').length
  lines.push(`\nProgress: ${completed}/${todos.length}`)
  return lines.join('\n')
}

function requiredInteractionMode(value: string): InteractionMode {
  const mode = resolveInteractionMode(value)
  if (mode === undefined) throw new ValidationError('mode', 'must be code, researcher, plan, or objective', value)
  return mode
}

function normalizePlanSteps(steps: readonly WorkflowPlanStep[]): WorkflowPlanStep[] {
  const ids = new Set<string>()
  return steps.map((step, index) => {
    const id = step.id.trim()
    const agent = step.agent.trim()
    const description = step.description.trim()
    if (!id || !agent || !description) throw new ValidationError('plan', `step ${index} requires id, agent, and description`)
    if (ids.has(id)) throw new ValidationError('plan', `contains duplicate step id ${id}`)
    ids.add(id)
    return Object.freeze({ id, agent, description, depends: Object.freeze(step.depends.map(value => value.trim()).filter(Boolean)) })
  })
}

function planStepPrompt(step: WorkflowPlanStep, completed: ReadonlyMap<string, string>): string {
  const prior = step.depends.map(id => `### Step ${id}\n${completed.get(id)?.slice(0, 1_000) ?? ''}`).join('\n\n')
  return `## Task (Step ${step.id} of plan)\n\n${step.description}${prior ? `\n\n## Results from previous steps\n${prior}` : ''}`
}

function planStepWire(step: WorkflowPlanStep): Record<string, unknown> {
  return { id: step.id, agent: step.agent, depends: [...step.depends], description: step.description }
}

function planSnapshotWire(snapshot: SpawnedAgentSnapshot): Record<string, unknown> {
  return { task_id: snapshot.id, task_name: snapshot.name, status: snapshot.status, error: snapshot.error ?? null }
}

function xmlAttribute(attributes: string, name: string): string | undefined {
  const match = attributes.match(new RegExp(`\\b${name}="([^"]*)"`))
  return match?.[1]
}

function isTerminal(snapshot: SpawnedAgentSnapshot): boolean {
  return snapshot.status === 'cancelled' || snapshot.status === 'closed' || snapshot.status === 'completed' || snapshot.status === 'error'
}

function arrayValue(value: JsonValue | undefined, name: string): JsonValue[] {
  if (Array.isArray(value)) return value
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown
      if (Array.isArray(parsed)) return parsed as JsonValue[]
    } catch {
      // The structured error below includes the input field name.
    }
  }
  throw new ValidationError(name, 'must be an array or a JSON-encoded array', value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function runGit(arguments_: readonly string[], cwd: string): Promise<string> {
  let process: Bun.ReadableSubprocess
  try {
    process = Bun.spawn(['git', ...arguments_], { cwd, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
  } catch (error) {
    throw new ValidationError('git', error instanceof Error ? error.message : String(error))
  }
  const [exitCode, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ])
  if (exitCode !== 0) throw new ValidationError('git', stderr.trim() || `git exited with code ${exitCode}`)
  return stdout
}
