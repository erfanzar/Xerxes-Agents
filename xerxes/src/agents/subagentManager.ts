// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from './definitions.js'
import {
  MAX_AGENT_TITLE_LENGTH,
  normalizeAgentTitle,
  SpawnedAgentManager,
  type SpawnedAgentSnapshot,
} from '../operators/subagents.js'

export const SUBAGENT_CALLER_PROMPT = [
  'You are now running as a subagent. All the user messages are sent by the main agent.',
  'The main agent cannot see your context; it can only see your last message when you finish the task.',
  'You must treat the parent agent as your caller. Do not directly ask the end user questions.',
  'If something is unclear, explain the ambiguity in your final summary to the parent agent.',
  'Stay within the delegated objective and ownership boundary; do not duplicate or broaden the parent task.',
  'The filesystem is shared with the parent and other agents. Preserve unrelated and concurrent changes.',
  'Return a distilled final summary with the outcome, concrete evidence, files read or changed, verification, and any blocker or remaining risk.',
].join(' ')

export const SUBAGENT_BLOCKED_TOOLS = Object.freeze(new Set([
  'AgentTool',
  'AwaitAgents',
  'CheckAgentMessages',
  'HandoffTool',
  'PeekAgent',
  'ResetAgent',
  'SendMessageTool',
  'SetInteractionModeTool',
  'SpawnAgents',
  'TaskCreateTool',
  'TaskGetTool',
  'TaskListTool',
  'TaskOutputTool',
  'TaskStopTool',
  'TaskUpdateTool',
  'SkillTool',
]))

const READ_FILE_TOOLS = new Set(['ReadFile', 'read_file', 'analyze_code_structure'])
const WRITE_FILE_TOOLS = new Set([
  'AppendFile',
  'FileEditTool',
  'NotebookEditTool',
  'WriteFile',
  'delete_file',
  'replace_in_file',
  'write_file',
])
const FILE_PATH_KEYS = ['file_path', 'path', 'notebook_path'] as const
const TERMINAL_STATUSES = new Set<SubAgentStatus>(['cancelled', 'completed', 'failed'])

export type SubAgentStatus = 'cancelled' | 'completed' | 'failed' | 'pending' | 'running'

export interface SubAgentTaskSnapshot {
  readonly agentDef: string
  readonly apiCalls: number | undefined
  readonly completionSummary: string | undefined
  readonly creatorId: string
  readonly currentTool: string
  readonly depth: number
  readonly error: string
  readonly id: string
  readonly idleMilliseconds: number | undefined
  readonly inboxSize: number
  readonly inputTokens: number | undefined
  readonly messagesSent: number
  readonly model: string
  readonly name: string
  readonly outputTokens: number | undefined
  readonly parentId: string
  readonly prompt: string
  readonly readFiles: readonly string[]
  readonly reasoningTokens: number | undefined
  readonly recentOutput: string
  readonly result: string | undefined
  readonly rules: readonly string[]
  readonly status: SubAgentStatus
  readonly title: string
  readonly toolCallsCount: number
  readonly toolsets: readonly string[]
  readonly worktreeBranch: string
  readonly worktreePath: string
  readonly writtenFiles: readonly string[]
}

export interface SubAgentTaskOptions {
  readonly agentDefName?: string
  readonly creatorId?: string
  readonly depth?: number
  readonly id?: string
  readonly model?: string
  readonly name?: string
  readonly parentId?: string
  readonly prompt?: string
  readonly rules?: readonly string[]
  /** Parent daemon session that owns this delegated task. */
  readonly sourceId?: string
  readonly title?: string
  readonly toolsets?: readonly string[]
}

/** One delegated task plus the state exposed to its parent. */
export class SubAgentTask {
  readonly agentDefName: string
  readonly creatorId: string
  readonly depth: number
  readonly id: string
  model: string
  readonly name: string
  readonly parentId: string
  readonly prompt: string
  readonly rules: readonly string[]
  readonly sourceId: string
  readonly title: string
  readonly toolsets: readonly string[]
  apiCalls: number | undefined
  currentTool = ''
  error = ''
  inboxSize = 0
  lastActivityAt: number | undefined
  messagesSent = 0
  inputTokens: number | undefined
  outputTokens: number | undefined
  reasoningTokens: number | undefined
  result: string | undefined
  status: SubAgentStatus = 'pending'
  toolCallsCount = 0
  worktreeBranch = ''
  worktreePath = ''
  readonly readFiles = new Set<string>()
  readonly writtenFiles = new Set<string>()
  private readonly recentOutput: string[] = []

  constructor(options: SubAgentTaskOptions = {}) {
    this.id = options.id ?? ''
    this.prompt = options.prompt ?? ''
    this.sourceId = options.sourceId?.trim() ?? ''
    this.creatorId = options.creatorId?.trim() ?? ''
    this.parentId = options.parentId?.trim() || this.creatorId
    this.depth = options.depth ?? 0
    this.name = options.name?.trim() || this.id.slice(0, 8)
    this.title = normalizeAgentTitle(options.title ?? (this.name || 'Subagent'))
    this.agentDefName = options.agentDefName ?? ''
    this.model = options.model?.trim() ?? ''
    this.rules = freezeLabels(options.rules)
    this.toolsets = freezeLabels(options.toolsets)
  }

  /** Push one emitted text chunk into the bounded recent-output ring. */
  recordText(text: string, now = Date.now()): void {
    if (!text) return
    this.recentOutput.push(text)
    while (this.recentOutput.length > 32) this.recentOutput.shift()
    this.lastActivityAt = now
  }

  /** Return the tail of streamed output without expanding the caller's context unboundedly. */
  recentOutputText(maxChars = 2_000): string {
    const joined = this.recentOutput.join('')
    return joined.length > maxChars ? `…${joined.slice(-maxChars)}` : joined
  }

  snapshot(now = Date.now()): SubAgentTaskSnapshot {
    return Object.freeze({
      id: this.id,
      name: this.name,
      title: this.title,
      prompt: this.prompt.slice(0, 200),
      status: this.status,
      result: this.result === undefined ? undefined : this.result.slice(0, 500),
      depth: this.depth,
      agentDef: this.agentDefName,
      creatorId: this.creatorId,
      parentId: this.parentId,
      model: this.model,
      rules: this.rules,
      toolsets: this.toolsets,
      apiCalls: this.apiCalls,
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      reasoningTokens: this.reasoningTokens,
      completionSummary: this.result === undefined ? undefined : this.result.slice(0, 500),
      worktreePath: this.worktreePath,
      worktreeBranch: this.worktreeBranch,
      error: this.error,
      messagesSent: this.messagesSent,
      inboxSize: this.inboxSize,
      currentTool: this.currentTool,
      toolCallsCount: this.toolCallsCount,
      readFiles: Object.freeze([...this.readFiles].sort().slice(0, 20)),
      writtenFiles: Object.freeze([...this.writtenFiles].sort().slice(0, 20)),
      recentOutput: this.recentOutputText(),
      idleMilliseconds: this.lastActivityAt === undefined ? undefined : Math.max(0, now - this.lastActivityAt),
    })
  }
}

export interface SubagentWorktree {
  readonly branch: string
  readonly path: string
}

/** Explicit git/worktree boundary. The manager never invokes git or changes process cwd itself. */
export interface SubagentWorktreePort {
  create(request: {
    readonly taskId: string
    readonly taskName: string
  }): Promise<SubagentWorktree>
  isClean(worktree: SubagentWorktree): Promise<boolean>
  remove(worktree: SubagentWorktree): Promise<void>
}

export interface SubagentToolStart {
  readonly inputs: Readonly<Record<string, unknown>>
  readonly name: string
  readonly toolCallId: string
}

export interface SubagentToolEnd {
  readonly durationMs?: number
  readonly name: string
  readonly permitted: boolean
  readonly result: string
  readonly toolCallId: string
}

export interface SubagentRunReporter {
  text(text: string): void
  thinking(text: string): void
  toolEnd(event: SubagentToolEnd): void
  toolStart(event: SubagentToolStart): void
  usage(event: SubagentUsage): void
}

export interface SubagentUsage {
  readonly apiCalls?: number
  readonly inputTokens?: number
  readonly model: string
  readonly outputTokens?: number
  readonly reasoningTokens?: number
  readonly toolCalls: number
}

export interface SubagentTaskRunRequest {
  readonly cancelSignal: AbortSignal
  readonly config: Readonly<Record<string, unknown>>
  readonly depth: number
  readonly prompt: string
  readonly report: SubagentRunReporter
  readonly systemPrompt: string
  readonly task: SubAgentTask
  readonly worktree?: SubagentWorktree
}

export interface SubagentTaskRunResult {
  readonly content: string
}

/** Caller-owned execution boundary for a subagent turn. */
export type SubagentTaskRunner = (
  request: SubagentTaskRunRequest,
) => Promise<SubagentTaskRunResult | string> | SubagentTaskRunResult | string

export interface SubAgentEvent {
  readonly agent: string
  readonly agentType: string
  readonly apiCalls: number | undefined
  readonly completionSummary: string | undefined
  readonly creatorId: string
  readonly data: Readonly<Record<string, unknown>>
  readonly depth: number
  readonly filesRead: readonly string[]
  readonly filesWritten: readonly string[]
  readonly goal: string
  readonly inputTokens: number | undefined
  readonly model: string
  readonly outputTokens: number | undefined
  readonly parentId: string
  readonly reasoningTokens: number | undefined
  readonly rules: readonly string[]
  readonly sequence: number
  readonly sourceId: string
  readonly taskId: string
  readonly timestamp: string
  readonly title: string
  readonly toolCalls: number
  readonly toolsets: readonly string[]
  readonly type: string
}

export interface SubAgentManagerOptions {
  readonly idFactory?: () => string
  readonly maxConcurrent?: number
  readonly maxDepth?: number
  readonly now?: () => Date
  readonly onEvent?: (event: SubAgentEvent) => void
  readonly pathResolver?: (rawPath: string) => string | undefined
  readonly runner: SubagentTaskRunner
  readonly worktree?: SubagentWorktreePort
}

export interface SpawnSubAgentOptions {
  readonly agentDefinition?: AgentDefinition
  readonly config?: Readonly<Record<string, unknown>>
  readonly creatorId?: string
  readonly depth?: number
  readonly isolation?: string
  readonly model?: string
  readonly name?: string
  readonly parentId?: string
  readonly prompt: string
  readonly rules?: readonly string[]
  /** Parent daemon session used to route live events without cross-session leakage. */
  readonly sourceId?: string
  readonly systemPrompt?: string
  readonly title?: string
  readonly toolsets?: readonly string[]
}

export interface WaitForOptions {
  readonly extraWake?: () => boolean
  readonly timeoutMs?: number
}

export interface FilteredSubagentTools<T extends Record<string, unknown>> {
  readonly execute: SubagentToolExecutor | undefined
  readonly toolSchemas: readonly T[]
}

export type SubagentToolExecutor = (
  toolName: string,
  inputs: Readonly<Record<string, unknown>>,
) => Promise<string> | string

export interface FilterSubagentToolsOptions<T extends Record<string, unknown>> {
  readonly config?: Readonly<Record<string, unknown>>
  readonly isSubagent: boolean
  readonly toolExecutor?: SubagentToolExecutor
  readonly toolSchemas?: readonly T[]
}

interface TaskRuntime {
  readonly agentDefinition: AgentDefinition | undefined
  readonly config: Readonly<Record<string, unknown>>
  readonly isolation: string
  readonly originalPrompt: string
  readonly originalSystemPrompt: string
  readonly systemPrompt: string
  readonly toolInputs: Map<string, Readonly<Record<string, unknown>>>
  readonly worktree: SubagentWorktree | undefined
  cleanup: Promise<void> | undefined
  emittedSpawn: boolean
  monitor: Promise<void> | undefined
}

/**
 * Agent-facing compatibility manager built on the native `SpawnedAgentManager`.
 *
 * This layer supplies Python-era task snapshots, mailbox events, capability
 * filtering, file-change coordination, and worktree ports while leaving
 * concrete LLM execution, git actions, and file resolution to injected hosts.
 */
export class SubAgentManager {
  readonly maxDepth: number
  maxConcurrent: number
  private readonly eventSink: (event: SubAgentEvent) => void
  private readonly gate: ConcurrencyGate
  private readonly handleManager: SpawnedAgentManager
  private readonly idFactory: () => string
  private readonly mailbox: SubAgentEvent[] = []
  private sequence = 0
  private readonly now: () => Date
  private readonly pathResolver: (rawPath: string) => string | undefined
  private readonly recentText = new Map<string, string[]>()
  private runner: SubagentTaskRunner
  private readonly runtimes = new Map<string, TaskRuntime>()
  private readonly tasksByName = new Map<string, string>()
  private readonly textBurst = new Map<string, string[]>()
  private readonly waiters = new Set<() => void>()
  private readonly worktree: SubagentWorktreePort | undefined
  readonly tasks = new Map<string, SubAgentTask>()

  constructor(options: SubAgentManagerOptions) {
    if (typeof options.runner !== 'function') throw new TypeError('runner must be a function')
    this.runner = options.runner
    this.maxConcurrent = positiveInteger(options.maxConcurrent ?? 8, 'maxConcurrent')
    this.maxDepth = nonNegativeInteger(options.maxDepth ?? 5, 'maxDepth')
    this.gate = new ConcurrencyGate(this.maxConcurrent)
    this.now = options.now ?? (() => new Date())
    this.idFactory = options.idFactory ?? (() => `subagent_${crypto.randomUUID().replaceAll('-', '').slice(0, 12)}`)
    this.eventSink = options.onEvent ?? (() => undefined)
    this.pathResolver = options.pathResolver ?? (rawPath => rawPath)
    this.worktree = options.worktree
    this.handleManager = new SpawnedAgentManager({
      idFactory: this.idFactory,
      now: this.now,
      runner: async (request, signal) => this.runTaskInput(request.handleId, request.input, signal),
    })
  }

  /** Increase the execution permit pool without interrupting current tasks. */
  ensureCapacity(minConcurrent: number): boolean {
    const requested = positiveInteger(minConcurrent, 'minConcurrent')
    if (requested <= this.maxConcurrent) return true
    this.maxConcurrent = requested
    this.gate.increaseTo(requested)
    return true
  }

  /** Replace the caller-owned execution port for subsequently started subagent turns. */
  setRunner(runner: SubagentTaskRunner): void {
    if (typeof runner !== 'function') throw new TypeError('runner must be a function')
    this.runner = runner
  }

  /** Spawn one delegated task. The returned task is registered before execution begins. */
  async spawn(options: SpawnSubAgentOptions): Promise<SubAgentTask> {
    const depth = options.depth ?? 0
    const taskId = this.nextTaskId()
    const isolation = options.isolation?.trim() || options.agentDefinition?.isolation || ''
    const config = effectiveConfig(options.config ?? {}, options.agentDefinition)
    const systemPrompt = effectiveSystemPrompt(options.systemPrompt ?? '', options.agentDefinition)
    const title = options.title === undefined ? titleFromPrompt(options.prompt, options.name) : normalizeAgentTitle(options.title)
    const model = options.model?.trim() || stringValue(config.model)
    const rules = options.rules ?? runtimePolicyRules(config, isolation)
    const toolsets = options.toolsets ?? configuredToolsets(config)
    const task = new SubAgentTask({
      id: taskId,
      prompt: options.prompt,
      depth,
      title,
      model,
      rules,
      toolsets,
      ...(options.name === undefined ? {} : { name: options.name }),
      ...(options.agentDefinition === undefined ? {} : { agentDefName: options.agentDefinition.name }),
      ...(options.sourceId === undefined ? {} : { sourceId: options.sourceId }),
      ...(options.creatorId === undefined ? {} : { creatorId: options.creatorId }),
      ...(options.parentId === undefined ? {} : { parentId: options.parentId }),
    })
    this.tasks.set(task.id, task)
    if (options.name?.trim()) this.tasksByName.set(options.name.trim(), task.id)

    if (depth >= this.maxDepth) {
      this.fail(task, `Max depth (${this.maxDepth}) exceeded`)
      return task
    }

    let worktree: SubagentWorktree | undefined
    let prompt = options.prompt
    if (isolation === 'worktree') {
      if (this.worktree === undefined) {
        this.fail(task, "isolation='worktree' requires a configured worktree port")
        return task
      }
      try {
        worktree = await this.worktree.create({ taskId: task.id, taskName: task.name })
        task.worktreePath = worktree.path
        task.worktreeBranch = worktree.branch
        prompt = `${prompt}\n\n[Note: You are working in an isolated git worktree at ${worktree.path} (branch: ${worktree.branch}). Commit your changes before finishing so they can be reviewed and merged.]`
      } catch (error) {
        this.fail(task, `Failed to create worktree: ${errorMessage(error)}`)
        return task
      }
    }

    this.runtimes.set(task.id, {
      agentDefinition: options.agentDefinition,
      config,
      isolation,
      originalPrompt: options.prompt,
      originalSystemPrompt: options.systemPrompt ?? '',
      systemPrompt,
      toolInputs: new Map(),
      worktree,
      cleanup: undefined,
      emittedSpawn: false,
      monitor: undefined,
    })

    try {
      await this.handleManager.spawn({
        nickname: task.id,
        message: prompt,
        agent: {
          id: task.id,
          name: task.name,
          ...(task.model ? { model: task.model } : {}),
          ...(systemPrompt ? { systemPrompt } : {}),
        },
        title: task.title,
        ...(task.creatorId ? { creatorAgentId: task.creatorId } : {}),
        ...(task.parentId ? { parentAgentId: task.parentId } : {}),
        rules: task.rules,
        toolsets: task.toolsets,
      })
      const runtime = this.runtimes.get(task.id)
      if (runtime !== undefined) runtime.monitor = this.monitor(task)
    } catch (error) {
      this.fail(task, errorMessage(error))
      await this.cleanupWorktree(task)
    }
    return task
  }

  /** Await the named task without discarding its current snapshot. */
  async wait(taskIdOrName: string, timeoutMs?: number): Promise<SubAgentTask | undefined> {
    const task = this.resolveTask(taskIdOrName)
    if (task === undefined) return undefined
    if (TERMINAL_STATUSES.has(task.status)) return task
    const wait = await this.handleManager.wait([task.id], timeoutMs ?? 30_000)
    const snapshot = wait.completed[0] ?? wait.pending[0]
    if (snapshot !== undefined) this.synchronize(task, snapshot)
    return task
  }

  /** Await a batch and return terminal and still-active task snapshots. */
  async waitAll(
    taskIds?: readonly string[],
    timeoutMs?: number,
  ): Promise<{ readonly completed: readonly SubAgentTaskSnapshot[]; readonly pending: readonly SubAgentTaskSnapshot[] }> {
    const tasks = (taskIds ?? [...this.tasks.keys()])
      .map(id => this.resolveTask(id))
      .filter((task): task is SubAgentTask => task !== undefined)
    await Promise.all(tasks.map(task => this.wait(task.id, timeoutMs)))
    const completed: SubAgentTaskSnapshot[] = []
    const pending: SubAgentTaskSnapshot[] = []
    for (const task of tasks) {
      if (TERMINAL_STATUSES.has(task.status)) completed.push(task.snapshot(this.now().valueOf()))
      else pending.push(task.snapshot(this.now().valueOf()))
    }
    return Object.freeze({ completed: Object.freeze(completed), pending: Object.freeze(pending) })
  }

  /** Queue follow-up input for a live task. */
  async sendMessage(taskIdOrName: string, message: string): Promise<boolean> {
    const task = this.resolveTask(taskIdOrName)
    if (task === undefined || TERMINAL_STATUSES.has(task.status) || !message.trim()) return false
    try {
      const snapshot = await this.handleManager.sendInput(task.id, { message })
      task.messagesSent += 1
      this.synchronize(task, snapshot)
      this.postEvent(task, 'message', { messagePreview: message.slice(0, 200) })
      return true
    } catch {
      return false
    }
  }

  /** Cooperatively cancel a task through the native spawned-agent manager. */
  cancel(taskIdOrName: string): boolean {
    const task = this.resolveTask(taskIdOrName)
    if (task === undefined || TERMINAL_STATUSES.has(task.status)) return false
    this.handleManager.close(task.id)
    task.status = 'cancelled'
    task.result ??= '[Sub-agent was cancelled.]'
    task.lastActivityAt = this.now().valueOf()
    this.postEvent(task, 'cancelled', { reason: 'explicit_cancel' })
    void this.cleanupWorktree(task)
    return true
  }

  /** Cancel every live task and return the number transitioned. */
  cancelAll(): number {
    let count = 0
    for (const task of this.tasks.values()) {
      if (this.cancel(task.id)) count += 1
    }
    return count
  }

  /** Cancel then respawn the same definition/configuration with an optional replacement prompt. */
  async reset(taskIdOrName: string, newPrompt = ''): Promise<SubAgentTask | undefined> {
    const task = this.resolveTask(taskIdOrName)
    const runtime = task === undefined ? undefined : this.runtimes.get(task.id)
    if (task === undefined || runtime === undefined) return undefined
    this.cancel(task.id)
    return this.spawn({
      prompt: newPrompt.trim() || runtime.originalPrompt,
      config: runtime.config,
      systemPrompt: runtime.originalSystemPrompt,
      depth: task.depth,
      isolation: runtime.isolation,
      name: task.name,
      title: task.title,
      sourceId: task.sourceId,
      creatorId: task.creatorId,
      parentId: task.parentId,
      model: task.model,
      rules: task.rules,
      toolsets: task.toolsets,
      ...(runtime.agentDefinition === undefined ? {} : { agentDefinition: runtime.agentDefinition }),
    })
  }

  listTasks(): SubAgentTask[] {
    return [...this.tasks.values()]
  }

  listSnapshots(): SubAgentTaskSnapshot[] {
    const now = this.now().valueOf()
    return this.listTasks().map(task => task.snapshot(now))
  }

  getByName(name: string): SubAgentTask | undefined {
    return this.resolveTask(name)
  }

  getResult(taskIdOrName: string): string | undefined {
    return this.resolveTask(taskIdOrName)?.result
  }

  /** Append a bounded lifecycle event and notify asynchronous waiters. */
  postEvent(task: SubAgentTask, type: string, data: Readonly<Record<string, unknown>> = {}): void {
    const event = Object.freeze({
      sequence: ++this.sequence,
      taskId: task.id,
      agent: task.name,
      title: task.title,
      agentType: task.agentDefName,
      creatorId: task.creatorId,
      parentId: task.parentId,
      depth: task.depth,
      goal: task.prompt,
      sourceId: task.sourceId,
      model: task.model,
      rules: task.rules,
      toolsets: task.toolsets,
      toolCalls: task.toolCallsCount,
      apiCalls: task.apiCalls,
      inputTokens: task.inputTokens,
      outputTokens: task.outputTokens,
      reasoningTokens: task.reasoningTokens,
      filesRead: Object.freeze([...task.readFiles].sort()),
      filesWritten: Object.freeze([...task.writtenFiles].sort()),
      completionSummary: task.result === undefined ? undefined : task.result.slice(0, 500),
      type,
      timestamp: this.now().toISOString(),
      data: Object.freeze({ ...data }),
    })
    this.mailbox.push(event)
    if (this.mailbox.length > 512) this.mailbox.shift()
    task.lastActivityAt = this.now().valueOf()
    this.eventSink(event)
    this.notifyWaiters()
  }

  /** Drain events newer than a caller cursor. */
  drainMailbox(sinceSequence = 0): SubAgentEvent[] {
    const events = this.mailbox.filter(event => event.sequence > sinceSequence)
    this.mailbox.length = 0
    return events
  }

  /** Inspect events without consuming them. */
  peekMailbox(sinceSequence = 0): SubAgentEvent[] {
    return this.mailbox.filter(event => event.sequence > sinceSequence)
  }

  latestSequence(): number {
    return this.sequence
  }

  /** Wait reactively for a predicate, an externally supplied wake condition, or timeout. */
  async waitFor(predicate: () => boolean, options: WaitForOptions = {}): Promise<boolean> {
    const timeoutMs = options.timeoutMs
    if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs < 0)) {
      throw new TypeError('timeoutMs must be a non-negative finite number')
    }
    const deadline = timeoutMs === undefined ? undefined : Date.now() + timeoutMs
    while (true) {
      if (predicate()) return true
      if (options.extraWake?.()) return false
      const remaining = deadline === undefined ? 1_000 : Math.max(0, deadline - Date.now())
      if (remaining === 0) return false
      const notification = this.nextNotification()
      const delay = timeout(Math.min(remaining, 50))
      await Promise.race([notification.promise, delay.promise])
      notification.dispose()
      delay.dispose()
    }
  }

  /** Track explicit tool file reads/writes and notify active stale readers. */
  recordToolFileAccess(
    task: SubAgentTask,
    toolName: string,
    inputs: Readonly<Record<string, unknown>>,
    options: {
      readonly permitted?: boolean
      readonly phase: 'end' | 'start'
      readonly result?: string
    },
  ): void {
    const paths = this.toolFilePaths(toolName, inputs)
    if (!paths.length) return
    if (READ_FILE_TOOLS.has(toolName) && options.phase === 'start') {
      for (const path of paths) task.readFiles.add(path)
      return
    }
    if (!WRITE_FILE_TOOLS.has(toolName) || options.phase !== 'end') return
    if (options.permitted === false || options.result?.trimStart().startsWith('Error:')) return
    for (const path of paths) {
      task.writtenFiles.add(path)
      let notified = 0
      for (const reader of this.tasks.values()) {
        if (reader.id === task.id || TERMINAL_STATUSES.has(reader.status) || !reader.readFiles.has(path)) continue
        notified += 1
        const message = `[Swarm coordination] \`${path}\` changed in another subagent (${task.name}). Re-read or inspect the diff before relying on earlier file context.`
        void this.sendMessage(reader.id, message)
        this.postEvent(reader, 'coordination', { path, writer: task.name })
      }
      this.postEvent(task, 'file_write', { path, readersNotified: notified })
    }
  }

  /** Cancel active tasks and wait for their native task monitors to settle. */
  async shutdown(): Promise<void> {
    this.cancelAll()
    await Promise.all([...this.runtimes.values()].flatMap(runtime => runtime.monitor ? [runtime.monitor] : []))
  }

  async close(): Promise<void> {
    await this.shutdown()
  }

  summary(): string {
    const tasks = this.listTasks()
    const lines = [
      '# Sub-Agent Tasks',
      '',
      `Total: ${tasks.length}`,
      `Running: ${tasks.filter(task => task.status === 'running').length}`,
      `Completed: ${tasks.filter(task => task.status === 'completed').length}`,
      '',
    ]
    for (const task of tasks) {
      const worktree = task.worktreeBranch ? ` [worktree: ${task.worktreeBranch}]` : ''
      const agent = task.agentDefName ? ` (${task.agentDefName})` : ''
      lines.push(`- **${task.name}**${agent} [${task.status}]${worktree} — ${task.prompt.slice(0, 60)}`)
    }
    return lines.join('\n')
  }

  private async runTaskInput(handleId: string, input: string, signal: AbortSignal): Promise<string> {
    const task = this.tasks.get(handleId)
    const runtime = this.runtimes.get(handleId)
    if (task === undefined || runtime === undefined) throw new Error(`Unknown subagent task '${handleId}'`)
    const release = await this.gate.acquire(signal)
    try {
      if (signal.aborted) throw abortError(signal)
      task.status = 'running'
      task.lastActivityAt = this.now().valueOf()
      if (!runtime.emittedSpawn) {
        runtime.emittedSpawn = true
        this.postEvent(task, 'spawn', {
          agentType: task.agentDefName,
          prompt: task.prompt.slice(0, 200),
          depth: task.depth,
          isolation: runtime.isolation,
        })
      }
      const chunksBefore = task.recentOutputText().length
      const output = await this.runner({
        prompt: input,
        config: runtime.config,
        systemPrompt: runtime.systemPrompt,
        depth: task.depth + 1,
        task,
        cancelSignal: signal,
        report: {
          text: text => this.reportText(task, text),
          thinking: text => this.reportThinking(task, text),
          toolStart: event => this.reportToolStart(task, runtime, event),
          toolEnd: event => this.reportToolEnd(task, runtime, event),
          usage: event => this.reportUsage(task, event),
        },
        ...(runtime.worktree === undefined ? {} : { worktree: runtime.worktree }),
      })
      if (signal.aborted) throw abortError(signal)
      const content = outputText(output)
      task.result = content
      if (task.recentOutputText().length === chunksBefore) this.reportText(task, content)
      return content
    } catch (error) {
      if (signal.aborted || task.status === 'cancelled') {
        task.status = 'cancelled'
        task.result ??= '[Sub-agent was cancelled.]'
      } else {
        task.status = 'failed'
        task.error = errorMessage(error)
        task.result = `Error: ${task.error}`
        this.postEvent(task, 'error', { error: task.error })
      }
      throw error
    } finally {
      this.flushTextBurst(task)
      release()
    }
  }

  private async monitor(task: SubAgentTask): Promise<void> {
    while (!TERMINAL_STATUSES.has(task.status)) {
      const result = await this.handleManager.wait([task.id], 60_000)
      const snapshot = result.completed[0] ?? result.pending[0]
      if (snapshot !== undefined) this.synchronize(task, snapshot)
    }
    this.flushTextBurst(task)
    this.postEvent(task, 'done', {
      status: task.status,
      error: task.error,
      resultPreview: (task.result ?? '').slice(0, 500),
      toolCalls: task.toolCallsCount,
      ...(task.apiCalls === undefined ? {} : { apiCalls: task.apiCalls }),
      ...(task.inputTokens === undefined ? {} : { inputTokens: task.inputTokens }),
      ...(task.outputTokens === undefined ? {} : { outputTokens: task.outputTokens }),
      ...(task.reasoningTokens === undefined ? {} : { reasoningTokens: task.reasoningTokens }),
      filesRead: [...task.readFiles].sort(),
      filesWritten: [...task.writtenFiles].sort(),
    })
    await this.cleanupWorktree(task)
  }

  private synchronize(task: SubAgentTask, snapshot: SpawnedAgentSnapshot): void {
    if (snapshot.lastOutput !== undefined) task.result = snapshot.lastOutput
    if (snapshot.error !== undefined && snapshot.error !== 'cancelled') task.error = snapshot.error
    task.inboxSize = snapshot.queueSize
    task.lastActivityAt = this.now().valueOf()
    switch (snapshot.status) {
      case 'completed':
        task.status = 'completed'
        break
      case 'cancelled':
      case 'closed':
      case 'interrupted':
        task.status = 'cancelled'
        task.result ??= '[Sub-agent was cancelled.]'
        break
      case 'error':
        task.status = 'failed'
        task.result ??= `Error: ${task.error || 'subagent execution failed'}`
        break
      case 'idle':
      case 'running':
        if (!TERMINAL_STATUSES.has(task.status)) task.status = 'running'
        break
    }
  }

  private fail(task: SubAgentTask, error: string): void {
    task.status = 'failed'
    task.error = error
    task.result = error
    task.lastActivityAt = this.now().valueOf()
    this.postEvent(task, 'error', { error })
    this.postEvent(task, 'done', { status: task.status, error, resultPreview: error, toolCalls: 0 })
  }

  private reportText(task: SubAgentTask, text: string): void {
    if (!text) return
    task.recordText(text, this.now().valueOf())
    const chunks = this.textBurst.get(task.id) ?? []
    chunks.push(text)
    this.textBurst.set(task.id, chunks)
    if (chunks.join('').length >= 512) this.flushTextBurst(task)
  }

  private reportThinking(task: SubAgentTask, text: string): void {
    if (!text) return
    task.lastActivityAt = this.now().valueOf()
    this.postEvent(task, 'thinking', {
      preview: text.length > 400 ? `…${text.slice(-400)}` : text,
    })
  }

  private reportToolStart(task: SubAgentTask, runtime: TaskRuntime, event: SubagentToolStart): void {
    task.currentTool = event.name
    task.toolCallsCount += 1
    task.lastActivityAt = this.now().valueOf()
    runtime.toolInputs.set(event.toolCallId, event.inputs)
    this.flushTextBurst(task)
    this.recordToolFileAccess(task, event.name, event.inputs, { phase: 'start' })
    this.postEvent(task, 'tool_start', {
      tool: event.name,
      toolCallId: event.toolCallId,
      inputPreview: previewToolInput(event.inputs),
    })
  }

  private reportToolEnd(task: SubAgentTask, runtime: TaskRuntime, event: SubagentToolEnd): void {
    task.currentTool = ''
    task.lastActivityAt = this.now().valueOf()
    const inputs = runtime.toolInputs.get(event.toolCallId) ?? {}
    runtime.toolInputs.delete(event.toolCallId)
    this.recordToolFileAccess(task, event.name, inputs, {
      phase: 'end',
      permitted: event.permitted,
      result: event.result,
    })
    this.postEvent(task, 'tool_end', {
      tool: event.name,
      toolCallId: event.toolCallId,
      permitted: event.permitted,
      ...(event.durationMs === undefined ? {} : { durationMs: event.durationMs }),
      resultPreview: event.result.slice(0, 200),
    })
  }

  private reportUsage(task: SubAgentTask, usage: SubagentUsage): void {
    if (usage.apiCalls !== undefined) {
      task.apiCalls = usage.apiCalls
    }
    task.toolCallsCount = usage.toolCalls
    if (usage.model.trim()) {
      task.model = usage.model.trim()
    }
    task.inputTokens = usage.inputTokens
    task.outputTokens = usage.outputTokens
    task.reasoningTokens = usage.reasoningTokens
  }

  private flushTextBurst(task: SubAgentTask): void {
    const chunks = this.textBurst.get(task.id)
    if (!chunks?.length) return
    this.textBurst.delete(task.id)
    const text = chunks.join('')
    this.postEvent(task, 'text_burst', {
      chars: text.length,
      preview: text.length > 400 ? `…${text.slice(-400)}` : text,
    })
  }

  private async cleanupWorktree(task: SubAgentTask): Promise<void> {
    const runtime = this.runtimes.get(task.id)
    const worktree = runtime?.worktree
    const worktreePort = this.worktree
    if (runtime === undefined || worktree === undefined || worktreePort === undefined) return
    if (runtime.cleanup !== undefined) return runtime.cleanup
    runtime.cleanup = (async () => {
      try {
        if (await worktreePort.isClean(worktree)) {
          await worktreePort.remove(worktree)
          this.postEvent(task, 'worktree_removed', { path: worktree.path, branch: worktree.branch })
        }
      } catch (error) {
        this.postEvent(task, 'worktree_cleanup_error', { error: errorMessage(error) })
      }
    })()
    return runtime.cleanup
  }

  private toolFilePaths(toolName: string, inputs: Readonly<Record<string, unknown>>): string[] {
    if (!READ_FILE_TOOLS.has(toolName) && !WRITE_FILE_TOOLS.has(toolName)) return []
    const paths = new Set<string>()
    for (const key of FILE_PATH_KEYS) {
      const value = inputs[key]
      const candidates = typeof value === 'string' ? [value] : Array.isArray(value)
        ? value.filter((item): item is string => typeof item === 'string')
        : []
      for (const rawPath of candidates) {
        if (!rawPath || rawPath.includes('\0')) continue
        const resolved = this.pathResolver(rawPath)
        if (resolved) paths.add(resolved)
      }
    }
    return [...paths]
  }

  private resolveTask(taskIdOrName: string): SubAgentTask | undefined {
    return this.tasks.get(taskIdOrName) ?? this.tasks.get(this.tasksByName.get(taskIdOrName) ?? '')
  }

  private nextTaskId(): string {
    const id = this.idFactory()
    if (!id || this.tasks.has(id)) throw new Error(`Subagent id '${id}' is unavailable`)
    return id
  }

  private nextNotification(): { readonly dispose: () => void; readonly promise: Promise<void> } {
    let resolvePromise: (() => void) | undefined
    const promise = new Promise<void>(resolve => {
      resolvePromise = resolve
    })
    const resolve = (): void => resolvePromise?.()
    this.waiters.add(resolve)
    return Object.freeze({
      promise,
      dispose: () => this.waiters.delete(resolve),
    })
  }

  private notifyWaiters(): void {
    for (const resolve of this.waiters) resolve()
    this.waiters.clear()
  }
}

/** Filter schemas and execution through agent-specific restrictions and recursive-delegation policy. */
export function filterSubagentTools<T extends Record<string, unknown>>(
  options: FilterSubagentToolsOptions<T>,
): FilteredSubagentTools<T> {
  if (options.toolSchemas === undefined) {
    return Object.freeze({ toolSchemas: Object.freeze([]), execute: options.toolExecutor })
  }
  const schemas = options.toolSchemas
  const allNames = new Set(schemas.flatMap(schema => typeof schema.name === 'string' && schema.name ? [schema.name] : []))
  const config = options.config ?? {}
  const whitelist = stringList(config._toolsWhitelist)
  const allowedTools = stringList(config._toolsAllowed)
  const excluded = new Set(stringList(config._toolsExcluded))
  const allowed = new Set(whitelist.length ? whitelist : allNames)
  if (allowedTools.length) {
    for (const name of [...allowed]) if (!allowedTools.includes(name)) allowed.delete(name)
  }
  for (const name of excluded) allowed.delete(name)
  if (options.isSubagent && config._allowSubagentDelegation !== true) {
    for (const name of SUBAGENT_BLOCKED_TOOLS) allowed.delete(name)
  }
  const toolSchemas = Object.freeze(schemas.filter(schema => typeof schema.name === 'string' && allowed.has(schema.name)))
  const execute = options.toolExecutor === undefined
    ? undefined
    : (toolName: string, inputs: Readonly<Record<string, unknown>>) => {
      if (!allowed.has(toolName)) return `Error: tool '${toolName}' is not allowed for this agent.`
      return options.toolExecutor!(toolName, inputs)
    }
  return Object.freeze({ toolSchemas, execute })
}

function effectiveConfig(
  config: Readonly<Record<string, unknown>>,
  definition: AgentDefinition | undefined,
): Readonly<Record<string, unknown>> {
  const effective: Record<string, unknown> = { ...config }
  if (definition === undefined) return Object.freeze(effective)
  if (definition.model) effective.model = definition.model
  if (definition.allowedTools !== null) effective._toolsAllowed = [...definition.allowedTools]
  if (definition.excludeTools.length) effective._toolsExcluded = [...definition.excludeTools]
  if (definition.tools.length) effective._toolsWhitelist = [...definition.tools]
  return Object.freeze(effective)
}

function titleFromPrompt(prompt: string, name?: string): string {
  const firstLine = prompt.replace(/\r\n?/gu, '\n').split('\n').find(line => line.trim())
  const readable = (firstLine ?? name ?? 'Subagent')
    .trim()
    .replace(/^(?:#{1,6}|[-+*>]|\d+[.)])\s+/u, '')
  const clipped = [...(readable || name || 'Subagent')].slice(0, MAX_AGENT_TITLE_LENGTH).join('').trimEnd()
  return normalizeAgentTitle(clipped || 'Subagent')
}

function freezeLabels(values: readonly string[] | undefined): readonly string[] {
  if (!values?.length) return Object.freeze([])
  const labels = values
    .filter(value => typeof value === 'string')
    .map(value => value.replace(/[\t\r\n]+/gu, ' ').replace(/\s+/gu, ' ').trim())
    .filter(Boolean)
  return Object.freeze([...new Set(labels)])
}

function configuredToolsets(config: Readonly<Record<string, unknown>>): readonly string[] {
  const whitelist = stringList(config._toolsWhitelist)
  const explicitlyAllowed = stringList(config._toolsAllowed)
  const excluded = new Set(stringList(config._toolsExcluded))
  const constrained = whitelist.length
    ? whitelist.filter(name => !explicitlyAllowed.length || explicitlyAllowed.includes(name))
    : explicitlyAllowed
  return freezeLabels(constrained.filter(name => !excluded.has(name)))
}

function runtimePolicyRules(config: Readonly<Record<string, unknown>>, isolation: string): readonly string[] {
  const permission = stringValue(config.permissionMode)
  return freezeLabels([
    ...(permission ? [`permission:${permission}`] : []),
    config._allowSubagentDelegation === true ? 'delegation:allowed' : 'delegation:blocked',
    ...(isolation ? [`isolation:${isolation}`] : []),
  ])
}

function effectiveSystemPrompt(base: string, definition: AgentDefinition | undefined): string {
  if (definition?.systemPrompt) return `${SUBAGENT_CALLER_PROMPT}\n\n${definition.systemPrompt.trim()}\n\n${base}`
  return base
}

function outputText(output: SubagentTaskRunResult | string): string {
  const content = typeof output === 'string' ? output : output.content
  if (typeof content !== 'string') throw new TypeError('subagent runner must return a string or { content: string }')
  return content
}

function previewToolInput(inputs: Readonly<Record<string, unknown>>): string {
  const pieces: string[] = []
  for (const [key, value] of Object.entries(inputs)) {
    const text = String(value)
    pieces.push(`${key}=${text.length > 60 ? `${text.slice(0, 57)}…` : text}`)
    if (pieces.join(', ').length > 200) break
  }
  return pieces.join(', ').slice(0, 200)
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive safe integer`)
  return value
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${name} must be a non-negative safe integer`)
  return value
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error('Subagent was cancelled')
}

function timeout(milliseconds: number): { readonly dispose: () => void; readonly promise: Promise<void> } {
  let timer: ReturnType<typeof setTimeout> | undefined
  const promise = new Promise<void>(resolve => {
    timer = setTimeout(resolve, milliseconds)
  })
  return Object.freeze({
    promise,
    dispose: () => {
      if (timer !== undefined) clearTimeout(timer)
    },
  })
}

class ConcurrencyGate {
  private active = 0
  private readonly waiters: Array<{
    readonly reject: (reason: unknown) => void
    readonly resolve: (release: () => void) => void
    readonly signal: AbortSignal
    readonly onAbort: () => void
  }> = []

  constructor(private capacity: number) {}

  increaseTo(capacity: number): void {
    this.capacity = Math.max(this.capacity, capacity)
    this.dispatch()
  }

  acquire(signal: AbortSignal): Promise<() => void> {
    if (signal.aborted) return Promise.reject(abortError(signal))
    if (this.active < this.capacity) {
      this.active += 1
      return Promise.resolve(this.release())
    }
    return new Promise((resolve, reject) => {
      const onAbort = (): void => {
        const index = this.waiters.findIndex(waiter => waiter.signal === signal)
        if (index >= 0) this.waiters.splice(index, 1)
        reject(abortError(signal))
      }
      signal.addEventListener('abort', onAbort, { once: true })
      this.waiters.push({ resolve, reject, signal, onAbort })
    })
  }

  private release(): () => void {
    let released = false
    return (): void => {
      if (released) return
      released = true
      this.active = Math.max(0, this.active - 1)
      this.dispatch()
    }
  }

  private dispatch(): void {
    while (this.active < this.capacity && this.waiters.length) {
      const waiter = this.waiters.shift()
      if (waiter === undefined) return
      waiter.signal.removeEventListener('abort', waiter.onAbort)
      if (waiter.signal.aborted) {
        waiter.reject(abortError(waiter.signal))
        continue
      }
      this.active += 1
      waiter.resolve(this.release())
    }
  }
}
