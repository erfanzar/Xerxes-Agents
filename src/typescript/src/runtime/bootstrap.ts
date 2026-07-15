// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import { xerxesSubdirFor } from '../core/paths.js'
import { scanContextContent } from '../security/promptScanner.js'
import { ExecutionRegistry, type EntryHandler } from './executionRegistry.js'
import { loadProjectAgentWorkspace } from './projectWorkspace.js'

export const DEFAULT_BOOTSTRAP_COMMANDS = [
  'help',
  'clear',
  'history',
  'save',
  'load',
  'model',
  'config',
  'cost',
  'context',
  'memory',
  'agents',
  'skills'
] as const

/** Maximum UTF-8 payload reserved for the changed-file listing in bootstrap context. */
export const MAX_BOOTSTRAP_GIT_STATUS_BYTES = 8 * 1024

/** Injectable Git command boundary used by the system bootstrap and focused tests. */
export type GitCommandRunner = (arguments_: readonly string[], cwd: string) => Promise<string>

export type BootstrapStageStatus = 'failed' | 'ok' | 'skipped'

export interface BootstrapStage {
  readonly detail: string
  readonly durationMs: number
  readonly name: string
  readonly status: BootstrapStageStatus
}

export interface BootstrapHost {
  readonly cwd: () => string
  readonly date: () => Date
  readonly gitInfo: (cwd: string) => Promise<string>
  readonly monotonicNow: () => number
  readonly platform: () => string
  readonly projectWorkspace: (cwd: string) => Promise<string>
  readonly readText: (path: string) => Promise<string | undefined>
  readonly runtimeVersion: () => string
  readonly xerxesHomeFile: (name: string) => string
}

export interface BootstrapOptions {
  readonly commands?: Readonly<Record<string, EntryHandler | undefined>>
  readonly cwd?: string
  readonly extraContext?: string
  readonly host?: BootstrapHost
  readonly includeGitInfo?: boolean
  readonly includeXerxesMd?: boolean
  readonly model?: string
  readonly tools?: readonly unknown[]
}

export interface BootstrapResultOptions {
  readonly context: Readonly<Record<string, string>>
  readonly registry: ExecutionRegistry
  readonly stages: readonly BootstrapStage[]
  readonly systemPrompt: string
}

/** Aggregate output of the native session-bootstrap pipeline. */
export class BootstrapResult {
  readonly context: Readonly<Record<string, string>>
  readonly registry: ExecutionRegistry
  readonly stages: readonly BootstrapStage[]
  readonly systemPrompt: string

  constructor(options: BootstrapResultOptions) {
    this.context = Object.freeze({ ...options.context })
    this.registry = options.registry
    this.stages = Object.freeze([...options.stages])
    this.systemPrompt = options.systemPrompt
  }

  get ok(): boolean {
    return !this.stages.some(stage => stage.status === 'failed')
  }

  /** Render a small diagnostic report suitable for `/doctor` or a startup log. */
  asMarkdown(): string {
    const lines = [
      '# Bootstrap Report',
      '',
      'Status: ' + (this.ok ? 'OK' : 'FAILED'),
      'Stages: ' + this.stages.length,
      ''
    ]
    for (const stage of this.stages) {
      const icon = stage.status === 'ok' ? '+' : stage.status === 'skipped' ? '~' : '!'
      lines.push(`- [${icon}] ${stage.name}: ${stage.detail} (${stage.durationMs.toFixed(1)}ms)`)
    }
    return lines.join('\n')
  }
}

/**
 * Assemble startup context, the execution registry, and a safe default system prompt.
 *
 * The host owns all filesystem, git, clock, and project-workspace effects. Failed
 * optional context probes become explicit skipped stages rather than stopping a turn.
 */
export async function bootstrap(options: BootstrapOptions = {}): Promise<BootstrapResult> {
  const host = options.host ?? createSystemBootstrapHost()
  const registry = new ExecutionRegistry()
  const stages: BootstrapStage[] = []
  const environmentStarted = host.monotonicNow()
  const workingDirectory = resolve(options.cwd ?? host.cwd())
  const context: Record<string, string> = {
    cwd: workingDirectory,
    runtime_version: host.runtimeVersion(),
    platform: host.platform(),
    model: options.model ?? '',
    date: formatBootstrapDate(host.date())
  }

  stages.push(
    stageFromElapsed(host, 'environment', 'ok', `${context.runtime_version} on ${context.platform}`, environmentStarted)
  )

  const includeGitInfo = options.includeGitInfo ?? true
  const gitStarted = host.monotonicNow()
  const gitInfo = includeGitInfo ? await optionalContext(() => host.gitInfo(workingDirectory)) : ''
  if (includeGitInfo) context.git_info = gitInfo
  stages.push(
    stageFromElapsed(
      host,
      'git_info',
      gitInfo ? 'ok' : 'skipped',
      gitInfo ? gitInfo.slice(0, 80) : includeGitInfo ? 'Not a git repository' : 'Disabled',
      gitStarted
    )
  )

  const includeXerxesMd = options.includeXerxesMd ?? true
  const xerxesMdStarted = host.monotonicNow()
  const xerxesMd = includeXerxesMd ? await loadXerxesMd(workingDirectory, host) : ''
  if (includeXerxesMd) context.xerxes_md = xerxesMd
  stages.push(
    stageFromElapsed(
      host,
      'xerxes_md',
      xerxesMd ? 'ok' : 'skipped',
      xerxesMd ? xerxesMd.length + ' chars' : includeXerxesMd ? 'No XERXES.md found' : 'Disabled',
      xerxesMdStarted
    )
  )

  const projectWorkspaceStarted = host.monotonicNow()
  const projectWorkspace = await optionalContext(() => host.projectWorkspace(workingDirectory))
  context.project_agent_workspace = projectWorkspace
  stages.push(
    stageFromElapsed(
      host,
      'project_agent_workspace',
      projectWorkspace ? 'ok' : 'skipped',
      projectWorkspace ? projectWorkspace.length + ' chars' : 'No .agents workspace found',
      projectWorkspaceStarted
    )
  )

  const commandsStarted = host.monotonicNow()
  for (const [name, handler] of Object.entries(options.commands ?? {})) {
    registry.registerCommand(name, handler)
  }
  for (const name of DEFAULT_BOOTSTRAP_COMMANDS) {
    if (!registry.getCommand(name)) {
      registry.registerCommand(name, undefined, {
        description: '/' + name + ' command'
      })
    }
  }
  stages.push(stageFromElapsed(host, 'commands', 'ok', registry.commandCount + ' commands registered', commandsStarted))

  const toolsStarted = host.monotonicNow()
  if (options.tools?.length) registry.registerFromAgentFunctions(options.tools)
  stages.push(stageFromElapsed(host, 'tools', 'ok', registry.toolCount + ' tools registered', toolsStarted))

  const promptStarted = host.monotonicNow()
  const systemPrompt = buildBootstrapSystemPrompt(context, options.extraContext ?? '', options.tools ?? [])
  stages.push(stageFromElapsed(host, 'system_prompt', 'ok', systemPrompt.length + ' chars', promptStarted))

  return new BootstrapResult({ context, registry, stages, systemPrompt })
}

/** Build the default native prompt without doing filesystem or process I/O. */
export function buildBootstrapSystemPrompt(
  context: Readonly<Record<string, string>>,
  extraContext = '',
  toolDefinitions: readonly unknown[] = [],
): string {
  const tools = bootstrapPromptTools(toolDefinitions)
  const toolNames = new Set(tools.map(tool => tool.name))
  const sections: string[] = [
    'You are Xerxes, an AI coding assistant.',
    '',
    '# Tools available this turn',
  ]
  if (tools.length) {
    for (const tool of tools) {
      const required = requiredFieldSummary(tool.schema)
      sections.push(
        `- ${tool.name}: ${tool.description || 'No description supplied by the host.'}${required ? ` (required: ${required})` : ''}`,
      )
    }
  } else {
    sections.push('- None. Do not emit or simulate tool calls.')
  }
  sections.push(
    '',
    '# How to decide',
    '- Answer directly when knowledge and reasoning are sufficient.',
    '- Do not use tools for greetings, simple arithmetic, or facts you already know.',
    '- Never invoke Python or Node as a calculator.',
    '- Call only tools listed above, and follow each supplied JSON schema exactly.',
    '- If a needed capability is absent, explain the limitation instead of inventing a tool call.',
  )
  if (toolNames.has('Calculator')) {
    sections.push('- Use Calculator only when a calculation genuinely warrants a tool.')
  }
  if (toolNames.has('ReadFile')) {
    sections.push('- Read relevant files with ReadFile before editing them.')
  }
  const writingTools = ['WriteFile', 'FileEditTool', 'AppendFile', 'apply_patch'].filter(name => toolNames.has(name))
  if (writingTools.length) {
    sections.push(`- Available editing tools: ${writingTools.join(', ')}. Choose the narrowest one that fits the change.`)
  }
  if (toolNames.has('exec_command')) {
    const execCommand = tools.find(tool => tool.name === 'exec_command')
    const execProperties = bootstrapRecord(execCommand?.schema?.properties)
    sections.push(execProperties?.args === undefined
      ? '- Use exec_command only when a process is actually needed; follow its provider-supplied schema exactly.'
      : '- exec_command uses direct argv: cmd is one executable and each argument belongs in args; do not put shell syntax in cmd.')
  }
  if (toolNames.has('write_stdin')) {
    sections.push('- Poll or interact with a live terminal session through write_stdin.')
  }
  if (toolNames.has('close_terminal_session')) {
    sections.push('- Close terminal sessions with close_terminal_session when their work is complete.')
  }
  const searchTools = ['GlobTool', 'GrepTool', 'ListDir'].filter(name => toolNames.has(name))
  if (searchTools.length) {
    sections.push(`- Available workspace discovery tools: ${searchTools.join(', ')}.`)
  }
  const webTools = tools.map(tool => tool.name).filter(name => name.startsWith('web.') || name === 'DuckDuckGoSearch')
  if (webTools.length) {
    sections.push(`- Available public-information tools: ${webTools.join(', ')}.`)
  }
  sections.push(
    '',
    '# Context headroom',
    '- Oversized tool results are stored in project agent memory outside model context and replaced with a bounded preview.',
    '- Treat `[Large tool result stored outside model context]` as a valid tool result, not a failure.',
    '- Do not rerun noisy commands just to recover output that is already stored.',
  )
  if (toolNames.has('agent_memory_read')) {
    sections.push('- Read stored-result pointers with agent_memory_read using its supplied schema.')
  }
  if (toolNames.has('SetInteractionModeTool')) {
    sections.push(
      '',
      '# Interaction modes',
      '- code: normal implementation and direct engineering work.',
      '- researcher: read-only exploration, evidence gathering, and cited findings.',
      '- plan: architecture and task breakdown; do not edit code in this mode.',
      '- objective: hard-goal loop for measurable outcomes.',
      '- Switch modes with SetInteractionModeTool and follow its supplied schema.',
      '- Do not final-answer in objective mode while acceptance criteria are unmet; continue iterating or report a concrete blocker with evidence.',
    )
  }
  if (['Agent', 'AgentTool', 'SpawnAgents'].some(name => toolNames.has(name))) {
    sections.push(
      '',
      '# Multi-Agent Orchestration',
      '- Use available agent tools for genuinely independent work and keep task boundaries clear.',
      '- Prefer separate research, implementation, and review paths when parallelism helps.',
    )
  }
  sections.push(
    '',
    '# Critical',
    '- Be concise and direct.',
    '- Respect every tool schema and its workspace/path constraints.',
    '',
    '# Environment',
    '- Date: ' + (context.date ?? ''),
    '- CWD: ' + (context.cwd ?? ''),
    '- Platform: ' + (context.platform ?? ''),
    '- Model: ' + (context.model ?? ''),
  )
  if (context.git_info) sections.push('', '# Git', context.git_info)
  if (context.xerxes_md) sections.push('', '# Project Context', context.xerxes_md)
  if (context.project_agent_workspace) sections.push('', context.project_agent_workspace)
  if (extraContext) sections.push('', extraContext)
  return sections.join('\n')
}

interface BootstrapPromptTool {
  readonly description: string
  readonly name: string
  readonly schema: Readonly<Record<string, unknown>> | undefined
}

function bootstrapPromptTools(definitions: readonly unknown[]): BootstrapPromptTool[] {
  const tools = new Map<string, BootstrapPromptTool>()
  for (const value of definitions) {
    const record = bootstrapRecord(value)
    const functionRecord = bootstrapRecord(record?.function)
    const name = bootstrapString(functionRecord?.name) ?? bootstrapString(record?.name)
    if (!name) continue
    const description = boundedToolDescription(
      bootstrapString(functionRecord?.description) ?? bootstrapString(record?.description) ?? '',
    )
    const schema = bootstrapRecord(functionRecord?.parameters)
      ?? bootstrapRecord(record?.input_schema)
      ?? bootstrapRecord(record?.parameters)
    tools.set(name, { name, description, schema })
  }
  return [...tools.values()]
}

function requiredFieldSummary(schema: Readonly<Record<string, unknown>> | undefined): string {
  if (!Array.isArray(schema?.required)) return ''
  const fields = schema.required
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .map(value => value.trim().slice(0, 40))
  const shown = fields.slice(0, 8)
  return shown.join(', ') + (fields.length > shown.length ? `, +${fields.length - shown.length} more` : '')
}

function boundedToolDescription(value: string): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length <= 240 ? normalized : normalized.slice(0, 237) + '...'
}

function bootstrapRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function bootstrapString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

/** Create the Bun-backed host used by the public convenience entry point. */
export function createSystemBootstrapHost(
  environment: Record<string, string | undefined> = process.env
): BootstrapHost {
  return {
    cwd: () => process.cwd(),
    date: () => new Date(),
    gitInfo: collectGitInfo,
    monotonicNow: () => performance.now(),
    platform: () => process.platform,
    projectWorkspace: async cwd => (await loadProjectAgentWorkspace(cwd)).prompt,
    readText: readOptionalText,
    runtimeVersion: () => 'Bun ' + Bun.version,
    xerxesHomeFile: name => xerxesSubdirFor(environment, name)
  }
}

async function loadXerxesMd(cwd: string, host: BootstrapHost): Promise<string> {
  const parts: string[] = []
  const globalPath = host.xerxesHomeFile('XERXES.md')
  const global = await optionalContext(() => host.readText(globalPath))
  if (global) {
    parts.push('[Global XERXES.md]\n' + scanContextContent(global, 'Global XERXES.md'))
  }

  let current = cwd
  for (let index = 0; index < 10; index += 1) {
    const candidate = join(current, 'XERXES.md')
    const project = await optionalContext(() => host.readText(candidate))
    if (project) {
      const label = 'Project XERXES.md: ' + candidate
      parts.push('[' + label + ']\n' + scanContextContent(project, label))
      break
    }
    const parent = dirname(current)
    if (parent === current) break
    current = parent
  }
  return parts.join('\n\n')
}

/** Collect bounded repository context while running independent Git probes concurrently. */
export async function collectGitInfo(cwd: string, run: GitCommandRunner = runGit): Promise<string> {
  const [branch, rawStatus, log] = await Promise.all([
    run(['rev-parse', '--abbrev-ref', 'HEAD'], cwd),
    run(['status', '--short'], cwd),
    run(['log', '--oneline', '-5'], cwd)
  ])
  if (!branch) return ''
  const status = clipBootstrapGitStatus(rawStatus)
  const parts = ['Branch: ' + branch]
  if (status) parts.push('Status:\n' + status)
  if (log) parts.push('Recent commits:\n' + log)
  return parts.join('\n')
}

/** Clip a potentially huge porcelain listing without splitting a UTF-8 path or hiding truncation. */
export function clipBootstrapGitStatus(status: string): string {
  if (Buffer.byteLength(status, 'utf8') <= MAX_BOOTSTRAP_GIT_STATUS_BYTES) {
    return status
  }

  const lines = status.split('\n').filter(line => line.length > 0)
  const kept: string[] = []
  let usedBytes = 0

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (line === undefined) continue
    const separatorBytes = kept.length > 0 ? 1 : 0
    const candidateBytes = usedBytes + separatorBytes + Buffer.byteLength(line, 'utf8')
    const remaining = lines.length - index - 1
    const markerBytes = remaining ? 1 + Buffer.byteLength(gitStatusOmissionMarker(remaining), 'utf8') : 0
    if (candidateBytes + markerBytes > MAX_BOOTSTRAP_GIT_STATUS_BYTES) {
      break
    }
    kept.push(line)
    usedBytes = candidateBytes
  }

  const omitted = lines.length - kept.length
  const marker = gitStatusOmissionMarker(omitted)
  return kept.length > 0 ? kept.join('\n') + '\n' + marker : marker
}

function gitStatusOmissionMarker(omitted: number): string {
  return `[truncated: ${omitted} additional changes omitted; run git status --short for the complete list]`
}

async function runGit(arguments_: readonly string[], cwd: string): Promise<string> {
  try {
    const child = Bun.spawn(['git', ...arguments_], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const [exitCode, stdout] = await Promise.all([child.exited, new Response(child.stdout).text()])
    return exitCode === 0 ? stdout.trim() : ''
  } catch {
    return ''
  }
}

async function readOptionalText(path: string): Promise<string | undefined> {
  try {
    return await readFile(path, 'utf8')
  } catch {
    return undefined
  }
}

async function optionalContext(read: () => Promise<string | undefined>): Promise<string> {
  try {
    return (await read()) ?? ''
  } catch {
    return ''
  }
}

function formatBootstrapDate(date: Date): string {
  const year = date.getFullYear().toString().padStart(4, '0')
  const month = (date.getMonth() + 1).toString().padStart(2, '0')
  const day = date.getDate().toString().padStart(2, '0')
  const weekday = new Intl.DateTimeFormat('en-US', { weekday: 'long' }).format(date)
  return year + '-' + month + '-' + day + ' ' + weekday
}

function stageFromElapsed(
  host: BootstrapHost,
  name: string,
  status: BootstrapStageStatus,
  detail: string,
  started: number
): BootstrapStage {
  return {
    name,
    status,
    detail,
    durationMs: Math.max(0, host.monotonicNow() - started)
  }
}
