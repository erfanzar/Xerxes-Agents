// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import type { PermissionRequest } from './events.js'

export type PermissionDecision = 'approve' | 'approve_for_session' | 'reject'
export type PermissionMode = 'accept-all' | 'auto' | 'manual' | 'plan'
export type PolicyAction = 'allow' | 'deny'
export type PermissionDisposition = 'allow' | 'deny' | 'prompt'

/** Xerxes starts in YOLO mode unless an embedding host explicitly chooses a stricter policy. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = 'accept-all'

export interface ToolPolicy {
  check(toolName: string, agentId?: string): PolicyAction
}

export interface PermissionBroker {
  request(request: PermissionRequest, signal?: AbortSignal): Promise<PermissionDecision>
}

export const SAFE_TOOLS = new Set([
  'ReadFile', 'GlobTool', 'GrepTool', 'ListDir', 'APIClient', 'RSSReader', 'URLAnalyzer', 'DuckDuckGoSearch', 'SystemInfo',
  'skills_list', 'skill_view', 'session_search', 'search_memory', 'get_memory_statistics', 'consolidate_agent_memories',
  'agent_memory_read', 'agent_memory_list', 'agent_memory_search', 'agent_memory_status', 'TaskListTool', 'TaskGetTool',
  'TaskOutputTool', 'AwaitAgents', 'CheckAgentMessages', 'PeekAgent', 'ToolSearchTool', 'AskUserQuestionTool',
  'SetInteractionModeTool', 'JSONProcessor', 'CSVProcessor', 'TextProcessor', 'Calculator', 'StatisticalAnalyzer',
  'MathematicalFunctions', 'UnitConverter', 'DateTimeProcessor',
])

const SAFE_COMMANDS = [
  /^\s*(ls|pwd|whoami|date|uname|cat|head|tail|wc|file|which|type|echo)\b/,
  /^\s*cd(?:\s+(?:--\s+)?[^\n;&|`$()<>]+)?\s*(?:&&\s*pwd\s*)?$/,
  /^\s*git\s+(status|log|diff|branch|show|remote|tag|stash\s+list)\b/,
  /^\s*(find|grep|rg|fd|ag|ack|tree)\b/,
  /^\s*(python|python3|node|ruby|go|cargo|rustc)\s+--version\b/,
  /^\s*(npm|yarn|pnpm|pip|pip3|cargo|go)\s+(list|show|info|search|outdated)\b/,
  // env/printenv are deliberately absent: they dump the daemon's process environment,
  // including provider API keys, so they must always prompt instead of auto-approving.
  /^\s*(hostname|id|groups|locale|df|du|free|uptime|top\s+-l\s*1)\b/,
]

const DANGEROUS_COMMANDS = [
  /\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b/,
  /\bgit\s+(push\s+--force|reset\s+--hard|clean\s+-[a-zA-Z]*f)\b/,
  /\bfind\b[^\n;&|]*\s-(?:delete|exec(?:dir)?|ok(?:dir)?)(?:\s|$)/,
  /\b(mkfs|dd\s+if=|format|fdisk|parted)\b/,
  /\bsudo\b/,
  /\bcurl\b.*\|\s*(bash|sh|zsh)\b/,
  /\bos\.system\b/,
  /\bsubprocess\b/,
  /\beval\s*\(/,
  /\bexec\s*\(/,
  /\bexecSync\s*\(/,
  /\bspawnSync\s*\(/,
  /\bchild_process\b/,
  /\brequire\s*\(/,
  /\bimport\s*\(/,
]

const WRITING_TOOLS = new Set(['Write', 'WriteFile', 'Edit', 'FileEditTool', 'AppendFile'])

const SAFE_DIRECT_COMMANDS = new Set([
  'ack', 'ag', 'cat', 'date', 'df', 'du', 'echo', 'fd', 'free', 'grep', 'groups', 'head', 'hostname', 'id', 'locale',
  'ls', 'pwd', 'rg', 'tail', 'top', 'tree', 'type', 'uname', 'uptime', 'wc', 'which', 'whoami',
])

interface DirectCommand {
  readonly args: readonly string[]
  readonly command: string
  readonly workdir: string | undefined
}

type CommandInput =
  | { readonly kind: 'argv'; readonly value: DirectCommand }
  | { readonly kind: 'shell'; readonly value: string }

export function isSafeShellCommand(command: string): boolean {
  const normalized = command.trim()
  // Fail closed on anything that can execute or redirect outside a plain read-only
  // pipeline: command substitution, backticks, process substitution, redirection.
  if (hasUnsafeShellSyntax(normalized)) {
    return false
  }
  const cdPrefix = normalized.match(/^cd(?:\s+(?:--\s+)?[^\n;&|`$()<>]+)?\s*&&\s*(.+)$/)
  if (cdPrefix?.[1]) {
    return isSafeShellCommand(cdPrefix[1])
  }
  // Newlines and single `&` are command separators too; a benign first line must
  // never smuggle a payload past the segment checks.
  const segments = normalized.split(/&&|\|\||[;\n&]|\|/).map(segment => segment.trim()).filter(Boolean)
  return segments.length > 0
    && segments.every(segment => !DANGEROUS_COMMANDS.some(pattern => pattern.test(segment)))
    && segments.every(segment => SAFE_COMMANDS.some(pattern => pattern.test(segment)))
}

/**
 * Quote-aware scan for shell syntax that the segment allowlist cannot contain.
 *
 * Command substitution and backticks still execute inside double quotes, so only
 * single quotes neutralize them; redirection and process substitution are inert
 * inside either quote style. When in doubt the command prompts instead of running.
 */
function hasUnsafeShellSyntax(command: string): boolean {
  let quote: 'single' | 'double' | undefined
  for (let index = 0; index < command.length; index += 1) {
    const char = command[index]
    if (quote === 'single') {
      if (char === '\'') quote = undefined
      continue
    }
    if (char === '\\') {
      index += 1
      continue
    }
    if (quote === 'double') {
      if (char === '"') quote = undefined
      else if (char === '`' || (char === '$' && command[index + 1] === '(')) return true
      continue
    }
    if (char === '\'') quote = 'single'
    else if (char === '"') quote = 'double'
    else if (char === '`' || char === '>') return true
    else if (char === '$' && command[index + 1] === '(') return true
    else if (char === '<' && command[index + 1] === '(') return true
  }
  return false
}

/**
 * Resolve the static policy gate before applying the interactive permission mode.
 *
 * A policy denial is final. A policy allowance only admits the call to the
 * mode-level rules; it does not silently bypass manual approval.
 */
export function permissionDisposition(
  call: Pick<ToolCall, 'function'>,
  mode: PermissionMode = DEFAULT_PERMISSION_MODE,
  policy?: ToolPolicy,
  agentId?: string,
): PermissionDisposition {
  const name = call.function.name
  if (policy?.check(name, agentId) === 'deny') {
    return 'deny'
  }
  if (mode === 'accept-all') {
    return 'allow'
  }
  if (mode === 'manual') {
    return 'prompt'
  }
  if (SAFE_TOOLS.has(name)) {
    return 'allow'
  }
  const command = commandInput(name, call.function.arguments)
  if (mode === 'plan') {
    return command && isSafeCommandInput(command) ? 'allow' : 'prompt'
  }
  if (command) {
    return isSafeCommandInput(command) ? 'allow' : 'prompt'
  }
  return name === 'Agent'
    || name === 'AgentTool'
    || name === 'SendMessage'
    || name === 'SendMessageTool'
    || name === 'MemorySave'
    ? 'allow'
    : 'prompt'
}

/** Whether a call is immediately allowed; false means either prompt or hard deny. */
export function checkPermission(
  call: Pick<ToolCall, 'function'>,
  mode: PermissionMode = DEFAULT_PERMISSION_MODE,
  policy?: ToolPolicy,
  agentId?: string,
): boolean {
  return permissionDisposition(call, mode, policy, agentId) === 'allow'
}

export function permissionDescription(call: Pick<ToolCall, 'function'>): string {
  const name = call.function.name
  const inputs = call.function.arguments
  const command = commandInput(name, inputs)
  if (command) {
    return `Run: ${command.kind === 'argv' ? formatDirectCommand(command.value) : command.value}`
  }
  if (name === 'Write' || name === 'WriteFile') {
    return `Write to: ${stringInput(inputs, 'file_path')}`
  }
  if (name === 'Edit' || name === 'FileEditTool') {
    return `Edit: ${stringInput(inputs, 'file_path')}`
  }
  if (name === 'AppendFile') {
    return `Append to: ${stringInput(inputs, 'file_path')}`
  }
  if (name === 'SpawnAgents') {
    const labels = spawnAgentLabels(inputs.agents)
    if (!labels.length) return 'Spawn agents in parallel'
    const preview = labels.slice(0, 4).join(', ')
    const remainder = labels.length > 4 ? ` +${labels.length - 4} more` : ''
    return `Spawn ${labels.length} agents in parallel: ${preview}${remainder}`
  }
  if (name === 'Agent' || name === 'AgentTool') {
    const label = stringInput(inputs, 'name') || stringInput(inputs, 'subagent_type') || 'subagent'
    const prompt = stringInput(inputs, 'prompt').trim()
    return prompt ? `Spawn agent ${label}: ${prompt.slice(0, 60)}` : `Spawn agent ${label}`
  }
  if (name === 'SendMessage' || name === 'SendMessageTool') {
    const target = stringInput(inputs, 'target').trim() || 'subagent'
    const message = stringInput(inputs, 'message').trim()
    return message ? `Message ${target}: ${message.slice(0, 60)}` : `Message ${target}`
  }
  const firstValue = Object.values(inputs)[0]
  return `${name}(${String(firstValue ?? '').slice(0, 60)})`
}

export function deniedResult(call: ToolCall): string {
  return `Permission denied for ${call.function.name}.`
}

export function isWritingTool(name: string): boolean {
  return WRITING_TOOLS.has(name)
}

function commandInput(name: string, inputs: JsonObject): CommandInput | undefined {
  if (name === 'Bash') {
    const command = stringInput(inputs, 'command')
    return command ? { kind: 'shell', value: command } : undefined
  }
  if (name === 'exec_command') {
    return execCommandInput(inputs)
  }
  return undefined
}

function execCommandInput(inputs: JsonObject): CommandInput | undefined {
  const command = stringInput(inputs, 'cmd').trim()
  if (!command) return undefined

  const argsValue = inputs.args
  if (argsValue !== undefined) {
    if (!Array.isArray(argsValue) || !argsValue.every(value => typeof value === 'string')) return undefined
    return {
      kind: 'argv',
      value: {
        command,
        args: argsValue,
        workdir: optionalStringInput(inputs, 'workdir'),
      },
    }
  }

  // The native process tool uses a bare executable and optional argv. The PTY
  // operator uses one shell string. Preserve the shell parser whenever `cmd`
  // contains shell syntax or whitespace rather than trying to reinterpret it.
  if (/\s|[;&|`$()<>]/.test(command)) {
    return { kind: 'shell', value: command }
  }
  return {
    kind: 'argv',
    value: {
      command,
      args: [],
      workdir: optionalStringInput(inputs, 'workdir'),
    },
  }
}

function isSafeCommandInput(input: CommandInput): boolean {
  return input.kind === 'shell' ? isSafeShellCommand(input.value) : isSafeDirectCommand(input.value)
}

function isSafeDirectCommand(input: DirectCommand): boolean {
  if (!/^[A-Za-z0-9._+-]+$/.test(input.command)) return false
  if (input.workdir !== undefined && isOutsideWorkspaceReference(input.workdir)) return false
  if (input.args.some(argument => hasControlCharacters(argument) || isOutsideWorkspaceReference(argument))) return false

  const command = input.command.toLowerCase()
  if (command === 'git') return isSafeGitArguments(input.args)
  if (command === 'find') return !input.args.some(isDestructiveFindArgument)
  if (command === 'fd') return !input.args.some(argument => ['-x', '-X', '--exec', '--exec-batch'].includes(argument))
  if (command === 'rg') return !input.args.some(argument => argument === '--pre' || argument.startsWith('--pre='))
  if (command === 'tree') return !input.args.some(argument => argument === '-o' || argument === '--output' || argument.startsWith('--output='))
  if (command === 'file') return !input.args.some(argument => argument === '-C' || argument === '--compile')
  if (SAFE_DIRECT_COMMANDS.has(command)) return true
  if (
    ['python', 'python3', 'node', 'ruby', 'go', 'cargo', 'rustc'].includes(command)
    && input.args.length === 1
    && input.args[0] === '--version'
  ) return true
  if (['npm', 'yarn', 'pnpm', 'pip', 'pip3', 'cargo', 'go'].includes(command)) {
    return input.args.length > 0 && ['list', 'show', 'info', 'search', 'outdated'].includes(input.args[0] ?? '')
  }
  return false
}

function isSafeGitArguments(args: readonly string[]): boolean {
  const subcommand = args[0]
  if (subcommand === undefined) return true
  const rest = args.slice(1)
  if (['status', 'log', 'diff', 'show'].includes(subcommand)) {
    return !rest.some(argument =>
      argument === '--ext-diff'
      || argument === '--textconv'
      || argument === '--output'
      || argument.startsWith('--output='),
    )
  }
  if (subcommand === 'branch') {
    if (!rest.length) return true
    return rest.every(argument =>
      argument.startsWith('-')
      && !['-d', '-D', '-m', '-M', '-c', '-C', '--delete', '--move', '--copy', '--edit-description'].includes(argument),
    )
  }
  if (subcommand === 'remote') {
    return rest.length === 0 || rest.every(argument => argument === '-v' || argument === '--verbose')
  }
  if (subcommand === 'tag') {
    return rest.length === 0 || rest.every(argument => argument === '-l' || argument === '--list' || argument.startsWith('--list='))
  }
  return subcommand === 'stash' && rest[0] === 'list'
}

function isDestructiveFindArgument(argument: string): boolean {
  return /^-(?:delete|exec(?:dir)?|ok(?:dir)?|fls|fprint|fprintf)$/.test(argument)
}

function isOutsideWorkspaceReference(value: string): boolean {
  const candidates = [value.trim()]
  const equals = value.indexOf('=')
  if (equals >= 0) candidates.push(value.slice(equals + 1).trim())
  return candidates.some(candidate => {
    if (!candidate) return false
    const normalized = candidate.replaceAll('\\', '/')
    return normalized.startsWith('/')
      || normalized.startsWith('~/')
      || /^[A-Za-z]:\//.test(normalized)
      || normalized.split('/').includes('..')
  })
}

function hasControlCharacters(value: string): boolean {
  return /[\u0000-\u001f\u007f]/.test(value)
}

function formatDirectCommand(input: DirectCommand): string {
  return [input.command, ...input.args].map(formatArgument).join(' ')
}

function formatArgument(value: string): string {
  return /^[A-Za-z0-9_./:=+@%,-]+$/.test(value) ? value : JSON.stringify(value)
}

function spawnAgentLabels(value: unknown): string[] {
  let agents = value
  if (typeof agents === 'string') {
    try {
      agents = JSON.parse(agents)
    } catch {
      return []
    }
  }
  if (!Array.isArray(agents)) return []
  return agents.map((agent, index) => {
    if (!agent || typeof agent !== 'object' || Array.isArray(agent)) return `agent ${index + 1}`
    const record = agent as Record<string, unknown>
    const label = typeof record.name === 'string' && record.name.trim()
      ? record.name.trim()
      : typeof record.subagent_type === 'string' && record.subagent_type.trim()
        ? record.subagent_type.trim()
        : `agent ${index + 1}`
    return label
  })
}

function stringInput(inputs: JsonObject, key: string): string {
  const value = inputs[key]
  return typeof value === 'string' ? value : ''
}

function optionalStringInput(inputs: JsonObject, key: string): string | undefined {
  const value = inputs[key]
  return typeof value === 'string' ? value : undefined
}
