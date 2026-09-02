// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFileSync } from 'node:fs'

import { HOOK_POINTS, type HookCallback, type HookPayload, type HookPoint } from './hooks.js'

/**
 * User shell hooks — Claude Code's settings.json `hooks` for Xerxes.
 *
 * Config lives in the `hooks` section of `~/.xerxes/config.yaml` (and the
 * workspace config, only when workspace config is trusted — a repo must never
 * gain shell execution on a user's machine just by being opened):
 *
 * ```yaml
 * hooks:
 *   before_tool_call:            # native point names…
 *     - matcher: "exec_command"  # optional regex on the tool name
 *       command: "./scripts/guard.sh"
 *       timeout_ms: 30000
 *   PreToolUse:                  # …or Claude Code event names (aliases below)
 *     - command: "python3 scripts/audit.py"
 * ```
 *
 * Wire protocol, following Claude Code so existing hook scripts port over:
 * the hook receives the hook payload as JSON on stdin and answers with its
 * exit code — 0 allows/observes, 2 denies (stderr becomes the denial reason),
 * anything else is a hook failure. Blocking is only meaningful at
 * tool_permission_check (PreToolUse), which fails closed: a timeout or crash
 * there denies the call. Mutation hooks (before_tool_call, after_tool_call,
 * tool_result_persist) may additionally print a JSON object on stdout whose
 * `arguments` / `result` field replaces the threaded value.
 */

/** One configured shell hook entry. */
export interface ShellHookSpec {
  /** Shell command run via the platform shell (`sh -c` / `cmd /c`). */
  readonly command: string
  /** Optional regular expression matched against the tool name; absent = all. */
  readonly matcher?: string
  /** Kill the hook after this many milliseconds. Defaults to 60_000. */
  readonly timeout_ms?: number
}

export type ShellHookMap = Partial<Record<HookPoint, readonly ShellHookSpec[]>>

/** Claude Code event names accepted alongside the native hook-point names. */
const EVENT_ALIASES: Readonly<Record<string, HookPoint>> = {
  PostToolUse: 'after_tool_call',
  PreCompact: 'on_compact',
  // PreToolUse maps to the fail-closed permission point so exit-2 denies —
  // before_tool_call cannot refuse a call, it can only mutate arguments.
  PreToolUse: 'tool_permission_check',
  SessionEnd: 'on_session_end',
  SessionStart: 'on_session_start',
  Stop: 'on_turn_end',
  UserPromptSubmit: 'on_turn_start',
}

const VALID_EVENTS = [...HOOK_POINTS, ...Object.keys(EVENT_ALIASES)].join(', ')

/** The mutation field each mutation hook threads, mirroring HookRunner.run. */
const MUTATION_FIELDS: Readonly<Partial<Record<HookPoint, string>>> = {
  after_tool_call: 'result',
  before_tool_call: 'arguments',
  tool_result_persist: 'result',
}

const DEFAULT_TIMEOUT_MS = 60_000
const MAX_HOOK_OUTPUT_BYTES = 64 * 1024

export interface ShellHookConfigLoad {
  readonly errors: readonly string[]
  readonly hooks: ShellHookMap
}

/**
 * Load shell hooks from the user config (`~/.xerxes/config.yaml|.yml|.json`)
 * and, only when workspace config is explicitly trusted, the workspace
 * `xerxes.yaml|.yml|.json`. Workspace hooks run arbitrary shell on the
 * user's machine, so they load exclusively behind the same opt-in that
 * guards the rest of workspace config — a cloned repo must never execute
 * code just by being opened.
 *
 * Other keys in the documents are ignored (the strict core config owns
 * them); malformed hooks sections are reported in `errors` and skipped so a
 * typo never blocks daemon startup.
 */
export async function loadShellHookConfig(options: {
  readonly allowWorkspace: boolean
  readonly home: string
  readonly workspaceRoot?: string
}): Promise<ShellHookConfigLoad> {
  const errors: string[] = []
  let merged: ShellHookMap = {}

  const readHooksSection = async (paths: readonly string[], origin: string): Promise<ShellHookMap> => {
    for (const path of paths) {
      let text: string
      try {
        text = await Bun.file(path).text()
      } catch {
        continue
      }
      let document: unknown
      try {
        document = path.endsWith('.json') ? JSON.parse(text) : Bun.YAML.parse(text)
      } catch (error) {
        errors.push(`${origin}: could not parse ${path}: ${error instanceof Error ? error.message : String(error)}`)
        return {}
      }
      if (typeof document !== 'object' || document === null || Array.isArray(document)) return {}
      const hooks = (document as Record<string, unknown>).hooks
      try {
        return parseShellHookConfig(hooks, `${origin} (${path})`)
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error))
        return {}
      }
    }
    return {}
  }

  const user = await readHooksSection(
    ['config.yaml', 'config.yml', 'config.json'].map(name => `${options.home}/${name}`),
    'user config',
  )
  merged = mergeShellHookMaps(merged, user)

  if (options.allowWorkspace && options.workspaceRoot !== undefined) {
    const workspace = await readHooksSection(
      ['xerxes.yaml', 'xerxes.yml', 'xerxes.json'].map(name => `${options.workspaceRoot}/${name}`),
      'workspace config',
    )
    merged = mergeShellHookMaps(merged, workspace)
  }

  return { errors: Object.freeze(errors), hooks: merged }
}

/** Synchronous variant for daemon startup, which boots synchronously. */
export function loadShellHookConfigSync(options: {
  readonly allowWorkspace: boolean
  readonly home: string
  readonly workspaceRoot?: string
}): ShellHookConfigLoad {
  const errors: string[] = []
  let merged: ShellHookMap = {}

  const readHooksSection = (paths: readonly string[], origin: string): ShellHookMap => {
    for (const path of paths) {
      let text: string
      try {
        text = readFileSync(path, 'utf8')
      } catch {
        continue
      }
      let document: unknown
      try {
        document = path.endsWith('.json') ? JSON.parse(text) : Bun.YAML.parse(text)
      } catch (error) {
        errors.push(`${origin}: could not parse ${path}: ${error instanceof Error ? error.message : String(error)}`)
        return {}
      }
      if (typeof document !== 'object' || document === null || Array.isArray(document)) return {}
      const hooks = (document as Record<string, unknown>).hooks
      try {
        return parseShellHookConfig(hooks, `${origin} (${path})`)
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error))
        return {}
      }
    }
    return {}
  }

  merged = mergeShellHookMaps(
    merged,
    readHooksSection(['config.yaml', 'config.yml', 'config.json'].map(name => `${options.home}/${name}`), 'user config'),
  )
  if (options.allowWorkspace && options.workspaceRoot !== undefined) {
    merged = mergeShellHookMaps(
      merged,
      readHooksSection(['xerxes.yaml', 'xerxes.yml', 'xerxes.json'].map(name => `${options.workspaceRoot}/${name}`), 'workspace config'),
    )
  }
  return { errors: Object.freeze(errors), hooks: merged }
}

function mergeShellHookMaps(base: ShellHookMap, extra: ShellHookMap): ShellHookMap {
  const merged: ShellHookMap = { ...base }
  for (const point of HOOK_POINTS) {
    const additional = extra[point]
    if (additional?.length) {
      merged[point] = [...(merged[point] ?? []), ...additional]
    }
  }
  return merged
}

/**
 * Parse the raw `hooks` section of a config document into a validated map.
 * Unknown event names and malformed entries throw with an actionable message —
 * a silently ignored hook is a security feature that isn't there.
 */
export function parseShellHookConfig(raw: unknown, origin: string): ShellHookMap {
  if (raw === undefined || raw === null) return {}
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`${origin}: hooks must be a map of event name to hook list`)
  }
  const map: ShellHookMap = {}
  for (const [event, list] of Object.entries(raw as Record<string, unknown>)) {
    const point = resolveHookPoint(event, origin)
    if (!Array.isArray(list)) {
      throw new Error(`${origin}: hooks.${event} must be a list of {command, matcher?, timeout_ms?}`)
    }
    const specs = list.map((entry, index) => parseShellHookSpec(entry, `${origin}: hooks.${event}[${index}]`))
    map[point] = [...(map[point] ?? []), ...specs]
  }
  return map
}

function resolveHookPoint(event: string, origin: string): HookPoint {
  if ((HOOK_POINTS as readonly string[]).includes(event)) return event as HookPoint
  const alias = EVENT_ALIASES[event]
  if (alias) return alias
  throw new Error(`${origin}: unknown hook event '${event}' — valid events: ${VALID_EVENTS}`)
}

function parseShellHookSpec(raw: unknown, path: string): ShellHookSpec {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`${path} must be an object with a command`)
  }
  const entry = raw as Record<string, unknown>
  const command = entry.command
  if (typeof command !== 'string' || !command.trim()) {
    throw new Error(`${path}.command must be a non-empty string`)
  }
  const matcher = entry.matcher
  if (matcher !== undefined && typeof matcher !== 'string') {
    throw new Error(`${path}.matcher must be a string regex`)
  }
  if (matcher !== undefined) {
    // Compile eagerly so a bad regex fails config validation, not a turn.
    new RegExp(matcher)
  }
  const timeout = entry.timeout_ms
  if (timeout !== undefined && (typeof timeout !== 'number' || !Number.isFinite(timeout) || timeout <= 0)) {
    throw new Error(`${path}.timeout_ms must be a positive number`)
  }
  return {
    command: command.trim(),
    ...(matcher === undefined ? {} : { matcher }),
    ...(timeout === undefined ? {} : { timeout_ms: Math.trunc(timeout) }),
  }
}

export interface ShellHookRunnerOptions {
  /** Working directory for hook processes; defaults to process.cwd(). */
  readonly cwd?: string
  /** Injectable spawn for tests; defaults to Bun.spawn against the platform shell. */
  readonly run?: ShellHookExecutor
}

/** Runs one hook command; returns exit code and captured output. */
export type ShellHookExecutor = (
  command: string,
  input: string,
  timeoutMs: number,
) => Promise<{ readonly code: number; readonly stderr: string; readonly stdout: string }>

/**
 * Register every configured shell hook on a runner. Returns the number of
 * hooks registered. Tool-only matchers never gate session/turn events.
 */
export function registerShellHooks(
  runner: { register(point: HookPoint, callback: HookCallback): void },
  hooks: ShellHookMap,
  options: ShellHookRunnerOptions = {},
): number {
  const execute = options.run ?? defaultShellHookExecutor(options.cwd)
  let count = 0
  for (const point of HOOK_POINTS) {
    for (const spec of hooks[point] ?? []) {
      runner.register(point, shellHookCallback(point, spec, execute))
      count += 1
    }
  }
  return count
}

function shellHookCallback(point: HookPoint, spec: ShellHookSpec, execute: ShellHookExecutor): HookCallback {
  const matcher = spec.matcher === undefined ? undefined : new RegExp(spec.matcher)
  return async payload => {
    if (matcher !== undefined) {
      const toolName = typeof payload.toolName === 'string' ? payload.toolName : ''
      if (!matcher.test(toolName)) return undefined
    }
    const input = JSON.stringify({ hook_point: point, ...payload })
    const { code, stdout, stderr } = await execute(spec.command, input, spec.timeout_ms ?? DEFAULT_TIMEOUT_MS)

    if (point === 'tool_permission_check') {
      // Fail closed: the HookRunner treats a throw as a denial, and an
      // exit-2 is an explicit one. Exit 0 may carry a JSON verdict on stdout.
      if (code === 2) {
        return { allow: false, reason: stderr.trim() || `denied by hook: ${spec.command}`, source: 'shell_hook' }
      }
      if (code !== 0) {
        throw new Error(`permission hook '${spec.command}' exited ${code}: ${stderr.trim() || 'no output'}`)
      }
      const verdict = parseJson(stdout)
      if (verdict !== undefined && typeof verdict.allow === 'boolean') {
        return {
          allow: verdict.allow,
          ...(typeof verdict.reason === 'string' ? { reason: verdict.reason } : {}),
          source: 'shell_hook',
        }
      }
      return { allow: true, source: 'shell_hook' }
    }

    if (code !== 0) {
      throw new Error(`hook '${spec.command}' exited ${code}: ${stderr.trim() || 'no output'}`)
    }
    const field = MUTATION_FIELDS[point]
    if (field !== undefined) {
      const body = parseJson(stdout)
      if (body !== undefined && field in body) {
        return body[field]
      }
    }
    return undefined
  }
}

function parseJson(stdout: string): Record<string, unknown> | undefined {
  const trimmed = stdout.trim()
  if (!trimmed.startsWith('{')) return undefined
  try {
    const parsed: unknown = JSON.parse(trimmed)
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined
  } catch {
    return undefined
  }
}

function defaultShellHookExecutor(cwd?: string): ShellHookExecutor {
  return async (command, input, timeoutMs) => {
    const shell = process.platform === 'win32' ? 'cmd.exe' : '/bin/sh'
    const args = process.platform === 'win32' ? ['/d', '/s', '/c', command] : ['-c', command]
    const proc = Bun.spawn([shell, ...args], {
      cwd: cwd ?? process.cwd(),
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    proc.stdin.write(input)
    proc.stdin.end()
    const killer = setTimeout(() => proc.kill(), timeoutMs)
    try {
      const readCapped = async (stream: ReadableStream<Uint8Array>): Promise<string> => {
        const reader = stream.getReader()
        const chunks: Uint8Array[] = []
        let total = 0
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          if (value === undefined) continue
          const remaining = MAX_HOOK_OUTPUT_BYTES - total
          if (remaining <= 0) continue
          chunks.push(value.byteLength > remaining ? value.subarray(0, remaining) : value)
          total += value.byteLength
        }
        return new TextDecoder().decode(
          chunks.length === 1 ? chunks[0] : Uint8Array.from(chunks.flatMap(chunk => [...chunk])),
        )
      }
      const [stdout, stderr, code] = await Promise.all([
        readCapped(proc.stdout),
        readCapped(proc.stderr),
        proc.exited,
      ])
      return { code, stderr, stdout }
    } finally {
      clearTimeout(killer)
    }
  }
}
