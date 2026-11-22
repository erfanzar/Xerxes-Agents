// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpath } from 'node:fs/promises'
import { isAbsolute, resolve, relative, sep } from 'node:path'
import { pathToFileURL } from 'node:url'

const DEFAULT_TIMEOUT_MS = 30_000

/** An explicit request to call one exported function from a workspace module. */
export interface WorkspaceModuleInvocation {
  readonly args: readonly unknown[]
  readonly exportName: string
  readonly modulePath: string
  readonly workspaceDirectory: string
}

/** Native behavioral-execution boundary used by coding task graders. */
export interface WorkspaceModuleEvaluator {
  invoke(invocation: WorkspaceModuleInvocation): Promise<unknown>
}

/** Settings for the direct Bun behavioral evaluator. */
export interface BunWorkspaceModuleEvaluatorOptions {
  readonly executable?: string
  readonly timeoutMs?: number
}

/**
 * Evaluate agent-authored TypeScript in a direct, shell-free Bun child process.
 *
 * This is deliberately separate from the provider transport. It never invokes
 * Python and it never passes a command through a shell. Hosts that require a
 * stronger code sandbox can inject their own `WorkspaceModuleEvaluator`.
 */
export class BunWorkspaceModuleEvaluator implements WorkspaceModuleEvaluator {
  private readonly executable: string
  private readonly timeoutMs: number

  constructor(options: BunWorkspaceModuleEvaluatorOptions = {}) {
    this.executable = options.executable ?? process.execPath
    this.timeoutMs = positiveInteger(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, 'timeoutMs')
  }

  async invoke(invocation: WorkspaceModuleInvocation): Promise<unknown> {
    const workspaceDirectory = resolveRequired(invocation.workspaceDirectory, 'workspaceDirectory')
    const lexicalModulePath = workspaceFile(workspaceDirectory, invocation.modulePath)
    const [realWorkspace, realModule] = await Promise.all([
      realpath(workspaceDirectory),
      realpath(lexicalModulePath),
    ])
    assertInsideDirectory(realWorkspace, realModule, invocation.modulePath)
    const exportName = requiredName(invocation.exportName, 'exportName')
    const marker = `__XERXES_EVAL_${crypto.randomUUID()}__`
    const child = Bun.spawn({
      cmd: [
        this.executable,
        '--eval',
        EVALUATOR_PROGRAM,
        pathToFileURL(realModule).href,
        exportName,
        JSON.stringify(invocation.args),
        marker,
      ],
      cwd: realWorkspace,
      stderr: 'pipe',
      stdin: 'ignore',
      stdout: 'pipe',
    })
    const stdoutPromise = new Response(child.stdout).text().catch(() => '')
    const stderrPromise = new Response(child.stderr).text().catch(() => '')
    const exitCode = await processExitBeforeDeadline(child, this.timeoutMs)
    if (exitCode === undefined) {
      child.kill('SIGKILL')
      await settleBeforeDeadline(Promise.all([child.exited, stdoutPromise, stderrPromise]), 250)
      throw new Error(`behavioral evaluator timed out after ${this.timeoutMs}ms`)
    }

    const [stdout, stderr] = await settleWithFallback(stdoutPromise, stderrPromise, 250)
    if (exitCode !== 0) throw new Error(`behavioral evaluator failed: ${lastLine(stderr) || `exit ${exitCode}`}`)
    return parseEnvelope(stdout, marker)
  }
}

const EVALUATOR_PROGRAM = `
const [moduleUrl, exportName, encodedArgs, marker] = Bun.argv.slice(1)
const mod = await import(moduleUrl)
const fn = mod[exportName]
if (typeof fn !== 'function') throw new Error('missing function export: ' + exportName)
const value = await fn(...JSON.parse(encodedArgs))
const envelope = value === undefined ? { ok: true } : { ok: true, value }
process.stdout.write('\\n' + marker + JSON.stringify(envelope) + marker + '\\n')
`

async function processExitBeforeDeadline(child: ReturnType<typeof Bun.spawn>, timeoutMs: number): Promise<number | undefined> {
  let timeout: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      child.exited,
      new Promise<undefined>(resolve => {
        timeout = setTimeout(() => resolve(undefined), timeoutMs)
      }),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
  }
}

async function settleBeforeDeadline(operation: Promise<unknown>, graceMs: number): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      operation.catch(() => undefined),
      new Promise<void>(resolve => {
        timeout = setTimeout(resolve, graceMs)
      }),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
  }
}

async function settleWithFallback(
  stdoutPromise: Promise<string>,
  stderrPromise: Promise<string>,
  graceMs: number,
): Promise<readonly [string, string]> {
  let timeout: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      Promise.all([stdoutPromise, stderrPromise]),
      new Promise<readonly [string, string]>(resolve => {
        timeout = setTimeout(() => resolve(['', '']), graceMs)
      }),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
  }
}

function parseEnvelope(stdout: string, marker: string): unknown {
  const start = stdout.lastIndexOf(marker)
  const beforeEnd = start > 0 ? stdout.lastIndexOf(marker, start - 1) : -1
  const payload = beforeEnd >= 0 ? stdout.slice(beforeEnd + marker.length, start) : stdout
  try {
    const parsed: unknown = JSON.parse(payload)
    if (typeof parsed !== 'object' || parsed === null || !('ok' in parsed) || parsed.ok !== true) {
      throw new Error('missing result envelope')
    }
    return 'value' in parsed ? parsed.value : undefined
  } catch (error) {
    throw new Error(`behavioral evaluator returned invalid JSON: ${errorMessage(error)}`)
  }
}

function workspaceFile(workspaceDirectory: string, filePath: string): string {
  const trimmed = filePath.trim()
  if (!trimmed) throw new Error('modulePath must not be empty')
  const candidate = resolve(workspaceDirectory, trimmed)
  assertInsideDirectory(workspaceDirectory, candidate, filePath)
  return candidate
}

function assertInsideDirectory(workspaceDirectory: string, candidate: string, originalPath: string): void {
  const pathFromWorkspace = relative(workspaceDirectory, candidate)
  if (
    pathFromWorkspace === ''
    || pathFromWorkspace === '..'
    || pathFromWorkspace.startsWith(`..${sep}`)
    || pathFromWorkspace.startsWith('../')
    || isAbsolute(pathFromWorkspace)
  ) {
    throw new Error(`modulePath must remain inside the workspace: ${originalPath}`)
  }
}

function resolveRequired(value: string, label: string): string {
  const trimmed = value.trim()
  if (!trimmed) throw new Error(`${label} must not be empty`)
  return resolve(trimmed)
}

function requiredName(value: string, label: string): string {
  const name = value.trim()
  if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) throw new Error(`${label} must be a JavaScript identifier`)
  return name
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`)
  return value
}

function lastLine(value: string): string {
  return value.trim().split('\n').at(-1)?.slice(0, 240) ?? ''
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
