// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { access } from 'node:fs/promises'
import { isAbsolute, resolve, relative } from 'node:path'
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
    const modulePath = workspaceFile(workspaceDirectory, invocation.modulePath)
    await access(modulePath)
    const exportName = requiredName(invocation.exportName, 'exportName')
    const child = Bun.spawn({
      cmd: [
        this.executable,
        '--eval',
        EVALUATOR_PROGRAM,
        pathToFileURL(modulePath).href,
        exportName,
        JSON.stringify(invocation.args),
      ],
      cwd: workspaceDirectory,
      stderr: 'pipe',
      stdin: 'ignore',
      stdout: 'pipe',
    })
    const exitCode = await processExitBeforeDeadline(child, this.timeoutMs)
    if (exitCode === undefined) {
      child.kill('SIGKILL')
      await child.exited
      throw new Error(`behavioral evaluator timed out after ${this.timeoutMs}ms`)
    }

    const stdout = await new Response(child.stdout).text()
    const stderr = await new Response(child.stderr).text()
    if (exitCode !== 0) throw new Error(`behavioral evaluator failed: ${lastLine(stderr) || `exit ${exitCode}`}`)
    return parseEnvelope(stdout)
  }
}

const EVALUATOR_PROGRAM = `
const [moduleUrl, exportName, encodedArgs] = Bun.argv.slice(1)
const mod = await import(moduleUrl)
const fn = mod[exportName]
if (typeof fn !== 'function') throw new Error('missing function export: ' + exportName)
const value = await fn(...JSON.parse(encodedArgs))
process.stdout.write(JSON.stringify({ value }))
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

function parseEnvelope(stdout: string): unknown {
  try {
    const value: unknown = JSON.parse(stdout)
    if (typeof value !== 'object' || value === null || !('value' in value)) {
      throw new Error('missing result envelope')
    }
    return value.value
  } catch (error) {
    throw new Error(`behavioral evaluator returned invalid JSON: ${errorMessage(error)}`)
  }
}

function workspaceFile(workspaceDirectory: string, filePath: string): string {
  const trimmed = filePath.trim()
  if (!trimmed) throw new Error('modulePath must not be empty')
  const candidate = resolve(workspaceDirectory, trimmed)
  const pathFromWorkspace = relative(workspaceDirectory, candidate)
  if (
    pathFromWorkspace === ''
    || pathFromWorkspace === '..'
    || pathFromWorkspace.startsWith('../')
    || isAbsolute(pathFromWorkspace)
  ) {
    throw new Error(`modulePath must remain inside the workspace: ${filePath}`)
  }
  return candidate
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
