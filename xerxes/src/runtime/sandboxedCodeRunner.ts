// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

export interface SandboxedRunOptions {
  readonly code: string
  readonly timeoutMs?: number
  readonly memoryMb?: number
}

export interface SandboxedRunResult {
  readonly ok: boolean
  readonly output?: unknown
  readonly error?: string
  readonly bindingCalls?: readonly { readonly binding: string; readonly args: readonly unknown[]; readonly result: unknown }[]
}

export interface SandboxedCodeRunnerOptions {
  readonly bindings?: Readonly<Record<string, (...args: unknown[]) => unknown | Promise<unknown>>>
  readonly nodePath?: string
}

/**
 * Run guest TypeScript in a separate Bun process with a restricted global
 * environment.
 *
 * Read this before trusting it with genuinely hostile input: the child runs as
 * the same uid, in the parent's working directory, with the filesystem and
 * network of the host. What the shim removes is the guest's ability to reach
 * modules — dynamic import, require, and every runtime code-compilation route
 * — plus the ambient environment. That is a meaningful containment for buggy
 * or semi-trusted code, and it is NOT an OS boundary.
 *
 * For hostile input, run this behind the OS sandbox in `security/localSandbox`
 * (bubblewrap / Landlock / Seatbelt), which is the layer that actually confines
 * filesystem and network. The docstring here used to promise isolation this
 * class never provided, which is worse than promising nothing.
 */
export class SandboxedCodeRunner {
  private readonly bindings: Readonly<Record<string, (...args: unknown[]) => unknown | Promise<unknown>>>
  private readonly nodePath: string
  /**
   * Located at call time rather than module load, and checked.
   *
   * `import.meta.url` is `dist/cli.js` in a built install, so the sibling
   * `sandboxShim.ts` does not exist there and the spawn failed by writing to
   * stderr and returning a generic error — the sandbox silently stopped
   * sandboxing for every packaged user while working perfectly from source.
   */
  private static resolveShimPath(): string | undefined {
    const here = dirname(fileURLToPath(import.meta.url))
    const candidates = [
      join(here, 'sandboxShim.ts'),
      join(here, 'sandboxShim.js'),
      join(here, 'runtime', 'sandboxShim.ts'),
    ]
    return candidates.find(candidate => existsSync(candidate))
  }

  constructor(options: SandboxedCodeRunnerOptions = {}) {
    this.bindings = options.bindings ?? {}
    this.nodePath = options.nodePath ?? process.execPath
  }

  async run(options: SandboxedRunOptions): Promise<SandboxedRunResult> {
    const timeoutMs = options.timeoutMs ?? 30_000
    const shimPath = SandboxedCodeRunner.resolveShimPath()
    if (shimPath === undefined) {
      // Fail closed and say so. Running the guest in-process instead, or
      // reporting a vague spawn error, both end with untrusted code executing
      // somewhere nobody audited.
      return { ok: false, error: 'sandbox shim is missing from this installation; refusing to run guest code' }
    }
    const child = spawn(this.nodePath, [shimPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {},
    })

    const bindingCalls: Array<{ readonly binding: string; readonly args: readonly unknown[]; readonly result: unknown }> = []
    let stderr = ''
    let timedOut = false

    child.stderr!.on('data', chunk => { stderr += String(chunk) })
    // An unspawnable interpreter emits 'error', and an unhandled 'error' on a
    // ChildProcess is thrown as an uncaught exception — a bad nodePath took the
    // host down instead of returning a failed run.
    let spawnError: string | undefined
    child.on('error', error => { spawnError = error instanceof Error ? error.message : String(error) })

    const request = JSON.stringify({
      code: options.code,
      bindings: Object.keys(this.bindings),
      memoryMb: options.memoryMb,
    })
    child.stdin!.write(`${request}\n`)

    const timeout = setTimeout(() => {
      timedOut = true
      try { child.kill('SIGKILL') } catch { /* ignore */ }
    }, timeoutMs)

    try {
      let result: SandboxedRunResult | undefined
      for await (const line of lineIterator(child.stdout!)) {
        let message: unknown
        try { message = JSON.parse(line) as unknown } catch { continue }
        if (!isRecord(message)) continue

        if (message.type === 'binding_call'
          && typeof message.binding === 'string'
          && typeof message.id === 'number'
          && Array.isArray(message.args)) {
          const bindingResult = await this.executeBinding(message.binding, message.args)
          bindingCalls.push({ binding: message.binding, args: message.args, result: bindingResult })
          const response = JSON.stringify({ type: 'binding_result', id: message.id, result: bindingResult })
          child.stdin!.write(`${response}\n`)
        } else if (message.type === 'result') {
          const error = stringValue(message.error)
          result = error === undefined
            ? { ok: message.ok === true, output: message.output, bindingCalls }
            : { ok: message.ok === true, output: message.output, error, bindingCalls }
          break
        }
      }
      child.stdin!.end()
      if (timedOut) return { ok: false, error: 'sandboxed execution timed out', bindingCalls }
      if (spawnError !== undefined && result === undefined) {
        return { ok: false, error: `sandboxed runner could not start: ${spawnError}`, bindingCalls }
      }
      if (stderr.trim() && result === undefined) {
        return { ok: false, error: `sandboxed runner stderr: ${stderr.trim()}`, bindingCalls }
      }
      return result ?? { ok: false, error: 'sandboxed runner produced no result frame', bindingCalls }
    } catch (error) {
      if (timedOut) return { ok: false, error: 'sandboxed execution timed out', bindingCalls }
      // A child that never started closes its stdout immediately, so the stream
      // iterator raises "Premature close" and buries the real cause. Prefer the
      // spawn failure whenever one was reported.
      if (spawnError !== undefined) {
        return { ok: false, error: `sandboxed runner could not start: ${spawnError}`, bindingCalls }
      }
      return { ok: false, error: `sandboxed execution failed: ${errorMessage(error)}`, bindingCalls }
    } finally {
      clearTimeout(timeout)
      try { child.stdin?.end() } catch { /* ignore */ }
      if (!child.killed) try { child.kill('SIGKILL') } catch { /* ignore */ }
    }
  }

  private async executeBinding(name: string, args: unknown[]): Promise<unknown> {
    const binding = this.bindings[name]
    if (binding === undefined) return { error: `unknown binding ${name}` }
    try {
      return await binding(...args)
    } catch (error) {
      return { error: errorMessage(error) }
    }
  }
}

async function* lineIterator(stream: NodeJS.ReadableStream): AsyncGenerator<string> {
  let buffer = ''
  for await (const chunk of stream) {
    buffer += String(chunk)
    let index: number
    while ((index = buffer.indexOf('\n')) >= 0) {
      yield buffer.slice(0, index)
      buffer = buffer.slice(index + 1)
    }
  }
  if (buffer.length > 0) yield buffer
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
