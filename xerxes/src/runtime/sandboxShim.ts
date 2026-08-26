// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Readable } from 'node:stream'

/**
 * Captured before {@link sealCodeConstructors} disarms the public ones, so the
 * shim can still compile the single guest body it is asked to run while the
 * guest itself cannot compile anything further.
 */
const CompileGuestBody = Function

interface RunRequest {
  readonly code: string
  readonly bindings: readonly string[]
  readonly memoryMb?: number
}

interface BindingCall {
  readonly args: readonly unknown[]
  readonly binding: string
  readonly id: number
  readonly type: 'binding_call'
}

interface BindingResult {
  readonly id: number
  readonly result: unknown
  readonly type: 'binding_result'
}

interface ResultFrame {
  readonly error?: string
  readonly ok: boolean
  readonly output?: unknown
  readonly type: 'result'
}

function send(value: BindingCall | BindingResult | ResultFrame): void {
  process.stdout.write(`${JSON.stringify(value)}\n`)
}

function setupSandbox(request: RunRequest): void {
  // Strip the ambient environment and process utilities that a sandboxed
  // guest must not rely on. This is JavaScript-level hardening, not OS
  // isolation; a malicious native module or Bun escape can still bypass it.
  process.env = {}
  try {
    Object.defineProperty(globalThis, 'Bun', { value: undefined, configurable: false })
    Object.defineProperty(process, 'env', { value: {}, configurable: true, writable: true })
    Object.defineProperty(process, 'argv', { value: [process.argv[0], __filename], configurable: true, writable: true })
    Object.defineProperty(globalThis, 'import', {
      value: async () => { throw new Error('dynamic imports are disabled in the sandbox') },
      configurable: true,
      writable: true,
    })
  } catch {
    // Best-effort: some runtimes freeze globals.
  }

  sealCodeConstructors()

  const bindings: Record<string, (...args: unknown[]) => Promise<unknown>> = {}
  for (const name of request.bindings) {
    bindings[name] = (...args: unknown[]) => callBinding(name, args)
  }
  // @ts-expect-error injected binding surface
  globalThis.__sandbox_bindings__ = Object.freeze(bindings)
}

/**
 * Remove the guest's ability to compile new code at runtime.
 *
 * The source-text guard alone was bypassable in one line, because nothing stops
 * a guest assembling the forbidden text at runtime and handing it to a compiler
 * reached through a prototype rather than through a name:
 *
 *   (async () => {}).constructor("return (await imp" + "ort('node:fs'))…")
 *
 * That reaches AsyncFunction without ever writing `import(` in the scanned
 * source. Disarming Function and its async/generator siblings closes the whole
 * family at once — the guest body has already been compiled by the reference
 * captured at module load, so the shim keeps working while the guest loses the
 * route. Still JavaScript-level hardening, not an OS boundary.
 */
function sealCodeConstructors(): void {
  const refuse = function sandboxedFunctionConstructor(): never {
    throw new Error('compiling new code is disabled in the sandbox')
  }
  const asyncPrototype = Object.getPrototypeOf(async function noop() {}) as { constructor: unknown }
  const generatorPrototype = Object.getPrototypeOf(function* noop() {}) as { constructor: unknown }
  const asyncGeneratorPrototype = Object.getPrototypeOf(async function* noop() {}) as { constructor: unknown }
  for (const target of [Function.prototype, asyncPrototype, generatorPrototype, asyncGeneratorPrototype]) {
    try {
      Object.defineProperty(target, 'constructor', { value: refuse, configurable: false, writable: false })
    } catch {
      // A frozen prototype is already as closed as this can make it.
    }
  }
  try {
    // The bare names too, so `Function("…")` and `eval` fail the same way.
    Object.defineProperty(globalThis, 'Function', { value: refuse, configurable: false, writable: false })
    Object.defineProperty(globalThis, 'eval', { value: refuse, configurable: false, writable: false })
  } catch {
    // Best-effort: some runtimes freeze globals.
  }
}

async function callBinding(name: string, args: unknown[]): Promise<unknown> {
  const id = nextBindingId++
  return new Promise((resolve, reject) => {
    pendingBindings.set(id, { resolve, reject })
    send({ type: 'binding_call', id, binding: name, args })
  })
}

let nextBindingId = 1
const pendingBindings = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()

/**
 * Source-level rejection of the obvious module escapes.
 *
 * Deliberately NOT sticky: this carried the `g` flag while being used with
 * `.test()`, whose lastIndex persists across calls on the same regex object, so
 * the identical payload alternated pass/fail — the guard waved through every
 * second request the shim served.
 *
 * This is a courtesy check that turns a mistake into a clear message. It is not
 * the boundary; a regex over source text cannot be, because the guest can build
 * the string at runtime. {@link sealCodeConstructors} is what actually removes
 * that route.
 */
const DISALLOWED_SYNTAX = /\b(import\s*\(|import\s+['"]|require\s*\(|export\s+(default\s+|const\s+|let\s+|var\s+|function\s+|class\s+|async\s+function\s+))/

async function evaluateCode(code: string): Promise<ResultFrame> {
  try {
    if (DISALLOWED_SYNTAX.test(code)) {
      throw new Error('imports, requires, and module exports are disabled in the sandbox')
    }
    const factory = CompileGuestBody(`return (async () => {\n${code}\n})()`) as () => Promise<unknown>
    const output = await factory()
    return { type: 'result', ok: true, output }
  } catch (error) {
    return { type: 'result', ok: false, error: errorMessage(error) }
  }
}

async function run(request: RunRequest): Promise<void> {
  setupSandbox(request)

  const evaluationPromise = evaluateCode(request.code)
  const responseLoop = (async () => {
    for await (const line of lineIterator(process.stdin)) {
      let message: unknown
      try { message = JSON.parse(line) as unknown } catch { continue }
      if (!isRecord(message) || message.type !== 'binding_result' || typeof message.id !== 'number') continue
      const pending = pendingBindings.get(message.id)
      if (pending === undefined) continue
      pendingBindings.delete(message.id)
      if (isErrorResult(message.result)) {
        pending.reject(new Error(String(message.result.error)))
      } else {
        pending.resolve(message.result)
      }
    }
  })()

  const result = await evaluationPromise
  send(result)
  await responseLoop
}

async function* lineIterator(stream: Readable): AsyncGenerator<string> {
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isErrorResult(value: unknown): value is { error: string } {
  return isRecord(value) && typeof value.error === 'string'
}

async function main(): Promise<void> {
  for await (const line of lineIterator(process.stdin)) {
    if (!line.trim()) continue
    let request: unknown
    try { request = JSON.parse(line) as unknown } catch (error) {
      send({ type: 'result', ok: false, error: `invalid request JSON: ${errorMessage(error)}` })
      return
    }
    if (!isRecord(request) || typeof request.code !== 'string' || !Array.isArray(request.bindings)) {
      send({ type: 'result', ok: false, error: 'invalid sandbox request shape' })
      return
    }
    await run(request as unknown as RunRequest)
    return
  }
  send({ type: 'result', ok: false, error: 'no sandbox request received' })
}

void main()
