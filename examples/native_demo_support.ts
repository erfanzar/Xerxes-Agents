// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

import {
  OpenAiCompatibleClient,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
} from '../xerxes/src/index.js'

export type DemoResponder = (request: CompletionRequest) => Promise<string> | string

/**
 * Credential-free LLM port for examples. It retains the real streaming client
 * contract so examples can swap in an explicitly configured provider.
 */
export class DemoLlmClient implements LlmClient {
  constructor(private readonly respond: DemoResponder) {}

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    throwIfAborted(signal)
    const response = await this.respond(request)
    for (const chunk of textChunks(response)) {
      throwIfAborted(signal)
      yield { content: chunk }
    }
    yield {
      usage: {
        inputTokens: approximateTokens(request.messages.map(message => textOf(message.content)).join('\n')),
        outputTokens: approximateTokens(response),
      },
    }
  }
}

/**
 * Select the safe local demo client unless a caller deliberately supplies all
 * connection details for an OpenAI-compatible endpoint.
 *
 * Args:
 *   args: Command line arguments. Live mode needs --live --base-url --api-key.
 *   respond: Deterministic response used by the default demo client.
 */
export function exampleLlm(args: readonly string[], respond: DemoResponder): LlmClient {
  if (!hasFlag(args, '--live')) return new DemoLlmClient(respond)
  const baseUrl = requiredOption(args, '--base-url')
  const apiKey = requiredOption(args, '--api-key')
  return new OpenAiCompatibleClient({
    providerName: 'openai',
    baseUrl,
    apiKey,
  })
}

/** Return a string option, requiring a non-flag value when the flag exists. */
export function optionValue(args: readonly string[], flag: string): string | undefined {
  const index = args.indexOf(flag)
  if (index < 0) return undefined
  const value = args[index + 1]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

/** Return a positive integer option or a supplied default. */
export function positiveIntegerOption(args: readonly string[], flag: string, fallback: number): number {
  const value = optionValue(args, flag)
  if (value === undefined) return fallback
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${flag} must be a positive integer`)
  return parsed
}

/** Return a number option or a supplied default. */
export function numberOption(args: readonly string[], flag: string, fallback: number): number {
  const value = optionValue(args, flag)
  if (value === undefined) return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) throw new Error(`${flag} must be a finite number`)
  return parsed
}

export function hasFlag(args: readonly string[], flag: string): boolean {
  return args.includes(flag)
}

/** A deliberately simple, dependency-free token estimate for example sizing. */
export function approximateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.trim().length / 4))
}

/** Persist an artifact only when the caller requests --write. */
export async function writeJsonWhenRequested(
  args: readonly string[],
  defaultPath: string,
  value: unknown,
): Promise<string | undefined> {
  if (!hasFlag(args, '--write')) return undefined
  const outputPath = resolve(optionValue(args, '--output') ?? defaultPath)
  await mkdir(dirname(outputPath), { recursive: true })
  await writeFile(outputPath, JSON.stringify(value, null, 2) + '\n', 'utf8')
  return outputPath
}

/** Execute a Bun example with a conventional nonzero failure result. */
export function runMain(main: () => Promise<void> | void): void {
  void Promise.resolve(main()).catch(error => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error))
    process.exitCode = 1
  })
}

export function divider(title: string): void {
  console.log(`\n${'='.repeat(72)}\n${title}\n${'='.repeat(72)}`)
}

export function textOf(value: unknown): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value.flatMap(part => (
    typeof part === 'object' && part !== null && 'text' in part && typeof part.text === 'string'
      ? [part.text]
      : []
  )).join('')
}

function requiredOption(args: readonly string[], flag: string): string {
  const value = optionValue(args, flag)
  if (!value) throw new Error(`Live mode requires ${flag}`)
  return value
}

function textChunks(value: string): string[] {
  const words = value.match(/\S+\s*/g) ?? []
  const chunks: string[] = []
  let current = ''
  for (const word of words) {
    if (current.length + word.length > 72 && current) {
      chunks.push(current)
      current = ''
    }
    current += word
  }
  if (current) chunks.push(current)
  return chunks.length ? chunks : ['']
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('Example request was cancelled')
}
