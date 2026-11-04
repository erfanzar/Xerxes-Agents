// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Entry-point support shared by the one-shot CLI and the ACP server: the user's
 * configured hooks (the same loader the daemon uses) and durable one-shot
 * sessions saved in the daemon's own transcript store and format, so
 * `xerxes --resume <id> "prompt"` continues an interrupted one-shot run.
 */

import { join, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import type { HookRunner } from '../extensions/hooks.js'
import { workspaceShellHooks } from '../extensions/workspaceHooks.js'
import { DaemonTranscriptStore, type DaemonTranscript, type RawMessage } from '../session/daemonTranscript.js'
import type { AgentState } from '../streaming/events.js'
import type { ChatMessage } from '../types/messages.js'
import { getTurnOutcome } from '../types/turnOutcome.js'

type Environment = Readonly<Record<string, string | undefined>>

/**
 * The user's shell hooks for one working directory, loaded exactly as the
 * daemon loads them: `~/.xerxes/config.*`, plus the workspace `xerxes.*` file
 * only under `XERXES_ALLOW_WORKSPACE_CONFIG`.
 */
export function userHookRunnerForCwd(options: {
  readonly home: string
  readonly environment?: Environment
  readonly reportError?: (message: string) => void
}): (cwd: string) => HookRunner {
  const environment = options.environment ?? process.env
  const optIn = environment.XERXES_ALLOW_WORKSPACE_CONFIG ?? ''
  return workspaceShellHooks({
    home: options.home,
    allowWorkspace: optIn === '1' || /^(true|yes|on)$/i.test(optIn),
    reportError: options.reportError ?? (error => console.error(`hooks: ${error}`)),
  })
}

/**
 * Whether one-shot runs are saved: `XERXES_ONESHOT_SAVE` (wins), else
 * `runtime.oneshot.save`, else true. Invalid values throw a ValidationError.
 */
export function parseOneShotSaveConfig(runtime: Readonly<Record<string, unknown>>, env: Environment = process.env): boolean {
  const fromEnv = env.XERXES_ONESHOT_SAVE?.trim()
  if (fromEnv) {
    if (/^(1|true|yes|on)$/i.test(fromEnv)) return true
    if (/^(0|false|no|off)$/i.test(fromEnv)) return false
    throw new ValidationError('XERXES_ONESHOT_SAVE', 'must be 1/true/on or 0/false/off', fromEnv)
  }
  const section = runtime.oneshot
  if (section === undefined || section === null) return true
  if (typeof section !== 'object' || Array.isArray(section)) {
    throw new ValidationError('oneshot', 'must be an object such as { "save": false }', section)
  }
  const save = (section as Record<string, unknown>).save
  if (save === undefined) return true
  if (typeof save !== 'boolean') throw new ValidationError('oneshot.save', 'must be true or false', save)
  return save
}

/**
 * Remove the global `--no-save` flag from CLI arguments. Words after a `--`
 * separator are prompt text and are kept verbatim.
 */
export function extractNoSaveOption(args: readonly string[]): { readonly noSave: boolean; readonly rest: string[] } {
  let noSave = false
  const rest: string[] = []
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]!
    if (argument === '--') {
      rest.push(...args.slice(index))
      break
    }
    if (argument === '--no-save') {
      noSave = true
      continue
    }
    rest.push(argument)
  }
  return { noSave, rest }
}

/** A session id the daemon will resume (`looksLikeSessionId`). */
export function newOneShotSessionId(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}

/** The daemon's transcript store, configured as the CLI daemon configures it. */
export function oneShotTranscriptStore(home: string, projectDirectory: string): DaemonTranscriptStore {
  return new DaemonTranscriptStore({
    currentProjectDirectory: projectDirectory,
    directory: join(home, 'sessions'),
    workspaceRoot: join(home, 'agents'),
  })
}

export interface OneShotSessionRecorderOptions {
  readonly store: DaemonTranscriptStore
  readonly sessionId: string
  readonly cwd: string
  readonly agentId: string
  readonly model: string
  /** `<xerxes home>/agents`, as the daemon runtime's workspace root. */
  readonly agentsRoot: string
  readonly permissionMode?: string
  /** Kept out of the saved history: a resumed session builds its own prompt. */
  readonly systemPrompt?: string
}

/**
 * Saves a one-shot run's agent state as a daemon session.
 *
 * `schedule` coalesces saves requested while one is writing, so calling it at
 * every step boundary costs at most one extra write. `flush` waits for the
 * last state to be on disk and rethrows the most recent failure.
 */
export class OneShotSessionRecorder {
  private readonly options: OneShotSessionRecorderOptions
  private generation = 0
  private persistedMessageCount = 0
  private pending: AgentState | undefined
  private running: Promise<void> | undefined
  private failure: unknown

  constructor(options: OneShotSessionRecorderOptions) {
    this.options = options
  }

  get sessionId(): string {
    return this.options.sessionId
  }

  schedule(state: AgentState): void {
    this.pending = state
    this.running ??= this.drain()
  }

  async flush(state: AgentState): Promise<void> {
    this.schedule(state)
    while (this.running) await this.running
    if (this.failure !== undefined) {
      const failure = this.failure
      this.failure = undefined
      throw failure
    }
  }

  private async drain(): Promise<void> {
    try {
      while (this.pending) {
        const state = this.pending
        this.pending = undefined
        try {
          await this.write(state)
          this.failure = undefined
        } catch (error) {
          this.failure = error
        }
      }
    } finally {
      this.running = undefined
    }
  }

  private async write(state: AgentState): Promise<void> {
    const transcript = oneShotTranscript(state, this.options, this.generation)
    await this.options.store.save(transcript, {
      mode: 'append',
      expectedGeneration: this.generation,
      expectedMessageCount: this.persistedMessageCount,
      onSavedGeneration: generation => { this.generation = generation },
    })
    this.persistedMessageCount = transcript.messages.length
  }
}

/** The daemon transcript for a one-shot agent state. */
export function oneShotTranscript(
  state: AgentState,
  options: OneShotSessionRecorderOptions,
  generation = 0,
): DaemonTranscript {
  const messages = state.messages
    .filter(message => !(message.role === 'system' && options.systemPrompt !== undefined && message.content === options.systemPrompt))
    .map(storedMessage)
  return {
    agentId: options.agentId,
    apiCallsComplete: state.apiCallsComplete,
    cwd: resolve(options.cwd),
    extra: {},
    format: 'bun-v2',
    generation,
    interactionMode: 'code',
    key: options.sessionId,
    messages,
    metadata: {
      model: options.model,
      session_source: 'oneshot',
      ...(options.permissionMode ? { permission_mode: options.permissionMode } : {}),
    },
    pendingResumeReplays: [],
    planMode: false,
    schemaVersion: undefined,
    sessionId: options.sessionId,
    thinkingContent: [...state.thinkingContent],
    toolExecutions: [...state.toolExecutions],
    totalApiCalls: state.totalApiCalls,
    totalInputTokens: state.totalInputTokens,
    totalOutputTokens: state.totalOutputTokens,
    turnCount: state.turnCount,
    updatedAt: new Date().toISOString(),
    usageComplete: state.usageComplete,
    workspace: resolve(options.agentsRoot, options.agentId || 'default'),
  }
}

/** A state message in the form daemon sessions store it. */
function storedMessage(message: ChatMessage): RawMessage {
  const outcome = getTurnOutcome(message)
  const presentation = outcome ? { turn_outcome: outcome } : {}
  if (message.role !== 'user' || !message.displayText) return { ...message, ...presentation }
  const { displayText, ...providerMessage } = message
  return { ...providerMessage, text: displayText, ...presentation }
}
