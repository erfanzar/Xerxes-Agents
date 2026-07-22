// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  COMPACTION_SUMMARY_PREFIX,
  CompactionProvisioner,
  renderMessagesForSummary,
  type ContextMessage,
} from '../context/index.js'
import { SmartTokenCounter } from '../context/tokenCounter.js'

export const COMPACTION_LENGTH_INSTRUCTIONS = Object.freeze({
  brief: 'Create an extremely brief summary in 2-3 sentences focusing only on the most critical information.',
  concise: 'Create a concise summary that captures the key points and important details in a few paragraphs.',
  detailed: 'Create a detailed summary that preserves important context, key decisions, and relevant details.',
})

export type CompactionTargetLength = keyof typeof COMPACTION_LENGTH_INSTRUCTIONS

export interface CompactionCompletionRequest {
  readonly maxTokens: number
  readonly prompt: string
  readonly stream: false
  readonly temperature: number
}

export interface CompactionChoice {
  readonly message?: {
    readonly content?: unknown
  }
}

export interface CompactionCompletion {
  readonly choices?: readonly CompactionChoice[]
  readonly content?: unknown
  readonly text?: unknown
}

/** Explicit boundary through which a host invokes its chosen summarization model. */
export type CompactionCompletionPort = (
  request: CompactionCompletionRequest,
) => Promise<CompactionCompletion | string> | CompactionCompletion | string

export interface CompactionAgentOptions {
  readonly completion: CompactionCompletionPort
  readonly model?: string
  readonly targetLength?: CompactionTargetLength | string
  readonly tokenCounter?: SmartTokenCounter
}

const COMPACTION_PROMPT_PLACEHOLDER = '__XERXES_COMPACTION_SUMMARY_PLACEHOLDER__'

/**
 * Model-backed context compactor with an injected completion boundary.
 *
 * It never creates a provider client or derives credentials. Hosts choose the
 * model and expose the exact completion call through `completion`.
 */
export class CompactionAgent {
  readonly model: string
  readonly targetLength: string
  private readonly completion: CompactionCompletionPort
  private readonly tokenCounter: SmartTokenCounter

  constructor(options: CompactionAgentOptions) {
    if (typeof options.completion !== 'function') throw new TypeError('completion must be a function')
    this.completion = options.completion
    this.model = options.model?.trim() || 'compaction'
    this.targetLength = options.targetLength?.trim() || 'concise'
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: this.model })
  }

  /** Summarize a text context while preserving caller-requested topics. */
  async summarizeContext(context: string, preserveTopics: readonly string[] = []): Promise<string> {
    if (!context || context.length < 200) return context
    const response = await this.completion({
      prompt: buildCompactionPrompt(context, this.targetLength, preserveTopics),
      temperature: 0.3,
      maxTokens: 2_048,
      stream: false,
    })
    return completionText(response)
  }

  /**
   * Replace compactable history with a model-written summary.
   *
   * `CompactionProvisioner` continues to own the safety rules for preserving
   * system messages, live context, tool pairs, and summary placement. The
   * provisioner is first used to determine the exact compactable window, then
   * this asynchronous agent invokes the caller-owned completion port.
   */
  async summarizeMessages(messages: readonly ContextMessage[]): Promise<ContextMessage[]> {
    const original = [...messages]
    if (original.length < 2) return original

    const currentTokens = Math.max(1, this.tokenCounter.countTokens(original))
    let compactable: readonly ContextMessage[] | undefined
    const provision = new CompactionProvisioner({
      model: this.model,
      maxContextTokens: currentTokens,
      targetTokens: Math.max(1, Math.floor(currentTokens / 2)),
      tokenCounter: this.tokenCounter,
      summaryAgent: candidate => {
        compactable = candidate
        return COMPACTION_PROMPT_PLACEHOLDER
      },
    }).compact(original, { force: true })

    if (!provision.compacted || compactable === undefined) return original
    const summary = await this.summarizeContext(renderMessagesForSummary(compactable))
    if (!summary.trim()) return original
    const replaced = provision.messages.map(message => replaceSummaryPlaceholder(message, summary))
    if (this.tokenCounter.countTokens(replaced) >= currentTokens) return original
    return replaced
  }
}

/** Construct a compaction agent from a caller-owned completion port. */
export function createCompactionAgent(options: CompactionAgentOptions): CompactionAgent {
  return new CompactionAgent(options)
}

/** Build the instruction delivered to a completion port for one compaction request. */
export function buildCompactionPrompt(
  context: string,
  targetLength: string = 'concise',
  preserveTopics: readonly string[] = [],
): string {
  const lengthInstruction = COMPACTION_LENGTH_INSTRUCTIONS[
    targetLength as CompactionTargetLength
  ] ?? COMPACTION_LENGTH_INSTRUCTIONS.concise
  const topicInstruction = preserveTopics.length
    ? `\n- Ensure these topics are covered: ${preserveTopics.join(', ')}`
    : ''
  return [
    'You are a context compaction specialist. Your job is to summarize conversation context while preserving the most important information.',
    '',
    lengthInstruction,
    '',
    'IMPORTANT GUIDELINES:',
    '- Preserve key facts, decisions, and outcomes',
    '- Maintain chronological order where relevant',
    '- Keep technical details that are likely to be referenced later',
    '- Remove redundant information and verbose explanations',
    '- Use clear, direct language',
    topicInstruction,
    '',
    'CONTEXT TO SUMMARIZE:',
    context,
    '',
    'COMPACTED SUMMARY:',
  ].join('\n')
}

function completionText(response: CompactionCompletion | string): string {
  if (typeof response === 'string') return response
  const choice = response.choices?.[0]?.message?.content
  if (typeof choice === 'string') return choice
  if (typeof response.content === 'string') return response.content
  if (typeof response.text === 'string') return response.text
  throw new TypeError('compaction completion must return a string, content, text, or choices[0].message.content')
}

function replaceSummaryPlaceholder(message: ContextMessage, summary: string): ContextMessage {
  if (typeof message.content !== 'string' || !message.content.includes(COMPACTION_PROMPT_PLACEHOLDER)) {
    return message
  }
  const content = message.content.replace(COMPACTION_PROMPT_PLACEHOLDER, summary)
  if (!content.startsWith(COMPACTION_SUMMARY_PREFIX)) {
    throw new Error('compaction provisioner returned an unexpected summary message')
  }
  return { ...message, content }
}
