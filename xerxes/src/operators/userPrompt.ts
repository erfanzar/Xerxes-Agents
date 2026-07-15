// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { PendingUserPrompt, UserPromptAnswer, UserPromptOption } from './types.js'

export interface UserPromptManagerOptions {
  readonly idFactory?: () => string
  readonly now?: () => Date
}

export interface UserPromptRequest {
  readonly allowFreeform?: boolean
  readonly options?: readonly string[]
  readonly placeholder?: string
  readonly question: string
}

interface Deferred<T> {
  readonly promise: Promise<T>
  readonly reject: (reason?: unknown) => void
  readonly resolve: (value: T) => void
}

/** Coordinates exactly one outstanding human clarification for a session. */
export class UserPromptManager {
  private readonly idFactory: () => string
  private readonly now: () => Date
  private pending: PendingUserPrompt | undefined
  private deferred: Deferred<UserPromptAnswer> | undefined

  constructor(options: UserPromptManagerOptions = {}) {
    this.idFactory = options.idFactory ?? (() => `user_prompt_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`)
    this.now = options.now ?? (() => new Date())
  }

  getPending(): PendingUserPrompt | undefined {
    return this.pending === undefined ? undefined : copyPrompt(this.pending)
  }

  hasPending(): boolean {
    return this.pending !== undefined
  }

  async request(request: UserPromptRequest, signal?: AbortSignal): Promise<UserPromptAnswer> {
    if (this.pending !== undefined) {
      throw new Error('Another user question is already pending')
    }
    const question = request.question.trim()
    if (!question) throw new TypeError('Question must not be empty')

    const pending: PendingUserPrompt = Object.freeze({
      requestId: this.idFactory(),
      question,
      options: Object.freeze(normalizeOptions(request.options)),
      allowFreeform: request.allowFreeform ?? true,
      ...(request.placeholder?.trim() ? { placeholder: request.placeholder.trim() } : {}),
      createdAt: this.now().toISOString(),
    })
    const deferred = createDeferred<UserPromptAnswer>()
    this.pending = pending
    this.deferred = deferred
    const abort = () => deferred.reject(signal?.reason ?? new Error('User prompt was cancelled'))
    if (signal?.aborted) abort()
    else signal?.addEventListener('abort', abort, { once: true })

    try {
      return await deferred.promise
    } finally {
      signal?.removeEventListener('abort', abort)
      if (this.pending?.requestId === pending.requestId) {
        this.pending = undefined
        this.deferred = undefined
      }
    }
  }

  answer(rawInput: string): UserPromptAnswer {
    const pending = this.requirePending()
    const cleaned = rawInput.trim()
    if (!cleaned) throw new TypeError('Answer cannot be empty.')

    const selectedOption = selectOption(pending.options, cleaned)
    if (selectedOption === undefined && !pending.allowFreeform) {
      throw new TypeError(invalidChoiceMessage(pending))
    }
    const result: UserPromptAnswer = Object.freeze({
      requestId: pending.requestId,
      question: pending.question,
      answer: selectedOption?.value ?? cleaned,
      rawInput: cleaned,
      ...(selectedOption === undefined ? {} : { selectedOption }),
      usedFreeform: selectedOption === undefined,
    })
    this.deferred?.resolve(result)
    return result
  }

  cancel(reason = 'User prompt was cancelled'): void {
    if (this.pending === undefined) return
    this.deferred?.reject(new Error(reason))
  }

  private requirePending(): PendingUserPrompt {
    if (this.pending === undefined) throw new Error('No pending user question.')
    return this.pending
  }
}

function normalizeOptions(options: readonly string[] | undefined): UserPromptOption[] {
  const normalized: UserPromptOption[] = []
  for (const option of options ?? []) {
    const value = option.trim()
    if (value) normalized.push(Object.freeze({ label: value, value }))
  }
  return normalized
}

function selectOption(options: readonly UserPromptOption[], input: string): UserPromptOption | undefined {
  if (input.match(/^\d+$/)) {
    const index = Number(input) - 1
    return index >= 0 && index < options.length ? options[index] : undefined
  }
  const comparison = input.toLowerCase()
  return options.find(option => option.label.toLowerCase() === comparison || option.value.toLowerCase() === comparison)
}

function invalidChoiceMessage(pending: PendingUserPrompt): string {
  if (!pending.options.length) return 'This question requires choosing one of the provided options.'
  return `Choose one of the listed options: ${pending.options.map((option, index) => `${index + 1}:${option.label}`).join(', ')}`
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function copyPrompt(prompt: PendingUserPrompt): PendingUserPrompt {
  return Object.freeze({ ...prompt, options: Object.freeze(prompt.options.map(option => Object.freeze({ ...option }))) })
}
