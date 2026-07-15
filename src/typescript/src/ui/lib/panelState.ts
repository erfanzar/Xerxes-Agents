// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/**
 * State primitives shared by modal, single-select TUI panels.
 *
 * These helpers deliberately own only selection and timeout state. Terminal input
 * dispatch and prompt rendering stay with the components that mount them.
 */

export const ApprovalChoice = {
  APPROVE: 'approve',
  APPROVE_ALWAYS: 'approve_always',
  APPROVE_FOR_SESSION: 'approve_for_session',
  APPROVE_ONCE: 'approve_once',
  DENY: 'deny',
  VIEW: 'view'
} as const

export type ApprovalChoice = (typeof ApprovalChoice)[keyof typeof ApprovalChoice]

export const DEFAULT_APPROVAL_OPTIONS = [
  ApprovalChoice.APPROVE,
  ApprovalChoice.APPROVE_ONCE,
  ApprovalChoice.APPROVE_FOR_SESSION,
  ApprovalChoice.APPROVE_ALWAYS,
  ApprovalChoice.DENY
] as const satisfies readonly ApprovalChoice[]

const finiteInteger = (value: number, label: string): number => {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new RangeError(`${label} must be a finite integer`)
  }

  return value
}

const modulo = (value: number, length: number): number => ((value % length) + length) % length

/** Mutable cursor over a fixed option list. */
export class PanelSelection<T extends string = string> {
  readonly options: readonly T[]
  index: number

  constructor(options: readonly T[], index = 0) {
    this.options = [...options]
    this.index = this.options.length === 0 ? 0 : modulo(finiteInteger(index, 'index'), this.options.length)
  }

  /** Moves by an arbitrary number of rows and wraps at either edge. */
  move(direction: number): number {
    if (this.options.length === 0) {
      return 0
    }

    this.index = modulo(this.index + finiteInteger(direction, 'direction'), this.options.length)

    return this.index
  }

  up(): number {
    return this.move(-1)
  }

  down(): number {
    return this.move(1)
  }

  /** Selects an absolute row, wrapping negative and oversized indices. */
  set(index: number): number {
    if (this.options.length === 0) {
      return 0
    }

    this.index = modulo(finiteInteger(index, 'index'), this.options.length)

    return this.index
  }

  selected(): T | '' {
    return this.options[this.index] ?? ''
  }
}

/** Selection state for the full approval surface. */
export class ApprovalPanelState {
  readonly choices: readonly ApprovalChoice[]
  readonly selection: PanelSelection<ApprovalChoice>

  constructor(choices: readonly ApprovalChoice[] = DEFAULT_APPROVAL_OPTIONS) {
    this.choices = [...choices]
    this.selection = new PanelSelection(this.choices)
  }

  up(): ApprovalChoice | '' {
    this.selection.up()

    return this.selection.selected()
  }

  down(): ApprovalChoice | '' {
    this.selection.down()

    return this.selection.selected()
  }

  current(): ApprovalChoice {
    const choice = this.selection.selected()

    if (!choice) {
      throw new Error('Approval panel has no choices')
    }

    return choice
  }
}

export interface ApprovalCountdownScheduler {
  cancel(handle: unknown): void
  now(): number
  schedule(callback: () => void, delayMs: number): unknown
}

const systemScheduler: ApprovalCountdownScheduler = {
  cancel: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
  now: () => Date.now(),
  schedule: (callback, delayMs) => setTimeout(callback, delayMs)
}

/**
 * Event-loop countdown for an approval request.
 *
 * The timer is intentionally renderer-independent. A component can redraw
 * from `remaining()` and call `cancel()` when an approval resolves; this
 * helper never attempts to emulate prompt_toolkit's input loop.
 */
export class ApprovalCountdown {
  private active = false
  private generation = 0
  private startedAt: number | null = null
  private timer: unknown | null = null

  constructor(
    readonly timeoutSeconds = 60,
    private readonly scheduler: ApprovalCountdownScheduler = systemScheduler
  ) {
    if (!Number.isFinite(timeoutSeconds)) {
      throw new RangeError('timeoutSeconds must be finite')
    }
  }

  /** Arms a fresh timeout, replacing any previous callback. */
  start(onTimeout: () => void): void {
    this.cancel()
    this.startedAt = this.scheduler.now()
    this.active = true
    const generation = ++this.generation
    const handle = this.scheduler.schedule(() => {
      if (generation !== this.generation) {
        return
      }

      this.active = false
      this.timer = null
      onTimeout()
    }, this.timeoutSeconds * 1_000)

    if (generation === this.generation && this.active) {
      this.timer = handle
    }
  }

  /** Cancels the current timeout. Calling it repeatedly is safe. */
  cancel(): void {
    this.generation += 1

    if (this.timer !== null) {
      this.scheduler.cancel(this.timer)
    }

    this.active = false
    this.startedAt = null
    this.timer = null
  }

  elapsed(): number {
    if (this.startedAt === null) {
      return 0
    }

    return Math.max(0, (this.scheduler.now() - this.startedAt) / 1_000)
  }

  remaining(): number {
    if (this.startedAt === null) {
      return 0
    }

    return Math.max(0, this.timeoutSeconds - this.elapsed())
  }

  isActive(): boolean {
    return this.active
  }
}
