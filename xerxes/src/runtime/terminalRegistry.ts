// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// One place to see every shell the agent is driving.
//
// Three execution paths could each start a process and none of them left a
// trace anyone could look at: a foreground `exec_command` returned its output
// into the model's context and threw the process away, a background command
// kept a `proc_id` only the model knew, and a PTY session lived entirely inside
// the operator state. From outside the turn there was no way to answer "what is
// it running right now, and what is that thing printing".
//
// The hard constraint is that watching must not consume. The model's own output
// buffers are drain-on-read by design — `check_command` returns *new* output so
// successive polls show progress — so a viewer that read from them would
// silently steal the lines the model was about to see. Every entry here
// therefore keeps its own tail mirror, written to as output flows past and
// never drained by anyone.

/** Where a terminal entry came from. */
export type TerminalKind = 'background' | 'foreground' | 'pty'

export type TerminalSignal = 'SIGKILL' | 'SIGTERM'

/**
 * Optional control surface published by whoever owns the process.
 *
 * The registry deliberately owns no subprocess handles: the background manager
 * and the PTY manager already own theirs and know how to shut them down
 * correctly (drain cancellation, terminal close, SIGTERM-then-SIGKILL). They
 * hand those operations over here so an inspector can reach them without the
 * daemon having to depend on either manager.
 */
export interface TerminalControl {
  /** Send an interrupt (Ctrl+C) to a live interactive session. */
  readonly interrupt?: () => Promise<void>
  /** Write characters to a live interactive session's stdin. */
  readonly write?: (chars: string) => Promise<void>
  /** Terminate the process. */
  readonly kill?: (signal: TerminalSignal) => Promise<void>
}

export interface TerminalOpenOptions {
  readonly command: string
  readonly control?: TerminalControl
  readonly cwd: string
  /** Session that created this terminal. Kept internal to avoid changing RPC rows. */
  readonly ownerSessionId: string
  /** Stable id; the owning manager's own handle (proc_id, PTY session id). */
  readonly id: string
  readonly kind: TerminalKind
  /** Human label, when the caller has something better than the command line. */
  readonly label?: string
  readonly pid?: number
}

/** Live handle held by the manager that opened the entry. */
export interface TerminalHandle {
  /** Mirror output as it flows past; never blocks and never rejects. */
  append(text: string): void
  /** Mark the process finished. The entry survives as history. */
  close(exitCode: number | null): void
  readonly id: string
}

/** List-view row: everything except the output itself. */
export interface TerminalSummary {
  readonly canInterrupt: boolean
  readonly canKill: boolean
  readonly canWrite: boolean
  readonly command: string
  readonly cwd: string
  /** Epoch ms, absent while running. */
  readonly endedAt?: number
  readonly exitCode: number | null
  readonly id: string
  readonly kind: TerminalKind
  readonly label: string
  /** Total characters observed, including any dropped from the mirror. */
  readonly outputChars: number
  readonly pid?: number
  readonly running: boolean
  /** Epoch ms. */
  readonly startedAt: number
}

/** Detail view: a summary plus the retained tail of what it printed. */
export interface TerminalInspection extends TerminalSummary {
  readonly output: string
  /** Older output was dropped from the mirror before this tail. */
  readonly outputTruncated: boolean
}

export interface TerminalRegistryOptions {
  /** Closed entries retained for inspection before the oldest is forgotten. */
  readonly historyLimit?: number
  /** Characters mirrored per terminal. */
  readonly mirrorCapacity?: number
  /** Epoch-ms clock. */
  readonly now?: () => number
}

const DEFAULT_HISTORY_LIMIT = 40
const DEFAULT_MIRROR_CAPACITY = 200_000
const DEFAULT_INSPECT_CHARS = 20_000

interface TerminalEntry {
  readonly command: string
  readonly control: TerminalControl
  readonly cwd: string
  endedAt: number | undefined
  readonly ownerSessionId: string
  exitCode: number | null
  readonly id: string
  readonly kind: TerminalKind
  readonly label: string
  readonly mirror: TailBuffer
  readonly pid: number | undefined
  running: boolean
  readonly startedAt: number
}

/**
 * Session-scoped view of every terminal the agent has driven.
 *
 * Insertion order is preserved so the list reads as a history; callers that
 * want running work first sort on the `running` flag rather than relying on
 * arrival order, because a background build started early usually outlives
 * a dozen foreground commands.
 */
export class TerminalRegistry {
  private readonly entries = new Map<string, TerminalEntry>()
  private readonly historyLimit: number
  private readonly mirrorCapacity: number
  private readonly now: () => number

  constructor(options: TerminalRegistryOptions = {}) {
    this.historyLimit = positiveInteger(options.historyLimit ?? DEFAULT_HISTORY_LIMIT, 'historyLimit')
    this.mirrorCapacity = positiveInteger(options.mirrorCapacity ?? DEFAULT_MIRROR_CAPACITY, 'mirrorCapacity')
    this.now = options.now ?? (() => Date.now())
  }

  /**
   * Start tracking a live process.
   *
   * Re-opening an existing id replaces the entry: managers reuse ids from their
   * own namespaces, and a recycled `pty_…` should not show the previous
   * session's output.
   */
  open(options: TerminalOpenOptions): TerminalHandle {
    const id = options.id.trim()
    if (!id) throw new TypeError('terminal id must be a non-empty string')
    const ownerSessionId = options.ownerSessionId.trim()
    if (!ownerSessionId) throw new TypeError('terminal ownerSessionId must be a non-empty string')
    const entry: TerminalEntry = {
      id,
      kind: options.kind,
      command: options.command,
      cwd: options.cwd,
      ownerSessionId,
      label: options.label?.trim() || firstWords(options.command),
      control: options.control ?? {},
      mirror: new TailBuffer(this.mirrorCapacity),
      running: true,
      exitCode: null,
      startedAt: this.now(),
      endedAt: undefined,
      pid: options.pid,
    }
    this.entries.delete(id)
    this.entries.set(id, entry)
    this.trim()
    return {
      id,
      append: text => {
        if (text) entry.mirror.append(text)
      },
      close: exitCode => {
        if (!entry.running) return
        entry.running = false
        entry.exitCode = normalizedExit(exitCode)
        entry.endedAt = this.now()
        this.trim()
      },
    }
  }

  /**
   * Record a command that has already finished.
   *
   * Foreground `exec_command` is the case: by the time anyone could inspect it
   * the process is gone, so it is registered and closed in one step purely so
   * the output survives in the history list.
   */
  record(
    options: Omit<TerminalOpenOptions, 'control'> & { readonly exitCode: number | null; readonly output: string },
  ): void {
    const handle = this.open(options)
    handle.append(options.output)
    handle.close(options.exitCode)
  }

  /** Every tracked terminal owned by one session, oldest first. */
  list(ownerSessionId: string): TerminalSummary[] {
    return [...this.entries.values()]
      .filter(entry => entry.ownerSessionId === ownerSessionId)
      .map(entry => summarize(entry))
  }

  /** One terminal with the retained tail of its output, only for its owner. */
  inspect(ownerSessionId: string, id: string, maxChars: number = DEFAULT_INSPECT_CHARS): TerminalInspection | undefined {
    const entry = this.entries.get(id)
    if (entry === undefined || entry.ownerSessionId !== ownerSessionId) return undefined
    const tail = entry.mirror.tail(positiveInteger(maxChars, 'maxChars'))
    return Object.freeze({
      ...summarize(entry),
      output: tail.text,
      outputTruncated: tail.truncated || entry.mirror.dropped,
    })
  }

  /** Write to a live interactive terminal. Rejects when it cannot be written to. */
  async write(ownerSessionId: string, id: string, chars: string): Promise<void> {
    const control = this.liveControl(ownerSessionId, id)
    if (!control.write) throw new Error('this terminal does not accept input')
    await control.write(chars)
  }

  /** Interrupt a live interactive terminal. */
  async interrupt(ownerSessionId: string, id: string): Promise<void> {
    const control = this.liveControl(ownerSessionId, id)
    if (!control.interrupt) throw new Error('this terminal cannot be interrupted')
    await control.interrupt()
  }

  /**
   * Signal a live terminal.
   *
   * The entry is not closed here — the owning manager closes it when the
   * process actually exits, so a kill that a process ignores does not make the
   * list claim it is gone.
   */
  async kill(ownerSessionId: string, id: string, signal: TerminalSignal = 'SIGTERM'): Promise<void> {
    const control = this.liveControl(ownerSessionId, id)
    if (!control.kill) throw new Error('this terminal cannot be killed from here')
    await control.kill(signal)
  }

  /** Forget everything. Session teardown; does not signal anything. */
  clear(): void {
    this.entries.clear()
  }

  private liveControl(ownerSessionId: string, id: string): TerminalControl {
    const entry = this.entries.get(id)
    if (entry === undefined || entry.ownerSessionId !== ownerSessionId) throw new Error(`unknown terminal: ${id}`)
    if (!entry.running) throw new Error('this terminal has already exited')
    return entry.control
  }

  /**
   * Drop the oldest closed entries past the history limit.
   *
   * Running entries are never dropped regardless of age: a build that has been
   * going for an hour is the single most interesting row in the list, and
   * ageing it out would leave a live process with no way to reach it.
   */
  private trim(): void {
    let closed = 0
    for (const entry of this.entries.values()) if (!entry.running) closed += 1
    if (closed <= this.historyLimit) return
    let excess = closed - this.historyLimit
    for (const [id, entry] of this.entries) {
      if (excess === 0) break
      if (entry.running) continue
      this.entries.delete(id)
      excess -= 1
    }
  }
}

/**
 * Bounded buffer that keeps the most recent output and is never drained.
 *
 * Distinct from `BoundedOutputBuffer` on purpose: that one is drain-on-read and
 * returns from the head, which is right for a poller consuming progress and
 * wrong for a viewer, which wants the last screenful of a process that has been
 * printing for an hour.
 */
class TailBuffer {
  private chunks: string[] = []
  private droppedChars = 0
  private length = 0
  private total = 0

  constructor(private readonly capacity: number) {}

  get dropped(): boolean {
    return this.droppedChars > 0
  }

  /** Characters ever observed, including those since dropped. */
  get observed(): number {
    return this.total
  }

  append(text: string): void {
    if (!text) return
    this.chunks.push(text)
    this.length += text.length
    this.total += text.length
    while (this.length > this.capacity && this.chunks.length > 0) {
      const first = this.chunks[0]
      if (first === undefined) break
      const excess = this.length - this.capacity
      if (first.length <= excess) {
        this.chunks.shift()
        this.length -= first.length
        this.droppedChars += first.length
      } else {
        this.chunks[0] = first.slice(excess)
        this.length -= excess
        this.droppedChars += excess
      }
    }
  }

  /** The last `maxChars` characters, without consuming anything. */
  tail(maxChars: number): { readonly text: string; readonly truncated: boolean } {
    const joined = this.chunks.join('')
    if (this.chunks.length > 1) this.chunks = joined ? [joined] : []
    return joined.length <= maxChars
      ? { text: joined, truncated: false }
      : { text: joined.slice(joined.length - maxChars), truncated: true }
  }
}

function summarize(entry: TerminalEntry): TerminalSummary {
  return Object.freeze({
    id: entry.id,
    kind: entry.kind,
    label: entry.label,
    command: entry.command,
    cwd: entry.cwd,
    running: entry.running,
    exitCode: entry.exitCode,
    startedAt: entry.startedAt,
    outputChars: entry.mirror.observed,
    canWrite: entry.running && entry.control.write !== undefined,
    canInterrupt: entry.running && entry.control.interrupt !== undefined,
    canKill: entry.running && entry.control.kill !== undefined,
    ...(entry.pid === undefined ? {} : { pid: entry.pid }),
    ...(entry.endedAt === undefined ? {} : { endedAt: entry.endedAt }),
  })
}

function firstWords(command: string, max = 48): string {
  const line = command.replace(/\s+/g, ' ').trim()
  if (!line) return 'shell'
  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

function normalizedExit(value: number | null): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new TypeError(`${name} must be a positive integer`)
  return value
}
