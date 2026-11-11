// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A run's safety floor: its output folder, its inputs, and the command sandbox
 * bound to them.
 *
 * - The output folder is the working directory. Writing a new file there, or a
 *   file the run itself created, needs no approval in `auto` mode.
 * - The inputs are the files that were in the folder when the run started.
 *   They stay read-only: an edit asks, the sandbox refuses it, and a result
 *   check reports any input that changed anyway.
 * - exec_command runs in the OS sandbox when one works here, and then needs no
 *   approval. A call that asks to run unsandboxed goes through approval.
 *
 * This only ever turns a prompt into an allow, and only in `auto` mode; the
 * permission layer still decides everything else.
 */
import { existsSync, realpathSync } from 'node:fs'
import { mkdtemp, readdir, realpath, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { findCommandSandbox, type BoundCommandSandbox, type CommandSandbox } from '../security/commandSandbox.js'
import type { ToolCall } from '../types/toolCalls.js'
import type { CheckFinding } from './resultChecks.js'

/** More inputs than this and the folder is a project tree, not a task's files: no lock, no sandbox. */
export const MAX_LOCKED_INPUTS = 2_000

/** Folders that are never inputs or outputs: nothing in a run writes them without asking. */
const PROTECTED_FOLDER_NAMES = new Set(['.git', '.hg', '.svn', 'node_modules', '.venv', 'venv'])

/** Tools that write one file, and the argument naming it. */
const FILE_WRITERS: Readonly<Record<string, string>> = {
  Write: 'file_path',
  WriteFile: 'file_path',
  AppendFile: 'file_path',
  Edit: 'file_path',
  FileEditTool: 'file_path',
  XlsxWriter: 'path',
}

export interface RunSafetyOptions {
  /** The working directory; outputs go here. */
  readonly outputFolder: string
  /** The host's command sandbox, when one works here. */
  readonly sandbox?: CommandSandbox
  /** Why there is no sandbox, for the run's notice. */
  readonly sandboxUnavailable?: string
}

/** A file a run wrote, relative to its output folder. */
export interface RunOutput {
  readonly path: string
  readonly size: number
  /** Milliseconds since the epoch. */
  readonly modifiedAt: number
}

/** The inputs a run started with: real path to a size-and-mtime signature. */
interface InputSnapshot {
  readonly files: ReadonlyMap<string, string>
  readonly protectedFolders: readonly string[]
  /** False when the folder held more than {@link MAX_LOCKED_INPUTS} files. */
  readonly complete: boolean
}

async function signature(path: string): Promise<string | undefined> {
  try {
    const info = await stat(path)
    return `${info.size}:${info.mtimeMs}`
  } catch (error) {
    if (errorCode(error) === 'ENOENT') return undefined
    throw error
  }
}

function errorCode(error: unknown): unknown {
  return typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
}

async function snapshotFolder(root: string): Promise<InputSnapshot> {
  const files = new Map<string, string>()
  const protectedFolders: string[] = []
  const pending = [root]
  for (let folder = pending.pop(); folder !== undefined; folder = pending.pop()) {
    let entries
    try {
      entries = await readdir(folder, { withFileTypes: true })
    } catch (error) {
      // A subfolder this process can't read is one it can't write either:
      // keep it out of reach of sandboxed commands too.
      const code = errorCode(error)
      if (folder !== root && (code === 'EACCES' || code === 'EPERM')) {
        protectedFolders.push(folder)
        continue
      }
      throw error
    }
    for (const entry of entries) {
      const path = join(folder, entry.name)
      if (entry.isDirectory()) {
        if (PROTECTED_FOLDER_NAMES.has(entry.name)) protectedFolders.push(path)
        else pending.push(path)
      } else if (entry.isFile()) {
        if (files.size >= MAX_LOCKED_INPUTS) return { files, protectedFolders, complete: false }
        const value = await signature(path)
        if (value !== undefined) files.set(path, value)
      }
    }
  }
  return { files, protectedFolders, complete: true }
}

export class RunSafety implements RunSafetyPort {
  readonly outputFolder: string
  /** Why commands still ask, when they do; undefined when the sandbox is on. */
  readonly sandboxUnavailable: string | undefined
  readonly #inputs: Map<string, string>
  readonly #protectedFolders: readonly string[]
  readonly #complete: boolean
  readonly #tempFolder: string
  readonly #sandbox: CommandSandbox | undefined
  #bound: BoundCommandSandbox | undefined

  private constructor(outputFolder: string, snapshot: InputSnapshot, tempFolder: string, options: RunSafetyOptions) {
    this.outputFolder = outputFolder
    this.#inputs = new Map(snapshot.files)
    this.#protectedFolders = snapshot.protectedFolders
    this.#complete = snapshot.complete
    this.#tempFolder = tempFolder
    this.#sandbox = snapshot.complete ? options.sandbox : undefined
    this.sandboxUnavailable = !snapshot.complete
      ? `the folder has more than ${MAX_LOCKED_INPUTS.toLocaleString('en-US')} files, so its inputs can't be locked`
      : options.sandbox === undefined
        ? options.sandboxUnavailable ?? 'no command sandbox is available'
        : undefined
  }

  /** Take the run's inputs from what is in the output folder now. */
  static async start(options: RunSafetyOptions): Promise<RunSafety> {
    const outputFolder = await realpath(options.outputFolder)
    const snapshot = await snapshotFolder(outputFolder)
    const tempFolder = await mkdtemp(join(await realpath(tmpdir()), 'xerxes-run-'))
    return new RunSafety(outputFolder, snapshot, tempFolder, options)
  }

  /** The input files, relative to the output folder. */
  inputs(): string[] {
    return [...this.#inputs.keys()].map(path => relative(this.outputFolder, path)).sort()
  }

  /**
   * Every file the run created or was allowed to change, written by any tool
   * or script: what is in the output folder now and is not one of its inputs.
   * Undefined when the folder is too large for the run to know its inputs.
   */
  async outputs(): Promise<RunOutput[] | undefined> {
    if (!this.#complete) return undefined
    const now = await snapshotFolder(this.outputFolder)
    if (!now.complete) return undefined
    const written: RunOutput[] = []
    for (const [path, value] of now.files) {
      if (this.#inputs.has(path)) continue
      const [size, modified] = value.split(':')
      written.push({ path: relative(this.outputFolder, path), size: Number(size), modifiedAt: Math.round(Number(modified)) })
    }
    return written.sort((a, b) => b.modifiedAt - a.modifiedAt || a.path.localeCompare(b.path))
  }

  /** The sandbox exec_command runs in, bound to this run's folders; undefined when commands ask. */
  commandSandbox(): BoundCommandSandbox | undefined {
    if (this.#sandbox === undefined) return undefined
    this.#bound ??= this.#sandbox.bind({
      writable: [this.outputFolder, this.#tempFolder],
      readOnlyFolders: this.#protectedFolders,
      readOnlyFiles: [...this.#inputs.keys()],
      tempFolder: this.#tempFolder,
    })
    return this.#bound
  }

  /**
   * Where a tool path really lands: resolved against the output folder, with
   * symlinks in the part that exists followed, so a link can't carry a write
   * outside it.
   */
  #target(path: string): string {
    let existing = isAbsolute(path) ? path : resolve(this.outputFolder, path)
    const rest: string[] = []
    while (!existsSync(existing)) {
      const parent = dirname(existing)
      if (parent === existing) break
      rest.unshift(basename(existing))
      existing = parent
    }
    return join(realpathSync(existing), ...rest)
  }

  /** True for a write that lands in the output folder and touches no input or protected folder. */
  isOutputWrite(path: string): boolean {
    const target = this.#target(path)
    const inside = (folder: string) => target === folder || target.startsWith(folder + sep)
    if (!inside(this.outputFolder)) return false
    if (this.#protectedFolders.some(inside)) return false
    return !this.#inputs.has(target)
  }

  /**
   * `allow` for a call the floor makes safe without asking; undefined for no
   * opinion. Only the `auto` permission mode asks this.
   */
  toolDisposition(call: Pick<ToolCall, 'function'>): 'allow' | undefined {
    const { name, arguments: inputs } = call.function
    if (name === 'exec_command') {
      return inputs.unsandboxed !== true && this.commandSandbox() !== undefined ? 'allow' : undefined
    }
    if (name === 'CSVProcessor') {
      return inputs.operation === 'write' && typeof inputs.file_path === 'string' && this.isOutputWrite(inputs.file_path)
        ? 'allow'
        : undefined
    }
    const key = FILE_WRITERS[name]
    if (key === undefined) return undefined
    const path = inputs[key]
    return typeof path === 'string' && this.isOutputWrite(path) ? 'allow' : undefined
  }

  /** The user approved `call`: when it writes an input, that input is released. */
  approved(call: Pick<ToolCall, 'function'>): void {
    const { name, arguments: inputs } = call.function
    const key = name === 'CSVProcessor' ? 'file_path' : FILE_WRITERS[name]
    const path = key === undefined ? undefined : inputs[key]
    if (typeof path === 'string') this.release(path)
  }

  /**
   * The user approved a write to `path`: if it is an input, it stops being one,
   * so the sandbox and the input check let the run change it.
   */
  release(path: string): void {
    const target = this.#target(path)
    if (this.#inputs.delete(target)) this.#bound = undefined
  }

  /** An `input_modified` error for each input that changed or disappeared since the last check. */
  async inputFindings(): Promise<CheckFinding[]> {
    const findings: CheckFinding[] = []
    for (const [path, before] of this.#inputs) {
      const now = await signature(path)
      const shown = relative(this.outputFolder, path)
      // Reported once: a retry cannot undo the change, so the next check
      // compares against what the file is now.
      if (now === undefined) {
        findings.push({ code: 'input_modified', severity: 'error', path: shown, message: 'An input file was deleted; inputs stay read-only.' })
        this.#inputs.delete(path)
      } else if (now !== before) {
        findings.push({ code: 'input_modified', severity: 'error', path: shown, message: 'An input file was changed; inputs stay read-only. Write results to a new file.' })
        this.#inputs.set(path, now)
      }
    }
    return findings
  }

  /** Remove the run's temp folder. */
  async dispose(): Promise<void> {
    await rm(this.#tempFolder, { recursive: true, force: true })
  }
}

/** What the turn loop asks of a run's safety floor. */
export interface RunSafetyPort {
  toolDisposition(call: Pick<ToolCall, 'function'>): 'allow' | undefined
  commandSandbox(): BoundCommandSandbox | undefined
  /** The user approved this call; a write to an input releases that input. */
  approved(call: Pick<ToolCall, 'function'>): void
}

/**
 * One safety floor per session and output folder, started at the session's
 * first turn there. The sandbox is probed once per process.
 */
export class RunSafetyRegistry {
  readonly #find: () => ReturnType<typeof findCommandSandbox>
  #sandbox: ReturnType<typeof findCommandSandbox> | undefined
  readonly #runs = new Map<string, Promise<RunSafety>>()

  constructor(find: () => ReturnType<typeof findCommandSandbox> = () => findCommandSandbox()) {
    this.#find = find
  }

  forSession(sessionId: string, outputFolder: string): Promise<RunSafety> {
    const key = `${sessionId}\0${outputFolder}`
    let run = this.#runs.get(key)
    if (run === undefined) {
      run = (async () => {
        const found = await (this.#sandbox ??= this.#find())
        return RunSafety.start({
          outputFolder,
          ...(found.sandbox ? { sandbox: found.sandbox } : { sandboxUnavailable: found.reason }),
        })
      })()
      this.#runs.set(key, run)
      // A floor that could not start is tried again at the next turn.
      run.catch(() => this.#runs.delete(key))
    }
    return run
  }

  /** Remove every run's temp folder. */
  async dispose(): Promise<void> {
    const runs = [...this.#runs.values()]
    this.#runs.clear()
    await Promise.all(runs.map(async run => (await run.catch(() => undefined))?.dispose()))
  }
}
