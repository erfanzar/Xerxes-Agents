// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { safeSlicePrefix } from './failureSalientPreview.js'

const COMPACTION_REFERENCE_PREFIX_LITERAL = '[CONTEXT COMPACTION — REFERENCE ONLY]'
const COMPACTION_SUMMARY_MARKER_LITERAL = 'xerxes_compaction_summary'

export const SWE_WORKING_SET_OPEN_TAG = '<swe-working-set>'
export const SWE_WORKING_SET_CLOSE_TAG = '</swe-working-set>'
export const DEFAULT_SWE_WORKING_SET_BUDGET_CHARS = 1_600
export const MAX_ACTIVE_FAILURE_COORDINATES = 5
const MAX_TRACKED_FILES = 32
const MAX_TRACKED_VERIFICATIONS = 8

export type SweFileOperation = 'created' | 'edited' | 'patched' | 'moved' | 'deleted'
export type SweFileTouchStatus = 'applied' | 'failed'
export type SweVerificationStatus = 'passed' | 'failed'

export interface SweTouchedFile {
  readonly path: string
  readonly operation: SweFileOperation
  readonly status: SweFileTouchStatus
  readonly movedFrom?: string
  readonly tool?: string
}

export interface SweFailureCoordinate {
  /** Workspace path and 1-based line (e.g. `src/parser.ts:184`). */
  readonly location: string
  /** Concise one-line error summary. */
  readonly summary: string
  /** Normalized verification command that surfaced this failure. */
  readonly command: string
}

export interface SweVerificationEntry {
  readonly command: string
  readonly status: SweVerificationStatus
  readonly exitCode?: number
  readonly failures: readonly SweFailureCoordinate[]
}

export interface SweWorkingSetLedger {
  readonly files: readonly SweTouchedFile[]
  readonly verifications: readonly SweVerificationEntry[]
  readonly activeFailures: readonly SweFailureCoordinate[]
}

export interface ExtractSweWorkingSetOptions {
  /** Prior compaction summary text whose `<swe-working-set>` block should be carried forward first. */
  readonly priorSummary?: string
  /** Optional preceding messages used only to resolve tool call metadata for tool results at the start of `messages`. */
  readonly callContextMessages?: readonly Record<string, unknown>[]
}

const FILE_MUTATING_TOOLS = new Set([
  'WriteFile',
  'AppendFile',
  'FileEditTool',
  'ApplyPatchTool',
  'git_apply_patch',
  'move_file',
  'delete_file',
])

const COMMAND_EXECUTION_TOOLS = new Set([
  'exec_command',
  'Bash',
  'bash',
  'check_command',
])

const VERIFICATION_COMMAND_RE = new RegExp(
  [
    String.raw`\b(?:bun|npm|pnpm|yarn|deno)\s+(?:run\s+(?:--\S+\s+\S+\s+)*)?(?:test(?::\S+)?|check(?::\S+)?|typecheck(?::\S+)?|lint(?::\S+)?|build(?::\S+)?)\b`,
    String.raw`\b(?:pytest|py\.test|vitest|jest|mocha|ava|tap|playwright\s+test|cypress\s+run)\b`,
    String.raw`\bpython3?\s+-m\s+(?:pytest|unittest|mypy|ruff|flake8|pylint)\b`,
    String.raw`\b(?:tsc|tsgo|vue-tsc|svelte-check|pyright|mypy|ruff\s+check|flake8|pylint|eslint|biome\s+(?:check|lint|ci)|oxlint|golangci-lint)\b`,
    String.raw`\bgo\s+(?:test|vet|build)\b`,
    String.raw`\bcargo\s+(?:test|nextest|check|clippy|build)\b`,
    String.raw`\b(?:mvn|gradle|gradlew)\s+(?:test|check|compile|build)\b`,
    String.raw`\bdotnet\s+(?:test|build)\b`,
    String.raw`\b(?:ctest|rspec|phpunit|swift\s+test|mix\s+test|zig\s+build(?:\s+test)?)\b`,
    String.raw`\bmake\s+(?:test|check|build|all)\b`,
    String.raw`\bassertBunOnly\b`,
  ].join('|'),
  'i',
)

const OUTPUT_LOOKS_LIKE_VERIFICATION_RE =
  /(?:={3,}\s*(?:FAILURES|ERRORS|test session starts|short test summary)\b|\b\d+\s+(?:passed|failed|pass|fail)\b|\bRan\s+\d+\s+tests\b|\berror TS\d+:|\berror\[E\d+\]:|\btest result:\s*(?:ok|FAILED)\b)/i

interface ParsedToolCall {
  readonly id: string
  readonly name: string
  readonly args: Record<string, unknown>
  readonly rawArgsText: string
}

interface ParsedToolResult {
  readonly callId: string
  readonly toolName?: string
  readonly rawContent: unknown
  readonly text: string
  readonly failedFlag: boolean
}

export function hasSweWorkingSetEntries(ledger: SweWorkingSetLedger): boolean {
  return ledger.files.length > 0 || ledger.verifications.length > 0 || ledger.activeFailures.length > 0
}

/** Remove all `<swe-working-set>...</swe-working-set>` blocks from a summary string. */
export function stripSweWorkingSet(text: string): string {
  if (!text || !text.includes(SWE_WORKING_SET_OPEN_TAG)) return text.trim()
  let current = text
  for (;;) {
    const open = current.indexOf(SWE_WORKING_SET_OPEN_TAG)
    if (open < 0) break
    const close = current.indexOf(SWE_WORKING_SET_CLOSE_TAG, open + SWE_WORKING_SET_OPEN_TAG.length)
    if (close < 0) {
      current = current.slice(0, open)
      break
    }
    const before = current.slice(0, open).trimEnd()
    const after = current.slice(close + SWE_WORKING_SET_CLOSE_TAG.length).trimStart()
    current = before && after ? `${before}\n\n${after}` : before || after
  }
  return current.trim()
}

/** Parse `<swe-working-set>...</swe-working-set>` blocks from text into a structured ledger. */
export function parseSweWorkingSet(text: string): SweWorkingSetLedger {
  const builder = new SweWorkingSetBuilder()
  if (!text || !text.includes(SWE_WORKING_SET_OPEN_TAG)) return builder.build()

  let searchFrom = 0
  for (;;) {
    const open = text.indexOf(SWE_WORKING_SET_OPEN_TAG, searchFrom)
    if (open < 0) break
    const bodyStart = open + SWE_WORKING_SET_OPEN_TAG.length
    const close = text.indexOf(SWE_WORKING_SET_CLOSE_TAG, bodyStart)
    const block = close < 0 ? text.slice(bodyStart) : text.slice(bodyStart, close)
    builder.mergeParsedBlock(block)
    if (close < 0) break
    searchFrom = close + SWE_WORKING_SET_CLOSE_TAG.length
  }
  return builder.build()
}

/**
 * Deterministically extract the active SWE working-set ledger from a sequence
 * of messages, merging any carried-forward `<swe-working-set>` summary state.
 */
export function extractSweWorkingSet(
  messages: readonly Record<string, unknown>[],
  options: ExtractSweWorkingSetOptions = {},
): SweWorkingSetLedger {
  const builder = new SweWorkingSetBuilder()

  if (options.priorSummary) {
    builder.mergeLedger(parseSweWorkingSet(options.priorSummary))
  }

  const callById = new Map<string, ParsedToolCall>()
  const procCommandById = new Map<string, string>()

  if (options.callContextMessages) {
    for (const ctxMsg of options.callContextMessages) {
      if (!isRecord(ctxMsg)) continue
      for (const call of extractToolCallsFromMessage(ctxMsg)) {
        if (call.id) callById.set(call.id, call)
      }
    }
  }

  // First pass over `messages`: index all tool results by callId so paired calls know their outcome in order.
  const resultsByCallId = new Map<string, ParsedToolResult[]>()
  for (const message of messages) {
    if (!isRecord(message)) continue
    for (const toolResult of extractToolResultsFromMessage(message)) {
      if (!toolResult.callId) continue
      const bucket = resultsByCallId.get(toolResult.callId) ?? []
      bucket.push(toolResult)
      resultsByCallId.set(toolResult.callId, bucket)
    }
  }

  const consumedResults = new Set<ParsedToolResult>()

  for (const message of messages) {
    if (!isRecord(message)) continue

    // Carry-forward from prior compaction summary messages embedded in `messages`.
    if (isSummaryMessageWithWorkingSet(message)) {
      builder.mergeLedger(parseSweWorkingSet(String(message.content)))
      continue
    }

    const calls = extractToolCallsFromMessage(message)
    for (const call of calls) {
      if (call.id) callById.set(call.id, call)
      const matchingResults = call.id ? resultsByCallId.get(call.id) : undefined
      const pairedResult = matchingResults?.shift()
      if (pairedResult) consumedResults.add(pairedResult)
      builder.applyToolInteraction(call, pairedResult, procCommandById)
    }

    // Handle orphan tool results in `messages` whose assistant `tool_call` preceded this slice.
    for (const toolResult of extractToolResultsFromMessage(message)) {
      if (consumedResults.has(toolResult)) continue
      consumedResults.add(toolResult)
      const knownCall = toolResult.callId ? callById.get(toolResult.callId) : undefined
      if (knownCall) {
        builder.applyToolInteraction(knownCall, toolResult, procCommandById)
      } else {
        builder.applyOrphanToolResult(toolResult, procCommandById)
      }
    }
  }

  return builder.build()
}

/**
 * Render a bounded `<swe-working-set>...</swe-working-set>` block (`<= budgetChars`),
 * or `''` when the ledger holds no files or verification state.
 */
export function renderSweWorkingSet(
  ledger: SweWorkingSetLedger,
  budgetChars: number = DEFAULT_SWE_WORKING_SET_BUDGET_CHARS,
): string {
  if (!hasSweWorkingSetEntries(ledger)) return ''
  const maxChars = Math.max(0, Math.trunc(budgetChars))
  const minEnvelope = SWE_WORKING_SET_OPEN_TAG.length + SWE_WORKING_SET_CLOSE_TAG.length + 4
  if (maxChars < minEnvelope) return ''

  let files = [...ledger.files]
  let verifications = [...ledger.verifications]
  let summaryCharCap = 120

  for (let attempt = 0; attempt < 10; attempt += 1) {
    const candidate = formatWorkingSetBlock(files, verifications, ledger.files.length - files.length, summaryCharCap)
    if (candidate.length <= maxChars) return candidate

    if (summaryCharCap > 56) {
      summaryCharCap = Math.max(56, Math.floor(summaryCharCap * 0.7))
      continue
    }
    const passingIdx = verifications.findIndex(entry => entry.status === 'passed' && verifications.length > 1)
    if (passingIdx >= 0) {
      verifications.splice(passingIdx, 1)
      continue
    }
    if (files.length > 4) {
      files = files.slice(-Math.max(4, Math.floor(files.length * 0.65)))
      continue
    }
    if (verifications.length > 2) {
      verifications = verifications.slice(-2)
      continue
    }
    if (files.length > 1) {
      files = files.slice(-1)
      continue
    }
    break
  }

  // Ultra-tight budget fallback: render minimal lines and clamp inside tags without breaking tag closure.
  const innerBudget = maxChars - SWE_WORKING_SET_OPEN_TAG.length - SWE_WORKING_SET_CLOSE_TAG.length - 2
  if (innerBudget <= 0) return ''
  const minimalInner = formatWorkingSetInner(
    files.slice(-2),
    verifications.slice(-1),
    Math.max(0, ledger.files.length - Math.min(2, files.length)),
    40,
  )
  const clampedInner = safeSlicePrefix(minimalInner, innerBudget).trimEnd()
  if (!clampedInner) return ''
  return `${SWE_WORKING_SET_OPEN_TAG}\n${clampedInner}\n${SWE_WORKING_SET_CLOSE_TAG}`
}

function formatWorkingSetBlock(
  files: readonly SweTouchedFile[],
  verifications: readonly SweVerificationEntry[],
  omittedEarlierFiles: number,
  summaryCharCap: number,
): string {
  const inner = formatWorkingSetInner(files, verifications, omittedEarlierFiles, summaryCharCap)
  return `${SWE_WORKING_SET_OPEN_TAG}\n${inner}\n${SWE_WORKING_SET_CLOSE_TAG}`
}

function formatWorkingSetInner(
  files: readonly SweTouchedFile[],
  verifications: readonly SweVerificationEntry[],
  omittedEarlierFiles: number,
  summaryCharCap: number,
): string {
  const lines: string[] = []
  if (files.length > 0) {
    lines.push('Files touched:')
    if (omittedEarlierFiles > 0) {
      lines.push(`- ... (+${omittedEarlierFiles} earlier file${omittedEarlierFiles === 1 ? '' : 's'})`)
    }
    for (const file of files) {
      const pathDisplay = file.operation === 'moved' && file.movedFrom
        ? `${file.movedFrom} -> ${file.path}`
        : file.path
      lines.push(`- ${pathDisplay} (${file.operation}, ${file.status})`)
    }
  }

  if (verifications.length > 0) {
    lines.push('Verification:')
    let failuresShown = 0
    for (const entry of verifications) {
      const badge = entry.status === 'passed' ? 'PASS' : 'FAIL'
      const exitSuffix = entry.exitCode !== undefined ? ` (exit ${entry.exitCode})` : ''
      lines.push(`- [${badge}] ${entry.command}${exitSuffix}`)
      if (entry.status === 'failed') {
        for (const failure of entry.failures) {
          if (failuresShown >= MAX_ACTIVE_FAILURE_COORDINATES) break
          const cleanSummary = safeSlicePrefix(singleLine(failure.summary), summaryCharCap)
          lines.push(`  - failure: ${failure.location} — ${cleanSummary}`)
          failuresShown += 1
        }
      }
    }
  }
  return lines.join('\n')
}

class SweWorkingSetBuilder {
  private readonly filesByPath = new Map<string, SweTouchedFile>()
  private readonly verificationsByKey = new Map<string, SweVerificationEntry>()

  mergeLedger(ledger: SweWorkingSetLedger): void {
    for (const file of ledger.files) {
      this.recordFileTouch(file)
    }
    for (const verification of ledger.verifications) {
      this.recordVerification(verification)
    }
  }

  mergeParsedBlock(block: string): void {
    let section: 'none' | 'files' | 'verification' = 'none'
    let currentVerificationKey: string | undefined

    for (const rawLine of block.split(/\r?\n/)) {
      const trimmed = rawLine.trim()
      if (!trimmed) continue
      if (trimmed === 'Files touched:') {
        section = 'files'
        currentVerificationKey = undefined
        continue
      }
      if (trimmed === 'Verification:') {
        section = 'verification'
        currentVerificationKey = undefined
        continue
      }

      if (section === 'files' && trimmed.startsWith('- ')) {
        if (trimmed.startsWith('- ... (')) continue
        const parsedFile = parseFileBullet(trimmed.slice(2).trim())
        if (parsedFile) this.recordFileTouch(parsedFile)
        continue
      }

      if (section === 'verification') {
        if (/^\s{2,}-\s+failure:\s+/u.test(rawLine) && currentVerificationKey !== undefined) {
          const existing = this.verificationsByKey.get(currentVerificationKey)
          if (existing && existing.status === 'failed') {
            const parsedFailure = parseFailureBullet(trimmed, existing.command)
            if (parsedFailure) {
              const deduped = dedupeFailureList([...existing.failures, parsedFailure])
              this.verificationsByKey.set(currentVerificationKey, {
                ...existing,
                failures: deduped.slice(0, MAX_ACTIVE_FAILURE_COORDINATES),
              })
            }
          }
          continue
        }
        if (trimmed.startsWith('- [')) {
          const parsedVer = parseVerificationBullet(trimmed)
          if (parsedVer) {
            currentVerificationKey = normalizeVerificationCommand(parsedVer.command)
            this.recordVerification(parsedVer)
          }
        }
      }
    }
  }

  applyToolInteraction(
    call: ParsedToolCall,
    result: ParsedToolResult | undefined,
    procCommandById: Map<string, string>,
  ): void {
    if (FILE_MUTATING_TOOLS.has(call.name)) {
      const failed = result ? isFailedToolExecution(result) : false
      const status: SweFileTouchStatus = failed ? 'failed' : 'applied'
      const touches = extractFileTouchesFromCall(call, status, result)
      for (const touch of touches) {
        this.recordFileTouch(touch)
      }
      return
    }

    if (COMMAND_EXECUTION_TOOLS.has(call.name)) {
      const commandFromCall = extractCommandStringFromCall(call, procCommandById)
      if (result) {
        this.applyCommandExecution(call.name, commandFromCall, result, procCommandById)
      }
    }
  }

  applyOrphanToolResult(
    result: ParsedToolResult,
    procCommandById: Map<string, string>,
  ): void {
    const unwrapped = unwrapPersistedOutput(result.text)
    const parsedJson = tryParseJsonRecord(unwrapped)
    if (parsedJson && Array.isArray(parsedJson.command)) {
      const cmdParts = parsedJson.command.filter((part): part is string => typeof part === 'string')
      if (cmdParts.length > 0) {
        this.applyCommandExecution(result.toolName ?? 'exec_command', cmdParts.join(' '), result, procCommandById)
        return
      }
    }
    if (result.toolName && COMMAND_EXECUTION_TOOLS.has(result.toolName)) {
      this.applyCommandExecution(result.toolName, undefined, result, procCommandById)
    }
  }

  private applyCommandExecution(
    toolName: string,
    rawCommand: string | undefined,
    result: ParsedToolResult,
    procCommandById: Map<string, string>,
  ): void {
    const unwrapped = unwrapPersistedOutput(result.text)
    const parsedJson = tryParseJsonRecord(unwrapped)

    // Record background process handles (`procId` / `proc_id`) so subsequent `check_command` calls map back to the command.
    const procId = firstNonEmptyString(parsedJson?.procId, parsedJson?.proc_id)
    const commandFromResult = parsedJson && Array.isArray(parsedJson.command)
      ? parsedJson.command.filter((item): item is string => typeof item === 'string').join(' ')
      : undefined
    const effectiveCommand = (rawCommand || commandFromResult || '').trim()

    if (procId && effectiveCommand) {
      procCommandById.set(procId, effectiveCommand)
    }

    // Ignore commands that are still running in the background and have not yet produced a terminal exit status.
    if (parsedJson?.running === true && parsedJson.exitCode === undefined && parsedJson.exit_code === undefined) {
      return
    }
    if (parsedJson?.running === true && (parsedJson.exitCode === null || parsedJson.exit_code === null)) {
      return
    }

    const isVerification = effectiveCommand
      ? VERIFICATION_COMMAND_RE.test(effectiveCommand)
      : toolName === 'check_command' && OUTPUT_LOOKS_LIKE_VERIFICATION_RE.test(unwrapped)
    if (!isVerification) return

    const commandLabel = normalizeVerificationCommand(effectiveCommand || `${toolName}`)
    const outcome = evaluateVerificationOutcome(result, unwrapped, parsedJson)
    if (outcome.status === 'passed') {
      // Passing run clears any active failures previously recorded for this command!
      this.recordVerification({
        command: commandLabel,
        status: 'passed',
        ...(outcome.exitCode !== undefined ? { exitCode: outcome.exitCode } : { exitCode: 0 }),
        failures: [],
      })
      return
    }

    const failures = extractFailureCoordinates(outcome.combinedOutput, commandLabel)
    this.recordVerification({
      command: commandLabel,
      status: 'failed',
      ...(outcome.exitCode !== undefined ? { exitCode: outcome.exitCode } : {}),
      failures,
    })
  }

  private recordFileTouch(touch: SweTouchedFile): void {
    const normalizedPath = normalizeWorkspacePath(touch.path)
    if (!normalizedPath) return
    const normalizedMovedFrom = touch.movedFrom ? normalizeWorkspacePath(touch.movedFrom) : undefined

    if (touch.operation === 'moved' && normalizedMovedFrom && touch.status === 'applied') {
      this.filesByPath.delete(normalizedMovedFrom)
    }

    const entry: SweTouchedFile = {
      path: normalizedPath,
      operation: touch.operation,
      status: touch.status,
      ...(normalizedMovedFrom ? { movedFrom: normalizedMovedFrom } : {}),
      ...(touch.tool ? { tool: touch.tool } : {}),
    }

    if (!this.filesByPath.has(normalizedPath) && this.filesByPath.size >= MAX_TRACKED_FILES) {
      const oldestKey = this.filesByPath.keys().next().value
      if (oldestKey !== undefined) this.filesByPath.delete(oldestKey)
    }
    this.filesByPath.set(normalizedPath, entry)
  }

  private recordVerification(entry: SweVerificationEntry): void {
    const key = normalizeVerificationCommand(entry.command)
    if (!key) return
    const normalizedEntry: SweVerificationEntry = {
      command: key,
      status: entry.status,
      ...(entry.exitCode !== undefined ? { exitCode: entry.exitCode } : {}),
      failures: entry.status === 'passed'
        ? []
        : dedupeFailureList(entry.failures).slice(0, MAX_ACTIVE_FAILURE_COORDINATES),
    }

    // Refresh position to latest execution order when re-run so newest commands stay protected against eviction.
    if (this.verificationsByKey.has(key)) {
      this.verificationsByKey.delete(key)
    } else if (this.verificationsByKey.size >= MAX_TRACKED_VERIFICATIONS) {
      // Evict oldest passing verification first, otherwise oldest entry.
      let evictKey: string | undefined
      for (const [candidateKey, candidateVal] of this.verificationsByKey) {
        if (candidateVal.status === 'passed') {
          evictKey = candidateKey
          break
        }
      }
      evictKey ??= this.verificationsByKey.keys().next().value
      if (evictKey !== undefined) this.verificationsByKey.delete(evictKey)
    }
    this.verificationsByKey.set(key, normalizedEntry)
  }

  build(): SweWorkingSetLedger {
    const files = Object.freeze([...this.filesByPath.values()])
    const verifications = Object.freeze([...this.verificationsByKey.values()])
    const activeFailures: SweFailureCoordinate[] = []
    const seenKeys = new Set<string>()
    // Collect active failures preferring the most recently executed failing commands first, then restore command order.
    const failingCommands = verifications.filter(entry => entry.status === 'failed')
    for (const entry of failingCommands) {
      for (const failure of entry.failures) {
        if (activeFailures.length >= MAX_ACTIVE_FAILURE_COORDINATES) break
        const dedupeKey = `${failure.location}\u0000${failure.summary}`
        if (seenKeys.has(dedupeKey)) continue
        seenKeys.add(dedupeKey)
        activeFailures.push(failure)
      }
    }
    return Object.freeze({
      files,
      verifications,
      activeFailures: Object.freeze(activeFailures),
    })
  }
}

function isSummaryMessageWithWorkingSet(message: Record<string, unknown>): boolean {
  if (typeof message.content !== 'string' || !message.content.includes(SWE_WORKING_SET_OPEN_TAG)) {
    return false
  }
  if (message[COMPACTION_SUMMARY_MARKER_LITERAL] === true) return true
  if (message.role === 'user' && message.content.startsWith(COMPACTION_REFERENCE_PREFIX_LITERAL)) return true
  return message.role === 'user' && /^\s*<swe-working-set>\r?\n/u.test(message.content)
}

function extractToolCallsFromMessage(message: Record<string, unknown>): ParsedToolCall[] {
  const calls: ParsedToolCall[] = []
  if (Array.isArray(message.tool_calls)) {
    for (const raw of message.tool_calls) {
      const parsed = parseToolCallObject(raw)
      if (parsed) calls.push(parsed)
    }
  }
  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (isRecord(part) && part.type === 'tool_use') {
        const parsed = parseToolCallObject(part)
        if (parsed) calls.push(parsed)
      }
    }
  }
  return calls
}

function parseToolCallObject(raw: unknown): ParsedToolCall | undefined {
  if (!isRecord(raw)) return undefined
  const fn = isRecord(raw.function) ? raw.function : undefined
  const name = firstNonEmptyString(fn?.name, raw.name, raw.tool_name)
  if (!name) return undefined
  const id = firstNonEmptyString(raw.id, raw.tool_call_id) ?? ''
  const rawArgs = [fn?.arguments, raw.arguments, raw.input, raw.args].find(value => value !== undefined)
  if (typeof rawArgs === 'string') {
    const parsedRecord = tryParseJsonRecord(rawArgs)
    return {
      id,
      name,
      args: parsedRecord ?? {},
      rawArgsText: rawArgs,
    }
  }
  if (isRecord(rawArgs)) {
    return {
      id,
      name,
      args: rawArgs,
      rawArgsText: safeJsonString(rawArgs),
    }
  }
  return { id, name, args: {}, rawArgsText: '' }
}

function extractToolResultsFromMessage(message: Record<string, unknown>): ParsedToolResult[] {
  if (message.role === 'tool' || message.role === 'error') {
    const text = messageContentAsText(message.content)
    const toolName = firstNonEmptyString(message.name, message.tool_name)
    return [{
      callId: firstNonEmptyString(message.tool_call_id) ?? '',
      ...(toolName ? { toolName } : {}),
      rawContent: message.content,
      text,
      failedFlag: message.role === 'error' || hasFailureFlag(message),
    }]
  }
  if (Array.isArray(message.content)) {
    const results: ParsedToolResult[] = []
    for (const part of message.content) {
      if (isRecord(part) && part.type === 'tool_result') {
        const text = messageContentAsText(part.content)
        const toolName = firstNonEmptyString(part.name, part.tool_name)
        results.push({
          callId: firstNonEmptyString(part.tool_use_id, part.tool_call_id) ?? '',
          ...(toolName ? { toolName } : {}),
          rawContent: part.content,
          text,
          failedFlag: hasFailureFlag(part),
        })
      }
    }
    return results
  }
  return []
}

function extractFileTouchesFromCall(
  call: ParsedToolCall,
  status: SweFileTouchStatus,
  result: ParsedToolResult | undefined,
): SweTouchedFile[] {
  const args = call.args
  switch (call.name) {
    case 'WriteFile': {
      const path = firstNonEmptyString(args.file_path, args.path)
      if (!path) return []
      const resultJson = result ? tryParseJsonRecord(unwrapPersistedOutput(result.text)) : undefined
      const overwritten = resultJson?.overwritten === true || resultJson?.created === false
      return [{ path, operation: overwritten ? 'edited' : 'created', status, tool: call.name }]
    }
    case 'AppendFile':
    case 'FileEditTool': {
      const path = firstNonEmptyString(args.file_path, args.path)
      return path ? [{ path, operation: 'edited', status, tool: call.name }] : []
    }
    case 'delete_file': {
      const path = firstNonEmptyString(args.path, args.file_path)
      return path ? [{ path, operation: 'deleted', status, tool: call.name }] : []
    }
    case 'move_file': {
      const source = firstNonEmptyString(args.source, args.from, args.old_path)
      const destination = firstNonEmptyString(args.destination, args.to, args.new_path)
      if (!destination) return []
      return [{
        path: destination,
        operation: 'moved',
        status,
        ...(source ? { movedFrom: source } : {}),
        tool: call.name,
      }]
    }
    case 'git_apply_patch': {
      if (args.check_only === true) return []
      const patchText = firstNonEmptyString(args.patch_content, args.patch, args.diff) ?? call.rawArgsText
      const repoPath = firstNonEmptyString(args.repo_path)
      const prefix = repoPath && repoPath !== '.' ? repoPath.replace(/\/+$/u, '') + '/' : ''
      const parsed = parsePatchTouches(patchText, call.name, status)
      return prefix
        ? parsed.map(entry => ({
            ...entry,
            path: `${prefix}${entry.path}`,
            ...(entry.movedFrom ? { movedFrom: `${prefix}${entry.movedFrom}` } : {}),
          }))
        : parsed
    }
    case 'ApplyPatchTool': {
      const patchText = firstNonEmptyString(args.patch, args.patch_content, args.diff, args.input, args.content)
        ?? call.rawArgsText
      const parsed = parsePatchTouches(patchText, call.name, status)
      if (parsed.length > 0) return parsed
      const singlePath = firstNonEmptyString(args.file_path, args.path)
      return singlePath ? [{ path: singlePath, operation: 'patched', status, tool: call.name }] : []
    }
    default:
      return []
  }
}

/** Parse touched paths & operation kinds from V4A (`*** Update/Add/Delete/Move`) and unified diff patches. */
export function parsePatchTouches(
  patchText: string,
  toolName: string,
  status: SweFileTouchStatus,
): SweTouchedFile[] {
  if (!patchText) return []
  const lines = patchText.split(/\r?\n/)
  const touches: SweTouchedFile[] = []
  const seen = new Set<string>()

  const pushTouch = (touch: SweTouchedFile) => {
    const key = `${touch.path}\u0000${touch.operation}`
    if (seen.has(key)) return
    seen.add(key)
    touches.push(touch)
  }

  // Pass 1: V4A headers (`*** Update File: ...`, `*** Add File: ...`, `*** Delete File: ...`, `*** Move to: ...`)
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx] ?? ''
    const addMatch = /^\*\*\*\s+Add File:\s+(.+?)\s*$/u.exec(line)
    if (addMatch?.[1]) {
      pushTouch({ path: addMatch[1], operation: 'created', status, tool: toolName })
      continue
    }
    const deleteMatch = /^\*\*\*\s+Delete File:\s+(.+?)\s*$/u.exec(line)
    if (deleteMatch?.[1]) {
      pushTouch({ path: deleteMatch[1], operation: 'deleted', status, tool: toolName })
      continue
    }
    const updateMatch = /^\*\*\*\s+Update File:\s+(.+?)\s*$/u.exec(line)
    if (updateMatch?.[1]) {
      const nextLine = lines[idx + 1] ?? ''
      const moveMatch = /^\*\*\*\s+Move to:\s+(.+?)\s*$/u.exec(nextLine)
      if (moveMatch?.[1]) {
        pushTouch({
          path: moveMatch[1],
          movedFrom: updateMatch[1],
          operation: 'moved',
          status,
          tool: toolName,
        })
        idx += 1
      } else {
        pushTouch({ path: updateMatch[1], operation: 'patched', status, tool: toolName })
      }
    }
  }
  if (touches.length > 0) return touches

  // Pass 2: Standard unified diff headers (`diff --git`, `--- `, `+++ `, `rename from/to`)
  let pendingOldPath: string | undefined
  let pendingRenameFrom: string | undefined
  for (const line of lines) {
    const renameFrom = /^rename from\s+(.+?)\s*$/u.exec(line)
    if (renameFrom?.[1]) {
      pendingRenameFrom = renameFrom[1]
      continue
    }
    const renameTo = /^rename to\s+(.+?)\s*$/u.exec(line)
    if (renameTo?.[1]) {
      pushTouch({
        path: renameTo[1],
        ...(pendingRenameFrom ? { movedFrom: pendingRenameFrom } : {}),
        operation: 'moved',
        status,
        tool: toolName,
      })
      pendingRenameFrom = undefined
      pendingOldPath = undefined
      continue
    }
    if (line.startsWith('--- ')) {
      pendingOldPath = parseUnifiedHeaderPath(line.slice(4))
      continue
    }
    if (line.startsWith('+++ ')) {
      const newPath = parseUnifiedHeaderPath(line.slice(4))
      if (pendingOldPath === '/dev/null' && newPath && newPath !== '/dev/null') {
        pushTouch({ path: newPath, operation: 'created', status, tool: toolName })
      } else if (newPath === '/dev/null' && pendingOldPath && pendingOldPath !== '/dev/null') {
        pushTouch({ path: pendingOldPath, operation: 'deleted', status, tool: toolName })
      } else if (newPath && newPath !== '/dev/null') {
        pushTouch({ path: newPath, operation: 'patched', status, tool: toolName })
      }
      pendingOldPath = undefined
      continue
    }
  }
  if (touches.length > 0) return touches

  // Fallback to `diff --git a/foo b/bar` lines when `---`/`+++` were omitted (e.g. mode/empty file diff)
  for (const line of lines) {
    const gitMatch = /^diff --git a\/(.+?) b\/(.+?)\s*$/u.exec(line)
    if (gitMatch?.[2]) {
      pushTouch({ path: gitMatch[2], operation: 'patched', status, tool: toolName })
    }
  }
  return touches
}

function parseUnifiedHeaderPath(rawHeaderRest: string): string {
  const withoutTimestamp = rawHeaderRest.split('\t', 1)[0]?.trim() ?? ''
  const unquoted = withoutTimestamp.replace(/^"(.*)"$/u, '$1')
  if (unquoted === '/dev/null') return '/dev/null'
  if (unquoted.startsWith('a/') || unquoted.startsWith('b/')) {
    return unquoted.slice(2)
  }
  return unquoted
}

function extractCommandStringFromCall(
  call: ParsedToolCall,
  procCommandById: ReadonlyMap<string, string>,
): string | undefined {
  const args = call.args
  if (call.name === 'exec_command') {
    const cmd = firstNonEmptyString(args.cmd, args.command)
    if (!cmd) return undefined
    const argList = Array.isArray(args.args)
      ? args.args.filter((item): item is string => typeof item === 'string')
      : []
    return [cmd, ...argList].join(' ').trim()
  }
  if (call.name === 'Bash' || call.name === 'bash') {
    return firstNonEmptyString(args.command, args.cmd)
  }
  if (call.name === 'check_command') {
    const procId = firstNonEmptyString(args.proc_id, args.procId)
    return procId ? procCommandById.get(procId) : undefined
  }
  return undefined
}

function evaluateVerificationOutcome(
  result: ParsedToolResult,
  unwrappedText: string,
  parsedJson: Record<string, unknown> | undefined,
): {
  readonly status: SweVerificationStatus
  readonly exitCode?: number
  readonly combinedOutput: string
} {
  if (parsedJson) {
    const rawExit = typeof parsedJson.exitCode === 'number'
      ? parsedJson.exitCode
      : typeof parsedJson.exit_code === 'number'
        ? parsedJson.exit_code
        : undefined
    const stdout = typeof parsedJson.stdout === 'string' ? parsedJson.stdout : ''
    const stderr = typeof parsedJson.stderr === 'string' ? parsedJson.stderr : ''
    const combined = [stderr, stdout].filter(Boolean).join('\n') || unwrappedText
    if (rawExit !== undefined && Number.isFinite(rawExit)) {
      return {
        status: rawExit === 0 && !result.failedFlag ? 'passed' : 'failed',
        exitCode: Math.trunc(rawExit),
        combinedOutput: combined,
      }
    }
    if (hasFailureFlag(parsedJson) || result.failedFlag) {
      return { status: 'failed', combinedOutput: combined }
    }
  }

  // Check explicit exit code patterns in plain-text or previewed JSON output.
  const exitMatch =
    /(?:"exitCode"\s*:\s*(-?\d+)|"exit_code"\s*:\s*(-?\d+)|\bexit\s+(?:code|status)\s*[:=]?\s*(-?\d+)|\bexit\s*[:=]\s*(-?\d+)|\bexited with (?:code|status)\s+(-?\d+))/i.exec(
      unwrappedText,
    )
  const parsedExit = exitMatch
    ? Number(exitMatch[1] ?? exitMatch[2] ?? exitMatch[3] ?? exitMatch[4] ?? exitMatch[5])
    : undefined
  if (parsedExit !== undefined && Number.isFinite(parsedExit)) {
    return {
      status: parsedExit === 0 && !result.failedFlag ? 'passed' : 'failed',
      exitCode: Math.trunc(parsedExit),
      combinedOutput: unwrappedText,
    }
  }

  if (result.failedFlag) {
    return { status: 'failed', combinedOutput: unwrappedText }
  }

  const hasNonZeroFailures =
    /\b[1-9]\d*\s+(?:failed|fail|errors?)\b/i.test(unwrappedText)
    || /={3,}\s*(?:FAILURES|ERRORS)\s*={3,}/.test(unwrappedText)
    || /\berror TS\d+:/.test(unwrappedText)
    || /\berror\[E\d+\]:/.test(unwrappedText)
    || /(?:^|\n)\s*(?:FAIL|FAILED)\s+\S/.test(unwrappedText)
    || /(?:^|\n)\s*(?:✕|✗)\s+\S/u.test(unwrappedText)
    || /Traceback \(most recent call last\):/.test(unwrappedText)
    || /\b(?:AssertionError|TypeError|SyntaxError|ReferenceError):/.test(unwrappedText)
    || /\btest result:\s*FAILED\b/.test(unwrappedText)

  return {
    status: hasNonZeroFailures ? 'failed' : 'passed',
    combinedOutput: unwrappedText,
  }
}

/**
 * Deterministically extract up to `MAX_ACTIVE_FAILURE_COORDINATES` (`5`)
 * `file:line` + one-line error summaries from compiler / test runner output.
 */
export function extractFailureCoordinates(
  output: string,
  command: string,
): SweFailureCoordinate[] {
  if (!output) return []
  const lines = output.split(/\r?\n/)
  const found: SweFailureCoordinate[] = []
  const seenLocations = new Set<string>()

  const addCoordinate = (rawPath: string, lineNum: string | number, rawSummary: string) => {
    if (found.length >= MAX_ACTIVE_FAILURE_COORDINATES) return
    const cleanPath = normalizeDiagnosticPath(rawPath)
    if (!cleanPath || isInternalRuntimePath(cleanPath)) return
    const parsedLine = typeof lineNum === 'number' ? lineNum : Number(lineNum)
    const location = Number.isSafeInteger(parsedLine) && parsedLine > 0
      ? `${cleanPath}:${parsedLine}`
      : cleanPath
    if (seenLocations.has(location)) return
    const summary = cleanErrorSummary(rawSummary)
    if (!summary) return
    seenLocations.add(location)
    found.push({ location, summary, command })
  }

  // 1. TypeScript `tsc`, Rust `cargo`, and inline `path:line(:col): error ...` diagnostics
  for (let idx = 0; idx < lines.length && found.length < MAX_ACTIVE_FAILURE_COORDINATES; idx += 1) {
    const line = lines[idx] ?? ''

    // tsc paren format: src/foo.ts(184,12): error TS2322: Message
    const tscParen = /^\s*([^\s():]+\.[A-Za-z0-9]{1,10})\((\d+),\d+\):\s*(error\s+TS\d+:\s*.+)$/u.exec(line)
    if (tscParen?.[1] && tscParen[2] && tscParen[3]) {
      addCoordinate(tscParen[1], tscParen[2], tscParen[3])
      continue
    }

    // tsc colon format: src/foo.ts:184:12 - error TS2322: Message
    const tscColon = /^\s*([^\s:]+\.[A-Za-z0-9]{1,10}):(\d+):\d+\s*-\s*(error\s+TS\d+:\s*.+)$/u.exec(line)
    if (tscColon?.[1] && tscColon[2] && tscColon[3]) {
      addCoordinate(tscColon[1], tscColon[2], tscColon[3])
      continue
    }

    // Rust diagnostic: error[E0308]: mismatched types \n --> src/lib.rs:184:12
    const rustErr = /^\s*(error(?:\[E\d+\])?:\s*.+)$/u.exec(line)
    if (rustErr?.[1]) {
      for (let look = idx + 1; look <= Math.min(lines.length - 1, idx + 3); look += 1) {
        const arrow = /^\s*-->\s*([^\s:]+\.rs):(\d+):\d+/u.exec(lines[look] ?? '')
        if (arrow?.[1] && arrow[2]) {
          addCoordinate(arrow[1], arrow[2], rustErr[1])
          break
        }
      }
      continue
    }
  }

  // 2. JS/TS/Bun/Vitest/Jest error headline followed by stack frames (`at ... (file:line:col)`)
  for (let idx = 0; idx < lines.length && found.length < MAX_ACTIVE_FAILURE_COORDINATES; idx += 1) {
    const line = lines[idx] ?? ''
    const headlineMatch =
      /^\s*(?:Uncaught\s+)?((?:AssertionError|TypeError|SyntaxError|ReferenceError|RangeError|Error|[A-Z][A-Za-z0-9_]*(?:Error|Exception)):\s*.+)$/u.exec(
        line,
      )
    if (headlineMatch?.[1]) {
      const frame = findNearestWorkspaceFrameAfter(lines, idx + 1, Math.min(lines.length, idx + 10))
      if (frame) {
        addCoordinate(frame.path, frame.line, headlineMatch[1])
        continue
      }
    }
  }

  // 3. Python standard tracebacks (`File "path.py", line N` ... `<Error>: <msg>`) and Pytest (`E   <Error>` + `path.py:N:`)
  for (let idx = 0; idx < lines.length && found.length < MAX_ACTIVE_FAILURE_COORDINATES; idx += 1) {
    const line = lines[idx] ?? ''

    // Pytest `E   AssertionError: ...` followed by `path/file.py:184: AssertionError`
    const pytestErr = /^E\s{2,}(.+)$/u.exec(line)
    if (pytestErr?.[1]) {
      let matchedFrame = false
      for (let look = idx + 1; look <= Math.min(lines.length - 1, idx + 6); look += 1) {
        const pytestLoc = /^\s*([^\s:]+\.py):(\d+):/u.exec(lines[look] ?? '')
        if (pytestLoc?.[1] && pytestLoc[2]) {
          addCoordinate(pytestLoc[1], pytestLoc[2], pytestErr[1])
          matchedFrame = true
          break
        }
      }
      if (!matchedFrame) {
        for (let look = idx - 1; look >= Math.max(0, idx - 8); look -= 1) {
          const pytestLoc = /^\s*([^\s:]+\.py):(\d+):/u.exec(lines[look] ?? '')
          if (pytestLoc?.[1] && pytestLoc[2]) {
            addCoordinate(pytestLoc[1], pytestLoc[2], pytestErr[1])
            break
          }
        }
      }
      continue
    }

    // Standard Python traceback ending in `<Error>: <msg>` preceded by `File "...", line N`
    const pyFinalErr = /^\s*([A-Z][A-Za-z0-9_]*(?:Error|Exception):\s*.+)$/u.exec(line)
    if (pyFinalErr?.[1]) {
      for (let look = idx - 1; look >= Math.max(0, idx - 12); look -= 1) {
        const pyFrame = /^\s*File "([^"]+)", line (\d+)\b/u.exec(lines[look] ?? '')
        if (pyFrame?.[1] && pyFrame[2] && !isInternalRuntimePath(pyFrame[1])) {
          addCoordinate(pyFrame[1], pyFrame[2], pyFinalErr[1])
          break
        }
      }
    }
  }

  // 4. Generic `path/to/file.ext:line(:col): <error summary>` (Go tests, linters, C/C++, general diagnostics)
  for (let idx = 0; idx < lines.length && found.length < MAX_ACTIVE_FAILURE_COORDINATES; idx += 1) {
    const line = lines[idx] ?? ''
    const genericDiag =
      /^\s*([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,10}):(\d+)(?::\d+)?:\s*(?!in\s+\S+$)(.+)$/u.exec(line)
    if (genericDiag?.[1] && genericDiag[2] && genericDiag[3]) {
      const msg = genericDiag[3].trim()
      if (msg && !/^\d+$/.test(msg)) {
        addCoordinate(genericDiag[1], genericDiag[2], msg)
      }
    }
  }

  // 5. Fallback: standalone stack frame paired with nearby `FAIL` / `✕` / `FAILED` line if no error-headline match succeeded
  if (found.length === 0) {
    for (let idx = 0; idx < lines.length && found.length < MAX_ACTIVE_FAILURE_COORDINATES; idx += 1) {
      const frame = parseWorkspaceStackFrame(lines[idx] ?? '')
      if (frame) {
        const contextSummary = findNearbyFailureSummary(lines, idx) ?? 'test/diagnostic failure'
        addCoordinate(frame.path, frame.line, contextSummary)
      }
    }
  }

  // 6. Last-resort fallback: `FAILED path/to/test.py::test_name - Reason` or `FAIL path/to/test.ts`
  if (found.length === 0) {
    for (const line of lines) {
      if (found.length >= MAX_ACTIVE_FAILURE_COORDINATES) break
      const pytestShort = /^\s*FAILED\s+([^\s:]+)(?:::(\S+))?(?:\s+-\s+(.+))?$/u.exec(line)
      if (pytestShort?.[1]) {
        const testName = pytestShort[2] ? `${pytestShort[2]}: ` : ''
        const reason = pytestShort[3] ?? 'test failed'
        addCoordinate(pytestShort[1], 1, `${testName}${reason}`)
        continue
      }
      const jestFail = /^\s*FAIL\s+([^\s]+\.[A-Za-z0-9]{1,10})\b(.*)$/u.exec(line)
      if (jestFail?.[1]) {
        addCoordinate(jestFail[1], 1, jestFail[2]?.trim() || 'test suite failed')
      }
    }
  }

  return found
}

function findNearestWorkspaceFrameAfter(
  lines: readonly string[],
  startIdx: number,
  endExclusive: number,
): { readonly path: string; readonly line: number } | undefined {
  for (let idx = startIdx; idx < endExclusive; idx += 1) {
    const parsed = parseWorkspaceStackFrame(lines[idx] ?? '')
    if (parsed) return parsed
  }
  return undefined
}

function parseWorkspaceStackFrame(line: string): { readonly path: string; readonly line: number } | undefined {
  // Matches `at fn (path/to/file.ts:184:12)`, `at path/to/file.ts:184:12`, or bare `path/to/file.ts:184:12`
  const match = /(?:^|[\s(])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,10}):(\d+):\d+\)?/u.exec(line)
  if (!match?.[1] || !match[2]) return undefined
  const clean = normalizeDiagnosticPath(match[1])
  if (!clean || isInternalRuntimePath(clean)) return undefined
  const lineNum = Number(match[2])
  return Number.isSafeInteger(lineNum) && lineNum > 0 ? { path: clean, line: lineNum } : undefined
}

function findNearbyFailureSummary(lines: readonly string[], centerIdx: number): string | undefined {
  for (let offset = -3; offset <= 3; offset += 1) {
    if (offset === 0) continue
    const candidate = (lines[centerIdx + offset] ?? '').trim()
    if (!candidate) continue
    if (/(?:Error|FAIL|FAILED|✕|✗|panic:)/u.test(candidate)) {
      return cleanErrorSummary(candidate)
    }
  }
  return undefined
}

function parseFileBullet(body: string): SweTouchedFile | undefined {
  const match = /^(.+?)\s+\((created|edited|patched|moved|deleted),\s*(applied|failed)\)$/u.exec(body)
  if (!match?.[1] || !match[2] || !match[3]) return undefined
  const rawPathPart = match[1].trim()
  const operation = match[2] as SweFileOperation
  const status = match[3] as SweFileTouchStatus
  if (operation === 'moved' && rawPathPart.includes(' -> ')) {
    const [from, to] = rawPathPart.split(' -> ', 2)
    if (from && to) {
      return { path: to.trim(), movedFrom: from.trim(), operation, status }
    }
  }
  return { path: rawPathPart, operation, status }
}

function parseVerificationBullet(trimmedLine: string): SweVerificationEntry | undefined {
  const match = /^-\s+\[(PASS|FAIL)\]\s+(.+?)(?:\s+\(exit\s+(-?\d+)\))?$/u.exec(trimmedLine)
  if (!match?.[1] || !match[2]) return undefined
  const status: SweVerificationStatus = match[1] === 'PASS' ? 'passed' : 'failed'
  const exitCode = match[3] !== undefined ? Number(match[3]) : undefined
  return {
    command: match[2].trim(),
    status,
    ...(exitCode !== undefined && Number.isFinite(exitCode) ? { exitCode } : {}),
    failures: [],
  }
}

function parseFailureBullet(trimmedLine: string, command: string): SweFailureCoordinate | undefined {
  const match = /^-\s+failure:\s+([^\s—]+)\s+—\s+(.+)$/u.exec(trimmedLine)
  if (!match?.[1] || !match[2]) return undefined
  return {
    location: match[1].trim(),
    summary: match[2].trim(),
    command,
  }
}

function isFailedToolExecution(result: ParsedToolResult): boolean {
  if (result.failedFlag) return true
  const unwrapped = unwrapPersistedOutput(result.text).trimStart()
  if (
    unwrapped.startsWith('Tool execution failed:')
    || unwrapped.startsWith('Error:')
    || unwrapped.startsWith('ValidationError')
    || unwrapped.startsWith('Permission denied')
    || unwrapped.startsWith('[Tool result unavailable')
  ) {
    return true
  }
  const parsed = tryParseJsonRecord(unwrapped)
  if (parsed) {
    if (hasFailureFlag(parsed)) return true
    if (parsed.applied === false || parsed.success === false) return true
    if (typeof parsed.error === 'string' && parsed.error.trim().length > 0) return true
  }
  return false
}

function hasFailureFlag(record: Record<string, unknown>): boolean {
  return (
    record.is_error === true
    || record.isError === true
    || record.ok === false
    || record.status === 'error'
    || record.status === 'failed'
  )
}

function unwrapPersistedOutput(text: string): string {
  const trimmed = text.trim()
  if (!trimmed.startsWith('<persisted-output')) return text
  const firstNewline = trimmed.indexOf('\n')
  const closeIdx = trimmed.lastIndexOf('</persisted-output>')
  if (firstNewline < 0 || closeIdx <= firstNewline) return text
  const inner = trimmed.slice(firstNewline + 1, closeIdx).trimEnd()
  return inner
    .replace(/\nThe full output (?:is on disk at the path above[^\n]*|was not retained\.)\s*$/u, '')
    .trimEnd()
}

export function normalizeVerificationCommand(command: string): string {
  return command
    .trim()
    .replace(/\s+/g, ' ')
    .split(' ')
    .map((token, idx) => (idx > 0 && token.startsWith('./') && token.length > 2 ? token.slice(2) : token))
    .join(' ')
}

function normalizeWorkspacePath(rawPath: string): string {
  let clean = rawPath.trim().replace(/^['"]|['"]$/gu, '')
  if (!clean.startsWith('/')) {
    clean = clean.replace(/\\/gu, '/')
  }
  while (clean.startsWith('./')) {
    clean = clean.slice(2)
  }
  return clean
}

function normalizeDiagnosticPath(rawPath: string): string {
  let clean = rawPath.trim().replace(/^['"(`]+|['")`:]+$/gu, '')
  if (clean.startsWith('file://')) {
    clean = clean.slice('file://'.length)
  }
  while (clean.startsWith('./')) {
    clean = clean.slice(2)
  }
  return clean
}

function isInternalRuntimePath(path: string): boolean {
  return (
    path.startsWith('node:')
    || path.startsWith('bun:')
    || path.startsWith('internal/')
    || path.includes('/node_modules/')
    || path.startsWith('node_modules/')
    || path === '<anonymous>'
  )
}

function cleanErrorSummary(raw: string): string {
  const collapsed = singleLine(raw).replace(/^[-–—:\s]+/u, '').trim()
  return safeSlicePrefix(collapsed, 140)
}

function singleLine(text: string): string {
  return text.replace(/[\r\n\t]+/gu, ' ').replace(/\s+/gu, ' ').trim()
}

function dedupeFailureList(items: readonly SweFailureCoordinate[]): SweFailureCoordinate[] {
  const seen = new Set<string>()
  const out: SweFailureCoordinate[] = []
  for (const item of items) {
    const key = `${item.location}\u0000${item.summary}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(item)
  }
  return out
}

function messageContentAsText(content: unknown): string {
  if (typeof content === 'string') return content
  if (content === undefined || content === null) return ''
  if (Array.isArray(content)) return content.map(messageContentAsText).join('\n')
  if (isRecord(content)) {
    if (typeof content.text === 'string') return content.text
    return safeJsonString(content)
  }
  return String(content)
}

function tryParseJsonRecord(text: string): Record<string, unknown> | undefined {
  const trimmed = text.trim()
  if (!trimmed.startsWith('{')) return undefined
  try {
    const parsed: unknown = JSON.parse(trimmed)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function safeJsonString(value: unknown): string {
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return ''
  }
}

function firstNonEmptyString(...values: readonly unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim()
    }
  }
  return undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
