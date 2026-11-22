// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, lstatSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { ValidationError } from '../core/errors.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import type { FileStateTracker, FileToolContext } from './fileState.js'
import { assertTrackedFileFresh, atomicWriteSync, fileStateTracker } from './fileState.js'
import { optionalBoolean, requiredString } from './inputs.js'
import type { WorkspacePathResolver } from './pathSafety.js'
import { discoverRelatedTests } from './testDiscovery.js'

export interface PatchHunkLine {
  readonly kind: 'context' | 'del' | 'add'
  readonly text: string
}

export interface PatchHunk {
  readonly header: string
  readonly anchorHeader?: string
  readonly oldStartHint?: number
  readonly eofAnchored?: boolean
  readonly noNewlineAtEnd?: boolean
  readonly lines: readonly PatchHunkLine[]
}

export type PatchOperation =
  | {
      readonly kind: 'add'
      readonly filePath: string
      readonly content: string
    }
  | {
      readonly kind: 'delete'
      readonly filePath: string
      readonly hunks?: readonly PatchHunk[]
    }
  | {
      readonly kind: 'update'
      readonly filePath: string
      readonly moveTo?: string
      readonly hunks: readonly PatchHunk[]
    }

export interface ParsedPatch {
  readonly format: 'v4a' | 'unified'
  readonly operations: readonly PatchOperation[]
}

export interface PatchTargetPaths {
  readonly allPaths: readonly string[]
  readonly readPaths: readonly string[]
  readonly writePaths: readonly string[]
}

const PATCH_WRITE_CAPABILITIES = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: true,
  interruptBehavior: 'block',
  openWorld: false,
  readOnly: false,
} as const)

const PATCH_WRITE_GUIDANCE =
  'ApplyPatchTool executes multi-hunk and multi-file edits atomically: if any path or hunk fails, '
  + 'zero files are modified on disk and a pinpoint line diagnostic is returned. Tracked files that '
  + 'changed externally since your last read are refused rather than overwritten, and all committed '
  + 'paths refresh the session read state automatically.'

export const APPLY_PATCH_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ApplyPatchTool',
    description:
      'Apply an atomic multi-hunk or multi-file patch inside the workspace. Accepts V4A patches '
      + '(*** Begin Patch / *** Update File: <path> / *** Add File: <path> / *** Delete File: <path> / '
      + '*** Move to: <path> with @@ context-anchored space/-/+' + ' lines and *** End Patch) as well as '
      + 'standard unified diffs (--- a/path, +++ b/path, @@ -l,s +l,s @@). All target paths, freshness '
      + 'guards, and hunks are validated in memory first; if any hunk fails, zero files are modified on '
      + 'disk and the exact file, hunk header, whitespace hint, and closest line number are reported.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        patch: {
          type: 'string',
          minLength: 1,
          description: 'Complete V4A or unified diff patch text.',
        },
        dry_run: {
          type: 'boolean',
          default: false,
          description: 'Validate all paths and hunks in memory without writing to disk.',
        },
      },
      required: ['patch'],
    },
  },
}

export function registerApplyPatchTool(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(
    APPLY_PATCH_TOOL_DEFINITION,
    (inputs, context) => applyPatch(inputs, paths, context),
    'default',
    PATCH_WRITE_CAPABILITIES,
    PATCH_WRITE_GUIDANCE,
  )
}

/**
 * Extract all workspace paths referenced or mutated by a patch string.
 * Falls back to header scanning if the patch body is malformed so security
 * gates (`DataBoundary`) still catch private or read-only path references.
 */
export function inspectPatchTargets(patchText: string): PatchTargetPaths {
  const readSet = new Set<string>()
  const writeSet = new Set<string>()

  try {
    const parsed = parsePatch(patchText)
    for (const op of parsed.operations) {
      if (op.kind === 'add') {
        writeSet.add(op.filePath)
      } else if (op.kind === 'delete') {
        readSet.add(op.filePath)
        writeSet.add(op.filePath)
      } else {
        readSet.add(op.filePath)
        writeSet.add(op.filePath)
        if (op.moveTo !== undefined) {
          readSet.add(op.moveTo)
          writeSet.add(op.moveTo)
        }
      }
    }
  } catch {
    for (const rawLine of unwrapOptionalCodeFence(patchText.replaceAll('\r\n', '\n')).split('\n')) {
      const v4aUpdate = rawLine.match(/^\*\*\*\s+(?:Update|Delete)\s+File:\s*(.+)$/)
      if (v4aUpdate?.[1]) {
        const cleaned = cleanHeaderPath(v4aUpdate[1])
        if (cleaned) {
          readSet.add(cleaned)
          writeSet.add(cleaned)
        }
        continue
      }
      const v4aAddOrMove = rawLine.match(/^\*\*\*\s+(?:Add\s+File|Move\s+to):\s*(.+)$/)
      if (v4aAddOrMove?.[1]) {
        const cleaned = cleanHeaderPath(v4aAddOrMove[1])
        if (cleaned) writeSet.add(cleaned)
        continue
      }
      const renameMatch = rawLine.match(/^rename\s+(?:from|to)\s+(.+)$/)
      if (renameMatch?.[1]) {
        const cleaned = cleanHeaderPath(renameMatch[1])
        if (cleaned) {
          readSet.add(cleaned)
          writeSet.add(cleaned)
        }
        continue
      }
      const unifiedHeader = rawLine.match(/^(?:---|\+\+\+)\s+(.+)$/)
      if (unifiedHeader?.[1]) {
        const raw = stripUnifiedTimestamp(unifiedHeader[1].trim())
        if (raw && raw !== '/dev/null') {
          const stripped = rawLine.startsWith('--- ') && raw.startsWith('a/')
            ? raw.slice(2)
            : rawLine.startsWith('+++ ') && raw.startsWith('b/')
              ? raw.slice(2)
              : raw
          const cleaned = cleanHeaderPath(stripped)
          if (cleaned) {
            readSet.add(cleaned)
            writeSet.add(cleaned)
          }
        }
      }
    }
  }

  const readPaths = [...readSet]
  const writePaths = [...writeSet]
  const allPaths = [...new Set([...readPaths, ...writePaths])]
  return { allPaths, readPaths, writePaths }
}

/** Parse a V4A or unified diff patch string into normalized operations. */
export function parsePatch(rawPatch: string): ParsedPatch {
  const normalized = unwrapOptionalCodeFence(rawPatch.replaceAll('\r\n', '\n'))
  if (!normalized.trim()) {
    throw new ValidationError('patch', 'must not be empty', rawPatch)
  }

  const lines = normalized.split('\n')
  if (normalized.endsWith('\n') && lines.length > 0 && lines[lines.length - 1] === '') {
    lines.pop()
  }

  const firstNonEmpty = lines.find(line => line.trim().length > 0)?.trim() ?? ''
  if (
    firstNonEmpty === '*** Begin Patch'
    || firstNonEmpty.startsWith('*** Update File:')
    || firstNonEmpty.startsWith('*** Add File:')
    || firstNonEmpty.startsWith('*** Delete File:')
  ) {
    return { format: 'v4a', operations: parseV4aPatch(lines) }
  }
  return { format: 'unified', operations: parseUnifiedPatch(lines) }
}

function parseV4aPatch(lines: readonly string[]): readonly PatchOperation[] {
  const operations: PatchOperation[] = []
  let index = 0

  while (index < lines.length && lines[index]!.trim() === '') {
    index += 1
  }

  let sawBegin = false
  if (index < lines.length && lines[index]!.trim() === '*** Begin Patch') {
    sawBegin = true
    index += 1
  }

  let sawEnd = false
  while (index < lines.length) {
    const line = lines[index]!
    const trimmed = line.trim()
    if (trimmed === '') {
      index += 1
      continue
    }
    if (trimmed === '*** End Patch') {
      sawEnd = true
      index += 1
      break
    }

    if (line.startsWith('*** Add File:')) {
      const filePath = requirePatchPath(line.slice('*** Add File:'.length), 'Add File')
      index += 1
      const addedLines: string[] = []
      let noNewlineAtEnd = false
      while (index < lines.length) {
        const current = lines[index]!
        if (isV4aDirective(current)) break
        if (current.startsWith('\\ No newline at end of file')) {
          noNewlineAtEnd = true
          index += 1
          continue
        }
        if (current === '' && !hasMoreAddLinesBeforeDirective(lines, index + 1)) {
          index += 1
          continue
        }
        if (!current.startsWith('+')) {
          throw new ValidationError(
            'patch',
            `invalid line in "*** Add File: ${filePath}" at patch line ${index + 1}: expected '+' prefix, got ${JSON.stringify(current.slice(0, 60))}`,
            filePath,
          )
        }
        addedLines.push(current.slice(1))
        index += 1
      }
      const content = addedLines.length === 0
        ? ''
        : addedLines.join('\n') + (noNewlineAtEnd ? '' : '\n')
      operations.push({ kind: 'add', filePath, content })
      continue
    }

    if (line.startsWith('*** Delete File:')) {
      const filePath = requirePatchPath(line.slice('*** Delete File:'.length), 'Delete File')
      index += 1
      while (index < lines.length && lines[index]!.trim() === '' && !isV4aDirective(lines[index]!)) {
        index += 1
      }
      if (index < lines.length && !isV4aDirective(lines[index]!)) {
        throw new ValidationError(
          'patch',
          `unexpected content under "*** Delete File: ${filePath}" at patch line ${index + 1}`,
          filePath,
        )
      }
      operations.push({ kind: 'delete', filePath })
      continue
    }

    if (line.startsWith('*** Update File:')) {
      const filePath = requirePatchPath(line.slice('*** Update File:'.length), 'Update File')
      index += 1
      let moveTo: string | undefined
      if (index < lines.length && lines[index]!.startsWith('*** Move to:')) {
        moveTo = requirePatchPath(lines[index]!.slice('*** Move to:'.length), 'Move to')
        index += 1
      }

      const hunks: PatchHunk[] = []
      while (index < lines.length && !isV4aSectionBoundary(lines[index]!)) {
        const current = lines[index]!
        if (current.trim() === '' && !hasHunkLineBeforeBoundary(lines, index + 1)) {
          index += 1
          continue
        }
        let header = '@@'
        let anchorHeader: string | undefined
        let oldStartHint: number | undefined
        if (current === '@@' || current.startsWith('@@ ') || current.startsWith('@@\t') || /^@@\s*-\d+/.test(current)) {
          header = current
          const parsedHeader = parseHunkHeaderMetadata(current)
          anchorHeader = parsedHeader.anchorHeader
          oldStartHint = parsedHeader.oldStartHint
          index += 1
        } else if (current.startsWith(' ') || current.startsWith('-') || current.startsWith('+')) {
          header = '@@'
        } else {
          throw new ValidationError(
            'patch',
            `invalid hunk header in "*** Update File: ${filePath}" at patch line ${index + 1}: expected "@@", got ${JSON.stringify(current.slice(0, 60))}`,
            filePath,
          )
        }

        const hunkLines: PatchHunkLine[] = []
        let eofAnchored = false
        let noNewlineAtEnd = false

        while (index < lines.length && !isV4aSectionBoundary(lines[index]!)) {
          const item = lines[index]!
          if (item === '@@' || item.startsWith('@@ ') || item.startsWith('@@\t') || /^@@\s*-\d+/.test(item)) {
            break
          }
          if (item.trim() === '*** End of File') {
            eofAnchored = true
            index += 1
            break
          }
          if (item.startsWith('\\ No newline at end of file')) {
            noNewlineAtEnd = true
            index += 1
            continue
          }
          if (item === '') {
            if (hasContinuationInSameV4aHunk(lines, index + 1)) {
              hunkLines.push({ kind: 'context', text: '' })
              index += 1
              continue
            }
            index += 1
            break
          }
          const prefix = item[0]
          if (prefix === ' ') {
            hunkLines.push({ kind: 'context', text: item.slice(1) })
          } else if (prefix === '-') {
            hunkLines.push({ kind: 'del', text: item.slice(1) })
          } else if (prefix === '+') {
            hunkLines.push({ kind: 'add', text: item.slice(1) })
          } else {
            throw new ValidationError(
              'patch',
              `invalid hunk line in "${filePath}" (${header}) at patch line ${index + 1}: expected ' ', '-', or '+' prefix, got ${JSON.stringify(item.slice(0, 60))}`,
              filePath,
            )
          }
          index += 1
        }

        if (hunkLines.length === 0) {
          throw new ValidationError('patch', `empty hunk (${header}) in "${filePath}"`, filePath)
        }

        hunks.push({
          header,
          ...(anchorHeader === undefined ? {} : { anchorHeader }),
          ...(oldStartHint === undefined ? {} : { oldStartHint }),
          ...(eofAnchored ? { eofAnchored: true } : {}),
          ...(noNewlineAtEnd ? { noNewlineAtEnd: true } : {}),
          lines: hunkLines,
        })
      }

      if (hunks.length === 0 && moveTo === undefined) {
        throw new ValidationError('patch', `"*** Update File: ${filePath}" contains no hunks`, filePath)
      }

      operations.push({
        kind: 'update',
        filePath,
        ...(moveTo === undefined ? {} : { moveTo }),
        hunks,
      })
      continue
    }

    throw new ValidationError(
      'patch',
      `unrecognized V4A patch directive at line ${index + 1}: ${JSON.stringify(line.slice(0, 80))}`,
      line,
    )
  }

  if (sawBegin && !sawEnd) {
    // Require explicit *** End Patch when *** Begin Patch was opened so truncated patches fail closed.
    throw new ValidationError('patch', 'missing "*** End Patch" terminator (patch may be truncated)')
  }
  if (operations.length === 0) {
    throw new ValidationError('patch', 'patch contains no file operations')
  }
  return operations
}

function parseUnifiedPatch(lines: readonly string[]): readonly PatchOperation[] {
  const operations: PatchOperation[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]!
    if (line.trim() === '') {
      index += 1
      continue
    }

    let gitHeaderUsedAb = false
    let gitOldPath: string | undefined
    let gitNewPath: string | undefined
    let renameFrom: string | undefined
    let renameTo: string | undefined
    let newFileMode = false
    let deletedFileMode = false

    if (line.startsWith('diff --git ')) {
      const parsedGit = parseDiffGitHeader(line)
      gitHeaderUsedAb = parsedGit.usedAb
      gitOldPath = parsedGit.oldPath
      gitNewPath = parsedGit.newPath
      index += 1
      while (index < lines.length) {
        const meta = lines[index]!
        if (meta.startsWith('diff --git ') || meta.startsWith('--- ')) break
        if (meta.startsWith('rename from ')) {
          renameFrom = requirePatchPath(meta.slice('rename from '.length), 'rename from')
        } else if (meta.startsWith('rename to ')) {
          renameTo = requirePatchPath(meta.slice('rename to '.length), 'rename to')
        } else if (meta.startsWith('new file mode ')) {
          newFileMode = true
        } else if (meta.startsWith('deleted file mode ')) {
          deletedFileMode = true
        }
        index += 1
      }

      // Handle metadata-only git patches (pure rename, empty new file, or mode-only delete without ---/+++).
      if (index >= lines.length || !lines[index]!.startsWith('--- ')) {
        if (renameFrom !== undefined && renameTo !== undefined) {
          operations.push({ kind: 'update', filePath: renameFrom, moveTo: renameTo, hunks: [] })
          continue
        }
        if (newFileMode && gitNewPath !== undefined) {
          operations.push({ kind: 'add', filePath: gitNewPath, content: '' })
          continue
        }
        if (deletedFileMode && gitOldPath !== undefined) {
          operations.push({ kind: 'delete', filePath: gitOldPath })
          continue
        }
        throw new ValidationError('patch', `incomplete diff --git section starting with ${JSON.stringify(line)}`)
      }
    } else if (
      line.startsWith('Index: ')
      || line.startsWith('===')
      || line.startsWith('index ')
    ) {
      index += 1
      continue
    } else if (line.startsWith('rename from ')) {
      renameFrom = requirePatchPath(line.slice('rename from '.length), 'rename from')
      index += 1
      if (index < lines.length && lines[index]!.startsWith('rename to ')) {
        renameTo = requirePatchPath(lines[index]!.slice('rename to '.length), 'rename to')
        index += 1
      }
      if (index >= lines.length || !lines[index]!.startsWith('--- ')) {
        if (renameFrom !== undefined && renameTo !== undefined) {
          operations.push({ kind: 'update', filePath: renameFrom, moveTo: renameTo, hunks: [] })
          continue
        }
      }
    }

    const oldHeaderLine = lines[index]
    if (oldHeaderLine === undefined || !oldHeaderLine.startsWith('--- ')) {
      throw new ValidationError(
        'patch',
        `expected unified diff file header ("--- <path>") at patch line ${index + 1}, got ${JSON.stringify((oldHeaderLine ?? '').slice(0, 80))}`,
      )
    }
    const newHeaderLine = lines[index + 1]
    if (newHeaderLine === undefined || !newHeaderLine.startsWith('+++ ')) {
      throw new ValidationError(
        'patch',
        `expected "+++ <path>" immediately after "${oldHeaderLine}" at patch line ${index + 2}`,
      )
    }
    index += 2

    const rawOld = stripUnifiedTimestamp(oldHeaderLine.slice(4).trim())
    const rawNew = stripUnifiedTimestamp(newHeaderLine.slice(4).trim())
    const stripAb = gitHeaderUsedAb
      || (rawOld === '/dev/null' && rawNew.startsWith('b/'))
      || (rawNew === '/dev/null' && rawOld.startsWith('a/'))
      || (rawOld.startsWith('a/') && rawNew.startsWith('b/'))

    const oldPath = rawOld === '/dev/null'
      ? undefined
      : requirePatchPath(stripAb && rawOld.startsWith('a/') ? rawOld.slice(2) : rawOld, '---')
    const newPath = rawNew === '/dev/null'
      ? undefined
      : requirePatchPath(stripAb && rawNew.startsWith('b/') ? rawNew.slice(2) : rawNew, '+++')

    const hunks: PatchHunk[] = []
    while (index < lines.length) {
      const current = lines[index]!
      if (
        current.startsWith('diff --git ')
        || current.startsWith('--- ')
        || current.startsWith('Index: ')
        || current.startsWith('rename from ')
      ) {
        break
      }
      if (current.trim() === '') {
        index += 1
        continue
      }
      const headerMatch = current.match(/^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@(.*)$/)
      if (!headerMatch) {
        throw new ValidationError(
          'patch',
          `invalid unified diff hunk header at patch line ${index + 1}: ${JSON.stringify(current.slice(0, 80))}`,
        )
      }
      const oldStartHint = Number.parseInt(headerMatch[1]!, 10)
      const declaredOldCount = headerMatch[2] === undefined ? 1 : Number.parseInt(headerMatch[2], 10)
      const declaredNewCount = headerMatch[4] === undefined ? 1 : Number.parseInt(headerMatch[4], 10)
      const rawAnchor = headerMatch[5]?.trim() ?? ''
      const anchorHeader = rawAnchor || undefined
      const header = current
      index += 1

      const hunkLines: PatchHunkLine[] = []
      let seenOld = 0
      let seenNew = 0
      let noNewlineAtEnd = false

      while (index < lines.length) {
        const item = lines[index]!
        if (item.startsWith('\\ No newline at end of file')) {
          noNewlineAtEnd = true
          index += 1
          continue
        }
        const countsSatisfied = seenOld >= declaredOldCount && seenNew >= declaredNewCount
        if (
          item.startsWith('diff --git ')
          || item.startsWith('@@ ')
          || item === '@@'
          || (countsSatisfied && (item.startsWith('--- ') || item.startsWith('Index: ') || item.startsWith('rename from ')))
        ) {
          break
        }
        if (item === '') {
          if (!countsSatisfied && hasMoreUnifiedHunkLines(lines, index + 1)) {
            hunkLines.push({ kind: 'context', text: '' })
            seenOld += 1
            seenNew += 1
            index += 1
            continue
          }
          if (countsSatisfied) {
            index += 1
            break
          }
          hunkLines.push({ kind: 'context', text: '' })
          seenOld += 1
          seenNew += 1
          index += 1
          continue
        }
        const prefix = item[0]
        if (prefix === ' ') {
          hunkLines.push({ kind: 'context', text: item.slice(1) })
          seenOld += 1
          seenNew += 1
        } else if (prefix === '-') {
          hunkLines.push({ kind: 'del', text: item.slice(1) })
          seenOld += 1
        } else if (prefix === '+') {
          hunkLines.push({ kind: 'add', text: item.slice(1) })
          seenNew += 1
        } else {
          if (countsSatisfied) break
          throw new ValidationError(
            'patch',
            `invalid unified diff line in "${newPath ?? oldPath ?? 'unknown'}" (${header}) at patch line ${index + 1}: ${JSON.stringify(item.slice(0, 60))}`,
          )
        }
        index += 1
      }

      hunks.push({
        header,
        ...(anchorHeader === undefined ? {} : { anchorHeader }),
        oldStartHint,
        ...(noNewlineAtEnd ? { noNewlineAtEnd: true } : {}),
        lines: hunkLines,
      })
    }

    if (oldPath === undefined && newPath !== undefined) {
      // Unified diff file creation
      const addedLines: string[] = []
      let noNewlineAtEnd = false
      for (const hunk of hunks) {
        for (const lineItem of hunk.lines) {
          if (lineItem.kind !== 'add') {
            throw new ValidationError(
              'patch',
              `new file diff for "${newPath}" may only contain '+' lines (${hunk.header})`,
              newPath,
            )
          }
          addedLines.push(lineItem.text)
        }
        if (hunk.noNewlineAtEnd) noNewlineAtEnd = true
      }
      const content = addedLines.length === 0
        ? ''
        : addedLines.join('\n') + (noNewlineAtEnd ? '' : '\n')
      operations.push({ kind: 'add', filePath: newPath, content })
    } else if (oldPath !== undefined && newPath === undefined) {
      operations.push({
        kind: 'delete',
        filePath: oldPath,
        ...(hunks.length > 0 ? { hunks } : {}),
      })
    } else if (oldPath !== undefined && newPath !== undefined) {
      const effectiveSource = renameFrom ?? oldPath
      const effectiveDest = renameTo ?? newPath
      const moveTo = effectiveDest !== effectiveSource ? effectiveDest : undefined
      if (hunks.length === 0 && moveTo === undefined) {
        throw new ValidationError('patch', `unified diff for "${effectiveSource}" contains no hunks`, effectiveSource)
      }
      operations.push({
        kind: 'update',
        filePath: effectiveSource,
        ...(moveTo === undefined ? {} : { moveTo }),
        hunks,
      })
    } else {
      throw new ValidationError('patch', 'unified diff cannot have both --- /dev/null and +++ /dev/null')
    }
  }

  if (operations.length === 0) {
    throw new ValidationError('patch', 'patch contains no recognizable V4A or unified diff operations')
  }
  return operations
}

interface VirtualFile {
  readonly absolutePath: string
  readonly displayPath: string
  readonly exists: boolean
  readonly content: string
  readonly mode: number
  readonly originalExists: boolean
  readonly originalContent: string | undefined
  readonly originalMode: number | undefined
}

interface ResolvedOpPaths {
  readonly sourceAbs: string
  readonly sourceRel: string
  readonly destAbs?: string
  readonly destRel?: string
}

/**
 * Validate and apply a multi-file patch transactionally.
 * Zero disk mutations happen until every path, freshness check, and hunk succeeds in memory.
 */
export async function applyPatch(
  inputs: JsonObject,
  paths: WorkspacePathResolver,
  context?: FileToolContext,
  tracker: FileStateTracker = fileStateTracker,
): Promise<string> {
  const patchText = requiredString(inputs, 'patch')
  const dryRun = optionalBoolean(inputs, 'dry_run', false)
  const parsed = parsePatch(patchText)

  // Step 1: Resolve all paths through WorkspacePathResolver first (rejects lexical & symlink escapes).
  const resolvedOps: ResolvedOpPaths[] = []
  const allTargets = new Set<string>()
  for (const op of parsed.operations) {
    const sourceAbs = await paths.resolve(op.filePath)
    const sourceRel = (await paths.relative(sourceAbs)).replaceAll('\\', '/')
    allTargets.add(sourceAbs)
    if (op.kind === 'update' && op.moveTo !== undefined) {
      const destAbs = await paths.resolve(op.moveTo)
      const destRel = (await paths.relative(destAbs)).replaceAll('\\', '/')
      allTargets.add(destAbs)
      resolvedOps.push({ sourceAbs, sourceRel, destAbs, destRel })
    } else {
      resolvedOps.push({ sourceAbs, sourceRel })
    }
  }

  // Step 2: Recheck every resolved target immediately before entering the synchronous validation+commit region.
  const recheckedMap = new Map<string, string>()
  for (const target of allTargets) {
    recheckedMap.set(target, await paths.recheck(target))
  }

  // Step 3: Synchronous guarded region — inspect disk, enforce freshness when tracked, and compute ALL hunks in memory.
  const virtualFs = new Map<string, VirtualFile>()
  const sessionId = context?.sessionId

  const loadVirtualFile = (rawAbs: string, displayPath: string, verifyFreshness: boolean): VirtualFile => {
    const abs = recheckedMap.get(rawAbs) ?? rawAbs
    const existing = virtualFs.get(abs)
    if (existing !== undefined) {
      return existing
    }
    if (!existsSync(abs)) {
      const missing: VirtualFile = {
        absolutePath: abs,
        displayPath,
        exists: false,
        content: '',
        mode: 0o644,
        originalExists: false,
        originalContent: undefined,
        originalMode: undefined,
      }
      virtualFs.set(abs, missing)
      return missing
    }
    const linkStats = lstatSync(abs)
    if (linkStats.isDirectory()) {
      throw new ValidationError('patch', `"${displayPath}" is a directory, not a regular file`, displayPath)
    }
    const fileStats = statSync(abs)
    if (!fileStats.isFile()) {
      throw new ValidationError('patch', `"${displayPath}" must refer to a regular file`, displayPath)
    }
    const currentContent = readFileSync(abs, 'utf8')
    if (currentContent.includes('\0')) {
      throw new ValidationError('patch', `"${displayPath}" appears to be a binary file and cannot be patched`, displayPath)
    }
    if (verifyFreshness) {
      assertTrackedFileFresh(
        {
          absolutePath: abs,
          currentContent,
          displayPath,
          mtimeMs: fileStats.mtimeMs,
          sessionId,
          size: fileStats.size,
          toolName: 'ApplyPatchTool',
        },
        tracker,
      )
    }
    const loaded: VirtualFile = {
      absolutePath: abs,
      displayPath,
      exists: true,
      content: currentContent,
      mode: fileStats.mode,
      originalExists: true,
      originalContent: currentContent,
      originalMode: fileStats.mode,
    }
    virtualFs.set(abs, loaded)
    return loaded
  }

  const summaryActions: string[] = []
  const editedSourceRelPaths: string[] = []
  let totalHunksApplied = 0
  let indentFallbackCount = 0

  for (let opIndex = 0; opIndex < parsed.operations.length; opIndex += 1) {
    const op = parsed.operations[opIndex]!
    const resolved = resolvedOps[opIndex]!

    if (op.kind === 'add') {
      const targetState = loadVirtualFile(resolved.sourceAbs, op.filePath, false)
      if (targetState.exists) {
        throw new ValidationError(
          'patch',
          `cannot add "${op.filePath}": file already exists (use "*** Update File: ${op.filePath}" to modify an existing file)`,
          op.filePath,
        )
      }
      ensureNoFileBlockingParent(targetState.absolutePath, op.filePath, virtualFs)
      virtualFs.set(targetState.absolutePath, {
        ...targetState,
        displayPath: resolved.sourceRel,
        exists: true,
        content: op.content,
        mode: 0o644,
      })
      summaryActions.push(`A ${resolved.sourceRel}`)
      editedSourceRelPaths.push(resolved.sourceRel)
      continue
    }

    if (op.kind === 'delete') {
      const targetState = loadVirtualFile(resolved.sourceAbs, op.filePath, true)
      if (!targetState.exists) {
        throw new ValidationError('patch', `cannot delete "${op.filePath}": file does not exist`, op.filePath)
      }
      if (op.hunks && op.hunks.length > 0) {
        const afterDeleteHunks = applyHunksToContent(targetState.content, op.hunks, op.filePath)
        if (afterDeleteHunks.content.trim() !== '') {
          throw new ValidationError(
            'patch',
            `deletion diff for "${op.filePath}" did not match the complete file contents on disk`,
            op.filePath,
          )
        }
      }
      virtualFs.set(targetState.absolutePath, {
        ...targetState,
        displayPath: resolved.sourceRel,
        exists: false,
        content: '',
      })
      summaryActions.push(`D ${resolved.sourceRel}`)
      continue
    }

    // op.kind === 'update'
    const sourceState = loadVirtualFile(resolved.sourceAbs, op.filePath, true)
    if (!sourceState.exists) {
      throw new ValidationError('patch', `cannot update "${op.filePath}": file does not exist`, op.filePath)
    }

    const applied = op.hunks.length > 0
      ? applyHunksToContent(sourceState.content, op.hunks, op.filePath)
      : { content: sourceState.content, indentMatches: 0 }

    totalHunksApplied += op.hunks.length
    indentFallbackCount += applied.indentMatches

    if (resolved.destAbs !== undefined && resolved.destRel !== undefined) {
      const destChecked = recheckedMap.get(resolved.destAbs) ?? resolved.destAbs
      if (destChecked !== sourceState.absolutePath) {
        const destState = loadVirtualFile(resolved.destAbs, op.moveTo ?? resolved.destRel, false)
        if (destState.exists) {
          throw new ValidationError(
            'patch',
            `cannot move "${op.filePath}" to "${op.moveTo ?? resolved.destRel}": destination already exists`,
            op.moveTo ?? resolved.destRel,
          )
        }
        ensureNoFileBlockingParent(destState.absolutePath, op.moveTo ?? resolved.destRel, virtualFs)
        virtualFs.set(sourceState.absolutePath, {
          ...sourceState,
          displayPath: resolved.sourceRel,
          exists: false,
          content: '',
        })
        virtualFs.set(destState.absolutePath, {
          ...destState,
          displayPath: resolved.destRel,
          exists: true,
          content: applied.content,
          mode: sourceState.mode,
        })
        summaryActions.push(`R ${resolved.sourceRel} -> ${resolved.destRel}`)
        editedSourceRelPaths.push(resolved.destRel)
        continue
      }
    }

    virtualFs.set(sourceState.absolutePath, {
      ...sourceState,
      displayPath: resolved.sourceRel,
      exists: true,
      content: applied.content,
    })
    summaryActions.push(`M ${resolved.sourceRel}`)
    editedSourceRelPaths.push(resolved.sourceRel)
  }

  if (dryRun) {
    return `Dry run succeeded: ${summaryActions.length} operation(s), ${totalHunksApplied} hunk(s) verified (${summaryActions.join(', ')}).`
  }

  // Step 4: Commit all mutations to disk with rollback protection if an OS write error occurs midway.
  const committedPaths: VirtualFile[] = []
  try {
    // Delete / move-away sources first so case-rename or path-reuse within one patch cannot collide.
    for (const entry of virtualFs.values()) {
      if (entry.originalExists && !entry.exists) {
        rmSync(entry.absolutePath)
        committedPaths.push(entry)
      }
    }
    for (const entry of virtualFs.values()) {
      if (!entry.exists) continue
      if (entry.originalExists && entry.content === entry.originalContent) {
        continue
      }
      mkdirSync(dirname(entry.absolutePath), { recursive: true })
      if (entry.originalExists) {
        atomicWriteSync(entry.absolutePath, entry.content, entry.mode)
      } else {
        writeFileSync(entry.absolutePath, entry.content, { flag: 'wx', mode: entry.mode })
      }
      committedPaths.push(entry)
    }
  } catch (commitError) {
    // Best-effort rollback of already-committed entries so disk stays consistent on unexpected I/O failure.
    for (const entry of committedPaths.reverse()) {
      try {
        if (entry.originalExists && entry.originalContent !== undefined) {
          mkdirSync(dirname(entry.absolutePath), { recursive: true })
          atomicWriteSync(entry.absolutePath, entry.originalContent, entry.originalMode ?? 0o644)
        } else if (!entry.originalExists) {
          rmSync(entry.absolutePath, { force: true })
        }
      } catch {
        // Preserve original commit error.
      }
    }
    throw commitError
  }

  // Step 5: Refresh / invalidate fileStateTracker for every touched path.
  if (sessionId !== undefined && sessionId !== '') {
    for (const entry of virtualFs.values()) {
      if (!entry.exists) {
        tracker.forget(sessionId, entry.absolutePath)
      } else {
        const stats = statSync(entry.absolutePath)
        tracker.record(sessionId, entry.absolutePath, entry.content, {
          mtimeMs: stats.mtimeMs,
          partialView: false,
          size: stats.size,
        })
      }
    }
  }

  const indentNote = indentFallbackCount > 0
    ? ` (${indentFallbackCount} hunk${indentFallbackCount === 1 ? '' : 's'} matched via uniform indentation normalization)`
    : ''
  const related = editedSourceRelPaths.length > 0
    ? discoverRelatedTests(paths.root, editedSourceRelPaths)
    : undefined
  const relatedSuffix = related?.hint ? `\n${related.hint}` : ''

  return `Applied patch across ${summaryActions.length} operation(s), ${totalHunksApplied} hunk(s)${indentNote}: ${summaryActions.join(', ')}.${relatedSuffix}`
}

interface SplitTextState {
  readonly lines: string[]
  readonly hasTrailingNewline: boolean
  readonly lineEnding: '\n' | '\r\n'
}

function splitFileLines(content: string): SplitTextState {
  const lineEnding: '\n' | '\r\n' = content.includes('\r\n') && !/(^|[^\r])\n/.test(content) ? '\r\n' : '\n'
  const normalized = lineEnding === '\r\n' ? content.replaceAll('\r\n', '\n') : content
  if (normalized === '') {
    return { lines: [], hasTrailingNewline: false, lineEnding }
  }
  const hasTrailingNewline = normalized.endsWith('\n')
  const body = hasTrailingNewline ? normalized.slice(0, -1) : normalized
  return {
    lines: body === '' ? [''] : body.split('\n'),
    hasTrailingNewline,
    lineEnding,
  }
}

function joinFileLines(lines: readonly string[], hasTrailingNewline: boolean, lineEnding: '\n' | '\r\n'): string {
  if (lines.length === 0) return ''
  const joined = lines.join(lineEnding)
  return hasTrailingNewline ? joined + lineEnding : joined
}

function applyHunksToContent(
  originalContent: string,
  hunks: readonly PatchHunk[],
  displayPath: string,
): { readonly content: string; readonly indentMatches: number } {
  const state = splitFileLines(originalContent)
  const workingLines = [...state.lines]
  let hasTrailingNewline = state.hasTrailingNewline
  let cursor = 0
  let lineDelta = 0
  let indentMatches = 0

  for (let hunkIndex = 0; hunkIndex < hunks.length; hunkIndex += 1) {
    const hunk = hunks[hunkIndex]!
    const oldLines = hunk.lines.filter(line => line.kind === 'context' || line.kind === 'del').map(line => line.text)
    const rawNewLines = hunk.lines.filter(line => line.kind === 'context' || line.kind === 'add').map(line => line.text)

    if (oldLines.length === 0) {
      // Pure insertion hunk (only valid on an empty file or when unified diff supplies an explicit line hint).
      let insertAt: number
      if (workingLines.length === 0) {
        insertAt = 0
        hasTrailingNewline = !hunk.noNewlineAtEnd
      } else if (hunk.oldStartHint !== undefined) {
        insertAt = Math.max(cursor, Math.min(workingLines.length, hunk.oldStartHint === 0 ? 0 : hunk.oldStartHint - 1 + lineDelta))
      } else if (hunk.eofAnchored) {
        insertAt = workingLines.length
      } else {
        throw new ValidationError(
          'patch',
          `failed to apply ${displayPath} hunk #${hunkIndex + 1} (${hunk.header}): pure insertion hunk has no surrounding context lines on a non-empty file`,
          displayPath,
        )
      }
      workingLines.splice(insertAt, 0, ...rawNewLines)
      cursor = insertAt + rawNewLines.length
      lineDelta += rawNewLines.length
      if (insertAt + rawNewLines.length === workingLines.length && hunk.noNewlineAtEnd) {
        hasTrailingNewline = false
      }
      continue
    }

    const match = locateHunkMatch(workingLines, oldLines, rawNewLines, hunk, cursor, lineDelta)
    if (!match.ok) {
      throw new ValidationError(
        'patch',
        formatHunkFailureDiagnostic(displayPath, hunkIndex + 1, hunk, workingLines, oldLines, match.reason),
        displayPath,
      )
    }

    if (match.viaUniformIndent) {
      indentMatches += 1
    }

    const touchesEof = match.startIndex + oldLines.length === workingLines.length
    workingLines.splice(match.startIndex, oldLines.length, ...match.adaptedNewLines)
    cursor = match.startIndex + match.adaptedNewLines.length
    lineDelta += match.adaptedNewLines.length - oldLines.length

    if (touchesEof) {
      if (hunk.noNewlineAtEnd) {
        hasTrailingNewline = false
      } else if (workingLines.length > 0 && originalContent === '') {
        hasTrailingNewline = true
      }
    }
  }

  return {
    content: joinFileLines(workingLines, workingLines.length > 0 ? hasTrailingNewline : false, state.lineEnding),
    indentMatches,
  }
}

type HunkMatchResult =
  | {
      readonly ok: true
      readonly startIndex: number
      readonly adaptedNewLines: readonly string[]
      readonly viaUniformIndent: boolean
    }
  | {
      readonly ok: false
      readonly reason:
        | { readonly kind: 'ambiguous_exact'; readonly candidateLines: readonly number[] }
        | { readonly kind: 'ambiguous_indent'; readonly candidateLines: readonly number[] }
        | { readonly kind: 'overlap_prior_hunk'; readonly matchedLine: number; readonly cursorLine: number }
        | { readonly kind: 'not_found' }
    }

function locateHunkMatch(
  fileLines: readonly string[],
  oldLines: readonly string[],
  newLines: readonly string[],
  hunk: PatchHunk,
  cursor: number,
  lineDelta: number,
): HunkMatchResult {
  const windowLen = oldLines.length
  if (windowLen > fileLines.length) {
    return { ok: false, reason: { kind: 'not_found' } }
  }

  // Pass 1: Exact match at or after cursor.
  const exactFromCursor: number[] = []
  for (let start = cursor; start <= fileLines.length - windowLen; start += 1) {
    if (hunk.eofAnchored && start + windowLen !== fileLines.length) continue
    if (windowMatchesExact(fileLines, start, oldLines)) {
      exactFromCursor.push(start)
    }
  }

  if (exactFromCursor.length === 1) {
    return { ok: true, startIndex: exactFromCursor[0]!, adaptedNewLines: newLines, viaUniformIndent: false }
  }
  if (exactFromCursor.length > 1) {
    const disambiguated = disambiguateCandidates(fileLines, exactFromCursor, hunk, lineDelta)
    if (disambiguated !== undefined) {
      return { ok: true, startIndex: disambiguated, adaptedNewLines: newLines, viaUniformIndent: false }
    }
    return {
      ok: false,
      reason: { kind: 'ambiguous_exact', candidateLines: exactFromCursor.map(idx => idx + 1) },
    }
  }

  // Check if an exact match existed before cursor (out-of-order or overlapping hunk).
  for (let start = 0; start < Math.min(cursor, fileLines.length - windowLen + 1); start += 1) {
    if (windowMatchesExact(fileLines, start, oldLines)) {
      return {
        ok: false,
        reason: { kind: 'overlap_prior_hunk', matchedLine: start + 1, cursorLine: cursor + 1 },
      }
    }
  }

  // Pass 2: Safe uniform-leading-indentation fallback — ONLY when unambiguously 1 unique match across the file!
  const indentMatches: Array<{ readonly startIndex: number; readonly adaptedNewLines: readonly string[] }> = []
  const allTrimStartMatches: number[] = []
  for (let start = 0; start <= fileLines.length - windowLen; start += 1) {
    if (hunk.eofAnchored && start + windowLen !== fileLines.length) continue
    const adapted = tryUniformLeadingIndentationMatch(fileLines, start, oldLines, newLines)
    if (adapted !== undefined) {
      allTrimStartMatches.push(start)
      if (start >= cursor) {
        indentMatches.push({ startIndex: start, adaptedNewLines: adapted })
      }
    }
  }

  // Require unambiguous uniqueness across the entire file so we never silently guess between duplicate blocks.
  if (allTrimStartMatches.length === 1 && indentMatches.length === 1) {
    const single = indentMatches[0]!
    return {
      ok: true,
      startIndex: single.startIndex,
      adaptedNewLines: single.adaptedNewLines,
      viaUniformIndent: true,
    }
  }
  if (allTrimStartMatches.length > 1) {
    return {
      ok: false,
      reason: { kind: 'ambiguous_indent', candidateLines: allTrimStartMatches.map(idx => idx + 1) },
    }
  }

  return { ok: false, reason: { kind: 'not_found' } }
}

function windowMatchesExact(fileLines: readonly string[], start: number, oldLines: readonly string[]): boolean {
  for (let offset = 0; offset < oldLines.length; offset += 1) {
    if (fileLines[start + offset] !== oldLines[offset]) return false
  }
  return true
}

function disambiguateCandidates(
  fileLines: readonly string[],
  candidates: readonly number[],
  hunk: PatchHunk,
  lineDelta: number,
): number | undefined {
  if (hunk.anchorHeader) {
    const anchorNeedle = hunk.anchorHeader.trim()
    if (anchorNeedle) {
      const anchorMatches: number[] = []
      for (let lineIdx = 0; lineIdx < fileLines.length; lineIdx += 1) {
        if (fileLines[lineIdx]!.includes(anchorNeedle)) {
          anchorMatches.push(lineIdx)
        }
      }
      if (anchorMatches.length === 1) {
        const anchorLine = anchorMatches[0]!
        const afterAnchor = candidates.filter(candidate => candidate >= anchorLine)
        if (afterAnchor.length > 0) {
          return afterAnchor[0]
        }
      }
    }
  }

  if (hunk.oldStartHint !== undefined && hunk.oldStartHint > 0) {
    const expected = hunk.oldStartHint - 1 + lineDelta
    const exactHint = candidates.filter(candidate => candidate === expected)
    if (exactHint.length === 1) return exactHint[0]
    const nearby = candidates.filter(candidate => Math.abs(candidate - expected) <= 3)
    if (nearby.length === 1) return nearby[0]
  }

  return undefined
}

/**
 * Check whether `fileLines[start .. start + oldLines.length)` differs from `oldLines`
 * strictly by a uniform leading whitespace prefix across all non-blank lines.
 * When matched, adapts `newLines` by applying the exact same base-indentation shift.
 */
function tryUniformLeadingIndentationMatch(
  fileLines: readonly string[],
  start: number,
  oldLines: readonly string[],
  newLines: readonly string[],
): readonly string[] | undefined {
  const nonBlankIndices: number[] = []

  for (let offset = 0; offset < oldLines.length; offset += 1) {
    const diskLine = fileLines[start + offset]!
    const patchLine = oldLines[offset]!
    const diskTrimmedStart = diskLine.trimStart()
    const patchTrimmedStart = patchLine.trimStart()

    // Non-leading content (including trailing whitespace) must match character-for-character.
    if (diskTrimmedStart !== patchTrimmedStart) {
      return undefined
    }
    if (patchTrimmedStart.length > 0) {
      nonBlankIndices.push(offset)
    }
  }

  if (nonBlankIndices.length === 0) {
    return undefined
  }

  const diskIndents = nonBlankIndices.map(offset => leadingWhitespace(fileLines[start + offset]!))
  const patchIndents = nonBlankIndices.map(offset => leadingWhitespace(oldLines[offset]!))
  const minDiskIndent = commonStringPrefix(diskIndents)
  const minPatchIndent = commonStringPrefix(patchIndents)

  // Every non-blank line must preserve identical relative indentation past the common base prefix.
  for (let i = 0; i < nonBlankIndices.length; i += 1) {
    const diskRelative = diskIndents[i]!.slice(minDiskIndent.length)
    const patchRelative = patchIndents[i]!.slice(minPatchIndent.length)
    if (diskRelative !== patchRelative) {
      return undefined
    }
  }

  // Ensure every non-blank replacement line also carries `minPatchIndent` so stripping never eats code or intentional outdents.
  for (const newLine of newLines) {
    if (newLine.trimStart().length === 0) continue
    if (!newLine.startsWith(minPatchIndent)) {
      return undefined
    }
  }

  return newLines.map(newLine => {
    if (newLine.trimStart().length === 0) {
      return ''
    }
    return minDiskIndent + newLine.slice(minPatchIndent.length)
  })
}

function formatHunkFailureDiagnostic(
  displayPath: string,
  hunkNumber: number,
  hunk: PatchHunk,
  fileLines: readonly string[],
  oldLines: readonly string[],
  reason: Extract<HunkMatchResult, { readonly ok: false }>['reason'],
): string {
  const basePrefix = `ApplyPatchTool failed on "${displayPath}" at hunk #${hunkNumber} (${hunk.header}) — zero files were modified.`

  if (fileLines.length === 0) {
    return `${basePrefix} File "${displayPath}" is currently empty (0 lines).`
  }

  if (reason.kind === 'ambiguous_exact') {
    return `${basePrefix} Hunk context is ambiguous and matches ${reason.candidateLines.length} locations (lines ${reason.candidateLines.join(', ')}); include more surrounding context lines or a specific @@ header.`
  }
  if (reason.kind === 'ambiguous_indent') {
    return `${basePrefix} Whitespace/indentation mismatch: matches ${reason.candidateLines.length} locations when leading indentation is normalized (lines ${reason.candidateLines.join(', ')}); match exact on-disk indentation and add context.`
  }
  if (reason.kind === 'overlap_prior_hunk') {
    return `${basePrefix} Hunk matched at line ${reason.matchedLine}, which overlaps or precedes an earlier hunk ending at line ${reason.cursorLine}; order hunks top-to-bottom without overlapping context.`
  }

  // Check for whitespace / non-uniform indentation mismatch (trimmed lines match cleanly).
  const trimmedMatches = findTrimmedWindowMatches(fileLines, oldLines)
  if (trimmedMatches.length > 0) {
    const candidateLine = trimmedMatches[0]! + 1
    const preview = renderNumberedPreview(fileLines, trimmedMatches[0]!, Math.min(oldLines.length, 6))
    return `${basePrefix} Indentation/whitespace mismatch near line ${candidateLine} (non-uniform indentation or trailing whitespace differs from disk). On-disk lines:\n${preview}`
  }

  // Locate closest candidate line window in the file by line similarity.
  const closest = findClosestCandidateWindow(fileLines, oldLines, hunk.oldStartHint)
  const preview = renderNumberedPreview(fileLines, closest.startIndex, Math.min(Math.max(oldLines.length, 3), 6))
  return `${basePrefix} Context/deletion lines did not match disk; closest candidate near line ${closest.startIndex + 1}:\n${preview}`
}

function findTrimmedWindowMatches(fileLines: readonly string[], oldLines: readonly string[]): number[] {
  if (oldLines.length === 0 || oldLines.length > fileLines.length) return []
  const trimmedNeedle = oldLines.map(line => line.trim())
  if (!trimmedNeedle.some(Boolean)) return []
  const matches: number[] = []
  for (let start = 0; start <= fileLines.length - oldLines.length; start += 1) {
    let allEqual = true
    for (let offset = 0; offset < oldLines.length; offset += 1) {
      if (fileLines[start + offset]!.trim() !== trimmedNeedle[offset]) {
        allEqual = false
        break
      }
    }
    if (allEqual) matches.push(start)
  }
  return matches
}

function findClosestCandidateWindow(
  fileLines: readonly string[],
  oldLines: readonly string[],
  oldStartHint: number | undefined,
): { readonly startIndex: number; readonly score: number } {
  const windowLen = Math.min(Math.max(oldLines.length, 1), fileLines.length)
  const normalizedNeedles = oldLines.map(line => line.trim())
  let bestStart = oldStartHint !== undefined && oldStartHint > 0
    ? Math.min(Math.max(0, oldStartHint - 1), Math.max(0, fileLines.length - windowLen))
    : 0
  let bestScore = -1

  for (let start = 0; start <= fileLines.length - windowLen; start += 1) {
    let score = 0
    for (let offset = 0; offset < windowLen; offset += 1) {
      const needle = normalizedNeedles[offset] ?? ''
      const candidate = fileLines[start + offset]!.trim()
      if (!needle && !candidate) {
        score += 0.1
      } else if (needle && candidate === needle) {
        score += 3
      } else if (needle.length >= 4 && (candidate.includes(needle) || needle.includes(candidate)) && candidate.length >= 4) {
        score += 1.5
      } else if (needle && candidate) {
        score += tokenOverlapScore(needle, candidate)
      }
    }
    if (oldStartHint !== undefined && oldStartHint > 0) {
      const distance = Math.abs(start - (oldStartHint - 1))
      score += Math.max(0, 0.25 - distance * 0.005)
    }
    if (score > bestScore) {
      bestScore = score
      bestStart = start
    }
  }

  return { startIndex: bestStart, score: bestScore }
}

function tokenOverlapScore(left: string, right: string): number {
  const leftTokens = left.split(/[^A-Za-z0-9_]+/).filter(token => token.length >= 2)
  if (leftTokens.length === 0) return 0
  const rightSet = new Set(right.split(/[^A-Za-z0-9_]+/).filter(token => token.length >= 2))
  let shared = 0
  for (const token of leftTokens) {
    if (rightSet.has(token)) shared += 1
  }
  return shared / leftTokens.length
}

function renderNumberedPreview(fileLines: readonly string[], startIndex: number, count: number): string {
  const end = Math.min(fileLines.length, startIndex + count)
  const rendered: string[] = []
  for (let i = startIndex; i < end; i += 1) {
    const raw = fileLines[i] ?? ''
    const clipped = raw.length <= 140 ? raw : `${raw.slice(0, 140)}…`
    rendered.push(`  ${i + 1} | ${clipped}`)
  }
  return rendered.join('\n')
}

function ensureNoFileBlockingParent(
  targetAbs: string,
  displayPath: string,
  virtualFs: ReadonlyMap<string, VirtualFile>,
): void {
  let current = dirname(targetAbs)
  while (true) {
    const virtual = virtualFs.get(current)
    if (virtual !== undefined) {
      if (virtual.exists) {
        throw new ValidationError(
          'patch',
          `cannot create "${displayPath}" because ancestor "${virtual.displayPath}" is a regular file`,
          displayPath,
        )
      }
    } else if (existsSync(current)) {
      if (!statSync(current).isDirectory()) {
        throw new ValidationError(
          'patch',
          `cannot create "${displayPath}" because parent path "${current}" is not a directory`,
          displayPath,
        )
      }
      return
    }
    const parent = dirname(current)
    if (parent === current) return
    current = parent
  }
}

function leadingWhitespace(line: string): string {
  const match = line.match(/^[ \t]*/)
  return match?.[0] ?? ''
}

function commonStringPrefix(values: readonly string[]): string {
  if (values.length === 0) return ''
  let prefix = values[0]!
  for (let i = 1; i < values.length && prefix.length > 0; i += 1) {
    const candidate = values[i]!
    let cursor = 0
    while (cursor < prefix.length && cursor < candidate.length && prefix[cursor] === candidate[cursor]) {
      cursor += 1
    }
    prefix = prefix.slice(0, cursor)
  }
  return prefix
}

function unwrapOptionalCodeFence(text: string): string {
  const trimmed = text.trim()
  if (!trimmed.startsWith('```') || !trimmed.endsWith('```')) {
    return text
  }
  const firstNewline = trimmed.indexOf('\n')
  if (firstNewline < 0) return text
  const inner = trimmed.slice(firstNewline + 1, -3)
  const firstLine = inner.trimStart()
  if (
    firstLine.startsWith('*** ')
    || firstLine.startsWith('diff --git ')
    || firstLine.startsWith('--- ')
    || firstLine.startsWith('rename from ')
  ) {
    return inner
  }
  return text
}

function isV4aDirective(line: string): boolean {
  return (
    line.startsWith('*** Update File:')
    || line.startsWith('*** Add File:')
    || line.startsWith('*** Delete File:')
    || line.trim() === '*** End Patch'
  )
}

function isV4aSectionBoundary(line: string): boolean {
  return isV4aDirective(line)
}

function hasMoreAddLinesBeforeDirective(lines: readonly string[], startIndex: number): boolean {
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i]!
    if (isV4aDirective(line)) return false
    if (line.trim() !== '') return true
  }
  return false
}

function hasHunkLineBeforeBoundary(lines: readonly string[], startIndex: number): boolean {
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i]!
    if (isV4aSectionBoundary(line)) return false
    if (line.trim() !== '') return true
  }
  return false
}

function hasContinuationInSameV4aHunk(lines: readonly string[], startIndex: number): boolean {
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i]!
    if (isV4aSectionBoundary(line)) return false
    if (line === '@@' || line.startsWith('@@ ') || line.startsWith('@@\t') || /^@@\s*-\d+/.test(line)) return false
    if (line.trim() === '*** End of File') return true
    if (line === '') continue
    return line.startsWith(' ') || line.startsWith('-') || line.startsWith('+')
  }
  return false
}

function hasMoreUnifiedHunkLines(lines: readonly string[], startIndex: number): boolean {
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i]!
    if (
      line.startsWith('diff --git ')
      || line.startsWith('@@ ')
      || line === '@@'
      || line.startsWith('--- ')
      || line.startsWith('Index: ')
      || line.startsWith('rename from ')
    ) {
      return false
    }
    if (line === '') continue
    return line.startsWith(' ') || line.startsWith('-') || line.startsWith('+')
  }
  return false
}

function parseHunkHeaderMetadata(header: string): {
  readonly anchorHeader?: string
  readonly oldStartHint?: number
} {
  const unified = header.match(/^@@\s*-(\d+)(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@(.*)$/)
  if (unified) {
    const oldStartHint = Number.parseInt(unified[1]!, 10)
    const anchor = unified[2]?.trim() ?? ''
    return {
      oldStartHint,
      ...(anchor ? { anchorHeader: anchor } : {}),
    }
  }
  const rest = header.slice(2).trim()
  return rest ? { anchorHeader: rest } : {}
}

function parseDiffGitHeader(line: string): {
  readonly usedAb: boolean
  readonly oldPath?: string
  readonly newPath?: string
} {
  const match = line.match(/^diff --git "?a\/(.+?)"? "?b\/(.+?)"?$/)
  if (match?.[1] && match[2]) {
    return { usedAb: true, oldPath: match[1], newPath: match[2] }
  }
  return { usedAb: false }
}

function stripUnifiedTimestamp(rawPath: string): string {
  const tabIndex = rawPath.indexOf('\t')
  return tabIndex >= 0 ? rawPath.slice(0, tabIndex).trim() : rawPath
}

function cleanHeaderPath(raw: string): string {
  const trimmed = raw.trim()
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith('\'') && trimmed.endsWith('\''))) {
    return trimmed.slice(1, -1).trim()
  }
  return trimmed
}

function requirePatchPath(raw: string, directive: string): string {
  const cleaned = cleanHeaderPath(raw)
  if (!cleaned || cleaned.includes('\0')) {
    throw new ValidationError('patch', `invalid or empty path in "${directive}" header`, raw)
  }
  return cleaned
}
