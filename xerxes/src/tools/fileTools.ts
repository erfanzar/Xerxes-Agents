// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile as appendTextFile, lstat, mkdir, readdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import type { FileToolContext } from './fileState.js'
import { guardedCreate, guardedWrite, recordFileRead, withStaleNotice } from './fileState.js'
import { optionalBoolean, optionalInteger, optionalString, requireRange, requiredString } from './inputs.js'
import { WorkspacePathError, WorkspacePathResolver } from './pathSafety.js'

export const DEFAULT_READ_LINE_LIMIT = 400
export const DEFAULT_MAX_RESULTS = 500
/** Byte ceiling for a single read before any operator override; see resolveMaxReadFileBytes. */
export const DEFAULT_MAX_READ_FILE_BYTES = 262_144
/**
 * Characters a single read may return, whichever line window produced them.
 *
 * The line window alone is not a bound: a minified bundle, a one-line JSON dump,
 * or a long lockfile line is a single line, so the default limit happily returns
 * the entire file and reports no truncation. Refusing here costs a hundred bytes
 * and teaches the caller to narrow the read; succeeding costs the context window.
 */
export const MAX_READ_WINDOW_CHARS = 40_000
const MAX_GREP_FILE_BYTES = 1_000_000
const MAX_TOOL_RESULTS = 5_000

export const READ_FILE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a window of lines from one UTF-8 text file. file_path is workspace-relative; an absolute path '
      + 'inside the workspace also resolves, anything outside it is refused. Output is the file text verbatim with no '
      + 'line-number prefix, so it can be pasted straight back into FileEditTool old_string. Defaults to '
      + `${DEFAULT_READ_LINE_LIMIT} lines from offset 0 and reports the offset to continue from; limit=-1 means a `
      + `deliberate whole-file read. One call returns at most ${MAX_READ_WINDOW_CHARS} characters and files over `
      + `${DEFAULT_MAX_READ_FILE_BYTES} bytes are refused outright; both are errors, not truncation. A file with very `
      + 'long lines (minified bundles, single-line JSON, lockfiles) can exceed the character ceiling even at limit=1 '
      + 'because it is one line: locate what you need with GrepTool, or set max_chars to cap the return. A missing '
      + 'file is a normal recoverable outcome — confirm the path with GlobTool instead of retrying the same read. '
      + 'A directory is not readable here; use ListDir.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative file path.' },
        offset: { type: 'integer', default: 0, description: 'Zero-based line offset.' },
        limit: {
          type: 'integer',
          default: DEFAULT_READ_LINE_LIMIT,
          description: 'Line limit; -1 reads the whole file, subject to the same character ceiling.',
        },
        max_chars: {
          type: 'integer',
          description: 'Optional character cap applied to the window; -1 disables it. Use it to read part of a file '
            + 'whose lines are too long to page through by line count.',
        },
      },
      required: ['file_path'],
    },
  },
}

export const WRITE_FILE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'WriteFile',
    description: 'Create one workspace file from complete UTF-8 content. content is the whole file, never a patch, so '
      + 'replacing an existing file requires overwrite=true and silently discards everything not repeated in content — '
      + 'for a change to a file that already exists, prefer FileEditTool. file_path is workspace-relative; an absolute '
      + 'path inside the workspace also resolves, anything outside it is refused. Missing parent directories are '
      + 'created unless create_dirs=false. The target must be a regular file: an existing directory at that path is an '
      + 'error, not an overwrite.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative output path.' },
        content: { type: 'string', description: 'Complete text content to write.' },
        overwrite: { type: 'boolean', default: false, description: 'Allow replacing an existing regular file.' },
        create_dirs: { type: 'boolean', default: true, description: 'Create missing parent directories.' },
      },
      required: ['file_path', 'content'],
    },
  },
}

export const APPEND_FILE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'AppendFile',
    description: 'Append text and an optional newline to a workspace file, creating it when necessary.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative output path.' },
        lines: { type: 'string', description: 'Text to append; an empty string is allowed.' },
        newline: { type: 'string', default: '\n', description: 'Suffix appended after lines.' },
      },
      required: ['file_path', 'lines'],
    },
  },
}

export const LIST_DIR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ListDir',
    description: 'List workspace directory entries without following symlinked directories.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        directory_path: { type: 'string', default: '.', description: 'Workspace-relative directory path.' },
        extension_filter: { type: 'string', description: 'Optional case-insensitive file extension filter.' },
        recursive: { type: 'boolean', default: false, description: 'Include descendant entries.' },
        max_depth: { type: 'integer', default: 3, description: 'Maximum recursive directory depth.' },
        show_hidden: { type: 'boolean', default: false, description: 'Include names beginning with a period.' },
        max_results: { type: 'integer', default: DEFAULT_MAX_RESULTS, description: 'Maximum entries to return.' },
      },
    },
  },
}

export const GLOB_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'GlobTool',
    description: 'Find workspace files and directories matching a Bun glob pattern.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        pattern: { type: 'string', description: 'Glob pattern such as **/*.ts.' },
        path: { type: 'string', default: '.', description: 'Workspace-relative directory to search.' },
        include_hidden: { type: 'boolean', default: false, description: 'Allow glob matches beginning with a period.' },
        max_results: { type: 'integer', default: DEFAULT_MAX_RESULTS, description: 'Maximum paths to return.' },
      },
      required: ['pattern'],
    },
  },
}

export const GREP_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'GrepTool',
    description: 'Search workspace text files with a JavaScript regular expression.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        pattern: { type: 'string', description: 'JavaScript regular expression source.' },
        path: { type: 'string', default: '.', description: 'Workspace-relative directory or file to search.' },
        glob: { type: 'string', description: 'Optional glob filter, for example **/*.ts.' },
        output_mode: {
          type: 'string',
          enum: ['files_with_matches', 'count', 'content'],
          default: 'files_with_matches',
        },
        case_insensitive: { type: 'boolean', default: false, description: 'Search without case sensitivity.' },
        context: { type: 'integer', default: 0, description: 'Context lines for content mode.' },
        max_results: {
          type: 'integer',
          default: DEFAULT_MAX_RESULTS,
          description: 'Maximum matching results to return.',
        },
      },
      required: ['pattern'],
    },
  },
}

export const FILE_EDIT_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'FileEditTool',
    description: 'Replace one exact span of text in a workspace file (edit_mode=search_replace) or rewrite the file '
      + 'whole (edit_mode=whole_file). Read the file first: old_string must match what is on disk right now, copied '
      + 'verbatim from ReadFile output — ReadFile adds no line-number prefix, but any prefix or re-indentation you add '
      + 'yourself will stop it matching. old_string must be unique: if it occurs more than once the edit is refused '
      + 'untouched, so extend it with neighbouring lines until it is unique, or set replace_all=true when every '
      + 'occurrence genuinely should change. When nothing matches, one retry folds curly quotes, non-breaking spaces, '
      + 'and CRLF line endings and the file keeps its own typography; any other mismatch (stale content, wrong '
      + 'indentation) needs a fresh read rather than a retry. new_string must differ from old_string and is inserted '
      + 'literally — no regex, no $-group substitution.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative file path.' },
        old_string: {
          type: 'string',
          default: '',
          description: 'Exact on-disk text to replace in search_replace mode; must be unique unless replace_all.',
        },
        new_string: { type: 'string', default: '', description: 'Replacement text or complete file contents.' },
        replace_all: { type: 'boolean', default: false, description: 'Replace every occurrence instead of refusing.' },
        edit_mode: { type: 'string', enum: ['search_replace', 'whole_file'], default: 'search_replace' },
      },
      required: ['file_path'],
    },
  },
}

export const FILE_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  READ_FILE_DEFINITION,
  WRITE_FILE_DEFINITION,
  APPEND_FILE_DEFINITION,
  LIST_DIR_DEFINITION,
  GLOB_TOOL_DEFINITION,
  GREP_TOOL_DEFINITION,
  FILE_EDIT_TOOL_DEFINITION,
]

/** Register the Bun-native filesystem tools against one workspace path resolver. */
/**
 * Read-only workspace queries: independent by construction, so a round asking
 * for several at once runs them together instead of paying each one's latency
 * end to end.
 */
const READ_ONLY_FILE_CAPABILITIES = Object.freeze({
  concurrencySafe: true,
  defer: false,
  destructive: false,
  openWorld: false,
  readOnly: true,
} as const)

/**
 * Writers serialize against everything, and are deliberately NOT interruptible:
 * aborting midway through a write leaves the workspace in a state neither the
 * user nor the model asked for. Each is bounded by its own size ceiling, so
 * letting one finish cannot park a turn indefinitely.
 */
const FILE_WRITE_CAPABILITIES = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: true,
  interruptBehavior: 'block',
  openWorld: false,
  readOnly: false,
} as const)

/**
 * The freshness gate every writer shares, stated where the model meets the
 * tool instead of in a central prompt that can drift from the registry.
 */
const FILE_WRITE_GUIDANCE =
  'Writes are checked against your last read: ReadFile the target immediately before writing or '
    + 'editing it, and re-read after anything outside this session may have changed it — a stale '
    + 'read is refused rather than silently overwriting newer work.'

export function registerFileTools(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(READ_FILE_DEFINITION, (inputs, context) => readFile(inputs, paths, context), 'default', READ_ONLY_FILE_CAPABILITIES)
  registry.register(WRITE_FILE_DEFINITION, (inputs, context) => writeFile(inputs, paths, context), 'default', FILE_WRITE_CAPABILITIES, FILE_WRITE_GUIDANCE)
  registry.register(APPEND_FILE_DEFINITION, inputs => appendFile(inputs, paths), 'default', FILE_WRITE_CAPABILITIES)
  registry.register(LIST_DIR_DEFINITION, inputs => listDirectory(inputs, paths), 'default', READ_ONLY_FILE_CAPABILITIES)
  registry.register(GLOB_TOOL_DEFINITION, inputs => globFiles(inputs, paths), 'default', READ_ONLY_FILE_CAPABILITIES)
  registry.register(GREP_TOOL_DEFINITION, inputs => grepFiles(inputs, paths), 'default', READ_ONLY_FILE_CAPABILITIES)
  registry.register(FILE_EDIT_TOOL_DEFINITION, (inputs, context) => editFile(inputs, paths, context), 'default', FILE_WRITE_CAPABILITIES, FILE_WRITE_GUIDANCE)
}

export async function readFile(
  inputs: JsonObject,
  paths: WorkspacePathResolver,
  context?: FileToolContext,
): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const offset = requireRange(optionalNullableInteger(inputs, 'offset', 0), 'offset', 0, Number.MAX_SAFE_INTEGER)
  const limit = optionalNullableInteger(inputs, 'limit', DEFAULT_READ_LINE_LIMIT)
  if (limit !== -1) {
    requireRange(limit, 'limit', 1, Number.MAX_SAFE_INTEGER)
  }
  const maxChars = optionalNullableInteger(inputs, 'max_chars', -1)
  if (maxChars !== -1) {
    requireRange(maxChars, 'max_chars', 0, Number.MAX_SAFE_INTEGER)
  }

  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)
  const fileInfo = await stat(target)
  const maxBytes = resolveMaxReadFileBytes()
  if (fileInfo.size > maxBytes) {
    throw new ValidationError(
      'file_path',
      'is ' + fileInfo.size + ' bytes, exceeding the ' + maxBytes
        + '-byte ReadFile limit; search it with GrepTool or split it into smaller files first',
      filePath,
    )
  }
  const text = await Bun.file(target).text()
  // An explicit max_chars is itself bounded by the window ceiling: a caller
  // passing max_chars=1_000_000 over a 60k-character window used to skip
  // truncation and then be rejected by the ceiling — advised, absurdly, to cap
  // the return with the very parameter they had supplied. -1 keeps its
  // documented meaning (no char cap; oversized reads are an error, not a cut).
  const effectiveMaxChars = maxChars === -1 ? -1 : Math.min(maxChars, MAX_READ_WINDOW_CHARS)
  if (limit === -1) {
    const whole = truncateCharacters(text, effectiveMaxChars)
    enforceReadWindowCeiling(whole, { charCap: 'max_chars', filePath, lineParameter: 'limit', toolName: 'ReadFile' })
    recordFileRead(context, target, text, {
      mtimeMs: fileInfo.mtimeMs,
      partialView: whole !== text,
      size: fileInfo.size,
    })
    return whole
  }

  const lines = splitLines(text)
  if (offset >= lines.length && lines.length > 0) {
    return `[ReadFile] Offset ${offset} is past end of file (${lines.length} lines).`
  }
  const endOffset = Math.min(offset + limit, lines.length)
  const window = lines.slice(offset, endOffset).join('')
  const selected = truncateCharacters(window, effectiveMaxChars)
  enforceReadWindowCeiling(selected, { charCap: 'max_chars', filePath, lineParameter: 'limit', toolName: 'ReadFile' })
  recordFileRead(context, target, text, {
    mtimeMs: fileInfo.mtimeMs,
    partialView: offset > 0 || endOffset < lines.length || selected !== window,
    size: fileInfo.size,
  })
  const notice = endOffset < lines.length
    ? `\n\n[ReadFile] Showing lines ${offset + 1}-${endOffset} of ${lines.length}. `
      + `Continue with offset=${endOffset}, limit=${limit}. `
      + 'Use limit=-1 only when the whole file is intentionally required.'
    : ''
  return selected + notice
}

/** Where a returned read window came from, so the refusal can name the knob that narrows it. */
export interface ReadWindowContext {
  /** Parameter that caps returned characters directly, when the caller exposes one. */
  readonly charCap?: string
  readonly filePath: string
  /** Parameter the caller shrinks to fetch fewer lines, for example limit or end_line. */
  readonly lineParameter: string
  /**
   * Value the suggested line count is added to, so an absolute parameter such as
   * end_line is quoted back as a line number rather than a count.
   */
  readonly lineParameterBase?: number
  readonly toolName: string
}

/**
 * Refuse a read window over MAX_READ_WINDOW_CHARS instead of returning it.
 *
 * Truncating here would be worse than failing: the caller would receive a
 * plausible-looking prefix of a generated file and keep reasoning from it. The
 * message therefore has to carry enough arithmetic for the retry to be right the
 * first time, including the case where no line count is small enough.
 */
export function enforceReadWindowCeiling(text: string, context: ReadWindowContext): void {
  if (text.length <= MAX_READ_WINDOW_CHARS) {
    return
  }
  const lineCount = Math.max(1, countNewlines(text) + (text.endsWith('\n') ? 0 : 1))
  const perLine = Math.ceil(text.length / lineCount)
  const charCapHint = context.charCap === undefined ? '' : ', or cap the return with ' + context.charCap
  const remedy = perLine > MAX_READ_WINDOW_CHARS
    ? 'single lines are longer than the whole ceiling, so this file is minified or generated: find what you need '
      + 'with GrepTool' + charCapHint
    : 'retry with ' + context.lineParameter + '='
      + ((context.lineParameterBase ?? 0) + Math.max(1, Math.floor(MAX_READ_WINDOW_CHARS / perLine)))
      + ' or smaller, or search the file with GrepTool instead' + charCapHint
  throw new ValidationError(
    'file_path',
    'selected ' + text.length + ' characters across ' + lineCount + ' line(s) averaging ' + perLine
      + ' characters, exceeding the ' + MAX_READ_WINDOW_CHARS + '-character ' + context.toolName
      + ' window ceiling; ' + remedy,
    context.filePath,
  )
}

let configuredMaxReadFileBytes: number | undefined

/**
 * Let a host raise or lower the read ceiling once at startup.
 *
 * The environment variable still wins, so an operator can override a profile or
 * bundled config they cannot edit without rebuilding.
 */
export function setMaxReadFileBytes(bytes: number | undefined): void {
  if (bytes !== undefined && (!Number.isInteger(bytes) || bytes < 1)) {
    throw new ValidationError('max_read_file_bytes', 'must be a positive integer', bytes)
  }
  configuredMaxReadFileBytes = bytes
}

/** Resolve the read byte ceiling as environment override, then runtime setting, then default. */
export function resolveMaxReadFileBytes(
  environment: Readonly<Record<string, string | undefined>> = process.env,
): number {
  return positiveInteger(environment.XERXES_MAX_READ_FILE_BYTES)
    ?? configuredMaxReadFileBytes
    ?? DEFAULT_MAX_READ_FILE_BYTES
}

export async function writeFile(
  inputs: JsonObject,
  paths: WorkspacePathResolver,
  context?: FileToolContext,
): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const content = requiredContent(inputs, 'content')
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const createDirs = optionalBoolean(inputs, 'create_dirs', true)
  const target = await paths.resolve(filePath)
  const existing = await pathExists(target)

  if (existing && !overwrite) {
    throw new ValidationError('file_path', 'already exists; pass overwrite=true to replace it', filePath)
  }
  if (existing) {
    await requireRegularFile(target, filePath)
  }
  if (createDirs) {
    await mkdir(dirname(target), { recursive: true })
  } else if (!(await isDirectory(dirname(target)))) {
    throw new ValidationError('file_path', 'parent directory does not exist and create_dirs is false', filePath)
  }
  const relativePath = await paths.relative(target)
  // Last await before the guarded region: everything after this runs synchronously so
  // the file cannot change between the freshness check and the write that trusts it.
  const checked = await paths.recheck(target)

  if (!existing) {
    guardedCreate({ absolutePath: checked, content, displayPath: filePath, sessionId: context?.sessionId })
    return `Wrote ${content.length} characters to ${relativePath} (created).`
  }
  guardedWrite({
    absolutePath: checked,
    displayPath: filePath,
    mode: 'overwrite',
    sessionId: context?.sessionId,
    toolName: 'WriteFile',
    transform: () => content,
  })
  return `Wrote ${content.length} characters to ${relativePath} (overwrote).`
}

/** Append text to a workspace file while preserving the same path-containment boundary as WriteFile. */
export async function appendFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const lines = requiredContent(inputs, 'lines')
  const newline = optionalString(inputs, 'newline') ?? '\n'
  if (newline.includes('\0')) {
    throw new ValidationError('newline', 'must not contain a null byte', newline)
  }
  const target = await paths.resolve(filePath)
  const existing = await pathExists(target)
  if (existing) {
    await requireRegularFile(target, filePath)
  } else {
    await mkdir(dirname(target), { recursive: true })
  }
  const relativePath = await paths.relative(target)
  const checked = await paths.recheck(target)
  await appendTextFile(checked, lines + newline, 'utf8')
  return 'Appended ' + lines.length + ' characters to ' + relativePath + '.'
}

export async function listDirectory(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string[]> {
  const directoryPath = optionalString(inputs, 'directory_path') ?? '.'
  const extensionFilter = optionalString(inputs, 'extension_filter')?.toLowerCase()
  const recursive = optionalBoolean(inputs, 'recursive', false)
  const maxDepth = requireRange(optionalInteger(inputs, 'max_depth', 3), 'max_depth', 0, 100)
  const showHidden = optionalBoolean(inputs, 'show_hidden', false)
  const maxResults = requireRange(
    optionalInteger(inputs, 'max_results', DEFAULT_MAX_RESULTS),
    'max_results',
    1,
    MAX_TOOL_RESULTS,
  )
  const target = await paths.resolve(directoryPath)
  if (!(await isDirectory(target))) {
    throw new ValidationError('directory_path', 'must refer to an existing directory', directoryPath)
  }

  const entries: string[] = []
  let truncated = false

  const collect = async (current: string, prefix: string, depth: number): Promise<void> => {
    const children = await readdir(current, { withFileTypes: true })
    children.sort((left, right) => left.name.localeCompare(right.name))
    for (const child of children) {
      if (!showHidden && child.name.startsWith('.')) {
        continue
      }
      if (entries.length >= maxResults) {
        truncated = true
        return
      }
      const childPath = prefix ? `${prefix}/${child.name}` : child.name
      if (child.isDirectory()) {
        entries.push(`${childPath}/`)
        if (recursive && depth < maxDepth) {
          await collect(join(current, child.name), childPath, depth + 1)
          if (truncated) {
            return
          }
        }
      } else if (!extensionFilter || child.name.toLowerCase().endsWith(extensionFilter)) {
        entries.push(child.isSymbolicLink() ? `${childPath}@` : childPath)
      }
    }
  }

  await collect(target, '', 0)
  if (truncated) {
    entries.push(`... (${maxResults} result limit reached)`)
  }
  return entries
}

export async function globFiles(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string[]> {
  const pattern = requiredString(inputs, 'pattern')
  validateGlobPattern(pattern)
  const directoryPath = optionalString(inputs, 'path') ?? '.'
  const includeHidden = optionalBoolean(inputs, 'include_hidden', false)
  const maxResults = requireRange(
    optionalInteger(inputs, 'max_results', DEFAULT_MAX_RESULTS),
    'max_results',
    1,
    MAX_TOOL_RESULTS,
  )
  const target = await paths.resolve(directoryPath)
  if (!(await isDirectory(target))) {
    throw new ValidationError('path', 'must refer to an existing directory', directoryPath)
  }

  const matches: string[] = []
  let truncated = false
  const glob = new Bun.Glob(pattern)
  for await (const match of glob.scan({ cwd: target, dot: includeHidden, followSymlinks: false, onlyFiles: false })) {
    if (matches.length >= maxResults) {
      truncated = true
      break
    }
    const resolvedMatch = await resolveScannedPath(paths, join(target, match))
    if (!resolvedMatch) {
      continue
    }
    // Tool results use POSIX separators on every host so model-visible paths
    // round-trip through search/replace and patch tooling unchanged.
    matches.push((await paths.relative(resolvedMatch)).replaceAll('\\', '/'))
  }
  matches.sort((left, right) => left.localeCompare(right))
  if (truncated) {
    matches.push(`... (${maxResults} result limit reached)`)
  }
  return matches
}

export async function grepFiles(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const pattern = requiredString(inputs, 'pattern')
  const directoryPath = optionalString(inputs, 'path') ?? '.'
  const globPattern = optionalString(inputs, 'glob') ?? '**/*'
  validateGlobPattern(globPattern)
  const outputMode = optionalString(inputs, 'output_mode') ?? 'files_with_matches'
  if (outputMode !== 'files_with_matches' && outputMode !== 'count' && outputMode !== 'content') {
    throw new ValidationError('output_mode', 'must be files_with_matches, count, or content', outputMode)
  }
  const caseInsensitive = optionalBoolean(inputs, 'case_insensitive', false)
  const context = requireRange(optionalInteger(inputs, 'context', 0), 'context', 0, 100)
  const maxResults = requireRange(
    optionalInteger(inputs, 'max_results', DEFAULT_MAX_RESULTS),
    'max_results',
    1,
    MAX_TOOL_RESULTS,
  )
  const target = await paths.resolve(directoryPath)
  if (!(await isDirectory(target)) && !(await isRegularFile(target))) {
    throw new ValidationError('path', 'must refer to an existing file or directory', directoryPath)
  }

  let expression: RegExp
  try {
    expression = new RegExp(pattern, caseInsensitive ? 'i' : '')
  } catch (error) {
    throw new ValidationError(
      'pattern',
      `must be a valid JavaScript regular expression: ${errorMessage(error)}`,
      pattern,
    )
  }

  const files = await grepCandidates(target, globPattern, paths)
  const results: string[] = []
  let truncated = false

  for (const file of files) {
    if (results.length >= maxResults) {
      truncated = true
      break
    }
    const fileStats = await stat(file)
    if (fileStats.size > MAX_GREP_FILE_BYTES) {
      continue
    }
    const text = await Bun.file(file).text()
    if (text.includes('\0')) {
      continue
    }
    const lines = splitLines(text).map(line => line.endsWith('\n') ? line.slice(0, -1) : line)
    const matchingLines: number[] = []
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]
      if (line !== undefined && expression.test(line)) {
        matchingLines.push(index)
      }
    }
    if (matchingLines.length === 0) {
      continue
    }

    const relativePath = (await paths.relative(file)).replaceAll('\\', '/')
    if (outputMode === 'files_with_matches') {
      results.push(relativePath)
      continue
    }
    if (outputMode === 'count') {
      results.push(`${relativePath}:${matchingLines.length}`)
      continue
    }

    const displayed = new Set<number>()
    for (const matchIndex of matchingLines) {
      const start = Math.max(0, matchIndex - context)
      const end = Math.min(lines.length, matchIndex + context + 1)
      for (let index = start; index < end; index += 1) {
        if (displayed.has(index)) {
          continue
        }
        if (results.length >= maxResults) {
          truncated = true
          break
        }
        displayed.add(index)
        const line = lines[index] ?? ''
        results.push(`${relativePath}:${index + 1}:${line}`)
      }
      if (truncated) {
        break
      }
    }
    if (truncated) {
      break
    }
  }

  if (results.length === 0) {
    return 'No matches found.'
  }
  if (truncated) {
    results.push(`... (${maxResults} result limit reached)`)
  }
  return results.join('\n')
}

export async function editFile(
  inputs: JsonObject,
  paths: WorkspacePathResolver,
  context?: FileToolContext,
): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const oldString = optionalString(inputs, 'old_string') ?? ''
  const newString = optionalString(inputs, 'new_string') ?? ''
  const replaceAll = optionalBoolean(inputs, 'replace_all', false)
  const editMode = optionalString(inputs, 'edit_mode') ?? 'search_replace'
  if (editMode !== 'search_replace' && editMode !== 'whole_file') {
    throw new ValidationError('edit_mode', 'must be search_replace or whole_file', editMode)
  }
  if (editMode === 'whole_file' && !newString) {
    throw new ValidationError('new_string', 'must not be empty in whole_file mode', newString)
  }
  if (editMode === 'search_replace') {
    if (!oldString) {
      throw new ValidationError('old_string', 'must not be empty in search_replace mode', oldString)
    }
    if (oldString === newString) {
      throw new ValidationError('new_string', 'must differ from old_string', newString)
    }
  }

  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)
  const relativePath = await paths.relative(target)
  // Last await before the guarded region; the replacement below is computed from the
  // very bytes the freshness check inspected, not from a re-read that could differ.
  const checked = await paths.recheck(target)

  if (editMode === 'whole_file') {
    guardedWrite({
      absolutePath: checked,
      displayPath: filePath,
      mode: 'overwrite',
      sessionId: context?.sessionId,
      toolName: 'FileEditTool',
      transform: () => newString,
    })
    return `Replaced entire file ${relativePath}.`
  }

  let applied: ForgivingReplacement | undefined
  const written = guardedWrite({
    absolutePath: checked,
    displayPath: filePath,
    mode: 'targeted',
    sessionId: context?.sessionId,
    toolName: 'FileEditTool',
    transform: content => {
      const occurrences = countOccurrences(content, oldString)
      if (occurrences === 0) {
        applied = replaceForgivingly(content, oldString, newString, replaceAll)
        return applied.text
      }
      if (occurrences > 1 && !replaceAll) {
        throw new ValidationError(
          'old_string',
          `appears ${occurrences} times; provide more context or set replace_all=true`,
          oldString,
        )
      }
      return replaceAll ? content.split(oldString).join(newString) : content.replace(oldString, newString)
    },
  })
  const summary = applied === undefined
    ? `Applied ${replaceAll ? countOccurrences(written.previous, oldString) : 1} replacement(s) to ${relativePath}.`
    : `Applied ${applied.replacements} replacement(s) to ${relativePath} `
      + `after matching on ${applied.matchedOn}; the file keeps its original characters.`
  return withStaleNotice(written.staleNotice, summary)
}

/**
 * Characters an editor, a chat client, or a copy-paste round trip silently swaps.
 *
 * Every entry maps one character to exactly one replacement character, which keeps
 * the folded text index-aligned with the original — that alignment is what lets the
 * replacement be spliced back into the untouched original text.
 */
const INVISIBLE_FOLDS: ReadonlyMap<string, string> = new Map([
  ['‘', '\''],
  ['’', '\''],
  ['‚', '\''],
  ['‛', '\''],
  ['“', '"'],
  ['”', '"'],
  ['„', '"'],
  ['‟', '"'],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  [' ', ' '],
  ['　', ' '],
])

interface ForgivingReplacement {
  /** Human-readable description of which fold made the match, for the tool result. */
  readonly matchedOn: string
  readonly replacements: number
  readonly text: string
}

/**
 * Retry a failed exact edit once against typography-folded text.
 *
 * A model that quotes a file back from memory, or a file that was authored in a word
 * processor, differs from disk only in curly quotes, non-breaking spaces, or line
 * endings. Refusing those edits sends the caller into a re-read loop that produces the
 * same bytes it already had. The fold is only used to *locate* the span; the original
 * text outside the match is never rewritten, and the replacement is bent back to the
 * matched span's own characters so the file does not end up with mixed typography.
 */
function replaceForgivingly(
  content: string,
  oldString: string,
  newString: string,
  replaceAll: boolean,
): ForgivingReplacement {
  const crlfContent = usesCrlfLineEndings(content)
  const crlfNeedle = crlfContent && oldString.includes('\n') && !oldString.includes('\r\n')
    ? oldString.replaceAll('\n', '\r\n')
    : undefined
  const attempts: readonly { readonly matchedOn: string; readonly needle: string }[] = crlfNeedle === undefined
    ? [{ matchedOn: 'normalized quotes and spaces', needle: oldString }]
    : [
        { matchedOn: 'CRLF line endings', needle: crlfNeedle },
        { matchedOn: 'CRLF line endings and normalized quotes and spaces', needle: crlfNeedle },
      ]

  for (const [attemptIndex, attempt] of attempts.entries()) {
    const exact = attemptIndex === 0 && attempt.needle !== oldString
    const indices = exact ? indicesOf(content, attempt.needle) : foldedIndicesOf(content, attempt.needle)
    if (indices.length === 0) {
      continue
    }
    if (indices.length > 1 && !replaceAll) {
      throw new ValidationError(
        'old_string',
        `appears ${indices.length} times; provide more context or set replace_all=true`,
        oldString,
      )
    }
    const chosen = replaceAll ? indices : indices.slice(0, 1)
    return {
      matchedOn: attempt.matchedOn,
      replacements: chosen.length,
      text: spliceMatches(content, chosen, attempt.needle.length, newString, crlfContent),
    }
  }
  throw new ValidationError('old_string', 'was not found exactly; re-read the file before retrying', oldString)
}

/** Offsets where the folded needle occurs in the folded haystack; the fold preserves offsets. */
function foldedIndicesOf(content: string, needle: string): number[] {
  const foldedNeedle = foldInvisibles(needle)
  if (foldedNeedle === needle && !containsFoldable(content)) {
    return []
  }
  return indicesOf(foldInvisibles(content), foldedNeedle)
}

function indicesOf(haystack: string, needle: string): number[] {
  const found: number[] = []
  let index = haystack.indexOf(needle)
  while (index !== -1) {
    found.push(index)
    index = haystack.indexOf(needle, index + needle.length)
  }
  return found
}

function spliceMatches(
  content: string,
  indices: readonly number[],
  length: number,
  replacement: string,
  crlfContent: boolean,
): string {
  let result = ''
  let cursor = 0
  for (const index of indices) {
    const span = content.slice(index, index + length)
    result += content.slice(cursor, index) + adaptReplacement(replacement, span, crlfContent)
    cursor = index + length
  }
  return result + content.slice(cursor)
}

/**
 * Bend the replacement toward the characters the matched span actually used.
 *
 * Restoration is positional and only happens when the replacement uses a folded
 * character exactly as often as the span did. That parity requirement is what makes
 * an opening/closing pair work — “ and ” both fold to one quote, so only their order
 * distinguishes them — and it is also what stops a longer replacement from having
 * plain spaces silently rewritten into the non-breaking space the span happened to
 * contain, which would plant invisible characters in source code.
 */
function adaptReplacement(replacement: string, span: string, crlfContent: boolean): string {
  const lineAdjusted = crlfContent ? replacement.replaceAll('\r\n', '\n').replaceAll('\n', '\r\n') : replacement
  const originals = foldedOriginals(span)
  if (originals.size === 0) {
    return lineAdjusted
  }
  const characters = [...lineAdjusted]
  const counts = new Map<string, number>()
  for (const character of characters) {
    const folded = INVISIBLE_FOLDS.get(character) ?? character
    if (originals.has(folded)) {
      counts.set(folded, (counts.get(folded) ?? 0) + 1)
    }
  }
  const cursors = new Map<string, number>()
  let adapted = ''
  for (const character of characters) {
    const folded = INVISIBLE_FOLDS.get(character) ?? character
    const sequence = originals.get(folded)
    if (sequence === undefined || counts.get(folded) !== sequence.length) {
      adapted += character
      continue
    }
    const cursor = cursors.get(folded) ?? 0
    cursors.set(folded, cursor + 1)
    adapted += sequence[cursor] ?? character
  }
  return adapted
}

/** Folded character to the original characters it stood for, in the order the span used them. */
function foldedOriginals(span: string): Map<string, string[]> {
  const originals = new Map<string, string[]>()
  for (const character of span) {
    const folded = INVISIBLE_FOLDS.get(character)
    if (folded === undefined) {
      continue
    }
    const sequence = originals.get(folded)
    if (sequence === undefined) {
      originals.set(folded, [character])
      continue
    }
    sequence.push(character)
  }
  return originals
}

function foldInvisibles(text: string): string {
  let folded = ''
  for (const character of text) {
    folded += INVISIBLE_FOLDS.get(character) ?? character
  }
  return folded
}

function containsFoldable(text: string): boolean {
  for (const character of text) {
    if (INVISIBLE_FOLDS.has(character)) {
      return true
    }
  }
  return false
}

/** True when every newline in the file is part of a CRLF pair, so an LF-only edit would mix endings. */
function usesCrlfLineEndings(content: string): boolean {
  return content.includes('\r\n') && !/(^|[^\r])\n/.test(content)
}

function requiredContent(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

function splitLines(text: string): string[] {
  return text.match(/[^\n]*\n|[^\n]+$/g) ?? []
}

function countNewlines(text: string): number {
  let total = 0
  let index = text.indexOf('\n')
  while (index !== -1) {
    total += 1
    index = text.indexOf('\n', index + 1)
  }
  return total
}

function positiveInteger(raw: string | undefined): number | undefined {
  if (raw === undefined) {
    return undefined
  }
  const parsed = Number.parseInt(raw.trim(), 10)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined
}

/** Appended when max_chars cuts a read window short (2 newlines + 26 chars). */
const TRUNCATED_BY_MAX_CHARS_MARKER = '\n\n…[truncated by max_chars]…'

function truncateCharacters(text: string, maxChars: number): string {
  if (maxChars === -1 || text.length <= maxChars) {
    return text
  }
  // Clamp the slice so slice + marker never exceed MAX_READ_WINDOW_CHARS.
  // Otherwise a caller asking for max_chars=40000 got the truncation marker
  // appended past the ceiling and then rejected by enforceReadWindowCeiling —
  // told, absurdly, to raise the very cap they had just set.
  const sliceLength = Math.min(maxChars, MAX_READ_WINDOW_CHARS - TRUNCATED_BY_MAX_CHARS_MARKER.length)
  return `${text.slice(0, sliceLength)}${TRUNCATED_BY_MAX_CHARS_MARKER}`
}

/** JSON tool calls often serialize omitted optional integers as null. */
function optionalNullableInteger(inputs: JsonObject, name: string, defaultValue: number): number {
  const value = inputs[name]
  if (value === undefined || value === null) {
    return defaultValue
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new ValidationError(name, 'must be an integer or null', value)
  }
  return value
}

function countOccurrences(content: string, needle: string): number {
  return content.split(needle).length - 1
}

async function requireRegularFile(target: string, originalPath: string): Promise<void> {
  if (!(await isRegularFile(target))) {
    throw new ValidationError('file_path', 'must refer to an existing regular file', originalPath)
  }
}

async function isRegularFile(target: string): Promise<boolean> {
  try {
    return (await stat(target)).isFile()
  } catch (error) {
    if (isNotFound(error)) {
      return false
    }
    throw error
  }
}

async function isDirectory(target: string): Promise<boolean> {
  try {
    return (await stat(target)).isDirectory()
  } catch (error) {
    if (isNotFound(error)) {
      return false
    }
    throw error
  }
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await lstat(target)
    return true
  } catch (error) {
    if (isNotFound(error)) {
      return false
    }
    throw error
  }
}

async function resolveScannedPath(paths: WorkspacePathResolver, candidate: string): Promise<string | undefined> {
  try {
    return await paths.resolve(candidate)
  } catch (error) {
    if (error instanceof WorkspacePathError) {
      return undefined
    }
    throw error
  }
}

async function grepCandidates(
  directoryOrFile: string,
  pattern: string,
  paths: WorkspacePathResolver,
): Promise<string[]> {
  if (await isRegularFile(directoryOrFile)) {
    return [directoryOrFile]
  }
  const files: string[] = []
  const glob = new Bun.Glob(pattern)
  for await (const match of glob.scan({ cwd: directoryOrFile, followSymlinks: false, onlyFiles: true })) {
    const resolvedMatch = await resolveScannedPath(paths, join(directoryOrFile, match))
    if (resolvedMatch && await isRegularFile(resolvedMatch)) {
      files.push(resolvedMatch)
    }
  }
  return files.sort((left, right) => left.localeCompare(right))
}

function validateGlobPattern(pattern: string): void {
  const hasTraversal = pattern.split(/[\\/]/).includes('..')
  if (pattern.includes('\0') || pattern.startsWith('/') || pattern.startsWith('\\') || hasTraversal) {
    throw new ValidationError('pattern', 'must be workspace-relative and must not contain parent traversal', pattern)
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
