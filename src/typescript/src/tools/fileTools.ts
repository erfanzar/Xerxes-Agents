// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile as appendTextFile, lstat, mkdir, readdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, requireRange, requiredString } from './inputs.js'
import { WorkspacePathError, WorkspacePathResolver } from './pathSafety.js'

export const DEFAULT_READ_LINE_LIMIT = 400
export const DEFAULT_MAX_RESULTS = 500
const MAX_GREP_FILE_BYTES = 1_000_000
const MAX_TOOL_RESULTS = 5_000

export const READ_FILE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a workspace file in line chunks. '
      + 'Use file_path, offset, and limit=-1 only for intentional full reads.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative file path.' },
        offset: { type: 'integer', default: 0, description: 'Zero-based line offset.' },
        limit: {
          type: 'integer',
          default: DEFAULT_READ_LINE_LIMIT,
          description: 'Line limit; -1 reads the whole file.',
        },
        max_chars: { type: 'integer', description: 'Optional character cap; -1 disables the cap.' },
      },
      required: ['file_path'],
    },
  },
}

export const WRITE_FILE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'WriteFile',
    description: 'Create a workspace file. Existing files require overwrite=true.',
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
    description: 'Replace exact text in a workspace file or replace its entire contents.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: { type: 'string', description: 'Workspace-relative file path.' },
        old_string: { type: 'string', default: '', description: 'Exact text to replace in search_replace mode.' },
        new_string: { type: 'string', default: '', description: 'Replacement text or complete file contents.' },
        replace_all: { type: 'boolean', default: false, description: 'Replace every exact occurrence.' },
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
export function registerFileTools(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(READ_FILE_DEFINITION, inputs => readFile(inputs, paths))
  registry.register(WRITE_FILE_DEFINITION, inputs => writeFile(inputs, paths))
  registry.register(APPEND_FILE_DEFINITION, inputs => appendFile(inputs, paths))
  registry.register(LIST_DIR_DEFINITION, inputs => listDirectory(inputs, paths))
  registry.register(GLOB_TOOL_DEFINITION, inputs => globFiles(inputs, paths))
  registry.register(GREP_TOOL_DEFINITION, inputs => grepFiles(inputs, paths))
  registry.register(FILE_EDIT_TOOL_DEFINITION, inputs => editFile(inputs, paths))
}

export async function readFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
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
  const text = await Bun.file(target).text()
  if (limit === -1) {
    return truncateCharacters(text, maxChars)
  }

  const lines = splitLines(text)
  if (offset >= lines.length && lines.length > 0) {
    return `[ReadFile] Offset ${offset} is past end of file (${lines.length} lines).`
  }
  const endOffset = Math.min(offset + limit, lines.length)
  const selected = lines.slice(offset, endOffset).join('')
  const notice = endOffset < lines.length
    ? `\n\n[ReadFile] Showing lines ${offset + 1}-${endOffset} of ${lines.length}. `
      + `Continue with offset=${endOffset}, limit=${limit}. `
      + 'Use limit=-1 only when the whole file is intentionally required.'
    : ''
  return truncateCharacters(selected, maxChars) + notice
}

export async function writeFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
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

  await Bun.write(target, content)
  const relativePath = await paths.relative(target)
  const action = existing ? 'overwrote' : 'created'
  return `Wrote ${content.length} characters to ${relativePath} (${action}).`
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
  await appendTextFile(target, lines + newline, 'utf8')
  return 'Appended ' + lines.length + ' characters to ' + await paths.relative(target) + '.'
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
    matches.push(await paths.relative(resolvedMatch))
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

    const relativePath = await paths.relative(file)
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

export async function editFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const oldString = optionalString(inputs, 'old_string') ?? ''
  const newString = optionalString(inputs, 'new_string') ?? ''
  const replaceAll = optionalBoolean(inputs, 'replace_all', false)
  const editMode = optionalString(inputs, 'edit_mode') ?? 'search_replace'
  if (editMode !== 'search_replace' && editMode !== 'whole_file') {
    throw new ValidationError('edit_mode', 'must be search_replace or whole_file', editMode)
  }

  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)
  const content = await Bun.file(target).text()
  if (editMode === 'whole_file') {
    if (!newString) {
      throw new ValidationError('new_string', 'must not be empty in whole_file mode', newString)
    }
    await Bun.write(target, newString)
    return `Replaced entire file ${await paths.relative(target)}.`
  }
  if (!oldString) {
    throw new ValidationError('old_string', 'must not be empty in search_replace mode', oldString)
  }
  if (oldString === newString) {
    throw new ValidationError('new_string', 'must differ from old_string', newString)
  }

  const occurrences = countOccurrences(content, oldString)
  if (occurrences === 0) {
    throw new ValidationError('old_string', 'was not found exactly; re-read the file before retrying', oldString)
  }
  if (occurrences > 1 && !replaceAll) {
    throw new ValidationError(
      'old_string',
      `appears ${occurrences} times; provide more context or set replace_all=true`,
      oldString,
    )
  }
  const next = replaceAll ? content.split(oldString).join(newString) : content.replace(oldString, newString)
  await Bun.write(target, next)
  return `Applied ${replaceAll ? occurrences : 1} replacement(s) to ${await paths.relative(target)}.`
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

function truncateCharacters(text: string, maxChars: number): string {
  if (maxChars === -1 || text.length <= maxChars) {
    return text
  }
  return `${text.slice(0, maxChars)}\n\n…[truncated by max_chars]…`
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
