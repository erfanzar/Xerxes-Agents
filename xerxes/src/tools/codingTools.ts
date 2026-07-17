// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cp, lstat, mkdir, readdir, rename, rm, stat, unlink } from 'node:fs/promises'
import { basename, dirname, extname, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import {
  optionalBoolean,
  optionalInteger,
  optionalString,
  optionalStringArray,
  requireRange,
  requiredString,
} from './inputs.js'
import { WorkspacePathResolver } from './pathSafety.js'

export const DEFAULT_READ_LINE_LIMIT = 400
const DEFAULT_GIT_TIMEOUT = 30_000
const MAX_GIT_OUTPUT = 200_000
const MAX_DIRECTORY_DEPTH = 100
const MAX_DIRECTORY_RESULTS = 10_000
const MAX_READ_FILE_BYTES = 10_000_000
const MAX_DIFF_LINES = 1_000
const MAX_DIFF_BYTES = 500_000
const MAX_REGEX_SUBJECT_BYTES = 1_000_000

export const CODING_READ_FILE_DEFINITION = codingDefinition(
  'read_file',
  'Read a workspace file with one-indexed line numbers and bounded line ranges.',
  {
    file_path: { type: 'string', description: 'Workspace-relative file path.' },
    start_line: { type: 'integer', default: 1, description: 'First one-indexed line to include.' },
    end_line: {
      type: 'integer',
      description: 'Last inclusive line. Omit for a 400-line chunk; use -1 for a full read.',
    },
  },
  ['file_path'],
)

export const CODING_WRITE_FILE_DEFINITION = codingDefinition(
  'write_file',
  'Write UTF-8 text within the workspace and return a capped unified diff (skipped for oversized files).',
  {
    file_path: { type: 'string', description: 'Workspace-relative destination path.' },
    content: { type: 'string', description: 'Complete UTF-8 content to write.' },
    create_dirs: { type: 'boolean', default: true },
    overwrite: { type: 'boolean', default: false, description: 'Allow replacing an existing regular file.' },
  },
  ['file_path', 'content'],
)

export const LIST_DIRECTORY_DEFINITION = codingDefinition(
  'list_directory',
  'List workspace directory entries with a small glob pattern and bounded recursion.',
  {
    directory: { type: 'string', default: '.', description: 'Workspace-relative directory.' },
    pattern: { type: 'string', default: '*', description: 'Glob pattern such as *.ts or **/*.ts.' },
    recursive: { type: 'boolean', default: false },
    show_hidden: { type: 'boolean', default: false },
    max_depth: { type: 'integer', default: 3 },
  },
)

export const COPY_FILE_DEFINITION = codingDefinition(
  'copy_file',
  'Copy a workspace file or directory to another workspace path.',
  {
    source: { type: 'string', description: 'Workspace-relative source path.' },
    destination: { type: 'string', description: 'Workspace-relative destination path.' },
    overwrite: { type: 'boolean', default: false },
  },
  ['source', 'destination'],
)

export const MOVE_FILE_DEFINITION = codingDefinition(
  'move_file',
  'Move a workspace file or directory to another workspace path.',
  {
    source: { type: 'string', description: 'Workspace-relative source path.' },
    destination: { type: 'string', description: 'Workspace-relative destination path.' },
    overwrite: { type: 'boolean', default: false },
  },
  ['source', 'destination'],
)

export const DELETE_FILE_DEFINITION = codingDefinition(
  'delete_file',
  'Delete a workspace file or directory. Non-trivial directories require force=true.',
  {
    path: { type: 'string', description: 'Workspace-relative path to delete.' },
    force: { type: 'boolean', default: false },
  },
  ['path'],
)

export const GIT_STATUS_DEFINITION = codingDefinition(
  'git_status',
  'Show porcelain Git status for a workspace-contained repository.',
  { repo_path: { type: 'string', default: '.' } },
)

export const GIT_DIFF_DEFINITION = codingDefinition(
  'git_diff',
  'Show a bounded unified Git diff for a workspace-contained repository.',
  {
    repo_path: { type: 'string', default: '.' },
    file_path: { type: 'string', description: 'Optional file inside repo_path.' },
    staged: { type: 'boolean', default: false },
    context_lines: { type: 'integer', default: 3 },
  },
)

export const GIT_APPLY_PATCH_DEFINITION = codingDefinition(
  'git_apply_patch',
  'Check or apply a unified patch in a workspace-contained Git repository.',
  {
    patch_content: { type: 'string' },
    repo_path: { type: 'string', default: '.' },
    check_only: { type: 'boolean', default: false },
  },
  ['patch_content'],
)

export const GIT_LOG_DEFINITION = codingDefinition(
  'git_log',
  'Show bounded commit history for a workspace-contained Git repository.',
  {
    repo_path: { type: 'string', default: '.' },
    max_commits: { type: 'integer', default: 10 },
    oneline: { type: 'boolean', default: true },
    file_path: { type: 'string', description: 'Optional file inside repo_path.' },
  },
)

export const GIT_ADD_DEFINITION = codingDefinition(
  'git_add',
  'Stage named workspace files in a workspace-contained Git repository.',
  {
    files: { type: 'array', items: { type: 'string' }, minItems: 1 },
    repo_path: { type: 'string', default: '.' },
  },
  ['files'],
)

export const CREATE_DIFF_DEFINITION = codingDefinition(
  'create_diff',
  'Create a unified diff between two UTF-8 text values.',
  {
    original: { type: 'string' },
    modified: { type: 'string' },
    file_name: { type: 'string', default: 'file.txt' },
    from_file: { type: 'string', description: 'Optional old-side display filename.' },
    to_file: { type: 'string', description: 'Optional new-side display filename.' },
  },
  ['original', 'modified'],
)

export const APPLY_DIFF_DEFINITION = codingDefinition(
  'apply_diff',
  'Apply one unified diff to UTF-8 text, validating every hunk context line.',
  {
    original: { type: 'string' },
    diff: { type: 'string' },
  },
  ['original', 'diff'],
)

export const FIND_AND_REPLACE_DEFINITION = codingDefinition(
  'find_and_replace',
  'Replace literal text or JavaScript regular-expression matches in one workspace file.',
  {
    file_path: { type: 'string' },
    search: { type: 'string' },
    replace: { type: 'string' },
    regex: { type: 'boolean', default: false },
    case_sensitive: { type: 'boolean', default: true },
    backup: { type: 'boolean', default: true },
  },
  ['file_path', 'search', 'replace'],
)

export const ANALYZE_CODE_STRUCTURE_DEFINITION = codingDefinition(
  'analyze_code_structure',
  'Summarize code-file language, imports, classes, functions, comments, and blank lines.',
  { file_path: { type: 'string', description: 'Workspace-relative source file.' } },
  ['file_path'],
)

export const CODING_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  CODING_READ_FILE_DEFINITION,
  CODING_WRITE_FILE_DEFINITION,
  LIST_DIRECTORY_DEFINITION,
  COPY_FILE_DEFINITION,
  MOVE_FILE_DEFINITION,
  DELETE_FILE_DEFINITION,
  GIT_STATUS_DEFINITION,
  GIT_DIFF_DEFINITION,
  GIT_APPLY_PATCH_DEFINITION,
  GIT_LOG_DEFINITION,
  GIT_ADD_DEFINITION,
  CREATE_DIFF_DEFINITION,
  APPLY_DIFF_DEFINITION,
  FIND_AND_REPLACE_DEFINITION,
  ANALYZE_CODE_STRUCTURE_DEFINITION,
]

/** Register the lower-case compatibility surface used by the universal coding agent. */
export function registerCodingTools(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(CODING_READ_FILE_DEFINITION, inputs => readFile(inputs, paths))
  registry.register(CODING_WRITE_FILE_DEFINITION, inputs => writeFile(inputs, paths))
  registry.register(LIST_DIRECTORY_DEFINITION, inputs => listDirectory(inputs, paths))
  registry.register(COPY_FILE_DEFINITION, inputs => copyFile(inputs, paths))
  registry.register(MOVE_FILE_DEFINITION, inputs => moveFile(inputs, paths))
  registry.register(DELETE_FILE_DEFINITION, inputs => deleteFile(inputs, paths))
  registry.register(GIT_STATUS_DEFINITION, inputs => gitStatus(inputs, paths))
  registry.register(GIT_DIFF_DEFINITION, inputs => gitDiff(inputs, paths))
  registry.register(GIT_APPLY_PATCH_DEFINITION, inputs => gitApplyPatch(inputs, paths))
  registry.register(GIT_LOG_DEFINITION, inputs => gitLog(inputs, paths))
  registry.register(GIT_ADD_DEFINITION, inputs => gitAdd(inputs, paths))
  registry.register(CREATE_DIFF_DEFINITION, createDiff)
  registry.register(APPLY_DIFF_DEFINITION, applyDiff)
  registry.register(FIND_AND_REPLACE_DEFINITION, inputs => findAndReplace(inputs, paths))
  registry.register(ANALYZE_CODE_STRUCTURE_DEFINITION, inputs => analyzeCodeStructure(inputs, paths))
}

function codingDefinition(
  name: string,
  description: string,
  properties: Record<string, JsonObject>,
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length > 0 ? { required: [...required] } : {}),
      },
    },
  }
}

/** Read a line-numbered, bounded chunk of one regular file. */
export async function readFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const startLine = Math.max(1, optionalNullableInteger(inputs, 'start_line', 1))
  const requestedEnd = optionalNullableInteger(inputs, 'end_line', Number.NaN)
  if (!Number.isNaN(requestedEnd) && requestedEnd !== -1 && requestedEnd < startLine) {
    throw new ValidationError('end_line', 'must be -1 or at least start_line', requestedEnd)
  }

  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)
  const fileInfo = await stat(target)
  if (fileInfo.size > MAX_READ_FILE_BYTES) {
    throw new ValidationError(
      'file_path',
      'is ' + fileInfo.size + ' bytes, exceeding the ' + MAX_READ_FILE_BYTES
        + '-byte read_file limit; search it with grep or split it into smaller files first',
      filePath,
    )
  }
  const text = await Bun.file(target).text()
  const lines = splitTextLines(text)
  const fullFile = requestedEnd === -1
  const endLine = fullFile
    ? lines.length
    : Number.isNaN(requestedEnd)
      ? Math.min(startLine + DEFAULT_READ_LINE_LIMIT - 1, lines.length)
      : Math.min(requestedEnd, lines.length)
  const output = lines
    .slice(startLine - 1, endLine)
    .map((line, index) => String(startLine + index).padStart(6) + ' | ' + line.text)
    .join('\n')
  if (!output) {
    return 'No content in specified range'
  }
  if (!fullFile && endLine < lines.length) {
    return output + '\n\n[read_file] Showing lines ' + startLine + '-' + endLine + ' of ' + lines.length + '. '
      + 'Continue with start_line=' + (endLine + 1) + '. '
      + 'Pass end_line=-1 only when the whole file is intentionally required.'
  }
  return output
}

/** Write UTF-8 text, preserving Python compatibility while constraining writes to the workspace. */
export async function writeFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const content = requiredText(inputs, 'content')
  const createDirs = optionalBoolean(inputs, 'create_dirs', true)
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const target = await paths.resolve(filePath)
  const exists = await pathExists(target)
  if (exists && !overwrite) {
    throw new ValidationError('file_path', 'already exists; pass overwrite=true to replace it', filePath)
  }
  if (exists) {
    await requireRegularFile(target, filePath)
  }
  if (createDirs) {
    await mkdir(dirname(target), { recursive: true })
  } else if (!(await isDirectory(dirname(target)))) {
    throw new ValidationError('file_path', 'parent directory does not exist and create_dirs is false', filePath)
  }

  const previous = exists ? await Bun.file(target).text() : ''
  await Bun.write(target, content)
  const relativePath = await paths.relative(target)
  const lineCount = content.length === 0 ? 0 : content.split('\n').length
  const summary = 'Successfully wrote ' + content.length + ' characters (' + lineCount + ' lines) to ' + relativePath
  if (previous === content) {
    return summary
  }
  if (exceedsDiffLimits(previous, content)) {
    // The write already succeeded; only the best-effort diff preview is skipped so the
    // quadratic Myers trace can never block or exhaust the single-threaded daemon.
    return summary + ' (diff skipped: input exceeds the ' + MAX_DIFF_LINES + '-line or '
      + MAX_DIFF_BYTES + '-byte diff limit)'
  }
  const diff = createUnifiedDiff(previous, content, basename(target), basename(target))
  const lines = diff.split('\n')
  const capped = lines.length > 60
    ? lines.slice(0, 60).join('\n') + '\n... (' + (lines.length - 60) + ' more lines)'
    : diff
  return summary + ':\n\n' + capped
}

/** List workspace entries without following symlinked directories. */
export async function listDirectory(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const directory = optionalString(inputs, 'directory') ?? '.'
  const pattern = optionalString(inputs, 'pattern') ?? '*'
  const recursive = optionalBoolean(inputs, 'recursive', false)
  const showHidden = optionalBoolean(inputs, 'show_hidden', false)
  const maxDepth = requireRange(optionalInteger(inputs, 'max_depth', 3), 'max_depth', 0, MAX_DIRECTORY_DEPTH)
  const root = await paths.resolve(directory)
  if (!(await isDirectory(root))) {
    throw new ValidationError('directory', 'must refer to an existing directory', directory)
  }

  const entries: string[] = []
  const matcher = globMatcher(pattern)
  let truncated = false
  const collect = async (current: string, depth: number, relativePath: string): Promise<void> => {
    const children = await readdir(current, { withFileTypes: true })
    children.sort((left, right) => left.name.localeCompare(right.name))
    for (const child of children) {
      if (entries.length >= MAX_DIRECTORY_RESULTS) {
        truncated = true
        return
      }
      if (!showHidden && child.name.startsWith('.')) {
        continue
      }
      const childRelative = relativePath ? relativePath + '/' + child.name : child.name
      const visible = recursive ? childRelative : child.name
      const childAbsolute = resolve(current, child.name)
      if (child.isDirectory()) {
        if (matcher(visible)) {
          entries.push('  '.repeat(depth) + '📁 ' + childRelative + '/')
        }
        if (recursive && depth < maxDepth) {
          await collect(childAbsolute, depth + 1, childRelative)
        }
        continue
      }
      if (!matcher(visible)) {
        continue
      }
      try {
        const safe = await paths.resolve(childAbsolute)
        const fileInfo = await stat(safe)
        entries.push('  '.repeat(depth) + '📄 ' + childRelative + ' (' + formatSize(fileInfo.size) + ')')
      } catch (error) {
        entries.push('  '.repeat(depth) + '❌ ' + childRelative + ': ' + errorMessage(error))
      }
    }
  }

  await collect(root, 0, '')
  if (entries.length === 0) {
    return "No items matching pattern '" + pattern + "' in " + directory
  }
  const header = 'Directory listing for: ' + root + '\n'
    + 'Pattern: ' + pattern + ' | Recursive: ' + recursive + ' | Hidden: ' + showHidden + '\n'
    + '-'.repeat(60) + '\n'
  return header + entries.join('\n') + (truncated ? '\n… (' + MAX_DIRECTORY_RESULTS + ' result limit reached)' : '')
}

/** Copy one workspace-contained file tree to another workspace-contained target. */
export async function copyFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const source = requiredString(inputs, 'source')
  const destination = requiredString(inputs, 'destination')
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const sourcePath = await paths.resolve(source)
  const destinationPath = await paths.resolve(destination)
  const sourceInfo = await requireExistingPath(sourcePath, source)
  if (sourcePath === destinationPath) {
    throw new ValidationError('destination', 'must differ from source', destination)
  }
  if ((await pathExists(destinationPath)) && !overwrite) {
    throw new ValidationError('destination', 'already exists; pass overwrite=true to replace it', destination)
  }
  await mkdir(dirname(destinationPath), { recursive: true })
  await cp(sourcePath, destinationPath, {
    dereference: false,
    errorOnExist: !overwrite,
    force: overwrite,
    preserveTimestamps: true,
    recursive: sourceInfo.isDirectory(),
  })
  const kind = sourceInfo.isDirectory() ? 'directory' : 'file'
  return 'Successfully copied ' + kind + ' ' + (await paths.relative(sourcePath))
    + ' to ' + (await paths.relative(destinationPath))
}

/** Move one workspace-contained file tree to another workspace-contained target. */
export async function moveFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const source = requiredString(inputs, 'source')
  const destination = requiredString(inputs, 'destination')
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const sourcePath = await paths.resolve(source)
  const destinationPath = await paths.resolve(destination)
  const sourceInfo = await requireExistingPath(sourcePath, source)
  if (sourcePath === destinationPath) {
    throw new ValidationError('destination', 'must differ from source', destination)
  }
  if (await pathExists(destinationPath)) {
    if (!overwrite) {
      throw new ValidationError('destination', 'already exists; pass overwrite=true to replace it', destination)
    }
    await rm(destinationPath, { force: true, recursive: true })
  }
  await mkdir(dirname(destinationPath), { recursive: true })
  try {
    await rename(sourcePath, destinationPath)
  } catch (error) {
    if (!isCrossDevice(error)) {
      throw error
    }
    await cp(sourcePath, destinationPath, {
      dereference: false,
      errorOnExist: true,
      force: false,
      preserveTimestamps: true,
      recursive: sourceInfo.isDirectory(),
    })
    await rm(sourcePath, { force: true, recursive: sourceInfo.isDirectory() })
  }
  return 'Successfully moved ' + (await paths.relative(sourcePath)) + ' to ' + (await paths.relative(destinationPath))
}

/** Delete a workspace-contained path while protecting the workspace root and large directories by default. */
export async function deleteFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const candidate = requiredString(inputs, 'path')
  const force = optionalBoolean(inputs, 'force', false)
  const target = await paths.resolve(candidate)
  const relativePath = await paths.relative(target)
  if (relativePath === '.') {
    throw new ValidationError('path', 'refuses to delete the workspace root', candidate)
  }
  const targetInfo = await requireExistingPath(target, candidate)
  if (targetInfo.isDirectory() && !force) {
    const count = await countDescendants(target, 11)
    if (count > 10) {
      throw new ValidationError(
        'path',
        'directory contains ' + count + '+ items; pass force=true to delete it',
        candidate,
      )
    }
  }
  if (targetInfo.isDirectory()) {
    await rm(target, { force: false, recursive: true })
    return 'Successfully deleted directory: ' + relativePath
  }
  await unlink(target)
  return 'Successfully deleted file: ' + relativePath
}

/** Return a human-readable porcelain Git status without allowing repository escapes. */
export async function gitStatus(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const repo = await resolveRepository(inputs, paths)
  const output = await runGit(repo, ['status', '--porcelain=v1', '-b'])
  const lines = output.trimEnd().split('\n').filter(Boolean)
  if (lines.length === 0) {
    return 'Working directory clean'
  }
  const result: string[] = []
  for (const line of lines) {
    if (line.startsWith('##')) {
      result.push('Branch: ' + line.slice(3))
      continue
    }
    const status = line.slice(0, 2)
    const path = line.slice(3)
    result.push('  ' + describeGitStatus(status) + ': ' + path)
  }
  return result.join('\n')
}

/** Return a Git diff from a workspace-contained repository. */
export async function gitDiff(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const repo = await resolveRepository(inputs, paths)
  const staged = optionalBoolean(inputs, 'staged', false)
  const contextLines = requireRange(optionalInteger(inputs, 'context_lines', 3), 'context_lines', 0, 100)
  const arguments_ = ['diff', '-U' + contextLines]
  if (staged) {
    arguments_.push('--staged')
  }
  const filePath = optionalString(inputs, 'file_path')
  if (filePath !== undefined) {
    arguments_.push('--', await repositoryRelativePath(filePath, repo, paths))
  }
  const output = await runGit(repo, arguments_)
  return output || 'No changes detected'
}

/** Check or apply a unified patch through direct Git argv execution. */
export async function gitApplyPatch(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const patch = requiredText(inputs, 'patch_content')
  const repo = await resolveRepository(inputs, paths)
  const checkOnly = optionalBoolean(inputs, 'check_only', false)
  const arguments_ = ['apply']
  if (checkOnly) {
    arguments_.push('--check')
  }
  await runGit(repo, arguments_, patch)
  return checkOnly ? 'Patch can be applied cleanly' : 'Patch applied successfully'
}

/** Return bounded Git history from a workspace-contained repository. */
export async function gitLog(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const repo = await resolveRepository(inputs, paths)
  const maxCommits = requireRange(optionalInteger(inputs, 'max_commits', 10), 'max_commits', 1, 100)
  const oneline = optionalBoolean(inputs, 'oneline', true)
  const arguments_ = ['log', '-' + maxCommits]
  arguments_.push(oneline ? '--oneline' : '--pretty=format:%h - %an, %ar : %s')
  const filePath = optionalString(inputs, 'file_path')
  if (filePath !== undefined) {
    arguments_.push('--', await repositoryRelativePath(filePath, repo, paths))
  }
  const output = await runGit(repo, arguments_)
  return output || 'No commit history found'
}

/** Stage explicit workspace-contained files in a workspace-contained repository. */
export async function gitAdd(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const files = optionalStringArray(inputs, 'files')
  if (files.length === 0) {
    throw new ValidationError('files', 'must include at least one path', files)
  }
  const repo = await resolveRepository(inputs, paths)
  const relativeFiles = await Promise.all(files.map(filePath => repositoryRelativePath(filePath, repo, paths)))
  await runGit(repo, ['add', '--', ...relativeFiles])
  return 'Successfully staged ' + files.length + ' file(s)'
}

async function resolveRepository(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const repoPath = optionalString(inputs, 'repo_path') ?? '.'
  const repo = await paths.resolve(repoPath)
  if (!(await isDirectory(repo))) {
    throw new ValidationError('repo_path', 'must refer to an existing workspace directory', repoPath)
  }
  return repo
}

async function repositoryRelativePath(
  candidate: string,
  repository: string,
  paths: WorkspacePathResolver,
): Promise<string> {
  const target = await paths.resolve(candidate)
  const result = relative(repository, target)
  if (!isWithin(repository, target)) {
    throw new ValidationError('file_path', 'must be inside repo_path', candidate)
  }
  return result || '.'
}

async function runGit(repository: string, arguments_: readonly string[], input?: string): Promise<string> {
  let child: Bun.PipedSubprocess
  try {
    child = Bun.spawn(['git', ...arguments_], {
      cwd: repository,
      stdin: 'pipe',
      stderr: 'pipe',
      stdout: 'pipe',
      maxBuffer: MAX_GIT_OUTPUT * 4,
    })
  } catch (error) {
    throw new ValidationError('git', errorMessage(error))
  }
  child.stdin.write(input ?? '')
  child.stdin.end()

  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill()
  }, DEFAULT_GIT_TIMEOUT)
  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ])
    if (timedOut) {
      throw new ValidationError('git', 'command timed out after ' + DEFAULT_GIT_TIMEOUT + 'ms')
    }
    if (exitCode !== 0) {
      throw new ValidationError('git', stderr.trim() || 'git exited with code ' + exitCode)
    }
    if (stdout.length > MAX_GIT_OUTPUT) {
      return stdout.slice(0, MAX_GIT_OUTPUT) + '\n…[truncated]…'
    }
    return stdout
  } finally {
    clearTimeout(timer)
  }
}

function describeGitStatus(status: string): string {
  const descriptions: Record<string, string> = {
    ' M': 'Modified (unstaged)',
    'A ': 'Added',
    'D ': 'Deleted',
    'M ': 'Modified (staged)',
    '??': 'Untracked',
    '!!': 'Ignored',
    'C ': 'Copied',
    'MM': 'Modified (staged and unstaged)',
    'R ': 'Renamed',
  }
  return descriptions[status] ?? status
}

/** Create a complete unified diff suitable for applyDiff or git apply. */
export function createDiff(inputs: JsonObject): string {
  const original = requiredText(inputs, 'original')
  const modified = requiredText(inputs, 'modified')
  const fileName = optionalString(inputs, 'file_name') || 'file.txt'
  const fromFile = optionalString(inputs, 'from_file') || fileName
  const toFile = optionalString(inputs, 'to_file') || fileName
  return createUnifiedDiff(original, modified, fromFile, toFile)
}

/** Apply one unified diff while checking every hunk line against the supplied original text. */
export function applyDiff(inputs: JsonObject): string {
  const original = requiredText(inputs, 'original')
  const diff = requiredText(inputs, 'diff')
  const source = splitTextLines(original)
  const hunks = parseUnifiedDiff(diff)
  const output: SourceLine[] = []
  let sourceIndex = 0

  for (const hunk of hunks) {
    const hunkStart = hunk.oldStart === 0 ? 0 : hunk.oldStart - 1
    if (hunkStart < sourceIndex || hunkStart > source.length) {
      throw new ValidationError('diff', 'contains overlapping or out-of-range hunks')
    }
    output.push(...source.slice(sourceIndex, hunkStart))
    sourceIndex = hunkStart
    let oldCount = 0
    let newCount = 0
    for (const line of hunk.lines) {
      if (line.prefix === '+') {
        output.push({ hasNewline: line.hasNewline, text: line.text })
        newCount += 1
        continue
      }
      const sourceLine = source[sourceIndex]
      if (!sourceLine || sourceLine.text !== line.text || sourceLine.hasNewline !== line.hasNewline) {
        throw new ValidationError('diff', 'hunk context does not match original content')
      }
      sourceIndex += 1
      oldCount += 1
      if (line.prefix === ' ') {
        output.push(sourceLine)
        newCount += 1
      }
    }
    if (oldCount !== hunk.oldCount || newCount !== hunk.newCount) {
      throw new ValidationError('diff', 'hunk line counts do not match its header')
    }
  }
  output.push(...source.slice(sourceIndex))
  return joinSourceLines(output)
}

/**
 * The Myers trace keeps one frontier snapshot per edit-distance step, so memory grows
 * quadratically with the line counts; cap each side before building an in-memory diff.
 */
function exceedsDiffLimits(original: string, modified: string): boolean {
  if (original.length > MAX_DIFF_BYTES || modified.length > MAX_DIFF_BYTES) {
    return true
  }
  const originalLines = original.length === 0 ? 0 : original.split('\n').length
  const modifiedLines = modified.length === 0 ? 0 : modified.split('\n').length
  return originalLines > MAX_DIFF_LINES || modifiedLines > MAX_DIFF_LINES
}

export function createUnifiedDiff(original: string, modified: string, fromFile = 'file.txt', toFile = fromFile): string {
  if (exceedsDiffLimits(original, modified)) {
    throw new ValidationError(
      'diff',
      'inputs exceed the ' + MAX_DIFF_LINES + '-line or ' + MAX_DIFF_BYTES
        + '-byte diff limit; refusing to build an in-memory line diff',
    )
  }
  const before = splitTextLines(original)
  const after = splitTextLines(modified)
  const operations = myersDiff(before, after)
  if (operations.every(operation => operation.kind === ' ')) {
    return ''
  }

  const header = '--- a/' + normalizeDiffName(fromFile) + '\n'
    + '+++ b/' + normalizeDiffName(toFile) + '\n'
  const body: string[] = []
  for (const hunk of createDiffHunks(operations)) {
    body.push('@@ -' + formatRange(hunk.oldStart, hunk.oldCount)
      + ' +' + formatRange(hunk.newStart, hunk.newCount) + ' @@\n')
    for (const operation of hunk.operations) {
      body.push(operation.kind + operation.line.text + '\n')
      if (!operation.line.hasNewline) {
        body.push('\\ No newline at end of file\n')
      }
    }
  }
  return header + body.join('')
}

interface SourceLine {
  readonly hasNewline: boolean
  readonly text: string
}

interface DiffOperation {
  readonly kind: ' ' | '+' | '-'
  readonly line: SourceLine
}

interface ParsedPatchLine {
  hasNewline: boolean
  readonly prefix: ' ' | '+' | '-'
  readonly text: string
}

interface AnnotatedOperation {
  readonly newLine: number
  readonly oldLine: number
  readonly operation: DiffOperation
}

interface DiffHunk {
  readonly newCount: number
  readonly newStart: number
  readonly oldCount: number
  readonly oldStart: number
  readonly operations: readonly DiffOperation[]
}

interface ParsedHunk {
  readonly lines: ParsedPatchLine[]
  readonly newCount: number
  readonly oldCount: number
  readonly oldStart: number
}

function splitTextLines(value: string): SourceLine[] {
  if (value.length === 0) {
    return []
  }
  const parts = value.split('\n')
  const endsWithNewline = value.endsWith('\n')
  if (endsWithNewline) {
    parts.pop()
  }
  return parts.map((text, index) => ({
    hasNewline: endsWithNewline || index < parts.length - 1,
    text,
  }))
}

function joinSourceLines(lines: readonly SourceLine[]): string {
  return lines.map(line => line.text + (line.hasNewline ? '\n' : '')).join('')
}

function myersDiff(before: readonly SourceLine[], after: readonly SourceLine[]): DiffOperation[] {
  const maxDistance = before.length + after.length
  const trace: Map<number, number>[] = []
  let frontier = new Map<number, number>([[1, 0]])

  for (let distance = 0; distance <= maxDistance; distance += 1) {
    trace.push(new Map(frontier))
    for (let diagonal = -distance; diagonal <= distance; diagonal += 2) {
      const next = frontier.get(diagonal + 1) ?? Number.NEGATIVE_INFINITY
      const previous = frontier.get(diagonal - 1) ?? Number.NEGATIVE_INFINITY
      let beforeIndex = diagonal === -distance || (diagonal !== distance && previous < next)
        ? (frontier.get(diagonal + 1) ?? 0)
        : (frontier.get(diagonal - 1) ?? 0) + 1
      let afterIndex = beforeIndex - diagonal
      while (
        beforeIndex < before.length
        && afterIndex < after.length
        && sameLine(before[beforeIndex], after[afterIndex])
      ) {
        beforeIndex += 1
        afterIndex += 1
      }
      frontier.set(diagonal, beforeIndex)
      if (beforeIndex >= before.length && afterIndex >= after.length) {
        return backtrackMyers(trace, before, after)
      }
    }
  }
  throw new ValidationError('diff', 'could not construct a line diff')
}

function backtrackMyers(
  trace: readonly Map<number, number>[],
  before: readonly SourceLine[],
  after: readonly SourceLine[],
): DiffOperation[] {
  let beforeIndex = before.length
  let afterIndex = after.length
  const output: DiffOperation[] = []
  for (let distance = trace.length - 1; distance > 0; distance -= 1) {
    const frontier = trace[distance]
    if (!frontier) {
      throw new ValidationError('diff', 'diff trace is incomplete')
    }
    const diagonal = beforeIndex - afterIndex
    const previous = frontier.get(diagonal - 1) ?? Number.NEGATIVE_INFINITY
    const next = frontier.get(diagonal + 1) ?? Number.NEGATIVE_INFINITY
    const previousDiagonal = diagonal === -distance || (diagonal !== distance && previous < next)
      ? diagonal + 1
      : diagonal - 1
    const previousBeforeIndex = frontier.get(previousDiagonal) ?? 0
    const previousAfterIndex = previousBeforeIndex - previousDiagonal
    while (beforeIndex > previousBeforeIndex && afterIndex > previousAfterIndex) {
      const line = before[beforeIndex - 1]
      if (!line) {
        throw new ValidationError('diff', 'diff trace referenced a missing source line')
      }
      output.push({ kind: ' ', line })
      beforeIndex -= 1
      afterIndex -= 1
    }
    if (beforeIndex === previousBeforeIndex) {
      const line = after[afterIndex - 1]
      if (!line) {
        throw new ValidationError('diff', 'diff trace referenced a missing modified line')
      }
      output.push({ kind: '+', line })
      afterIndex -= 1
    } else {
      const line = before[beforeIndex - 1]
      if (!line) {
        throw new ValidationError('diff', 'diff trace referenced a missing source line')
      }
      output.push({ kind: '-', line })
      beforeIndex -= 1
    }
  }
  while (beforeIndex > 0 && afterIndex > 0) {
    const line = before[beforeIndex - 1]
    if (!line) {
      throw new ValidationError('diff', 'diff trace referenced a missing source line')
    }
    output.push({ kind: ' ', line })
    beforeIndex -= 1
    afterIndex -= 1
  }
  while (beforeIndex > 0) {
    const line = before[beforeIndex - 1]
    if (!line) {
      throw new ValidationError('diff', 'diff trace referenced a missing source line')
    }
    output.push({ kind: '-', line })
    beforeIndex -= 1
  }
  while (afterIndex > 0) {
    const line = after[afterIndex - 1]
    if (!line) {
      throw new ValidationError('diff', 'diff trace referenced a missing modified line')
    }
    output.push({ kind: '+', line })
    afterIndex -= 1
  }
  return output.reverse()
}

function parseUnifiedDiff(value: string): ParsedHunk[] {
  const rows = value.split('\n')
  const hunks: ParsedHunk[] = []
  let index = 0
  let sawHunk = false
  while (index < rows.length) {
    const row = rows[index] ?? ''
    if (!row.startsWith('@@')) {
      if (sawHunk && (row.startsWith('diff --git ') || row.startsWith('--- ') || row.startsWith('+++ '))) {
        throw new ValidationError('diff', 'must contain only one file')
      }
      index += 1
      continue
    }
    sawHunk = true
    const header = row.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/)
    if (!header) {
      throw new ValidationError('diff', 'contains an invalid unified-diff hunk header', row)
    }
    const oldStart = Number(header[1])
    const oldCount = header[2] === undefined ? 1 : Number(header[2])
    const newCount = header[4] === undefined ? 1 : Number(header[4])
    const lines: ParsedPatchLine[] = []
    index += 1
    while (index < rows.length) {
      const body = rows[index] ?? ''
      if (body.startsWith('@@')) {
        break
      }
      if (body === '' && index === rows.length - 1) {
        index += 1
        break
      }
      if (body === '\\ No newline at end of file') {
        const previous = lines[lines.length - 1]
        if (!previous) {
          throw new ValidationError('diff', 'has a newline marker without a preceding patch line')
        }
        previous.hasNewline = false
        index += 1
        continue
      }
      const prefix = body.charAt(0)
      if (prefix !== ' ' && prefix !== '+' && prefix !== '-') {
        throw new ValidationError('diff', 'contains an invalid unified-diff body line', body)
      }
      lines.push({ hasNewline: true, prefix, text: body.slice(1) })
      index += 1
    }
    hunks.push({ lines, newCount, oldCount, oldStart })
  }
  if (hunks.length === 0) {
    throw new ValidationError('diff', 'does not contain a unified-diff hunk')
  }
  return hunks
}

function sameLine(left: SourceLine | undefined, right: SourceLine | undefined): boolean {
  return left !== undefined && right !== undefined && left.text === right.text && left.hasNewline === right.hasNewline
}

function formatRange(start: number, count: number): string {
  if (count === 0) {
    return Math.max(0, start) + ',0'
  }
  return count === 1 ? String(start) : start + ',' + count
}

function normalizeDiffName(value: string): string {
  const normalized = value.replaceAll('\\', '/').replace(/^\/+/, '')
  return normalized || 'file.txt'
}

function createDiffHunks(operations: readonly DiffOperation[], context = 3): DiffHunk[] {
  const annotated = annotateOperations(operations)
  const changed = annotated
    .map((value, index) => value.operation.kind === ' ' ? -1 : index)
    .filter(index => index >= 0)
  if (changed.length === 0) {
    return []
  }

  const ranges: Array<{ end: number; start: number }> = []
  for (const index of changed) {
    const start = Math.max(0, index - context)
    const end = Math.min(annotated.length, index + context + 1)
    const latest = ranges[ranges.length - 1]
    if (latest && start <= latest.end) {
      latest.end = Math.max(latest.end, end)
    } else {
      ranges.push({ end, start })
    }
  }

  return ranges.map(range => {
    const members = annotated.slice(range.start, range.end)
    const first = members[0]
    if (!first) {
      throw new ValidationError('diff', 'could not construct a unified-diff hunk')
    }
    const hunkOperations = members.map(member => member.operation)
    const oldCount = hunkOperations.filter(operation => operation.kind !== '+').length
    const newCount = hunkOperations.filter(operation => operation.kind !== '-').length
    return {
      newCount,
      newStart: newCount === 0 ? first.newLine - 1 : first.newLine,
      oldCount,
      oldStart: oldCount === 0 ? first.oldLine - 1 : first.oldLine,
      operations: hunkOperations,
    }
  })
}

function annotateOperations(operations: readonly DiffOperation[]): AnnotatedOperation[] {
  let oldLine = 1
  let newLine = 1
  return operations.map(operation => {
    const annotated = { newLine, oldLine, operation }
    if (operation.kind !== '+') {
      oldLine += 1
    }
    if (operation.kind !== '-') {
      newLine += 1
    }
    return annotated
  })
}

/** Find and replace text in one workspace file, optionally preserving a sibling backup. */
export async function findAndReplace(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const search = requiredString(inputs, 'search')
  const replacement = requiredText(inputs, 'replace')
  const regex = optionalBoolean(inputs, 'regex', false)
  const caseSensitive = optionalBoolean(inputs, 'case_sensitive', true)
  const backup = optionalBoolean(inputs, 'backup', true)
  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)

  // Validate the model-supplied regex before any workspace mutation, including the backup write.
  let pattern: RegExp | undefined
  if (regex) {
    try {
      pattern = new RegExp(search, caseSensitive ? 'g' : 'gi')
    } catch (error) {
      throw new ValidationError(
        'search',
        'must be a valid JavaScript regular expression: ' + errorMessage(error),
        search,
      )
    }
  }

  const content = await Bun.file(target).text()
  if (pattern !== undefined && content.length > MAX_REGEX_SUBJECT_BYTES) {
    // A synchronous regex cannot be timed out; cap the subject size so catastrophic
    // backtracking cannot freeze the single-threaded daemon.
    throw new ValidationError(
      'file_path',
      'is ' + content.length + ' characters, exceeding the ' + MAX_REGEX_SUBJECT_BYTES
        + '-character regex subject limit; use literal mode or a smaller file',
      filePath,
    )
  }

  const backupPath = backup ? await paths.resolve(filePath + '.bak') : undefined
  if (backupPath !== undefined) {
    await Bun.write(backupPath, content)
  }

  let count: number
  let updated: string
  if (pattern !== undefined) {
    count = [...content.matchAll(pattern)].length
    // A function replacer keeps $-sequences in the replacement literal,
    // consistent with the literal search modes below.
    updated = content.replace(pattern, () => replacement)
  } else if (caseSensitive) {
    count = content.split(search).length - 1
    updated = content.replaceAll(search, () => replacement)
  } else {
    const literalPattern = new RegExp(escapeRegularExpression(search), 'gi')
    count = 0
    updated = content.replace(literalPattern, () => {
      count += 1
      return replacement
    })
  }
  if (count > 0) {
    await Bun.write(target, updated)
  }
  const backupMessage = backupPath === undefined ? '' : ' (backup saved as ' + basename(backupPath) + ')'
  return 'Replaced ' + count + ' occurrence(s) in ' + (await paths.relative(target)) + backupMessage
}

/** Analyze basic structural features of Python, JavaScript/TypeScript, Java, and common source files. */
export async function analyzeCodeStructure(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const target = await paths.resolve(filePath)
  await requireRegularFile(target, filePath)
  const text = await Bun.file(target).text()
  const lines = text.split('\n')
  if (text.endsWith('\n')) {
    lines.pop()
  }
  const language = detectLanguage(filePath)
  const analysis: CodeAnalysis = {
    blankLines: 0,
    classes: [],
    comments: 0,
    functions: [],
    imports: [],
  }
  if (language === 'Python') {
    analyzePython(lines, analysis)
  } else if (language === 'JavaScript' || language === 'TypeScript') {
    analyzeJavaScript(lines, analysis)
  } else if (language === 'Java') {
    analyzeJava(lines, analysis)
  } else {
    analyzeGeneric(lines, analysis)
  }

  const output = [
    'Code Structure Analysis: ' + basename(target),
    'Language: ' + language,
    'Total Lines: ' + lines.length,
    'Blank Lines: ' + analysis.blankLines,
    'Comment Lines: ' + analysis.comments,
    'Code Lines: ' + (lines.length - analysis.blankLines - analysis.comments),
  ]
  appendAnalysisSection(output, 'Imports', analysis.imports, 10)
  appendAnalysisSection(output, 'Classes', analysis.classes)
  appendAnalysisSection(output, 'Functions', analysis.functions, 20)
  return output.join('\n')
}

/** Detect a display language from an extension or filename. */
export function detectLanguage(fileNameOrExtension: string): string {
  const extension = fileNameOrExtension.startsWith('.') && !fileNameOrExtension.includes('/')
    ? fileNameOrExtension.toLowerCase()
    : extname(fileNameOrExtension).toLowerCase()
  const languages: Record<string, string> = {
    '.bash': 'Bash',
    '.c': 'C',
    '.cc': 'C++',
    '.cpp': 'C++',
    '.cs': 'C#',
    '.css': 'CSS',
    '.go': 'Go',
    '.h': 'C/C++ Header',
    '.html': 'HTML',
    '.java': 'Java',
    '.jl': 'Julia',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.json': 'JSON',
    '.kt': 'Kotlin',
    '.m': 'MATLAB',
    '.md': 'Markdown',
    '.php': 'PHP',
    '.py': 'Python',
    '.r': 'R',
    '.rb': 'Ruby',
    '.rs': 'Rust',
    '.scala': 'Scala',
    '.scss': 'SCSS',
    '.sh': 'Shell',
    '.sql': 'SQL',
    '.swift': 'Swift',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.xml': 'XML',
    '.yaml': 'YAML',
    '.yml': 'YAML',
  }
  return languages[extension] ?? 'Unknown'
}

interface CodeAnalysis {
  readonly classes: string[]
  readonly functions: string[]
  readonly imports: string[]
  blankLines: number
  comments: number
}

function analyzePython(lines: readonly string[], analysis: CodeAnalysis): void {
  let inDocstring: '"' | "'" | undefined
  for (const line of lines) {
    const stripped = line.trim()
    if (!stripped) {
      analysis.blankLines += 1
      continue
    }
    if (inDocstring !== undefined) {
      analysis.comments += 1
      if (stripped.includes(inDocstring.repeat(3))) {
        inDocstring = undefined
      }
      continue
    }
    if (stripped.startsWith('#')) {
      analysis.comments += 1
      continue
    }
    if (stripped.startsWith('"""') || stripped.startsWith("'''")) {
      analysis.comments += 1
      const quote = stripped.charAt(0) as '"' | "'"
      if (stripped.indexOf(quote.repeat(3), 3) < 0) {
        inDocstring = quote
      }
      continue
    }
    if (stripped.startsWith('import ') || stripped.startsWith('from ')) {
      analysis.imports.push(stripped)
      continue
    }
    const classMatch = stripped.match(/^class\s+([A-Za-z_]\w*)/)
    if (classMatch?.[1]) {
      analysis.classes.push(classMatch[1])
      continue
    }
    const functionMatch = stripped.match(/^(?:async\s+)?def\s+([A-Za-z_]\w*)/)
    if (functionMatch?.[1]) {
      analysis.functions.push(functionMatch[1])
    }
  }
}

function analyzeJavaScript(lines: readonly string[], analysis: CodeAnalysis): void {
  let inBlockComment = false
  for (const line of lines) {
    const stripped = line.trim()
    if (!stripped) {
      analysis.blankLines += 1
      continue
    }
    if (inBlockComment || stripped.startsWith('/*') || stripped.startsWith('*')) {
      analysis.comments += 1
      inBlockComment = !stripped.includes('*/')
      continue
    }
    if (stripped.startsWith('//')) {
      analysis.comments += 1
      continue
    }
    if (/^(?:import|export\s+[^=]+\s+from)\b/.test(stripped) || /\brequire\s*\(/.test(stripped)) {
      analysis.imports.push(stripped)
    }
    const classMatch = stripped.match(/(?:^|\s)class\s+([A-Za-z_$][\w$]*)/)
    if (classMatch?.[1]) {
      analysis.classes.push(classMatch[1])
    }
    const functionMatch = stripped.match(/(?:^|\s)(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/)
      ?? stripped.match(/(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/)
      ?? stripped.match(/(?:^|\s)([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{/)
    if (functionMatch?.[1] && !['catch', 'for', 'if', 'switch', 'while'].includes(functionMatch[1])) {
      analysis.functions.push(functionMatch[1])
    }
  }
}

function analyzeJava(lines: readonly string[], analysis: CodeAnalysis): void {
  let inBlockComment = false
  for (const line of lines) {
    const stripped = line.trim()
    if (!stripped) {
      analysis.blankLines += 1
      continue
    }
    if (inBlockComment || stripped.startsWith('/*') || stripped.startsWith('*')) {
      analysis.comments += 1
      inBlockComment = !stripped.includes('*/')
      continue
    }
    if (stripped.startsWith('//')) {
      analysis.comments += 1
      continue
    }
    if (stripped.startsWith('import ')) {
      analysis.imports.push(stripped)
    }
    const classMatch = stripped.match(/\bclass\s+([A-Za-z_$][\w$]*)/)
    if (classMatch?.[1]) {
      analysis.classes.push(classMatch[1])
    }
    const functionMatch = stripped.match(
      /(?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\], ?]+\s+([A-Za-z_$][\w$]*)\s*\(/,
    )
    if (functionMatch?.[1] && !['catch', 'for', 'if', 'switch', 'while'].includes(functionMatch[1])) {
      analysis.functions.push(functionMatch[1])
    }
  }
}

function analyzeGeneric(lines: readonly string[], analysis: CodeAnalysis): void {
  for (const line of lines) {
    const stripped = line.trim()
    if (!stripped) {
      analysis.blankLines += 1
    } else if (stripped.startsWith('#') || stripped.startsWith('//') || stripped.startsWith('/*')) {
      analysis.comments += 1
    }
  }
}

function appendAnalysisSection(
  output: string[],
  title: string,
  values: readonly string[],
  limit = Number.MAX_SAFE_INTEGER,
): void {
  if (values.length === 0) {
    return
  }
  output.push('\n' + title + ' (' + values.length + '):')
  for (const value of values.slice(0, limit)) {
    output.push('  • ' + value)
  }
}

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

function requiredText(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

async function requireRegularFile(path: string, input: string): Promise<void> {
  const info = await requireExistingPath(path, input)
  if (!info.isFile()) {
    throw new ValidationError('file_path', 'must refer to an existing regular file', input)
  }
}

async function requireExistingPath(path: string, input: string) {
  try {
    return await lstat(path)
  } catch (error) {
    if (isNotFound(error)) {
      throw new ValidationError('path', 'was not found in the workspace', input)
    }
    throw error
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path)
    return true
  } catch (error) {
    if (isNotFound(error)) {
      return false
    }
    throw error
  }
}

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch (error) {
    if (isNotFound(error)) {
      return false
    }
    throw error
  }
}

async function countDescendants(directory: string, maximum: number): Promise<number> {
  let count = 0
  const collect = async (current: string): Promise<void> => {
    if (count >= maximum) {
      return
    }
    const entries = await readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      count += 1
      if (count >= maximum) {
        return
      }
      if (entry.isDirectory()) {
        await collect(resolve(current, entry.name))
      }
    }
  }
  await collect(directory)
  return count
}

function globMatcher(pattern: string): (candidate: string) => boolean {
  const normalized = pattern.replaceAll('\\', '/')
  let source = '^'
  for (let index = 0; index < normalized.length; index += 1) {
    const character = normalized.charAt(index)
    if (character === '*') {
      if (normalized.charAt(index + 1) === '*') {
        index += 1
        if (normalized.charAt(index + 1) === '/') {
          index += 1
          source += '(?:.*/)?'
        } else {
          source += '.*'
        }
      } else {
        source += '[^/]*'
      }
      continue
    }
    if (character === '?') {
      source += '[^/]'
      continue
    }
    source += escapeRegularExpression(character)
  }
  const expression = new RegExp(source + '$')
  return candidate => expression.test(candidate.replaceAll('\\', '/'))
}

function escapeRegularExpression(value: string): string {
  const special = new Set(['\\', '^', '$', '.', '|', '?', '*', '+', '(', ')', '[', ']', '{', '}'])
  return [...value].map(character => special.has(character) ? '\\' + character : character).join('')
}

function formatSize(size: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let remaining = size
  for (const unit of units) {
    if (remaining < 1024 || unit === 'TB') {
      return remaining.toFixed(1) + unit
    }
    remaining /= 1024
  }
  return remaining.toFixed(1) + 'TB'
}

function isWithin(root: string, candidate: string): boolean {
  const fromRoot = relative(root, candidate)
  return fromRoot === ''
    || (!fromRoot.startsWith('..' + sep) && fromRoot !== '..' && !fromRoot.startsWith(sep))
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function isCrossDevice(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EXDEV'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
