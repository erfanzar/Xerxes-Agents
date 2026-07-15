// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import * as path from 'node:path'

export const AT_MENTION_TRIGGERS = ['@file:', '@folder:', '@diff', '@staged', '@git:', '@url:'] as const

const DEFAULT_GIT_REFS = ['HEAD', 'main', 'master', 'develop'] as const
const MENTION_TOKEN_RE = /@(?:file|folder|git|url):[^\s]+|@diff\b|@staged\b/g

export type AtMentionTrigger = (typeof AT_MENTION_TRIGGERS)[number]
export type AtMentionKind = 'diff' | 'file' | 'folder' | 'git' | 'staged' | 'unknown' | 'url'
export type AtMentionPathKind = 'directory' | 'file'
export type GitDiffMode = 'staged' | 'unstaged'

/** A parsed @ token active at a text cursor. */
export interface ParsedAtMention {
  readonly cursor: number
  readonly remainder: string
  readonly start: number
  readonly textBeforeToken: string
  readonly trigger: '@' | AtMentionTrigger
}

/** A completion replacement and the absolute text offset it replaces from. */
export interface AtMentionCompletion {
  readonly display: string
  readonly kind: 'file' | 'folder' | 'git' | 'trigger'
  readonly replacement: string
  readonly replaceStart: number
}

/** A canonical, inspected filesystem path supplied by the host. */
export interface AtMentionPathInfo {
  readonly kind: AtMentionPathKind
  /** Absolute, symlink-resolved path. */
  readonly path: string
}

/** Directory item supplied by the host filesystem port. */
export interface AtMentionDirectoryEntry {
  readonly name: string
  /** Absolute path to inspect before it can be surfaced to the user. */
  readonly path: string
}

/**
 * Filesystem capability required by file and folder mentions.
 *
 * `inspect` must return an absolute, symlink-resolved path. The module uses
 * that canonical path for every read and validates it remains under the
 * configured workspace root before surfacing any content.
 */
export interface AtMentionFileSystemPort {
  inspect(path: string): Promise<AtMentionPathInfo | undefined>
  readDirectory(path: string): Promise<readonly AtMentionDirectoryEntry[]>
  readTextFile(path: string): Promise<string>
}

/** Git capability required by diff, staged, and ref mentions. */
export interface AtMentionGitPort {
  diff(workspaceRoot: string, mode: GitDiffMode): Promise<string>
  logOne(workspaceRoot: string, ref: string): Promise<string>
  listRefs?(workspaceRoot: string): Promise<readonly string[]>
}

/** Explicit host capabilities and workspace boundary for @-mention resolution. */
export interface AtMentionOptions {
  readonly fileSystem: AtMentionFileSystemPort
  readonly git?: AtMentionGitPort
  readonly maxResults?: number
  readonly workspaceRoot: string
}

/** One @ token expanded into its referenced content or an explicit error. */
export interface ExpandedAtMention {
  readonly error: string
  readonly kind: AtMentionKind
  readonly payload: string
  readonly token: string
}

/**
 * Parse the active @ token before a cursor.
 *
 * Returns `undefined` when the cursor is not inside a whitespace-free mention
 * token. A bare `@` is retained so callers can offer all mention triggers.
 */
export function parseAtMentionToken(text: string, cursor = text.length): ParsedAtMention | undefined {
  if (!Number.isInteger(cursor) || cursor < 0 || cursor > text.length) {
    return undefined
  }
  const beforeCursor = text.slice(0, cursor)
  const start = beforeCursor.lastIndexOf('@')
  if (start < 0) {
    return undefined
  }
  const span = beforeCursor.slice(start)
  if (/\s/.test(span)) {
    return undefined
  }
  for (const trigger of AT_MENTION_TRIGGERS) {
    if (span.startsWith(trigger)) {
      return {
        cursor,
        trigger,
        remainder: span.slice(trigger.length),
        start,
        textBeforeToken: beforeCursor.slice(0, start)
      }
    }
  }
  return {
    cursor,
    trigger: '@',
    remainder: span.slice(1),
    start,
    textBeforeToken: beforeCursor.slice(0, start)
  }
}

/** Return expandable @ tokens in document order. */
export function atMentionTokens(text: string): string[] {
  return [...text.matchAll(MENTION_TOKEN_RE)].map(match => match[0])
}

/**
 * Complete the active @ token using only injected filesystem and git ports.
 *
 * Completion failures intentionally return no candidates, matching normal TUI
 * completion behavior; expansion APIs retain detailed errors for submission
 * flows that choose to consume them.
 */
export async function completeAtMention(
  text: string,
  options: AtMentionOptions,
  cursor = text.length
): Promise<AtMentionCompletion[]> {
  const parsed = parseAtMentionToken(text, cursor)
  if (!parsed) {
    return []
  }
  if (parsed.trigger === '@') {
    return AT_MENTION_TRIGGERS.map(trigger => ({
      kind: 'trigger' as const,
      display: trigger,
      replacement: trigger,
      replaceStart: parsed.start
    }))
  }
  if (parsed.trigger === '@file:' || parsed.trigger === '@folder:') {
    return completePathMention(parsed, options, parsed.trigger === '@file:' ? 'file' : 'directory')
  }
  if (parsed.trigger === '@git:') {
    return completeGitMention(parsed, options)
  }
  return []
}

/** Expand one supported @ token using only explicitly supplied ports. */
export async function expandAtMention(token: string, options: AtMentionOptions): Promise<ExpandedAtMention> {
  if (token === '@diff') {
    return expandGitMention(token, 'diff', 'unstaged', options)
  }
  if (token === '@staged') {
    return expandGitMention(token, 'staged', 'staged', options)
  }
  if (token.startsWith('@git:')) {
    return expandGitRef(token, token.slice('@git:'.length), options)
  }
  if (token.startsWith('@url:')) {
    return { token, kind: 'url', payload: token.slice('@url:'.length), error: '' }
  }
  if (token.startsWith('@file:')) {
    return expandPathMention(token, token.slice('@file:'.length), options)
  }
  if (token.startsWith('@folder:')) {
    return expandPathMention(token, token.slice('@folder:'.length), options)
  }
  return { token, kind: 'unknown', payload: '', error: 'unrecognized trigger' }
}

/** Expand every recognised @ token in document order. */
export async function expandAtMentionsInText(text: string, options: AtMentionOptions): Promise<ExpandedAtMention[]> {
  const expanded: ExpandedAtMention[] = []
  for (const token of atMentionTokens(text)) {
    expanded.push(await expandAtMention(token, options))
  }
  return expanded
}

async function completePathMention(
  parsed: ParsedAtMention,
  options: AtMentionOptions,
  wantedKind: AtMentionPathKind
): Promise<AtMentionCompletion[]> {
  try {
    const root = await workspaceRoot(options)
    const prefix = parsed.remainder
    const basePath = pathBasePath(root.path, prefix)
    if (!basePath) {
      return []
    }
    const base = await inspectedContainedPath(basePath, root, options.fileSystem)
    if (!base || base.kind !== 'directory') {
      return []
    }
    const stem = path.basename(prefix).toLowerCase()
    const maxResults = completionLimit(options.maxResults)
    const entries = await options.fileSystem.readDirectory(base.path)
    const results: AtMentionCompletion[] = []
    for (const entry of [...entries].filter(isSafeDirectoryEntry).sort(compareDirectoryEntries)) {
      if (stem && !entry.name.toLowerCase().startsWith(stem)) {
        continue
      }
      const info = await safelyInspectContained(entry.path, root, options.fileSystem)
      if (!info || info.kind !== wantedKind) {
        continue
      }
      const relative = relativeWorkspacePath(root.path, info.path)
      if (!relative) {
        continue
      }
      const isDirectory = info.kind === 'directory'
      const replacement = relative + (isDirectory ? '/' : '')
      results.push({
        kind: isDirectory ? 'folder' : 'file',
        display: entry.name + (isDirectory ? '/' : ''),
        replacement,
        replaceStart: parsed.start + parsed.trigger.length
      })
      if (results.length >= maxResults) {
        break
      }
    }
    return results
  } catch {
    return []
  }
}

async function completeGitMention(parsed: ParsedAtMention, options: AtMentionOptions): Promise<AtMentionCompletion[]> {
  const refs: string[] = [...DEFAULT_GIT_REFS]
  if (options.git?.listRefs) {
    try {
      const root = await workspaceRoot(options)
      refs.push(...(await options.git.listRefs(root.path)))
    } catch {
      // The static refs remain useful when a host cannot enumerate branches.
    }
  }
  const prefix = parsed.remainder.toLowerCase()
  const maxResults = completionLimit(options.maxResults)
  const seen = new Set<string>()
  const results: AtMentionCompletion[] = []
  for (const ref of refs) {
    if (!ref || seen.has(ref) || (prefix && !ref.toLowerCase().startsWith(prefix))) {
      continue
    }
    seen.add(ref)
    results.push({
      kind: 'git',
      display: ref,
      replacement: ref,
      replaceStart: parsed.start + parsed.trigger.length
    })
    if (results.length >= maxResults) {
      break
    }
  }
  return results
}

async function expandPathMention(
  token: string,
  rawPath: string,
  options: AtMentionOptions
): Promise<ExpandedAtMention> {
  const requestedKind: AtMentionKind = token.startsWith('@file:') ? 'file' : 'folder'
  try {
    const root = await workspaceRoot(options)
    const candidate = workspacePath(root.path, rawPath)
    if (!candidate) {
      return expansionError(token, requestedKind, 'escapes workspace root')
    }
    const target = await options.fileSystem.inspect(candidate)
    if (!target) {
      return expansionError(token, requestedKind, 'not found')
    }
    if (!path.isAbsolute(target.path)) {
      return expansionError(token, requestedKind, 'filesystem port returned a non-absolute path')
    }
    if (!isWithinWorkspace(root.path, target.path)) {
      return expansionError(token, requestedKind, 'escapes workspace root')
    }
    if (target.kind === 'file') {
      return { token, kind: 'file', payload: await options.fileSystem.readTextFile(target.path), error: '' }
    }
    const entries = await options.fileSystem.readDirectory(target.path)
    return {
      token,
      kind: 'folder',
      payload: entries
        .filter(isSafeDirectoryEntry)
        .map(entry => entry.name)
        .sort(compareStrings)
        .join('\n'),
      error: ''
    }
  } catch (error) {
    return expansionError(token, requestedKind, errorMessage(error))
  }
}

async function expandGitMention(
  token: string,
  kind: 'diff' | 'staged',
  mode: GitDiffMode,
  options: AtMentionOptions
): Promise<ExpandedAtMention> {
  if (!options.git) {
    return expansionError(token, kind, 'git port not configured')
  }
  try {
    const root = await workspaceRoot(options)
    return { token, kind, payload: await options.git.diff(root.path, mode), error: '' }
  } catch (error) {
    return expansionError(token, kind, errorMessage(error))
  }
}

async function expandGitRef(token: string, ref: string, options: AtMentionOptions): Promise<ExpandedAtMention> {
  if (!options.git) {
    return expansionError(token, 'git', 'git port not configured')
  }
  try {
    const root = await workspaceRoot(options)
    return { token, kind: 'git', payload: await options.git.logOne(root.path, ref), error: '' }
  } catch (error) {
    return expansionError(token, 'git', errorMessage(error))
  }
}

async function workspaceRoot(options: AtMentionOptions): Promise<AtMentionPathInfo> {
  const root = await options.fileSystem.inspect(options.workspaceRoot)
  if (!root) {
    throw new Error('workspace root not found')
  }
  if (root.kind !== 'directory') {
    throw new Error('workspace root is not a directory')
  }
  if (!path.isAbsolute(root.path)) {
    throw new Error('filesystem port returned a non-absolute workspace path')
  }
  return root
}

async function inspectedContainedPath(
  candidate: string,
  root: AtMentionPathInfo,
  fileSystem: AtMentionFileSystemPort
): Promise<AtMentionPathInfo | undefined> {
  if (!isWithinWorkspace(root.path, candidate)) {
    return undefined
  }
  return safelyInspectContained(candidate, root, fileSystem)
}

async function safelyInspectContained(
  candidate: string,
  root: AtMentionPathInfo,
  fileSystem: AtMentionFileSystemPort
): Promise<AtMentionPathInfo | undefined> {
  if (!path.isAbsolute(candidate)) {
    return undefined
  }
  try {
    const info = await fileSystem.inspect(candidate)
    return info && path.isAbsolute(info.path) && isWithinWorkspace(root.path, info.path) ? info : undefined
  } catch {
    return undefined
  }
}

function pathBasePath(root: string, prefix: string): string | undefined {
  const parent = path.dirname(prefix)
  const rawParent = parent === '.' ? '' : parent
  return workspacePath(root, rawParent)
}

function workspacePath(root: string, input: string): string | undefined {
  const candidate = path.isAbsolute(input) ? path.normalize(input) : path.resolve(root, input)
  return isWithinWorkspace(root, candidate) ? candidate : undefined
}

function isWithinWorkspace(root: string, candidate: string): boolean {
  if (!path.isAbsolute(root) || !path.isAbsolute(candidate)) {
    return false
  }
  const relative = path.relative(root, candidate)
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

function relativeWorkspacePath(root: string, target: string): string | undefined {
  if (!isWithinWorkspace(root, target)) {
    return undefined
  }
  const relative = path.relative(root, target)
  return relative && relative !== '.' ? relative.split(path.sep).join('/') : undefined
}

function completionLimit(value: number | undefined): number {
  const limit = value ?? 30
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError('maxResults must be a positive integer')
  }
  return limit
}

function expansionError(token: string, kind: AtMentionKind, error: string): ExpandedAtMention {
  return { token, kind, payload: '', error }
}

function compareDirectoryEntries(left: AtMentionDirectoryEntry, right: AtMentionDirectoryEntry): number {
  return compareStrings(left.name, right.name)
}

function isSafeDirectoryEntry(entry: AtMentionDirectoryEntry): boolean {
  return (
    Boolean(entry.name) &&
    entry.name !== '.' &&
    entry.name !== '..' &&
    !entry.name.includes('/') &&
    !entry.name.includes('\\')
  )
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error)
}
