// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { constants } from 'node:fs'
import { copyFile, lstat, mkdir, readdir, readFile, rm } from 'node:fs/promises'
import { basename, dirname, join, relative, resolve, sep } from 'node:path'

/** Runs git with an argv list (never a shell). Injected so tests do not reach the network. */
export type GitRunner = (args: readonly string[], options: { readonly cwd: string; readonly timeoutMs: number }) => Promise<{ readonly exitCode: number; readonly output: string }>

const CLONE_TIMEOUT_MS = 120_000
const TOKEN_TTL_MS = 30 * 60_000
const MAX_PREVIEW_FILES = 300
const MAX_LINE_COUNT_BYTES = 1_000_000
const OUTPUT_TAIL = 2_000
const EXPORT_SKIP = new Set(['.git', 'node_modules'])

/** Run git directly with a minimal environment and a hard timeout. */
export const defaultGitRunner: GitRunner = async (args, options) => {
  const env: Record<string, string> = { GIT_TERMINAL_PROMPT: '0' }
  for (const key of ['PATH', 'HOME', 'SSH_AUTH_SOCK', 'LANG']) {
    const value = process.env[key]
    if (value) env[key] = value
  }
  const child = Bun.spawn(['git', ...args], { cwd: options.cwd, env, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
  const timer = setTimeout(() => { try { child.kill('SIGKILL') } catch { /* gone */ } }, options.timeoutMs)
  try {
    const [stdout, stderr, exitCode] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    return { exitCode, output: `${stdout}${stderr}` }
  } finally { clearTimeout(timer) }
}

/** Whether text names a remote source rather than a local path. */
export function looksLikeGitSource(text: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(text) || /^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:/.test(text)
}

/**
 * Accept https://, ssh:// and scp-style `user@host:path` git URLs only.
 * file://, http://, git://, ext:: transports and local paths are refused, as
 * are URLs carrying credentials.
 */
export function validateGitUrl(raw: string): string {
  const url = typeof raw === 'string' ? raw.trim() : ''
  if (!url || /\s/.test(url) || url.startsWith('-') || url.includes('::')) throw new Error('Not an acceptable git URL')
  const refused = 'Only https://, ssh:// or user@host:path git URLs are accepted (file://, http://, git:// and local paths are refused)'
  if (/^(https|ssh):\/\//i.test(url)) {
    let parsed: URL
    try { parsed = new URL(url) } catch { throw new Error(refused) }
    if (!parsed.hostname) throw new Error(refused)
    if (parsed.protocol === 'https:' && (parsed.username || parsed.password)) throw new Error('Do not put credentials in the git URL; configure a git credential helper instead')
    if (parsed.password) throw new Error('Do not put credentials in the git URL')
    return url
  }
  if (/^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[A-Za-z0-9._~/-]+$/.test(url) && !url.split(':')[1]!.startsWith('/')) return url
  if (/^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:\/[A-Za-z0-9._~/-]+$/.test(url)) return url
  throw new Error(refused)
}

export interface PreviewFile {
  readonly path: string
  /** Line count, or undefined for large files and symlinks. */
  readonly lines?: number
  readonly symlink?: boolean
}

export interface GitPluginPreview {
  readonly url: string
  readonly directory: string
  readonly entry: string
  readonly files: readonly PreviewFile[]
  readonly truncated: boolean
  /** The `permissions: {...}` text found in the entry file — read, not executed, so unverified. */
  readonly declaredPermissions?: string
  readonly readmeHead?: string
  readonly token: string
  readonly expiresAt: string
}

/**
 * Clones plugin sources for review. A clone never loads or runs anything;
 * installing needs the one-time token printed with the preview.
 */
export class PluginSourceCache {
  private readonly pending = new Map<string, { readonly url: string; readonly directory: string; readonly entry: string; readonly expires: number }>()

  constructor(private readonly directory: string, private readonly options: { readonly git?: GitRunner; readonly now?: () => number } = {}) {}

  async preview(rawUrl: string): Promise<GitPluginPreview> {
    const url = validateGitUrl(rawUrl)
    const now = this.options.now?.() ?? Date.now()
    await mkdir(this.directory, { recursive: true, mode: 0o700 })
    // A fresh folder per clone: an earlier clone of the same URL may already be installed.
    const id = createHash('sha256').update(url).update('\0').update(randomBytes(8)).digest('hex').slice(0, 24)
    const target = join(this.directory, id)
    const git = this.options.git ?? defaultGitRunner
    const result = await git(['-c', 'protocol.file.allow=never', '-c', 'core.hooksPath=/dev/null', 'clone', '--depth', '1', '--', url, target], { cwd: this.directory, timeoutMs: CLONE_TIMEOUT_MS })
    if (result.exitCode !== 0) {
      await rm(target, { recursive: true, force: true })
      throw new Error(`git clone failed (exit ${result.exitCode}): ${result.output.trim().slice(-OUTPUT_TAIL)}`)
    }
    try {
      const entry = await findEntry(target)
      const { files, truncated } = await listFiles(target)
      const source = await readFile(entry, 'utf8')
      const declaredPermissions = staticPermissions(source)
      const readme = await readFile(join(target, 'README.md'), 'utf8').catch(() => undefined)
      const token = randomBytes(12).toString('hex')
      const expires = now + TOKEN_TTL_MS
      this.pending.set(token, { url, directory: target, entry, expires })
      return {
        url, directory: target, entry, files, truncated, token, expiresAt: new Date(expires).toISOString(),
        ...(declaredPermissions ? { declaredPermissions } : {}),
        ...(readme ? { readmeHead: readme.split(/\r?\n/).slice(0, 20).join('\n') } : {}),
      }
    } catch (error) {
      await rm(target, { recursive: true, force: true })
      throw error
    }
  }

  /** Redeem a preview token once; it cannot be used again. */
  consume(token: string): { readonly url: string; readonly directory: string; readonly entry: string } {
    const pending = this.pending.get(token)
    this.pending.delete(token)
    if (!pending) throw new Error('Unknown or already used install token; run /plugins install <git-url> again to get a new preview')
    if (pending.expires < (this.options.now?.() ?? Date.now())) throw new Error('This install token expired; run /plugins install <git-url> again')
    return pending
  }
}

async function findEntry(root: string): Promise<string> {
  for (const name of ['index.ts', 'index.js', 'index.mjs']) {
    const path = join(root, name)
    const info = await lstat(path).catch(() => undefined)
    if (info?.isFile()) return path
    if (info?.isSymbolicLink()) throw new Error(`${name} is a symlink; refusing to install it`)
  }
  throw new Error('The repository has no index.ts, index.js or index.mjs at its root')
}

async function listFiles(root: string): Promise<{ files: PreviewFile[]; truncated: boolean }> {
  const files: PreviewFile[] = []
  let truncated = false
  const walk = async (directory: string): Promise<void> => {
    const entries = (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)
    for (const entry of entries) {
      if (files.length >= MAX_PREVIEW_FILES) { truncated = true; return }
      const path = join(directory, entry.name)
      const shown = relative(root, path)
      if (entry.isDirectory()) { if (entry.name !== '.git') await walk(path); continue }
      if (entry.isSymbolicLink()) { files.push({ path: shown, symlink: true }); continue }
      if (!entry.isFile()) continue
      const info = await lstat(path)
      if (info.size > MAX_LINE_COUNT_BYTES) { files.push({ path: shown }); continue }
      const text = await readFile(path, 'utf8')
      files.push({ path: shown, lines: text === '' ? 0 : text.split('\n').length - (text.endsWith('\n') ? 1 : 0) })
    }
  }
  await walk(root)
  return { files, truncated }
}

/** The literal `permissions: { ... }` text in a module, found by brace matching. */
export function staticPermissions(source: string): string | undefined {
  const match = /permissions\s*:\s*\{/.exec(source)
  if (!match) return undefined
  const start = match.index
  let depth = 0
  for (let index = source.indexOf('{', start); index < source.length && index - start < 2_000; index += 1) {
    if (source[index] === '{') depth += 1
    else if (source[index] === '}' && --depth === 0) return source.slice(start, index + 1).replace(/\s+/g, ' ')
  }
  return undefined
}

export function renderGitPreview(preview: GitPluginPreview): string {
  return [
    `Cloned ${preview.url} for review. Nothing was loaded or run.`,
    `Entry: ${preview.entry}`,
    `Files (${preview.files.length}${preview.truncated ? '+, list truncated' : ''}):`,
    ...preview.files.map(file => `  ${file.path}${file.symlink ? ' (symlink, not followed)' : file.lines === undefined ? ' (large)' : ` · ${file.lines} lines`}`),
    preview.declaredPermissions
      ? `Declared permissions (read from the source text, not verified): ${preview.declaredPermissions}`
      : 'Declared permissions: none found in the entry file (exact permissions are shown after install).',
    ...(preview.readmeHead ? ['README (first lines):', ...preview.readmeHead.split('\n').map(line => `  ${line}`)] : ['No README.md.']),
    '',
    'Installing runs this code in its own process with your user privileges (not a sandbox). Review it first.',
    `To install: /plugins install --confirm ${preview.token}   (valid until ${preview.expiresAt}, one use)`,
  ].join('\n')
}

/**
 * Copy a plugin's sources to a new directory for sharing. A plugin that is a
 * folder's `index.*` brings the whole folder (minus .git, node_modules, .env
 * files and symlinks); a single-file plugin brings the files it loaded from its
 * folder and its `<name>.test.*` files. Secrets live in the Xerxes home and are
 * never part of a plugin's folder. Refuses to overwrite anything.
 */
export async function exportPluginSources(modulePath: string, loadedFiles: readonly string[], destination: string): Promise<string[]> {
  const target = resolve(destination)
  if (await lstat(target).then(() => true, () => false)) throw new Error(`${target} already exists; export needs a new directory`)
  const folder = dirname(modulePath)
  const prefix = folder.endsWith(sep) ? folder : folder + sep
  if (target === folder || target.startsWith(prefix)) throw new Error('Export the plugin outside its own folder')
  const selected: string[] = []
  if (/^index\.(?:[cm]?js|ts)$/.test(basename(modulePath))) {
    const walk = async (directory: string): Promise<void> => {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        if (EXPORT_SKIP.has(entry.name) || entry.name.startsWith('.env')) continue
        const path = join(directory, entry.name)
        if (entry.isDirectory()) await walk(path)
        else if (entry.isFile()) selected.push(path)
      }
    }
    await walk(folder)
  } else {
    const stem = basename(modulePath).replace(/\.(?:[cm]?js|ts)$/, '')
    const candidates = [modulePath, ...loadedFiles, ...['ts', 'js'].map(extension => join(folder, `${stem}.test.${extension}`))]
    for (const path of candidates) {
      if (!path.startsWith(prefix) || selected.includes(path)) continue
      const info = await lstat(path).catch(() => undefined)
      if (info?.isFile()) selected.push(path)
    }
  }
  await mkdir(dirname(target), { recursive: true })
  await mkdir(target)
  const copied: string[] = []
  for (const path of selected.sort()) {
    const name = relative(folder, path)
    await mkdir(dirname(join(target, name)), { recursive: true })
    await copyFile(path, join(target, name), constants.COPYFILE_EXCL)
    copied.push(name)
  }
  return copied
}
