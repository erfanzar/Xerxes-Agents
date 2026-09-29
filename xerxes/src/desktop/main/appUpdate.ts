// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Keeping the desktop app itself current from GitHub releases — only with the
 * user's say-so. The app checks the latest release, shows the version and its
 * notes, and installs only after the person clicks Install:
 *
 *  1. download the release's macOS DMG, checking its size and GitHub's
 *     published SHA-256 digest;
 *  2. mount it read-only, copy the app out, verify its code signature;
 *  3. hand a tiny detached script the swap: once this app has quit, move the
 *     old bundle aside, put the new one in its place, relaunch it.
 *
 * Nothing is installed without consent, and anything it cannot do (a dev
 * build, an unwritable Applications folder, another OS) is reported with the
 * release page to fall back to. Electron-free apart from the host ports the
 * caller injects, so every step is testable headlessly.
 */

import { createHash } from 'node:crypto'
import { access, mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { constants, createWriteStream } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'

export const RELEASES_REPOSITORY = 'erfanzar/Xerxes-Agents'
const MAX_NOTES_CHARS = 12_000

export interface ReleaseAsset {
  readonly name: string
  readonly url: string
  readonly size: number
  /** GitHub's `sha256:<hex>` digest, when the API reports one. */
  readonly sha256?: string
}

export interface AvailableRelease {
  readonly version: string
  readonly tag: string
  readonly notes: string
  readonly publishedAt?: string
  readonly pageUrl: string
  /** The installer for this machine; absent when the release has none for it. */
  readonly asset?: ReleaseAsset
}

export type UpdateCheck =
  | { readonly kind: 'current'; readonly version: string }
  | { readonly kind: 'available'; readonly release: AvailableRelease }

type Fetch = (url: string, init?: RequestInit) => Promise<Response>

/** Compare dotted numeric versions ("0.6.10" > "0.6.9"); a prerelease suffix sorts before its release. */
export function compareVersions(left: string, right: string): number {
  const parse = (value: string) => {
    const [core = '', pre] = value.replace(/^v/i, '').split('-', 2)
    return { parts: core.split('.').map(part => Number.parseInt(part, 10) || 0), pre }
  }
  const a = parse(left), b = parse(right)
  for (let index = 0; index < Math.max(a.parts.length, b.parts.length); index++) {
    const difference = (a.parts[index] ?? 0) - (b.parts[index] ?? 0)
    if (difference !== 0) return difference
  }
  if (a.pre && !b.pre) return -1
  if (!a.pre && b.pre) return 1
  return (a.pre ?? '').localeCompare(b.pre ?? '')
}

/** The installer asset this machine can use, by the release's naming. */
export function installerAssetName(version: string, platform: string, arch: string): string | undefined {
  if (platform !== 'darwin') return undefined
  return `Xerxes-Agents-${version}-macOS-${arch === 'arm64' ? 'arm64' : 'x64'}.dmg`
}

/** Ask GitHub for the latest release and whether it is newer than this app. */
export async function checkForAppUpdate(options: {
  readonly currentVersion: string
  readonly platform: string
  readonly arch: string
  readonly fetch: Fetch
  readonly repository?: string
  readonly signal?: AbortSignal
}): Promise<UpdateCheck> {
  const repository = options.repository ?? RELEASES_REPOSITORY
  const response = await options.fetch(`https://api.github.com/repos/${repository}/releases/latest`, {
    headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'xerxes-agents-desktop' },
    ...(options.signal ? { signal: options.signal } : {}),
  })
  if (!response.ok) throw new Error(`GitHub answered ${response.status} when checking for an update`)
  const body: unknown = await response.json()
  if (!body || typeof body !== 'object') throw new Error('GitHub returned an unreadable release')
  const release = body as Record<string, unknown>
  const tag = typeof release.tag_name === 'string' ? release.tag_name : ''
  const version = tag.replace(/^v/i, '')
  if (!/^\d+\.\d+\.\d+/.test(version)) throw new Error(`The latest release has an unexpected tag: ${tag || '(none)'}`)
  if (release.draft === true || release.prerelease === true || compareVersions(version, options.currentVersion) <= 0) {
    return { kind: 'current', version: options.currentVersion }
  }
  const wanted = installerAssetName(version, options.platform, options.arch)
  const assets = Array.isArray(release.assets) ? release.assets.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : []
  const match = wanted ? assets.find(item => item.name === wanted) : undefined
  const digest = typeof match?.digest === 'string' && /^sha256:[0-9a-f]{64}$/i.test(match.digest) ? match.digest.slice(7).toLowerCase() : undefined
  const notes = typeof release.body === 'string' ? release.body.trim() : ''
  return {
    kind: 'available',
    release: {
      version,
      tag,
      notes: notes.length > MAX_NOTES_CHARS ? `${notes.slice(0, MAX_NOTES_CHARS - 1)}…` : notes,
      ...(typeof release.published_at === 'string' ? { publishedAt: release.published_at } : {}),
      pageUrl: typeof release.html_url === 'string' ? release.html_url : `https://github.com/${repository}/releases/tag/${tag}`,
      ...(match && typeof match.browser_download_url === 'string' && typeof match.size === 'number'
        ? { asset: { name: String(match.name), url: match.browser_download_url, size: match.size, ...(digest ? { sha256: digest } : {}) } }
        : {}),
    },
  }
}

export interface InstallPorts {
  readonly fetch: Fetch
  /** Run a program to completion; rejects with its stderr on a non-zero exit. */
  readonly run: (argv: readonly string[]) => Promise<string>
  /** Start the detached swap script; it must outlive this process. */
  readonly spawnDetached: (argv: readonly string[]) => void
}

export interface InstallOptions {
  readonly asset: ReleaseAsset
  /** The running app bundle, e.g. `/Applications/Xerxes Agents.app`. */
  readonly appBundle: string
  /** This process, which the swap waits to exit. */
  readonly pid: number
  readonly ports: InstallPorts
  readonly onProgress?: (received: number, total: number) => void
  readonly signal?: AbortSignal
}

/** The `.app` bundle that contains an executable path, if any. */
export function appBundleOf(executablePath: string): string | undefined {
  const index = executablePath.indexOf('.app/Contents/')
  return index > 0 ? executablePath.slice(0, index + 4) : undefined
}

/**
 * Download, verify and stage the new app, then schedule the swap. Resolves
 * once the swap is armed; the caller quits the app to let it run.
 */
export async function installAppUpdate(options: InstallOptions): Promise<void> {
  const { asset, appBundle, ports } = options
  const parent = dirname(appBundle)
  try { await access(parent, constants.W_OK) }
  catch { throw new Error(`${parent} is not writable by this user, so the app cannot replace itself. Download the update from the release page instead.`) }
  const work = await mkdtemp(join(tmpdir(), 'xerxes-update-'))
  const dmg = join(work, asset.name)
  const mount = join(work, 'mount')
  let mounted = false
  try {
    const response = await ports.fetch(asset.url, { headers: { 'User-Agent': 'xerxes-agents-desktop' }, ...(options.signal ? { signal: options.signal } : {}) })
    if (!response.ok || !response.body) throw new Error(`The download failed (${response.status})`)
    // Streamed to disk while hashing: the installer is ~170 MB.
    const hash = createHash('sha256')
    const file = createWriteStream(dmg)
    let received = 0
    const reader = response.body.getReader()
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        hash.update(value)
        received += value.byteLength
        if (!file.write(value)) await new Promise<void>(resolve => file.once('drain', () => resolve()))
        options.onProgress?.(received, asset.size)
      }
    } finally {
      await new Promise<void>((resolve, reject) => file.end((error?: Error | null) => error ? reject(error) : resolve()))
    }
    if (received !== asset.size) throw new Error(`The download is incomplete (${received} of ${asset.size} bytes)`)
    if (asset.sha256 && hash.digest('hex') !== asset.sha256) throw new Error('The download does not match the checksum GitHub published for it')
    options.signal?.throwIfAborted()
    await mkdir(mount)
    await ports.run(['hdiutil', 'attach', '-nobrowse', '-readonly', '-noautoopen', '-mountpoint', mount, dmg])
    mounted = true
    const bundleName = (await readdir(mount)).find(name => name.endsWith('.app'))
    if (!bundleName) throw new Error('The installer does not contain an app')
    const staged = join(work, 'staged', basename(appBundle))
    await mkdir(dirname(staged), { recursive: true })
    await ports.run(['ditto', join(mount, bundleName), staged])
    await ports.run(['hdiutil', 'detach', mount, '-quiet'])
    mounted = false
    await ports.run(['codesign', '--verify', '--deep', '--strict', staged])
    options.signal?.throwIfAborted()
    const script = join(work, 'swap.sh')
    await writeFile(script, swapScript(options.pid, appBundle, staged, work), { mode: 0o700 })
    ports.spawnDetached(['/bin/sh', script])
  } catch (error) {
    if (mounted) await ports.run(['hdiutil', 'detach', mount, '-force', '-quiet']).catch(() => undefined)
    await rm(work, { recursive: true, force: true })
    throw error
  }
}

const quote = (value: string) => `'${value.replaceAll("'", `'\\''`)}'`

/** Waits for the app to exit, swaps the bundle (restoring the old one on failure), relaunches. */
export function swapScript(pid: number, appBundle: string, staged: string, work: string): string {
  const target = quote(appBundle), previous = quote(`${appBundle}.previous`), next = quote(staged)
  return [
    '#!/bin/sh',
    `while kill -0 ${pid} 2>/dev/null; do sleep 0.5; done`,
    `rm -rf ${previous}`,
    `if mv ${target} ${previous} && mv ${next} ${target}; then`,
    `  rm -rf ${previous}`,
    'else',
    `  [ -d ${target} ] || mv ${previous} ${target}`,
    'fi',
    `open ${target}`,
    `rm -rf ${quote(work)}`,
    '',
  ].join('\n')
}
