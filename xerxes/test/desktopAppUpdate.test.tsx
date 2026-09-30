// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { appBundleOf, checkForAppUpdate, compareVersions, installAppUpdate, swapScript } from '../src/desktop/main/appUpdate.js'
import { AppUpdatePrompt } from '../src/desktop/renderer/AppUpdatePrompt.js'
import { appUpdateView } from '../src/desktop/renderer/store.js'

const release = (tag: string, extra: Record<string, unknown> = {}) => ({
  tag_name: tag,
  html_url: `https://github.com/erfanzar/Xerxes-Agents/releases/tag/${tag}`,
  body: '- Desktop: agents stay visible.',
  published_at: '2026-09-29T12:00:00Z',
  assets: [
    { name: `Xerxes-Agents-${tag.slice(1)}-macOS-arm64.dmg`, size: 1234, browser_download_url: `https://github.com/x/${tag}.dmg`, digest: `sha256:${'a'.repeat(64)}` },
    { name: `xerxes-agents-${tag.slice(1)}.tgz`, size: 99, browser_download_url: 'https://github.com/x/tgz' },
  ],
  ...extra,
})
const github = (body: unknown, status = 200) => async () => new Response(JSON.stringify(body), { status })

test('versions compare numerically, and a prerelease sorts before its release', () => {
  expect(compareVersions('0.6.10', '0.6.9')).toBeGreaterThan(0)
  expect(compareVersions('v0.7.0', '0.6.99')).toBeGreaterThan(0)
  expect(compareVersions('0.6.6', '0.6.6')).toBe(0)
  expect(compareVersions('0.7.0-rc.1', '0.7.0')).toBeLessThan(0)
})

test('a newer release is offered with its notes and this machine\'s installer', async () => {
  const result = await checkForAppUpdate({ currentVersion: '0.6.6', platform: 'darwin', arch: 'arm64', fetch: github(release('v0.6.7')) })
  expect(result).toMatchObject({ kind: 'available', release: { version: '0.6.7', notes: '- Desktop: agents stay visible.', asset: { name: 'Xerxes-Agents-0.6.7-macOS-arm64.dmg', size: 1234, sha256: 'a'.repeat(64) } } })
  expect(await checkForAppUpdate({ currentVersion: '0.6.7', platform: 'darwin', arch: 'arm64', fetch: github(release('v0.6.7')) })).toEqual({ kind: 'current', version: '0.6.7' })
  expect(await checkForAppUpdate({ currentVersion: '0.6.6', platform: 'darwin', arch: 'arm64', fetch: github(release('v0.6.7', { prerelease: true })) })).toMatchObject({ kind: 'current' })
  // No installer for this computer: still offered, without an asset (the release page is the fallback).
  const linux = await checkForAppUpdate({ currentVersion: '0.6.6', platform: 'linux', arch: 'x64', fetch: github(release('v0.6.7')) })
  expect(linux.kind === 'available' && linux.release.asset).toBeUndefined()
  await expect(checkForAppUpdate({ currentVersion: '0.6.6', platform: 'darwin', arch: 'arm64', fetch: github({}, 403) })).rejects.toThrow('GitHub answered 403')
})

test('the bundle of a packaged executable is found', () => {
  expect(appBundleOf('/Applications/Xerxes Agents.app/Contents/MacOS/Xerxes Agents')).toBe('/Applications/Xerxes Agents.app')
  expect(appBundleOf('/usr/local/bin/electron')).toBeUndefined()
})

async function withApplications(body: (applications: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-appupdate-'))
  try {
    const applications = join(root, 'Applications')
    await mkdir(join(applications, 'Xerxes Agents.app'), { recursive: true })
    await body(applications)
  } finally { await rm(root, { recursive: true, force: true }) }
}

test('an agreed install downloads, verifies, stages and arms the swap', async () => {
  await withApplications(async applications => {
    const payload = new TextEncoder().encode('dmg bytes')
    const sha256 = createHash('sha256').update(payload).digest('hex')
    const commands: string[][] = []
    const spawned: string[][] = []
    const progress: number[] = []
    await installAppUpdate({
      asset: { name: 'Xerxes-Agents-0.6.7-macOS-arm64.dmg', url: 'https://github.com/x.dmg', size: payload.byteLength, sha256 },
      appBundle: join(applications, 'Xerxes Agents.app'),
      pid: 4242,
      onProgress: received => progress.push(received),
      ports: {
        fetch: async () => new Response(payload),
        run: async argv => {
          commands.push([...argv])
          if (argv[0] === 'hdiutil' && argv[1] === 'attach') await mkdir(join(argv[argv.indexOf('-mountpoint') + 1]!, 'Xerxes Agents.app'), { recursive: true })
          return ''
        },
        spawnDetached: argv => { spawned.push([...argv]) },
      },
    })
    expect(commands.map(argv => argv.slice(0, 2).join(' '))).toEqual(['hdiutil attach', 'ditto ' + commands[1]![1], 'hdiutil detach', 'codesign --verify'])
    expect(progress.at(-1)).toBe(payload.byteLength)
    expect(spawned).toHaveLength(1)
    expect(spawned[0]![0]).toBe('/bin/sh')
    // Nothing touched the installed bundle yet; the swap waits for this process.
    expect(await readdir(applications)).toEqual(['Xerxes Agents.app'])
  })
})

test('a download that does not match GitHub\'s checksum or size installs nothing', async () => {
  await withApplications(async applications => {
    const ports = { fetch: async () => new Response('tampered'), run: async () => '', spawnDetached: () => { throw new Error('must not arm') } }
    const base = { asset: { name: 'x.dmg', url: 'https://github.com/x.dmg', size: 8, sha256: 'b'.repeat(64) }, appBundle: join(applications, 'Xerxes Agents.app'), pid: 1, ports }
    await expect(installAppUpdate(base)).rejects.toThrow('checksum')
    await expect(installAppUpdate({ ...base, asset: { ...base.asset, size: 999 } })).rejects.toThrow('incomplete')
  })
})

test('the swap restores the old app if the new one cannot be moved in, then relaunches', () => {
  const script = swapScript(4242, "/Applications/Xerxes Agents.app", "/tmp/w/staged/Xerxes Agents.app", '/tmp/w')
  expect(script).toContain('while kill -0 4242')
  expect(script).toContain("mv '/Applications/Xerxes Agents.app' '/Applications/Xerxes Agents.app.previous'")
  expect(script).toContain("[ -d '/Applications/Xerxes Agents.app' ] || mv '/Applications/Xerxes Agents.app.previous' '/Applications/Xerxes Agents.app'")
  expect(script).toContain("open '/Applications/Xerxes Agents.app'")
})

test('the prompt asks before installing and falls back to the release page when it cannot', () => {
  const asked = renderToStaticMarkup(createElement(AppUpdatePrompt, { update: appUpdateView({ phase: 'available', installable: true, release: { version: '0.6.7', notes: '- Faster agents', pageUrl: 'https://github.com/x' } }) }))
  expect(asked).toContain('Xerxes Agents 0.6.7 is available')
  expect(asked).toContain('May I download and install it?')
  expect(asked).toContain('Install and restart')
  expect(asked).toContain('Skip this version')
  expect(asked).toContain('Faster agents')
  const blocked = renderToStaticMarkup(createElement(AppUpdatePrompt, { update: appUpdateView({ phase: 'available', installable: false, reason: 'This is a development build; update it from source.', release: { version: '0.6.7' } }) }))
  expect(blocked).not.toContain('Install and restart')
  expect(blocked).toContain('Open release page')
  const downloading = renderToStaticMarkup(createElement(AppUpdatePrompt, { update: appUpdateView({ phase: 'downloading', received: 52_428_800, total: 104_857_600, release: { version: '0.6.7' } }) }))
  expect(downloading).toContain('50%')
  expect(downloading).not.toContain('Not now')
  expect(appUpdateView({ phase: 'idle' })).toBeNull()
})

test('an explicit check that finds nothing newer says so instead of showing nothing', () => {
  const view = appUpdateView({ phase: 'current', version: '0.6.14' })
  expect(view).toMatchObject({ phase: 'current', version: '0.6.14' })
  const shown = renderToStaticMarkup(createElement(AppUpdatePrompt, { update: view }))
  expect(shown).toContain('Xerxes Agents is up to date')
  expect(shown).toContain('0.6.14 is the latest version.')
  expect(shown).toContain('>OK<')
  expect(shown).not.toContain('Install and restart')
})
