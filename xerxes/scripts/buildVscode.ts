// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Builds the VS Code extension: one .vsix per platform, with the Bun runtime
 * inside so nothing else needs installing.
 *
 *   bun run build:vscode                      # this machine's platform
 *   bun run build:vscode -- --target linux-x64
 *   bun run build:vscode -- --all             # every supported platform
 *
 * Needs `bun run build:runtime` and `bun scripts/buildDesktop.ts` first (the
 * package script runs both): the extension ships the desktop renderer and the
 * runtime CLI unchanged. A non-native target downloads the matching official
 * Bun build and checks it against Bun's published SHA-256 sums.
 */

import { createHash } from 'node:crypto'
import { chmod, cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { DAEMON_PROTOCOL_VERSION } from '../src/daemon/fingerprint.js'
import { sourceDaemonBuildId } from '../src/daemon/sourceBuild.js'
import { copyDesktopRuntimeAssets } from './packageDesktopMac.js'

const packageDirectory = join(import.meta.dir, '..')
const distDirectory = join(packageDirectory, 'dist')
const repoRoot = join(packageDirectory, '..')

/** VS Code platform target → the official Bun build for it. */
const TARGETS: Readonly<Record<string, string>> = {
  'darwin-arm64': 'bun-darwin-aarch64',
  'darwin-x64': 'bun-darwin-x64',
  'linux-x64': 'bun-linux-x64',
  'linux-arm64': 'bun-linux-aarch64',
  'win32-x64': 'bun-windows-x64',
}

function nativeTarget(): string {
  const os = process.platform === 'win32' ? 'win32' : process.platform
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'
  return `${os}-${arch}`
}

async function version(): Promise<string> {
  const manifest = JSON.parse(await readFile(join(packageDirectory, 'package.json'), 'utf8')) as { version: string }
  return manifest.version
}

/** The app's surfaces without a native VS Code home, opened from the view's overflow menu. Keep in step with SHEETS in src/vscode/extension.ts. */
const SHEETS: ReadonlyArray<readonly [string, string]> = [['activity', 'Activity'], ['usage', 'Usage']]

/** The extension manifest VS Code and the Marketplace read. */
function manifest(extensionVersion: string): Record<string, unknown> {
  return {
    name: 'xerxes-agents',
    displayName: 'Xerxes Agents',
    description: 'Multi-agent coding with Xerxes: tasks, agents, goals, approvals and providers in VS Code, with the runtime built in.',
    version: extensionVersion,
    publisher: 'xsimurgh',
    license: 'Apache-2.0',
    icon: 'media/icon.png',
    repository: { type: 'git', url: 'https://github.com/erfanzar/Xerxes-Agents' },
    homepage: 'https://github.com/erfanzar/Xerxes-Agents',
    bugs: { url: 'https://github.com/erfanzar/Xerxes-Agents/issues' },
    engines: { vscode: '^1.95.0' },
    categories: ['AI', 'Chat', 'Programming Languages'],
    keywords: ['agents', 'ai', 'coding agent', 'multi-agent', 'xerxes'],
    main: './extension.js',
    // Runs where the folder is: on the remote host under Remote-SSH, so the
    // runtime works on that machine's files.
    extensionKind: ['workspace'],
    activationEvents: ['onStartupFinished'],
    contributes: {
      viewsContainers: {
        activitybar: [{ id: 'xerxes', title: 'Xerxes', icon: 'media/xerxes.svg' }],
      },
      views: {
        xerxes: [{ type: 'webview', id: 'xerxes.chat', name: 'Xerxes' }],
      },
      commands: [
        { command: 'xerxes.newTask', title: 'New Task', category: 'Xerxes', icon: '$(add)' },
        { command: 'xerxes.history', title: 'Task History', category: 'Xerxes', icon: '$(history)' },
        { command: 'xerxes.settings', title: 'Settings', category: 'Xerxes', icon: '$(settings-gear)' },
        { command: 'xerxes.sendSelection', title: 'Send Selection to Xerxes', category: 'Xerxes' },
        { command: 'xerxes.search', title: 'Search Message History', category: 'Xerxes' },
        { command: 'xerxes.palette', title: 'Command Palette', category: 'Xerxes' },
        { command: 'xerxes.exportTranscript', title: 'Export Transcript', category: 'Xerxes' },
        ...SHEETS.map(([name, title]) => ({ command: `xerxes.panel.${name}`, title, category: 'Xerxes' })),
      ],
      menus: {
        'editor/context': [{ command: 'xerxes.sendSelection', when: 'editorHasSelection', group: 'xerxes@1' }],
        // The title bar carries the everyday actions; everything else is in its overflow menu.
        'view/title': [
          { command: 'xerxes.newTask', when: 'view == xerxes.chat', group: 'navigation@1' },
          { command: 'xerxes.history', when: 'view == xerxes.chat', group: 'navigation@2' },
          { command: 'xerxes.settings', when: 'view == xerxes.chat', group: 'navigation@3' },
          { command: 'xerxes.search', when: 'view == xerxes.chat', group: '1_task@1' },
          { command: 'xerxes.exportTranscript', when: 'view == xerxes.chat', group: '1_task@2' },
          ...SHEETS.map(([name], index) => ({ command: `xerxes.panel.${name}`, when: 'view == xerxes.chat', group: `2_panels@${index + 1}` })),
        ],
      },
      keybindings: [
        { command: 'xerxes.sendSelection', key: 'ctrl+alt+x', mac: 'cmd+alt+x', when: 'editorTextFocus' },
      ],
    },
  }
}

/** A monochrome activity-bar mark: the dotted ring of the welcome orb. */
function activityIcon(): string {
  const dots = Array.from({ length: 16 }, (_, index) => {
    const angle = (index / 16) * Math.PI * 2
    return `<circle cx="${(12 + Math.cos(angle) * 8).toFixed(2)}" cy="${(12 + Math.sin(angle) * 8).toFixed(2)}" r="${index % 2 ? 1 : 1.4}"/>`
  }).join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">${dots}</svg>\n`
}

/** The build constants the renderer reads, as buildDesktop.ts defines them. */
async function rendererDefines(): Promise<Record<string, string>> {
  const buildId = await sourceDaemonBuildId(join(packageDirectory, 'src'))
  if (!buildId) throw new Error('could not fingerprint the daemon source')
  return {
    'process.env.NODE_ENV': JSON.stringify('production'),
    __XERXES_DESKTOP_VERSION__: JSON.stringify(await version()),
    __XERXES_DESKTOP_PROTOCOL__: String(DAEMON_PROTOCOL_VERSION),
    __XERXES_EXPECTED_DAEMON_BUILD_ID__: JSON.stringify(buildId),
  }
}

async function bunBinary(target: string, destination: string): Promise<void> {
  if (target === nativeTarget()) {
    await cp(process.execPath, destination)
    return
  }
  const asset = TARGETS[target]!
  const base = `https://github.com/oven-sh/bun/releases/download/bun-v${Bun.version}`
  const [archive, sums] = await Promise.all([
    fetch(`${base}/${asset}.zip`).then(response => { if (!response.ok) throw new Error(`Bun ${asset} download failed (${response.status})`); return response.arrayBuffer() }),
    fetch(`${base}/SHASUMS256.txt`).then(response => { if (!response.ok) throw new Error(`Bun checksums download failed (${response.status})`); return response.text() }),
  ])
  const expected = sums.split('\n').find(line => line.endsWith(` ${asset}.zip`))?.split(' ')[0]
  const actual = createHash('sha256').update(new Uint8Array(archive)).digest('hex')
  if (!expected || expected !== actual) throw new Error(`Bun ${asset} did not match its published checksum`)
  const work = join(distDirectory, `vscode-bun-${asset}`)
  await rm(work, { recursive: true, force: true })
  await mkdir(work, { recursive: true })
  await writeFile(join(work, `${asset}.zip`), new Uint8Array(archive))
  const unzip = Bun.spawnSync(['unzip', '-oq', join(work, `${asset}.zip`), '-d', work])
  if (unzip.exitCode !== 0) throw new Error(`Could not unpack ${asset}: ${unzip.stderr.toString()}`)
  await cp(join(work, asset, target.startsWith('win32') ? 'bun.exe' : 'bun'), destination)
  await rm(work, { recursive: true, force: true })
}

async function build(target: string): Promise<string> {
  if (!TARGETS[target]) throw new Error(`Unsupported target ${target}. Choose one of: ${Object.keys(TARGETS).join(', ')}`)
  const extensionVersion = await version()
  const out = join(distDirectory, 'vscode', target)
  await rm(out, { recursive: true, force: true })
  await mkdir(join(out, 'media'), { recursive: true })

  const host = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'vscode', 'extension.ts')],
    outdir: out, target: 'node', format: 'cjs', minify: true,
    // VS Code supplies its API module at run time.
    external: ['vscode'],
  })
  if (!host.success) { for (const log of host.logs) console.error(log); throw new Error('extension host build failed') }
  const webview = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'vscode', 'webview.ts')],
    outdir: join(out, 'media'), target: 'browser', format: 'iife', minify: true,
  })
  if (!webview.success) { for (const log of webview.logs) console.error(log); throw new Error('webview bridge build failed') }
  // The chat view: the renderer's conversation pieces in a VS Code layout.
  const chat = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'vscode', 'ui', 'main.tsx')],
    outdir: join(out, 'media'), naming: 'chat.[ext]', target: 'browser', format: 'esm', minify: true,
    define: await rendererDefines(),
  })
  if (!chat.success) { for (const log of chat.logs) console.error(log); throw new Error('chat view build failed') }
  await cp(join(packageDirectory, 'src', 'vscode', 'ui', 'vscode.css'), join(out, 'media', 'vscode.css'))

  // The renderer's stylesheets and assets (buildDesktop.ts writes them).
  await cp(join(distDirectory, 'desktop', 'renderer'), join(out, 'media', 'renderer'), { recursive: true })
  await cp(join(repoRoot, 'assets', 'logo-128.png'), join(out, 'media', 'icon.png'))
  await writeFile(join(out, 'media', 'xerxes.svg'), activityIcon(), 'utf8')

  // The runtime: the CLI the panel's runtime connection launches, and Bun to run it.
  const runtime = join(out, 'runtime')
  await mkdir(runtime, { recursive: true })
  await cp(join(distDirectory, 'cli.js'), join(runtime, 'cli.js'))
  await cp(join(distDirectory, 'build-id'), join(runtime, 'build-id'))
  await copyDesktopRuntimeAssets(distDirectory, runtime)
  const bun = join(runtime, target.startsWith('win32') ? 'bun.exe' : 'bun')
  await bunBinary(target, bun)
  await chmod(bun, 0o755)
  await cp(join(packageDirectory, 'assets', 'notices', 'Bun-LICENSE.md'), join(runtime, 'Bun-LICENSE.md'))

  await writeFile(join(out, 'package.json'), JSON.stringify(manifest(extensionVersion), null, 2) + '\n', 'utf8')
  await cp(join(repoRoot, 'LICENSE'), join(out, 'LICENSE'))
  await cp(join(packageDirectory, 'src', 'vscode', 'README.md'), join(out, 'README.md'))

  const vsix = join(distDirectory, `xerxes-agents-${extensionVersion}-${target}.vsix`)
  const vsce = Bun.spawnSync([process.execPath, join(packageDirectory, 'node_modules', '@vscode', 'vsce', 'vsce'), 'package', '--target', target, '--no-dependencies', '--out', vsix], { cwd: out, stdout: 'inherit', stderr: 'inherit' })
  if (vsce.exitCode !== 0) throw new Error(`vsce package failed for ${target}`)
  return vsix
}

const args = process.argv.slice(2)
const named = args.indexOf('--target')
const targets = args.includes('--all') ? Object.keys(TARGETS) : [named >= 0 && args[named + 1] ? args[named + 1]! : nativeTarget()]
for (const target of targets) console.log(`VS Code extension ready: ${await build(target)}`)
