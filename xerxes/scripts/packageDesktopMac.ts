// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Wraps the compiled desktop files in a genuinely branded macOS application.
 * app.setName() changes Electron's runtime label, but macOS still identifies a
 * source launch from Electron.app by that host bundle's Info.plist. Packaging
 * replaces that outer identity so Dock, menu bar, Finder, and Activity Monitor
 * all see Xerxes Agents rather than Electron.
 */

import { cp, mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const productName = 'Xerxes Agents'
const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryDirectory = resolve(packageDirectory, '..')
const outputDirectory = join(packageDirectory, 'dist')
const applicationBundle = join(outputDirectory, `${productName}.app`)

async function run(command: string, arguments_: string[]): Promise<void> {
  const process = Bun.spawn([command, ...arguments_], { stdout: 'inherit', stderr: 'inherit' })
  const exitCode = await process.exited
  if (exitCode !== 0) throw new Error(`${command} exited with status ${exitCode}`)
}

function replacePlistString(plist: string, key: string, value: string): string {
  const expression = new RegExp(`(<key>${key}</key>\\s*<string>)[^<]*(</string>)`)
  if (!expression.test(plist)) throw new Error(`Electron Info.plist is missing ${key}`)
  return plist.replace(expression, `$1${value}$2`)
}

async function brandHelperBundles(frameworks: string): Promise<void> {
  for (const suffix of ['', ' (GPU)', ' (Plugin)', ' (Renderer)']) {
    const sourceName = `Electron Helper${suffix}`
    const productHelperName = `${productName} Helper${suffix}`
    const sourceBundle = join(frameworks, `${sourceName}.app`)
    const productBundle = join(frameworks, `${productHelperName}.app`)
    await rename(sourceBundle, productBundle)
    await rename(
      join(productBundle, 'Contents', 'MacOS', sourceName),
      join(productBundle, 'Contents', 'MacOS', productHelperName),
    )
    const plistPath = join(productBundle, 'Contents', 'Info.plist')
    let plist = await readFile(plistPath, 'utf8')
    const role = suffix.replaceAll(/[^A-Za-z]+/g, '').toLowerCase()
    plist = replacePlistString(plist, 'CFBundleIdentifier', `dev.xerxes.agents.helper${role ? `.${role}` : ''}`)
    plist = replacePlistString(plist, 'CFBundleName', productHelperName)
    await writeFile(plistPath, plist, 'utf8')
  }
}

async function createIcon(destination: string): Promise<void> {
  const source = join(repositoryDirectory, 'assets', 'logo.png')
  const iconset = join(outputDirectory, '.xerxes-agents.iconset')
  await rm(iconset, { recursive: true, force: true })
  await mkdir(iconset, { recursive: true })
  for (const size of [16, 32, 128, 256, 512]) {
    await run('sips', ['-z', String(size), String(size), source, '--out', join(iconset, `icon_${size}x${size}.png`)])
    const retina = size * 2
    await run('sips', ['-z', String(retina), String(retina), source, '--out', join(iconset, `icon_${size}x${size}@2x.png`)])
  }
  await run('iconutil', ['-c', 'icns', iconset, '-o', destination])
  await rm(iconset, { recursive: true, force: true })
}

async function packageDesktopMac(): Promise<void> {
  if (process.platform !== 'darwin') {
    console.log('skipped macOS desktop bundle packaging on this platform')
    return
  }

  const electronEntry = Bun.resolveSync('electron', packageDirectory)
  if (!electronEntry) throw new Error('desktop packaging could not resolve Electron')
  const sourceBundle = join(dirname(electronEntry), 'dist', 'Electron.app')
  const contents = join(applicationBundle, 'Contents')
  const resources = join(contents, 'Resources')
  const sourceExecutable = join(contents, 'MacOS', 'Electron')
  const productExecutable = join(contents, 'MacOS', productName)

  const packageRecord = JSON.parse(await readFile(join(packageDirectory, 'package.json'), 'utf8')) as {
    readonly version?: unknown
  }
  const version = typeof packageRecord.version === 'string' ? packageRecord.version : ''
  if (!version) throw new Error('desktop packaging needs package.json version')

  await rm(applicationBundle, { recursive: true, force: true })
  await cp(sourceBundle, applicationBundle, { recursive: true })
  await rename(sourceExecutable, productExecutable)
  await brandHelperBundles(join(contents, 'Frameworks'))
  await rm(join(resources, 'app'), { recursive: true, force: true })
  await cp(join(outputDirectory, 'desktop'), join(resources, 'app'), { recursive: true })
  // A branded .app cannot find the runtime from a workspace-relative path: the
  // bundle's main is Resources/app, not <checkout>/dist/desktop. Copy the
  // compiled CLI next to it so the packaged shell can launch a daemon for any
  // chosen workspace without XERXES_TUI_BUN_DAEMON.
  const runtimeDirectory = join(resources, 'runtime')
  await rm(runtimeDirectory, { recursive: true, force: true })
  await cp(join(outputDirectory, 'cli.js'), join(runtimeDirectory, 'cli.js'))
  // The source fingerprint stamped by writeBuildId: daemons spawned from this
  // bundled runtime announce it, matching the fingerprint the renderer build
  // baked in — without it every packaged daemon trips the compatibility
  // banner by construction.
  await cp(join(outputDirectory, 'build-id'), join(runtimeDirectory, 'build-id'))
  for (const entry of await readdir(outputDirectory, { withFileTypes: true })) {
    if (entry.isDirectory() && entry.name === 'skills') {
      await cp(join(outputDirectory, entry.name), join(runtimeDirectory, entry.name), { recursive: true })
    }
  }

  const iconName = 'xerxes-agents.icns'
  await createIcon(join(resources, iconName))

  const plistPath = join(contents, 'Info.plist')
  let plist = await readFile(plistPath, 'utf8')
  plist = replacePlistString(plist, 'CFBundleDisplayName', productName)
  plist = replacePlistString(plist, 'CFBundleExecutable', productName)
  plist = replacePlistString(plist, 'CFBundleIconFile', iconName)
  plist = replacePlistString(plist, 'CFBundleIdentifier', 'dev.xerxes.agents')
  plist = replacePlistString(plist, 'CFBundleName', productName)
  plist = replacePlistString(plist, 'CFBundleShortVersionString', version)
  plist = replacePlistString(plist, 'CFBundleVersion', version)
  await writeFile(plistPath, plist, 'utf8')

  // Modifying a vendor-signed Electron.app invalidates its outer signature.
  // An ad-hoc deep signature makes this local development bundle launchable.
  await run('codesign', ['--force', '--deep', '--sign', '-', applicationBundle])
  console.log(`packaged ${applicationBundle}`)
}

if (import.meta.main) await packageDesktopMac()
