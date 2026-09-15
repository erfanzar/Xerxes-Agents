// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cp, mkdir, mkdtemp, rename, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export function installerName(version: string, arch: string): string {
  if (!/^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$/.test(version))
    throw new Error('Invalid installer version')
  if (arch !== 'arm64' && arch !== 'x64') throw new Error('Unsupported macOS architecture')
  return `Xerxes-Agents-${version}-macOS-${arch}.dmg`
}

/** Finder settings belong to this image, never the user's global preferences. */
export function installerLayoutScript(mount: string): string {
  const folder = JSON.stringify(mount)
  const background = JSON.stringify(join(mount, '.background', 'background.tiff'))
  return `tell application "Finder"
    set installerFolder to POSIX file ${folder} as alias
    open installerFolder
    delay 1
    set installerWindow to front window
    set current view of installerWindow to icon view
    set toolbar visible of installerWindow to false
    set statusbar visible of installerWindow to false
    set bounds of installerWindow to {120, 120, 800, 588}
    set options to icon view options of installerWindow
    set arrangement of options to not arranged
    set icon size of options to 96
    set text size of options to 13
    set background picture of options to POSIX file ${background}
    set position of item "Xerxes Agents.app" of installerFolder to {170, 228}
    set position of item "Applications" of installerFolder to {510, 228}
    update installerFolder without registering applications
    delay 2
    close installerWindow
  end tell`
}

async function run(command: string[]): Promise<void> {
  const child = Bun.spawn(command, { stdout: 'inherit', stderr: 'inherit' })
  if (await child.exited !== 0) throw new Error(`Installer command failed: ${command[0]}`)
}

export async function buildDesktopInstaller(): Promise<string> {
  if (process.platform !== 'darwin') throw new Error('The macOS installer must be built on macOS')
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
  const { version } = await Bun.file(join(root, 'package.json')).json()
  const packageDirectory = process.env.XERXES_DESKTOP_PACKAGE_DIR?.trim()
    ? resolve(process.env.XERXES_DESKTOP_PACKAGE_DIR.trim()) : join(root, 'dist')
  const output = join(packageDirectory, installerName(version, process.arch))
  const temporary = await mkdtemp(join(tmpdir(), 'xerxes-installer-'))
  const staging = join(temporary, 'contents')
  const mount = join(temporary, 'mounted')
  const writable = join(temporary, 'layout.dmg')
  const finished = join(temporary, 'installer.dmg')
  let attached = false
  try {
    await mkdir(join(staging, '.background'), { recursive: true })
    await cp(join(packageDirectory, 'Xerxes Agents.app'), join(staging, 'Xerxes Agents.app'), { recursive: true })
    await symlink('/Applications', join(staging, 'Applications'))
    await cp(join(root, '..', 'assets', 'installer', 'background.tiff'), join(staging, '.background', 'background.tiff'))
    await run(['hdiutil', 'create', '-volname', `Xerxes Layout ${crypto.randomUUID().slice(0, 8)}`, '-srcfolder', staging, '-format', 'UDRW', '-fs', 'HFS+', writable])
    await run(['hdiutil', 'attach', '-nobrowse', '-noautoopen', '-mountpoint', mount, writable])
    attached = true
    await run(['osascript', '-e', installerLayoutScript(mount)])
    await run(['diskutil', 'renameVolume', mount, `Install Xerxes Agents ${version}`])
    // Detach flushes Finder's .DS_Store before creating the read-only image.
    await run(['hdiutil', 'detach', mount])
    attached = false
    await run(['hdiutil', 'convert', writable, '-format', 'UDZO', '-o', finished])
    await run(['hdiutil', 'verify', finished])
    // Replace an earlier installer only after the new image is complete.
    await rename(finished, output)
    console.log(`Installer ready: ${output}`)
    return output
  } finally {
    // A failed detach retains the mounted image for recovery rather than
    // recursively deleting files from a volume that is still mounted.
    if (attached) await run(['hdiutil', 'detach', mount])
    await rm(temporary, { recursive: true, force: true })
  }
}
if (import.meta.main) await buildDesktopInstaller()
