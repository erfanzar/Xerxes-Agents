// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cp, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export function installerName(version: string, arch: string): string {
  if (!/^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$/.test(version))
    throw new Error('Invalid installer version')
  if (arch !== 'arm64' && arch !== 'x64') throw new Error('Unsupported macOS architecture')
  return `Xerxes-Agents-${version}-macOS-${arch}.dmg`
}

export async function buildDesktopInstaller(): Promise<string> {
  if (process.platform !== 'darwin') throw new Error('The macOS installer must be built on macOS')
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
  const { version } = await Bun.file(join(root, 'package.json')).json()
  const output = join(root, 'dist', installerName(version, process.arch))
  const staging = await mkdtemp(join(tmpdir(), 'xerxes-installer-'))
  try {
    await cp(join(root, 'dist', 'Xerxes Agents.app'), join(staging, 'Xerxes Agents.app'), {
      recursive: true,
    })
    await symlink('/Applications', join(staging, 'Applications'))
    await writeFile(
      join(staging, 'Start here.txt'),
      `Welcome to Xerxes Agents\n\n1. Drag Xerxes Agents into Applications.\n2. Open it from Applications, then eject this disk.\n3. Choose a project folder and connect your model provider.\n\nBun is included. Git is needed only for Git features; SSH workspaces use your existing SSH configuration.\n\nTo update, quit the desktop app and replace it in Applications. Existing sessions and provider settings are kept outside the app.\nTo uninstall, move the app to Trash. Your saved data is retained in ~/.xerxes (or XERXES_HOME).\n`,
    )
    const child = Bun.spawn(
      [
        'hdiutil',
        'create',
        '-volname',
        'Install Xerxes Agents',
        '-srcfolder',
        staging,
        '-ov',
        '-format',
        'UDZO',
        output,
      ],
      { stdout: 'inherit', stderr: 'inherit' },
    )
    if ((await child.exited) !== 0) throw new Error('Could not build the installer disk image')
    console.log(`Installer ready: ${output}`)
    return output
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}
if (import.meta.main) await buildDesktopInstaller()
