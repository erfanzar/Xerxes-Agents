// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { cp, mkdir, readFile, rename, rm } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { dirname, join, relative, resolve, isAbsolute, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { copyDesktopRuntimeAssets } from './packageDesktopMac.js'

/** A self-contained portable Windows application; no global Bun installation needed. */
export async function packageDesktopWindows(): Promise<string | undefined> {
  if (process.platform !== 'win32') return undefined
  const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
  const output = process.env.XERXES_DESKTOP_PACKAGE_DIR?.trim()
    ? resolve(process.env.XERXES_DESKTOP_PACKAGE_DIR.trim()) : join(packageDirectory, 'dist')
  const destination = join(output, `Xerxes Agents-win32-${process.arch}`)
  const staging = destination + '.staging'
  const local = relative(output, staging)
  if (!local || local === '..' || local.startsWith('..' + sep) || isAbsolute(local)) throw new Error('Unsafe desktop package path')
  const electron = dirname(Bun.resolveSync('electron', packageDirectory))
  const { version } = JSON.parse(await readFile(join(electron, 'package.json'), 'utf8')) as { version: string }
  // Resolving Electron alone does not install its native binary. Fail explicitly.
  const native = join(electron, 'dist')
  if (!await Bun.file(join(native, 'electron.exe')).exists()) throw new Error(`Electron ${version} native binary is missing. Run bun ${join(electron, 'install.js')} and retry.`)
  await mkdir(output, { recursive: true })
  await rm(staging, { recursive: true, force: true })
  await cp(native, staging, { recursive: true })
  const resources = join(staging, 'resources')
  await cp(join(packageDirectory, 'dist', 'desktop'), join(resources, 'app'), { recursive: true })
  const runtime = join(resources, 'runtime')
  await mkdir(runtime, { recursive: true })
  for (const file of ['cli.js', 'build-id']) await cp(join(packageDirectory, 'dist', file), join(runtime, file))
  await copyDesktopRuntimeAssets(join(packageDirectory, 'dist'), runtime)
  await cp(process.execPath, join(resources, 'bun.exe'))
  await cp(join(packageDirectory, 'assets', 'notices', 'Bun-LICENSE.md'), join(resources, 'Bun-LICENSE.md'))
  await cp(join(packageDirectory, '..', 'LICENSE'), join(resources, 'Xerxes-LICENSE'))
  await rename(join(staging, 'electron.exe'), join(staging, 'Xerxes Agents.exe'))
  // Never partially delete a runnable package when Windows locks an open file.
  const previous = destination + '.previous-' + randomUUID()
  let moved = false
  try { await rename(destination, previous); moved = true }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
  try { await rename(staging, destination) }
  catch (error) { if (moved) await rename(previous, destination); throw error }
  if (moved) {
    try { await rm(previous, { recursive: true, force: true }) }
    catch (error) { console.warn(`Previous desktop package is still in use; close it before removing ${previous}: ${String(error)}`) }
  }
  console.log(`packaged portable Windows app: ${destination}`)
  return destination
}
if (import.meta.main) await packageDesktopWindows()
