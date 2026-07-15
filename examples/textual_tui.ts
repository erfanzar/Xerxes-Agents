// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Launch the native React + OpenTUI terminal UI.
 *
 * This replaces the former Textual example. The default mode is descriptive
 * and side-effect-free; pass --launch to start the bundled OpenTUI client.
 */

import { resolve } from 'node:path'

import { divider, hasFlag, runMain } from './native_demo_support.js'

export function nativeTuiLaunchCommand(projectRoot = process.cwd()): string[] {
  return [process.execPath, resolve(projectRoot, 'xerxes/dist/ui/entry.js')]
}

export async function launchNativeTui(projectRoot = process.cwd()): Promise<number> {
  const child = Bun.spawn({
    cmd: nativeTuiLaunchCommand(projectRoot),
    cwd: projectRoot,
    stdin: 'inherit',
    stdout: 'inherit',
    stderr: 'inherit',
  })
  return child.exited
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  divider('XERXES NATIVE TERMINAL UI')
  console.log('The TypeScript TUI uses React + OpenTUI and connects to the Bun daemon/runtime.')
  console.log(`Launch command: ${nativeTuiLaunchCommand().join(' ')}`)
  if (!hasFlag(args, '--launch')) {
    console.log('No UI was started. Re-run with --launch when you want an interactive terminal session.')
    return
  }
  process.exitCode = await launchNativeTui()
}

if (import.meta.main) runMain(main)
