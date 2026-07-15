// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { join } from 'node:path'

import {
  CLAUDE_CODE_PACKAGE,
  InstallCommandError,
  parseInstallCommandOptions,
  runInstallCommand,
} from '../src/runtime/companionInstall.js'

test('Bun install command parses Claude Code aliases and retires the managed Node installer', () => {
  expect(parseInstallCommandOptions(['--cloud-code', '--force', '--dry-run'])).toEqual({
    dryRun: true,
    force: true,
  })
  expect(() => parseInstallCommandOptions(['--node'])).toThrow(InstallCommandError)
  expect(() => parseInstallCommandOptions([])).toThrow('Choose an install target')
})

test('Bun install command uses a native global install, with safe dry-run and installed checks', async () => {
  const installedOutput: string[] = []
  const installed = await runInstallCommand(['--claude-code'], {
    findExecutable: name => name === 'claude' ? '/opt/bin/claude' : null,
    run: async () => {
      throw new Error('already-installed commands must not spawn')
    },
    write: line => installedOutput.push(line),
  })
  expect(installed).toEqual({ command: [], status: 'already-installed' })
  expect(installedOutput.join('\n')).toContain('/opt/bin/claude')

  const dryRunOutput: string[] = []
  const dryRun = await runInstallCommand(['--cloud-code', '--force', '--dry-run'], {
    bunExecutable: 'bun',
    findExecutable: () => '/opt/bin/claude',
    run: async () => {
      throw new Error('dry-run commands must not spawn')
    },
    write: line => dryRunOutput.push(line),
  })
  expect(dryRun).toEqual({
    command: ['bun', 'add', '--global', '--force', CLAUDE_CODE_PACKAGE],
    status: 'dry-run',
  })
  expect(dryRunOutput).toEqual(['Would run: bun add --global --force ' + CLAUDE_CODE_PACKAGE])

  const commands: string[][] = []
  const installOutput: string[] = []
  const installedByBun = await runInstallCommand(['--claude-code', '--force'], {
    bunExecutable: '/opt/bun',
    findExecutable: () => null,
    run: async command => {
      commands.push([...command])
      return 0
    },
    write: line => installOutput.push(line),
  })
  expect(installedByBun.status).toBe('installed')
  expect(commands).toEqual([['/opt/bun', 'add', '--global', '--force', CLAUDE_CODE_PACKAGE]])
  expect(installOutput).toEqual(['Claude Code installed.', 'Login with: claude auth login'])
})

test('Bun CLI exposes the companion installer without starting Python', async () => {
  const child = Bun.spawn([
    process.execPath,
    join(import.meta.dir, '../src/cli.ts'),
    'install',
    '--cloud-code',
    '--force',
    '--dry-run',
  ], {
    stderr: 'pipe',
    stdout: 'pipe',
  })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ])

  expect(exitCode).toBe(0)
  expect(stderr).toBe('')
  expect(stdout).toBe('Would run: ' + process.execPath + ' add --global --force ' + CLAUDE_CODE_PACKAGE + '\n')
})
