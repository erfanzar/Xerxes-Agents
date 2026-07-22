// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  UpdateCommandError,
  checkBunPackageUpdate,
  compareReleaseVersions,
  formatGitUpdateStatus,
  gitUpdateStatus,
  parseUpdateCommandOptions,
  planBunUpdate,
  runUpdateCommand,
  type UpdateProcessResult,
  type UpdateProcessRunner,
} from '../src/runtime/update.js'

function processResult(
  stdout = '',
  exitCode = 0,
  stderr = '',
  timedOut = false,
): UpdateProcessResult {
  return { exitCode, stderr, stdout, timedOut }
}

function currentGitRunner(calls: string[][]): UpdateProcessRunner {
  return async argv => {
    calls.push([...argv])
    const command = argv.slice(1).join(' ')
    if (command === 'rev-parse --is-inside-work-tree') return processResult('true\n')
    if (command === 'rev-parse --abbrev-ref HEAD') return processResult('main\n')
    if (command === 'rev-parse --short=12 HEAD') return processResult('local12345678\n')
    if (command === 'rev-parse --abbrev-ref --symbolic-full-name @{u}') return processResult('origin/main\n')
    if (command === 'rev-list --left-right --count HEAD...origin/main') return processResult('0\t0\n')
    if (command === 'rev-parse --short=12 origin/main') return processResult('remote1234567\n')
    if (command === 'fetch --quiet --no-tags origin') return processResult()
    throw new Error('Unexpected command: ' + argv.join(' '))
  }
}

test('git update status uses a local fallback upstream and fetches only when explicitly requested', async () => {
  const calls: string[][] = []
  const runner: UpdateProcessRunner = async argv => {
    calls.push([...argv])
    const command = argv.slice(1).join(' ')
    if (command === 'rev-parse --is-inside-work-tree') return processResult('true\n')
    if (command === 'rev-parse --abbrev-ref HEAD') return processResult('feature\n')
    if (command === 'rev-parse --short=12 HEAD') return processResult('local12345678\n')
    if (command === 'rev-parse --abbrev-ref --symbolic-full-name @{u}') return processResult('', 1, 'no upstream')
    if (command === 'rev-parse --verify origin/feature') return processResult('remote-tip\n')
    if (command === 'fetch --quiet --no-tags origin') return processResult()
    if (command === 'rev-list --left-right --count HEAD...origin/feature') return processResult('2\t3\n')
    if (command === 'rev-parse --short=12 origin/feature') return processResult('remote1234567\n')
    throw new Error('Unexpected command: ' + argv.join(' '))
  }

  const status = await gitUpdateStatus({ cwd: '/workspace', fetch: true, runner })
  expect(status).toEqual({
    isGit: true,
    branch: 'feature',
    headHash: 'local12345678',
    upstream: 'origin/feature',
    upstreamHash: 'remote1234567',
    aheadCount: 2,
    behindCount: 3,
    error: '',
  })
  expect(calls).toContainEqual(['git', 'fetch', '--quiet', '--no-tags', 'origin'])
  expect(formatGitUpdateStatus(status)).toContain('3 upstream commit(s) available')

  const noFetchCalls: string[][] = []
  await gitUpdateStatus({ cwd: '/workspace', runner: currentGitRunner(noFetchCalls) })
  expect(noFetchCalls.some(command => command.includes('fetch'))).toBe(false)
})

test('Bun update plans require a caller-provided package or source spec', () => {
  expect(() => planBunUpdate({ bunExecutable: 'bun', environment: {} })).toThrow(UpdateCommandError)
  expect(() => planBunUpdate({ packageSpec: '--unsafe' })).toThrow('cannot start with a flag')

  expect(planBunUpdate({
    bunExecutable: 'bun',
    environment: { XERXES_PACKAGE: 'github:example/xerxes#stable' },
  })).toEqual({
    argv: ['bun', 'add', '--global', 'github:example/xerxes#stable'],
    source: 'environment',
    spec: 'github:example/xerxes#stable',
  })
  expect(planBunUpdate({ bunExecutable: 'bun', packageSpec: 'file:./release' })).toEqual({
    argv: ['bun', 'add', '--global', 'file:./release'],
    source: 'argument',
    spec: 'file:./release',
  })
})

test('an invalid explicit spec fails loudly instead of falling back to the environment spec', () => {
  const environment = { XERXES_PACKAGE: 'github:example/xerxes#stable' }

  for (const invalid of ['--unsafe', '', '   ', 'bad\nspec']) {
    expect(() => planBunUpdate({ bunExecutable: 'bun', environment, packageSpec: invalid }))
      .toThrow(UpdateCommandError)
  }
  // The environment spec is consulted only when no explicit spec was given.
  expect(planBunUpdate({ bunExecutable: 'bun', environment }).spec).toBe('github:example/xerxes#stable')
  expect(planBunUpdate({ bunExecutable: 'bun', environment, packageSpec: 'file:./release' }).source)
    .toBe('argument')
})

test('named registry checks use the supplied fetch boundary and do not guess an installed version', async () => {
  const urls: string[] = []
  const available = await checkBunPackageUpdate({
    packageName: '@example/xerxes',
    currentVersion: '1.2.0',
    fetch: async url => {
      urls.push(url)
      return new Response(JSON.stringify({ version: '1.3.0' }), { status: 200 })
    },
  })
  expect(urls).toEqual(['https://registry.npmjs.org/%40example%2Fxerxes/latest'])
  expect(available).toEqual({
    packageName: '@example/xerxes',
    currentVersion: '1.2.0',
    latestVersion: '1.3.0',
    updateAvailable: true,
  })

  const unversioned = await checkBunPackageUpdate({
    packageName: 'example-xerxes',
    fetch: async () => new Response(JSON.stringify({ version: '2.0.0' }), { status: 200 }),
  })
  expect(unversioned).toEqual({ packageName: 'example-xerxes', latestVersion: '2.0.0' })
  expect(compareReleaseVersions('2.0.0', '2.0.0-rc.1')).toBe(1)
  expect(compareReleaseVersions('git-main', '1.0.0')).toBeUndefined()
})

test('update command remains status-only until dry-run or apply is explicitly supplied', async () => {
  const statusCalls: string[][] = []
  const statusOutput: string[] = []
  const status = await runUpdateCommand([], {
    cwd: '/workspace',
    fetch: async () => {
      throw new Error('status-only update must not query a registry')
    },
    runner: currentGitRunner(statusCalls),
    write: line => statusOutput.push(line),
  })
  expect(status.applied).toBe(false)
  expect(statusCalls.every(command => command[0] === 'git')).toBe(true)
  expect(statusOutput).toContain(
    'No Bun update command was run. Use --dry-run to review a spec or --apply to execute one.',
  )

  const dryRunCalls: string[][] = []
  const dryRunOutput: string[] = []
  const dryRun = await runUpdateCommand(['--dry-run', '--spec', 'file:./release'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: currentGitRunner(dryRunCalls),
    write: line => dryRunOutput.push(line),
  })
  expect(dryRun.applied).toBe(false)
  expect(dryRun.plan?.argv).toEqual(['bun', 'add', '--global', 'file:./release'])
  expect(dryRunCalls.every(command => command[0] === 'git')).toBe(true)
  expect(dryRunOutput).toContain('Would run: bun add --global file:./release')

  const applyCalls: string[][] = []
  const runner: UpdateProcessRunner = async argv => {
    if (argv[0] === 'bun') {
      applyCalls.push([...argv])
      return processResult('Bun command output')
    }
    return currentGitRunner(applyCalls)(argv, { cwd: '/workspace', timeout: 1_000 })
  }
  const applied = await runUpdateCommand(['--apply', '--spec', 'file:./release'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner,
    write: () => undefined,
  })
  expect(applied.applied).toBe(true)
  expect(applyCalls).toContainEqual(['bun', 'add', '--global', 'file:./release'])
})

test('update parser rejects Python-era and unsafe option combinations', () => {
  expect(() => parseUpdateCommandOptions(['--git'])).toThrow('not supported by the Bun update command')
  expect(() => parseUpdateCommandOptions(['--apply', '--dry-run'])).toThrow(UpdateCommandError)
  expect(() => parseUpdateCommandOptions(['--package', 'example-xerxes'])).toThrow('--package requires --check')
  expect(() => parseUpdateCommandOptions(['--current-version', '1.0.0'])).toThrow('--current-version requires --package')
})
