// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { resolve } from 'node:path'

import {
  UpdateCommandError,
  checkBunPackageUpdate,
  compareReleaseVersions,
  executeGitUpdate,
  formatGitUpdateStatus,
  gitUpdateStatus,
  parseUpdateCommandOptions,
  planBunUpdate,
  planGitUpdate,
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
  expect(() => parseUpdateCommandOptions(['--force'])).toThrow('not supported by the Bun update command')
  expect(() => parseUpdateCommandOptions(['--apply', '--dry-run'])).toThrow(UpdateCommandError)
  expect(() => parseUpdateCommandOptions(['--package', 'example-xerxes'])).toThrow('--package requires --check')
  expect(() => parseUpdateCommandOptions(['--current-version', '1.0.0'])).toThrow('--current-version requires --package')
})

test('update parser accepts --git alone, with --dry-run, or with --apply, but not with --spec', () => {
  expect(parseUpdateCommandOptions(['--git'])).toEqual({
    apply: false,
    check: false,
    currentVersion: undefined,
    cwd: undefined,
    dryRun: false,
    git: true,
    packageName: undefined,
    packageSpec: undefined,
  })
  expect(parseUpdateCommandOptions(['--git', '--dry-run'])).toMatchObject({ dryRun: true, git: true })
  expect(parseUpdateCommandOptions(['--git', '--apply'])).toMatchObject({ apply: true, git: true })
  expect(() => parseUpdateCommandOptions(['--git', '--spec', 'file:./release'])).toThrow(
    '--git updates the managed checkout and cannot be combined with --spec',
  )
  expect(() => parseUpdateCommandOptions(['--git', '--apply', '--dry-run'])).toThrow(
    '--apply and --dry-run cannot be used together',
  )
})

interface GitUpdateRunnerOverrides {
  readonly counts?: string
  readonly failStep?: 'fetch' | 'merge' | 'install' | 'build'
  readonly isGit?: boolean
  readonly upstream?: string
}

/** Fake runner covering the git-status probes plus the four git-update steps. */
function gitUpdateRunner(calls: string[][], overrides: GitUpdateRunnerOverrides = {}): UpdateProcessRunner {
  return async argv => {
    calls.push([...argv])
    const command = argv.slice(1).join(' ')
    const failed = (step: string): boolean => overrides.failStep === step
    if (argv[0] === 'git') {
      if (command === 'rev-parse --is-inside-work-tree') {
        return overrides.isGit === false ? processResult('', 1, 'not a git repo') : processResult('true\n')
      }
      if (command === 'rev-parse --abbrev-ref HEAD') return processResult('main\n')
      if (command === 'rev-parse --short=12 HEAD') return processResult('local12345678\n')
      if (command === 'rev-parse --abbrev-ref --symbolic-full-name @{u}') {
        return overrides.upstream === ''
          ? processResult('', 1, 'no upstream')
          : processResult('origin/main\n')
      }
      if (command === 'fetch --quiet --no-tags origin') {
        return failed('fetch') ? processResult('', 1, 'network down') : processResult()
      }
      if (command === 'rev-list --left-right --count HEAD...origin/main') {
        return processResult((overrides.counts ?? '0\t1') + '\n')
      }
      if (command === 'rev-parse --short=12 origin/main') return processResult('remote1234567\n')
      if (command === 'merge --ff-only origin/main') {
        return failed('merge') ? processResult('', 1, 'refusing to merge unrelated histories') : processResult()
      }
      throw new Error('Unexpected git command: ' + argv.join(' '))
    }
    if (argv[0] === 'bun') {
      const subcommand = argv.slice(1).join(' ')
      if (subcommand === 'install --frozen-lockfile') {
        return failed('install') ? processResult('', 1, 'lockfile mismatch') : processResult('installed\n')
      }
      if (subcommand === 'run build') {
        return failed('build') ? processResult('', 1, 'type error') : processResult('built\n')
      }
    }
    throw new Error('Unexpected command: ' + argv.join(' '))
  }
}

const GIT_UPDATE_STEP_CALLS = [
  ['git', 'fetch', '--quiet', '--no-tags', 'origin'],
  ['git', 'merge', '--ff-only', 'origin/main'],
  ['bun', 'install', '--frozen-lockfile'],
  ['bun', 'run', 'build'],
]

function mutationCalls(calls: string[][]): string[][] {
  return calls.filter(call =>
    (call[0] === 'git' && (call.includes('fetch') || call.includes('merge'))) || call[0] === 'bun',
  )
}

test('--git alone reports status plus a hint and runs nothing', async () => {
  const calls: string[][] = []
  const output: string[] = []
  const result = await runUpdateCommand(['--git'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(calls),
    write: line => output.push(line),
  })
  expect(result.applied).toBe(false)
  expect(result.gitPlan).toBeUndefined()
  expect(mutationCalls(calls)).toEqual([])
  expect(output).toContain(
    'Git update: not run; use --git --dry-run to review the plan or --git --apply to execute it.',
  )
})

test('--git --dry-run prints the step plan and runs nothing', async () => {
  const calls: string[][] = []
  const output: string[] = []
  const result = await runUpdateCommand(['--git', '--dry-run'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(calls),
    write: line => output.push(line),
  })
  expect(result.applied).toBe(false)
  // The checkout is canonicalized with node:path (drive letters on Windows).
  expect(result.gitPlan?.checkout).toBe(resolve('/workspace'))
  expect(result.gitPlan?.upstream).toBe('origin/main')
  expect(mutationCalls(calls)).toEqual([])
  expect(output).toContain(`Git update plan for ${resolve('/workspace')} (upstream origin/main):`)
  expect(output).toContain('Would run (fetch): git fetch --quiet --no-tags origin')
  expect(output).toContain('Would run (merge): git merge --ff-only origin/main')
  expect(output).toContain('Would run (install): bun install --frozen-lockfile')
  expect(output).toContain('Would run (build): bun run build')
  expect(output).toContain('Dry run only; re-run with --git --apply to execute these steps.')
})

test('--git --apply runs fetch, merge, install, and build in order through the injected runner', async () => {
  const calls: string[][] = []
  const output: string[] = []
  const result = await runUpdateCommand(['--git', '--apply'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(calls),
    write: line => output.push(line),
  })
  expect(result.applied).toBe(true)
  expect(result.gitExecution).toEqual({
    checkout: resolve('/workspace'),
    completedSteps: ['fetch', 'merge', 'install', 'build'],
    ok: true,
  })
  expect(mutationCalls(calls)).toEqual(GIT_UPDATE_STEP_CALLS)
  // The divergence check must run after fetch and before merge (the status probe also runs
  // rev-list earlier, so only the post-fetch occurrence counts).
  const fetchIndex = calls.findIndex(call => call.join(' ') === 'git fetch --quiet --no-tags origin')
  const mergeIndex = calls.findIndex(call => call.join(' ') === 'git merge --ff-only origin/main')
  const divergenceIndex = calls.findIndex(
    (call, index) =>
      index > fetchIndex && call.join(' ') === 'git rev-list --left-right --count HEAD...origin/main',
  )
  expect(fetchIndex).toBeGreaterThanOrEqual(0)
  expect(divergenceIndex).toBeGreaterThan(fetchIndex)
  expect(mergeIndex).toBeGreaterThan(divergenceIndex)
  expect(output).toContain('Git update completed; restart running Xerxes processes to use the new build.')
})

test('--git --apply aborts on the first failing step and names it', async () => {
  const calls: string[][] = []
  const failure = runUpdateCommand(['--git', '--apply'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(calls, { failStep: 'install' }),
    write: () => undefined,
  })
  await expect(failure).rejects.toThrow('git update step "install" failed: lockfile mismatch')
  expect(mutationCalls(calls)).toEqual(GIT_UPDATE_STEP_CALLS.slice(0, 3))

  const fetchCalls: string[][] = []
  const fetchFailure = runUpdateCommand(['--git', '--apply'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(fetchCalls, { failStep: 'fetch' }),
    write: () => undefined,
  })
  await expect(fetchFailure).rejects.toThrow('git update step "fetch" failed: network down')
  expect(mutationCalls(fetchCalls)).toEqual(GIT_UPDATE_STEP_CALLS.slice(0, 1))
})

test('--git --apply refuses a diverged checkout and never merges', async () => {
  const calls: string[][] = []
  const refusal = runUpdateCommand(['--git', '--apply'], {
    bunExecutable: 'bun',
    cwd: '/workspace',
    runner: gitUpdateRunner(calls, { counts: '2\t3' }),
    write: () => undefined,
  })
  await expect(refusal).rejects.toThrow(
    'local checkout has diverged from origin/main (2 ahead, 3 behind); refusing to merge --ff-only',
  )
  expect(mutationCalls(calls)).toEqual(GIT_UPDATE_STEP_CALLS.slice(0, 1))
})

test('planGitUpdate fails honestly for a non-git directory or a checkout without upstream', async () => {
  const notGit = planGitUpdate({
    cwd: '/tmp/source-install',
    runner: gitUpdateRunner([], { isGit: false }),
  })
  await expect(notGit).rejects.toThrow('is not a git checkout; a git update requires the managed clone')

  const noUpstream = planGitUpdate({
    cwd: '/workspace',
    runner: gitUpdateRunner([], { upstream: '' }),
  })
  await expect(noUpstream).rejects.toThrow('has no upstream ref')
})

test('executeGitUpdate requires the four named steps and aborts with step names on failure', async () => {
  const calls: string[][] = []
  const plan = await planGitUpdate({ bunExecutable: 'bun', cwd: '/workspace', runner: gitUpdateRunner([]) })
  expect(plan.steps.map(step => step.name)).toEqual(['fetch', 'merge', 'install', 'build'])
  expect(plan.steps[2]?.timeout).toBeGreaterThan(1_000)
  expect(plan.steps[3]?.timeout).toBeGreaterThan(1_000)

  const buildFailure = executeGitUpdate(plan, {
    runner: gitUpdateRunner(calls, { failStep: 'build' }),
  })
  await expect(buildFailure).rejects.toThrow('git update step "build" failed: type error')
  expect(mutationCalls(calls)).toEqual(GIT_UPDATE_STEP_CALLS)
})
