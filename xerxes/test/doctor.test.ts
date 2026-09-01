// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  checkAgentDefinitions,
  checkComputerUse,
  checkPlatform,
  checkProviderKeys,
  checkSubsystemStorage,
  checkWindowsTooling,
  checkXerxesHome,
  checkXerxesOnPath,
  formatDoctorReport,
  hasDoctorFailures,
  runAllDoctorChecks,
} from '../src/runtime/doctor.js'
import { loadAgentDefinitions } from '../src/agents/definitions.js'

test('Bun doctor checks use injected host facts and do not expose credential values', () => {
  const options = {
    bunVersion: '1.3.12',
    environment: { OPENAI_API_KEY: 'secret-value' },
    fileExists: (path: string) => path === '/home/xerxes' || path.startsWith('/home/xerxes/'),
    findExecutable: () => '/bin/xerxes',
    home: '/home/xerxes',
    platform: 'linux' as const,
  }
  const report = runAllDoctorChecks(options)

  // Asserted as a property rather than a positional list of severities: the old
  // form broke whenever a check was added and never said which one regressed.
  expect(report.filter(item => item.severity !== 'ok')).toEqual([])
  expect(report.length).toBeGreaterThanOrEqual(6)
  expect(formatDoctorReport(report)).toContain('OPENAI_API_KEY')
  expect(formatDoctorReport(report)).not.toContain('secret-value')
  expect(hasDoctorFailures(report)).toBe(false)
})

test('computer_use doctor check reports platform fit, missing tools, and permission guidance', () => {
  expect(checkComputerUse({ platform: 'linux' }).severity).toBe('ok')
  expect(checkComputerUse({ platform: 'linux' }).message).toContain('not applicable')

  const missing = checkComputerUse({ platform: 'darwin', fileExists: path => !path.includes('sips') })
  expect(missing.severity).toBe('warn')
  expect(missing.message).toContain('sips')

  const ready = checkComputerUse({ platform: 'darwin', fileExists: () => true })
  expect(ready.severity).toBe('ok')
  expect(ready.message).toContain('Accessibility')
})

test('Bun doctor warns for absent optional setup', () => {
  expect(checkXerxesOnPath({ findExecutable: () => null }).severity).toBe('warn')
  expect(checkProviderKeys({ environment: {} }).severity).toBe('warn')
  expect(checkXerxesHome({ home: '/missing', fileExists: () => false }).severity).toBe('warn')
})

test('subsystem storage doctor check reports present and missing directories', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-doctor-subsystems-'))
  try {
    await mkdir(join(directory, 'scheduler'))
    await mkdir(join(directory, 'governed-memory'))
    await mkdir(join(directory, 'capabilities'))
    await mkdir(join(directory, 'telemetry'))
    const allPresent = checkSubsystemStorage({ home: directory, fileExists: () => true })
    expect(allPresent.severity).toBe('ok')
    expect(allPresent.message).toContain('scheduler')

    const missing = checkSubsystemStorage({ home: directory, fileExists: () => false })
    expect(missing.severity).toBe('warn')
    expect(missing.message).toContain('missing')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('Bun doctor reports each platform as supported and names its control transport', () => {
  // Windows is a supported host: the control channel is a named pipe rather than
  // a Unix socket. This check previously warned that native Windows could not run
  // the daemon at all.
  const windows = checkPlatform({ platform: 'win32' })
  expect(windows.severity).toBe('ok')
  expect(windows.message).toContain('named pipe')

  const posix = checkPlatform({ platform: 'linux' })
  expect(posix.severity).toBe('ok')
  expect(posix.message).toContain('Unix socket')
})

test('Bun doctor flags missing Windows console tooling only on Windows', () => {
  // powershell.exe is how the runtime reads a process command line and how the
  // TUI reaches the clipboard; there is no `ps` to fall back to.
  const missing = checkWindowsTooling({ platform: 'win32', findExecutable: () => null })
  expect(missing.severity).toBe('warn')
  expect(missing.message).toContain('powershell.exe')

  const present = checkWindowsTooling({ platform: 'win32', findExecutable: () => 'C:\\Windows\\system32\\x.exe' })
  expect(present.severity).toBe('ok')

  expect(checkWindowsTooling({ platform: 'linux', findExecutable: () => null }).severity).toBe('ok')
})

test('agent-specs doctor check warns on broken specs and reports a clean catalog as ok', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-doctor-agent-specs-'))
  const home = join(directory, 'home')
  const priorHome = process.env.XERXES_HOME
  try {
    // The check reads the loader's ambient view (like config provenance), so
    // injection happens through XERXES_HOME and a forced fresh catalog load.
    await mkdir(join(home, 'agents'), { recursive: true })
    process.env.XERXES_HOME = home
    await writeFile(join(home, 'agents', 'broken.yaml'), '- this is not an agent mapping\n', 'utf8')
    loadAgentDefinitions()

    const warn = checkAgentDefinitions()
    expect(warn.name).toBe('agent-specs')
    expect(warn.severity).toBe('warn')
    expect(warn.message).toContain('broken.yaml')
    expect(warn.fixHint).toContain('broken.yaml')

    // Once the offending file is gone the same check reports ok.
    await rm(join(home, 'agents', 'broken.yaml'))
    loadAgentDefinitions()
    const ok = checkAgentDefinitions()
    expect(ok.name).toBe('agent-specs')
    expect(ok.severity).toBe('ok')
    expect(ok.fixHint).toBe('')
  } finally {
    if (priorHome === undefined) delete process.env.XERXES_HOME
    else process.env.XERXES_HOME = priorHome
    await rm(directory, { recursive: true, force: true })
    // Reset the shared loader state so later tests do not observe this env.
    loadAgentDefinitions()
  }
})
