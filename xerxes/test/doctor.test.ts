// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  checkComputerUse,
  checkPlatform,
  checkProviderKeys,
  checkXerxesHome,
  checkXerxesOnPath,
  formatDoctorReport,
  hasDoctorFailures,
  runAllDoctorChecks,
} from '../src/runtime/doctor.js'

test('Bun doctor checks use injected host facts and do not expose credential values', () => {
  const options = {
    bunVersion: '1.3.12',
    environment: { OPENAI_API_KEY: 'secret-value' },
    fileExists: (path: string) => path === '/home/xerxes',
    findExecutable: () => '/bin/xerxes',
    home: '/home/xerxes',
    platform: 'linux' as const,
  }
  const report = runAllDoctorChecks(options)

  expect(report.map(item => item.severity)).toEqual(['ok', 'ok', 'ok', 'ok', 'ok', 'ok'])
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

test('Bun doctor warns for absent optional setup and reports the win32 daemon transport', () => {
  expect(checkXerxesOnPath({ findExecutable: () => null }).severity).toBe('warn')
  expect(checkProviderKeys({ environment: {} }).severity).toBe('warn')
  expect(checkXerxesHome({ home: '/missing', fileExists: () => false }).severity).toBe('warn')
  // Native Windows runs the WebSocket control transport by default; forcing
  // the Unix socket there is the only warned configuration.
  expect(checkPlatform({ platform: 'win32' }).severity).toBe('ok')
  expect(checkPlatform({ platform: 'win32' }).message).toContain('websocket')
  expect(
    checkPlatform({ platform: 'win32', environment: { XERXES_DAEMON_TRANSPORT: 'unix' } }).severity,
  ).toBe('warn')
  expect(checkPlatform({ platform: 'linux' }).severity).toBe('ok')
})
