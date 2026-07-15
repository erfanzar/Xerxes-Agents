// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
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

  expect(report.map(item => item.severity)).toEqual(['ok', 'ok', 'ok', 'ok', 'ok'])
  expect(formatDoctorReport(report)).toContain('OPENAI_API_KEY')
  expect(formatDoctorReport(report)).not.toContain('secret-value')
  expect(hasDoctorFailures(report)).toBe(false)
})

test('Bun doctor warns for absent optional setup and Windows Unix-socket limitations', () => {
  expect(checkXerxesOnPath({ findExecutable: () => null }).severity).toBe('warn')
  expect(checkProviderKeys({ environment: {} }).severity).toBe('warn')
  expect(checkXerxesHome({ home: '/missing', fileExists: () => false }).severity).toBe('warn')
  expect(checkPlatform({ platform: 'win32' }).message).toContain('Unix-socket')
})
