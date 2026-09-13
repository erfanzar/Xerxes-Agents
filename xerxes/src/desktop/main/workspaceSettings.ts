// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute } from 'node:path'
import { randomUUID } from 'node:crypto'

export function saveDesktopWorkspace(file: string, directory: string): void {
  if (!isAbsolute(directory) || /[\0\r\n]/.test(directory)) throw new Error('Choose an absolute workspace folder')
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, `${JSON.stringify({workspace:directory})}\n`, {mode:0o600})
    renameSync(temporary, file)
  } finally { rmSync(temporary, {force:true}) }
}
