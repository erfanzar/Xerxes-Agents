// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, realpathSync, readFileSync, mkdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute } from 'node:path'
import { randomUUID } from 'node:crypto'

export function loadDesktopWorkspaces(file: string): string[] {
  if (!existsSync(file)) return []
  const value = JSON.parse(readFileSync(file, 'utf8'))
  const paths: unknown[] = [...(Array.isArray(value.directories) ? value.directories : []), value.workspace]
  return [...new Set(paths.filter((path): path is string => typeof path === 'string' && isAbsolute(path) && !/[\x00-\x1f]/.test(path)))]
}

export function saveDesktopWorkspace(file: string, directory: string): void {
  if (!isAbsolute(directory) || /[\0\r\n]/.test(directory)) throw new Error('Choose an absolute workspace folder')
  if (existsSync(directory)) directory = realpathSync(directory)
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, `${JSON.stringify({workspace:directory, directories:[...new Set([...loadDesktopWorkspaces(file), directory])]})}\n`, {mode:0o600})
    renameSync(temporary, file)
  } finally { rmSync(temporary, {force:true}) }
}
