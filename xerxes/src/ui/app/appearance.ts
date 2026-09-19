// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdir, rename } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { atom } from 'nanostores'

export type Appearance = 'chrome' | 'transparent'
export const $appearance = atom<Appearance>('chrome')
export const appearancePath = () => join(process.env.XERXES_HOME?.trim() || join(homedir(), '.xerxes'), 'tui-appearance.json')

export async function loadAppearance(path = appearancePath()): Promise<Appearance> {
  try {
    const value: unknown = await Bun.file(path).json()
    if (value && typeof value === 'object' && 'appearance' in value && value.appearance === 'transparent') return 'transparent'
    return 'chrome'
  } catch (error) {
    if ((error as { code?: string }).code === 'ENOENT' || error instanceof SyntaxError) return 'chrome'
    throw error
  }
}

// Write before publishing, so a failed save never claims the choice persists.
export async function saveAppearance(appearance: Appearance, path = appearancePath()): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = `${path}.${process.pid}.tmp`
  await Bun.write(temporary, JSON.stringify({ appearance }) + '\n', { mode: 0o600 })
  await rename(temporary, path)
  $appearance.set(appearance)
}
