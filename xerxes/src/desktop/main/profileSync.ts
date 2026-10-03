// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Copying this Mac's provider profiles to an SSH workspace host, so the keys
 * set up here work there without re-entering them. Pure of Electron so the
 * rules are testable headlessly; main.ts runs it after each SSH connect.
 */

import { watch } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { basename, dirname } from 'node:path'

export type ProfileImportCall = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>

export interface ProfileCopyResult {
  readonly imported: readonly string[]
  /** The profile the host started using because it had none of its own. */
  readonly selected?: string
  /** Why nothing was copied; the connection itself is unaffected. */
  readonly error?: string
}

/**
 * The profiles in `profilesFile` that carry a key. Sign-in profiles (Claude
 * Code, ChatGPT) have none — their credential is this Mac's login, which the
 * host has to make itself.
 */
export async function localKeyedProfiles(profilesFile: string): Promise<Record<string, unknown>[]> {
  let text: string
  try { text = await readFile(profilesFile, 'utf8') }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw error
  }
  const document: unknown = JSON.parse(text)
  const profiles = document && typeof document === 'object' && !Array.isArray(document) ? (document as Record<string, unknown>).profiles : undefined
  if (!profiles || typeof profiles !== 'object' || Array.isArray(profiles)) return []
  return Object.entries(profiles as Record<string, unknown>).flatMap(([name, value]) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const profile = value as Record<string, unknown>
    if (typeof profile.api_key !== 'string' || !profile.api_key.trim()) return []
    return [{
      name,
      provider: profile.provider,
      base_url: profile.base_url,
      api_key: profile.api_key,
      model: profile.model,
      ...(profile.sampling && typeof profile.sampling === 'object' ? { sampling: profile.sampling } : {}),
    }]
  })
}

/** This Mac's chosen profile, which a host with no choice of its own adopts. */
export async function localActiveProfile(profilesFile: string): Promise<string | undefined> {
  try {
    const document: unknown = JSON.parse(await readFile(profilesFile, 'utf8'))
    const active = document && typeof document === 'object' && !Array.isArray(document) ? (document as Record<string, unknown>).active : undefined
    return typeof active === 'string' && active.trim() ? active.trim() : undefined
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}

/** Send the keyed profiles to the host's runtime; failures are reported, never thrown. */
export async function copyProfilesToRemote(call: ProfileImportCall, profilesFile: string): Promise<ProfileCopyResult> {
  try {
    const profiles = await localKeyedProfiles(profilesFile)
    if (!profiles.length) return { imported: [] }
    const active = await localActiveProfile(profilesFile)
    const result = await call('provider.import', { profiles, ...(active ? { active } : {}) })
    if (result.ok !== true) return { imported: [], error: typeof result.error === 'string' ? result.error : 'The host runtime refused the profiles.' }
    return {
      imported: Array.isArray(result.imported) ? result.imported.filter((name): name is string => typeof name === 'string') : [],
      ...(typeof result.selected === 'string' ? { selected: result.selected } : {}),
    }
  } catch (error) {
    return { imported: [], error: error instanceof Error ? error.message : String(error) }
  }
}

/**
 * Call `changed` shortly after this Mac's profiles file is written, so a
 * provider added or a key changed here reaches a connected SSH host without
 * reconnecting. The folder is watched, not the file: saves replace the file
 * by rename, which ends a watch on the old one. Returns the stop function.
 */
export function watchProfiles(profilesFile: string, changed: () => void, debounceMs = 500): () => void {
  const name = basename(profilesFile)
  let timer: ReturnType<typeof setTimeout> | undefined
  let watcher: ReturnType<typeof watch> | undefined
  try {
    watcher = watch(dirname(profilesFile), (_event, file) => {
      if (file !== null && file !== name) return
      clearTimeout(timer)
      timer = setTimeout(changed, debounceMs)
    })
    watcher.on('error', error => console.warn(`Stopped watching ${profilesFile}: ${error instanceof Error ? error.message : String(error)}`))
  } catch (error) {
    console.warn(`Cannot watch ${profilesFile}; profiles reach the host on each connect and task open: ${error instanceof Error ? error.message : String(error)}`)
  }
  return () => { clearTimeout(timer); watcher?.close() }
}
