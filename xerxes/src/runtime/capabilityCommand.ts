// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { CapabilityRegistry, type CapabilityManifest } from './capabilityRegistry.js'

export type CapabilityCommandAction = 'register' | 'unregister' | 'list' | 'diff'

export interface CapabilityCommandOptions {
  readonly action: CapabilityCommandAction
  readonly id?: string | undefined
  readonly file?: string | undefined
  readonly manifestJson?: string | undefined
  readonly fromSnapshot?: string | undefined
  readonly toSnapshot?: string | undefined
  readonly directory?: string | undefined
}

export interface CapabilityCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runCapabilityCommand(options: CapabilityCommandOptions): Promise<CapabilityCommandResult> {
  const directory = options.directory ?? join(xerxesHome(), 'capabilities')
  await mkdir(directory, { recursive: true })
  const snapshotPath = join(directory, 'manifests.json')
  const registry = new CapabilityRegistry()
  const existing = await loadManifests(snapshotPath)
  for (const manifest of existing) registry.register(manifest)

  switch (options.action) {
    case 'register': {
      if (!options.id) return { ok: false, error: 'register requires --id' }
      let manifestJson = options.manifestJson
      if (options.file) {
        manifestJson = await readFile(options.file, 'utf8')
      }
      if (!manifestJson) return { ok: false, error: 'register requires --file or inline JSON via --manifest-json' }
      const parsed = JSON.parse(manifestJson) as unknown
      if (!isManifest(parsed)) return { ok: false, error: 'manifest must be { id, capabilities: [...] }' }
      if (parsed.id !== options.id) return { ok: false, error: 'manifest id does not match --id' }
      registry.register(parsed)
      await saveManifests(snapshotPath, registry)
      return { ok: true, message: `registered capabilities for ${parsed.id}` }
    }
    case 'unregister': {
      if (!options.id) return { ok: false, error: 'unregister requires --id' }
      registry.unregister(options.id)
      await saveManifests(snapshotPath, registry)
      return { ok: true, message: `unregistered capabilities for ${options.id}` }
    }
    case 'list': {
      const state = registry.snapshot()
      const lines: string[] = []
      for (const [id, grants] of state.entries) {
        lines.push(id)
        for (const grant of grants) {
          const resources = grant.resources?.join(',') ?? '*'
          const hosts = grant.hosts?.join(',') ?? '*'
          lines.push(`  ${grant.scope}:${grant.action} resources=[${resources}] hosts=[${hosts}]`)
        }
      }
      return { ok: true, message: lines.join('\n') || 'no capability manifests registered' }
    }
    case 'diff': {
      if (!options.fromSnapshot || !options.toSnapshot) return { ok: false, error: 'diff requires --from-snapshot and --to-snapshot' }
      const fromJson = JSON.parse(await readFile(options.fromSnapshot, 'utf8')) as unknown
      const toJson = JSON.parse(await readFile(options.toSnapshot, 'utf8')) as unknown
      const fromManifests = isManifests(fromJson) ? fromJson : []
      const toManifests = isManifests(toJson) ? toJson : []
      for (const manifest of fromManifests) registry.register(manifest)
      const diff = registry.diff(toManifests)
      const lines: string[] = []
      for (const { pluginId, capability } of diff.added) lines.push(`+ ${pluginId}: ${capability.scope}:${capability.action}`)
      for (const { pluginId, capability } of diff.removed) lines.push(`- ${pluginId}: ${capability.scope}:${capability.action}`)
      return { ok: true, message: lines.join('\n') || 'no changes' }
    }
  }
}

async function loadManifests(path: string): Promise<CapabilityManifest[]> {
  try {
    const contents = await readFile(path, 'utf8')
    const parsed = JSON.parse(contents) as unknown
    return isManifests(parsed) ? parsed : []
  } catch {
    return []
  }
}

async function saveManifests(path: string, registry: CapabilityRegistry): Promise<void> {
  const state = registry.snapshot()
  const manifests: CapabilityManifest[] = []
  for (const [id, grants] of state.entries) manifests.push({ id, capabilities: grants })
  await writeFile(path, JSON.stringify(manifests, null, 2), 'utf8')
}

function isManifest(value: unknown): value is CapabilityManifest {
  if (typeof value !== 'object' || value === null) return false
  const manifest = value as Partial<CapabilityManifest>
  return typeof manifest.id === 'string' && Array.isArray(manifest.capabilities)
}

function isManifests(value: unknown): value is CapabilityManifest[] {
  return Array.isArray(value) && value.every(isManifest)
}
