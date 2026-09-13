// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, rename, rm } from 'node:fs/promises'
import { dirname } from 'node:path'
import { withFileLock } from '../session/daemonTranscript.js'
import { machineDiscovery, type MachineDiscovery } from './machineDiscovery.js'

export interface MachineWorkspace {
  readonly alias: string
  readonly target: string
  readonly workspacePath: string
}

const USAGE = '/machine add <name> <ssh-target> <absolute-path> · /machine remove <name> · /machine connect <name>'

function validateMachine(value: unknown): MachineWorkspace {
  if (!value || typeof value !== 'object') throw new Error('Invalid machine entry')
  const row = value as Record<string, unknown>
  if (typeof row.alias !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(row.alias)) throw new Error('Machine name must contain letters, numbers, underscores or hyphens')
  if (typeof row.target !== 'string' || row.target.length > 255 || !/^(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/.test(row.target)) throw new Error('Use an SSH config alias or user@hostname; configure ports and identity files in ~/.ssh/config')
  if (typeof row.workspacePath !== 'string' || !row.workspacePath.startsWith('/') || /[\x00-\x1f\x7f]/.test(row.workspacePath) || row.workspacePath.length > 4096) throw new Error('Remote workspace must be an absolute path without control characters')
  return { alias: row.alias, target: row.target, workspacePath: row.workspacePath }
}

/** A small quoted-argument grammar. No interpolation or shell execution occurs here. */
export function machineArguments(input: string): string[] {
  const parts: string[] = []
  let value = '', quote = '', started = false
  for (const char of input.trim()) {
    if (quote) { if (char === quote) quote = ''; else value += char; started = true }
    else if (char === '"' || char === "'") { quote = char; started = true }
    else if (/\s/.test(char)) { if (started) { parts.push(value); value = ''; started = false } }
    else { value += char; started = true }
  }
  if (quote) throw new Error('Unclosed quote in machine command')
  if (started) parts.push(value)
  return parts
}

async function readMachines(path: string): Promise<MachineWorkspace[]> {
  let raw: string
  try { raw = await readFile(path, 'utf8') }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []; throw error }
  const data: unknown = JSON.parse(raw)
  if (!data || typeof data !== 'object' || !('machines' in data) || !Array.isArray(data.machines) || data.machines.length > 100) throw new Error('Invalid machine configuration; expected a machines array (up to 100 entries)')
  const machines = data.machines.map(validateMachine)
  if (new Set(machines.map(m => m.alias)).size !== machines.length) throw new Error('Duplicate machine names in configuration')
  return machines.sort((a, b) => a.alias.localeCompare(b.alias))
}

/** Persistent registry shared by daemons; returning a target does not claim a connection succeeded. */
export interface MachineRegistryPort {
  write(path: string, content: string): Promise<void>
}
const machineRegistry: MachineRegistryPort = {
  async write(path, content) { await Bun.write(path, content, { mode: 0o600 }) },
}
export async function runMachineCommand(path: string, input: string, discovery: MachineDiscovery = machineDiscovery, registry: MachineRegistryPort = machineRegistry): Promise<Record<string, unknown>> {
  try {
    const [action = 'list', ...args] = machineArguments(input)
    if (action === 'hosts' && !args.length) return { ok: true, hosts: await discovery.hosts() }
    if (action === 'browse' && (args.length === 1 || args.length === 2)) {
      const folder = args[1] ? Buffer.from(args[1], 'base64url').toString('utf8') : ''
      return { ok: true, ...await discovery.browse(args[0]!, folder) }
    }
    if (action === 'list' && !args.length) {
      const machines = await readMachines(path)
      return { ok: true, machines, output: ['REMOTE WORKSPACES', ...machines.map(m => `${m.alias} · ${m.target} · ${m.workspacePath}`), ...(machines.length ? [] : ['No remote workspaces saved yet.']), '', USAGE, 'Connect opens Xerxes over SSH in this terminal. Exit the remote TUI to return to your local chat.', 'Connect installs/updates a managed Xerxes build automatically. Configure provider authentication on the remote host. SSH uses your existing config.'].join('\n') }
    }
    if (action === 'connect' && args.length === 1) {
      const machine = (await readMachines(path)).find(m => m.alias === args[0])
      if (!machine) throw new Error('Unknown machine; use /machine to see saved workspaces')
      return { ok: true, machine }
    }
    if (!((action === 'add' && args.length === 3) || (action === 'remove' && args.length === 1))) throw new Error(USAGE)
    const machine = action === 'add' ? validateMachine({ alias: args[0], target: args[1], workspacePath: args[2] }) : undefined
    await mkdir(dirname(path), { recursive: true })
    return await withFileLock(`${path}.lock`, async () => {
      const machines = await readMachines(path)
      const index = machines.findIndex(m => m.alias === args[0])
      if (action === 'add') {
        if (index >= 0) throw new Error('Machine name already exists; remove it before replacing its target')
        if (machines.length >= 100) throw new Error('Machine registry is full (100 entries)')
        machines.push(machine!)
      } else {
        if (index < 0) throw new Error('Unknown machine')
        machines.splice(index, 1)
      }
      const temporary = `${path}.${crypto.randomUUID()}.tmp`
      try { await registry.write(temporary, JSON.stringify({ machines }, null, 2)); await rename(temporary, path) }
      finally { await rm(temporary, { force: true }) }
      return { ok: true, machines, output: action === 'add' ? `Saved ${machine!.alias}. Open /machine to connect.` : `Removed ${args[0]}.` }
    }, { label: 'machine settings', staleMs: 30000, waitMs: 5000 })
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}
