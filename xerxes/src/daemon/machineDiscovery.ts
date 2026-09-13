// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawn, type SpawnOptions, type ChildProcess } from 'node:child_process'

export interface RemoteFolders { path: string; directories: string[]; truncated: boolean }
export interface MachineDiscovery {
  hosts(): Promise<string[]>
  browse(target: string, path: string): Promise<RemoteFolders>
}

function sshWords(line: string): string[] {
  const words: string[] = []
  let word = '', quote = '', escaped = false
  for (const char of line) {
    if (escaped) { word += char; escaped = false; continue }
    if (char === '\\') { escaped = true; continue }
    if (quote) { if (char === quote) quote = ''; else word += char; continue }
    if (char === '"' || char === "'") { quote = char; continue }
    if (char === '#') break
    if (/\s/.test(char) || (char === '=' && words.length === 0)) {
      if (word) { words.push(word); word = '' }
    } else word += char
  }
  if (word) words.push(word)
  return words.filter(token => token !== '=')
}

/** Discover names only: never evaluate Match exec, expand identity files, or rewrite SSH configuration. */
export type SshConfigScan = (pattern: string) => AsyncIterable<string>
const scanSshConfig: SshConfigScan = pattern => new Bun.Glob(pattern).scan({ absolute: true, onlyFiles: true })
export async function readSshHosts(home = homedir(), scan: SshConfigScan = scanSshConfig): Promise<string[]> {
  const names = new Set<string>(), visited = new Set<string>()
  const base = join(home, '.ssh')
  async function read(file: string): Promise<void> {
    file = resolve(file)
    if (visited.has(file)) return
    if (visited.size >= 64) throw new Error('SSH config has too many included files (limit 64)')
    visited.add(file)
    let content: string
    try { content = await readFile(file, 'utf8') }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return; throw error }
    if (content.length > 1024 * 1024) throw new Error('SSH config is too large')
    for (const line of content.split(/\r?\n/)) {
      const normalized = sshWords(line)
      if (!normalized.length) continue
      const keyword = normalized.shift()!.toLowerCase()
      if (keyword === 'host') for (const name of normalized) {
        if (/^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/.test(name)) names.add(name)
      }
      if (keyword === 'include') for (const pattern of normalized) {
        const expanded = pattern.startsWith('~/') ? join(home, pattern.slice(2)) : resolve(base, pattern)
        const matches: string[] = []
        for await (const match of scan(expanded)) {
          matches.push(match)
          if (matches.length > 64) throw new Error('SSH Include matches too many files (limit 64)')
        }
        for (const match of matches.sort()) await read(match)
      }
    }
  }
  await read(join(base, 'config'))
  return [...names].sort((a, b) => a.localeCompare(b))
}

export function remoteFolderCommand(path: string): string {
  if (path && (!path.startsWith('/') || path.length > 4096 || /[\x00-\x1f\x7f]/.test(path))) throw new Error('Choose an absolute remote folder')
  const quoted = path ? "'" + path.replaceAll("'", "'\\''") + "'" : '"$HOME"'
  // NUL framing preserves spaces/quotes; globbing is expanded remotely, never by the local shell.
  const script = `cd ${quoted} || exit 1; printf '%s\\0' "$PWD"; n=0; for d in ./* ./.[!.]* ./..?*; do [ -d "$d" ] || continue; n=$((n+1)); if [ "$n" -gt 1000 ]; then printf '%s\\0' '__XERXES_TRUNCATED__'; break; fi; printf '%s\\0' "$d"; done`
  return "exec sh -c '" + script.replaceAll("'", "'\\''") + "'"
}

export function browseSshFolders(target: string, path: string, options: { signal?: AbortSignal; spawnProcess?: (file: string, args: readonly string[], options: SpawnOptions) => ChildProcess; timeoutMs?: number } = {}): Promise<RemoteFolders> {
  if (!/^(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/.test(target) || target.length > 255) return Promise.reject(new Error('Choose an SSH alias or user@hostname'))
  const command = remoteFolderCommand(path)
  return new Promise((accept, reject) => {
    if (options.signal?.aborted) { reject(new Error('Folder browse cancelled')); return }
    const child = (options.spawnProcess ?? spawn)('ssh', ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '--', target, command], { stdio: ['ignore', 'pipe', 'pipe'] })
    const output: Buffer[] = [], errors: Buffer[] = []
    let bytes = 0, settled = false
    const finish = (error?: Error, value?: RemoteFolders) => {
      if (settled) return
      settled = true; clearTimeout(timer); options.signal?.removeEventListener('abort', cancel)
      if (error) { child.kill('SIGKILL'); reject(error) } else accept(value!)
    }
    const cancel = () => finish(new Error('Folder browse cancelled'))
    const timer = setTimeout(() => finish(new Error('SSH folder browse timed out. Check the host and SSH connection.')), options.timeoutMs ?? 15000)
    options.signal?.addEventListener('abort', cancel, { once: true })
    for (const [stream, chunks] of [[child.stdout, output], [child.stderr, errors]] as const) stream?.on('data', (data: Buffer) => {
      bytes += data.length
      if (bytes > 1024 * 1024) finish(new Error('SSH folder response is too large; enter a more specific path'))
      else chunks.push(data)
    })
    child.on('error', error => finish(error))
    child.on('close', code => {
      if (settled) return
      if (code !== 0) { finish(new Error(`SSH browse failed: ${Buffer.concat(errors).toString('utf8').trim().slice(0, 1000) || `exit ${code}`}. Connect with ssh ${target} in a terminal to verify authentication and the host key.`)); return }
      const fields = Buffer.concat(output).toString('utf8').split('\0')
      const directory = fields.shift() ?? ''
      if (!directory.startsWith('/') || /[\x00-\x1f\x7f]/.test(directory)) { finish(new Error('SSH returned an invalid folder listing')); return }
      const truncated = fields.includes('__XERXES_TRUNCATED__')
      const directories = fields.filter(entry => entry.startsWith('./') && !/[\x00-\x1f\x7f]/.test(entry)).map(entry => entry.slice(2)).sort((a, b) => a.localeCompare(b))
      finish(undefined, { path: directory, directories, truncated })
    })
  })
}

export const machineDiscovery: MachineDiscovery = { hosts: readSshHosts, browse: browseSshFolders }
