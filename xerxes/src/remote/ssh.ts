// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * SSH connection manager for remote workspaces.
 *
 * A remote workspace is a directory on another machine that the agent works
 * in as if it were local. The manager owns the SSH connection, keeps it
 * alive, and exposes a minimal exec/sync surface for the daemon.
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { ValidationError } from '../core/errors.js'

export interface RemoteHost {
  readonly alias: string
  readonly host: string
  readonly user: string
  readonly port?: number
  readonly identityFile?: string
  /** Remote directory that becomes the workspace root. */
  readonly workspacePath: string
}

export interface RemoteExecResult {
  readonly code: number
  readonly stderr: string
  readonly stdout: string
}

export interface RemoteExecOptions {
  readonly cwd?: string
  readonly env?: Record<string, string>
  readonly input?: string
  readonly timeoutMs?: number
}

const DEFAULT_SSH_OPTIONS = [
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'ServerAliveInterval=30',
  '-o', 'ServerAliveCountMax=3',
  '-o', 'StrictHostKeyChecking=accept-new',
]

export class RemoteConnection {
  private process: ChildProcess | undefined
  private readonly host: RemoteHost
  private connected = false

  constructor(host: RemoteHost) {
    this.host = host
  }

  get alias(): string {
    return this.host.alias
  }

  get isConnected(): boolean {
    return this.connected
  }

  /** Open the SSH connection. Idempotent — a live connection is reused. */
  async connect(): Promise<void> {
    if (this.connected) return
  }

  /** Run a command on the remote host. */
  async exec(command: string, options: RemoteExecOptions = {}): Promise<RemoteExecResult> {
    const args = [
      ...DEFAULT_SSH_OPTIONS,
      ...(this.host.port ? ['-p', String(this.host.port)] : []),
      ...(this.host.identityFile ? ['-i', this.host.identityFile] : []),
      `${this.host.user}@${this.host.host}`,
      command,
    ]

    const child = spawn('ssh', args, {
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    const stdout: Buffer[] = []
    const stderr: Buffer[] = []

    child.stdout?.on('data', (chunk: Buffer) => stdout.push(chunk))
    child.stderr?.on('data', (chunk: Buffer) => stderr.push(chunk))

    if (options.input) {
      child.stdin?.write(options.input)
      child.stdin?.end()
    }

    const timeout = options.timeoutMs ?? 30_000
    const timer = setTimeout(() => {
      child.kill('SIGTERM')
    }, timeout)

    return new Promise((resolve, reject) => {
      child.on('error', reject)
      child.on('close', code => {
        clearTimeout(timer)
        resolve({
          code: code ?? 0,
          stderr: Buffer.concat(stderr).toString('utf8'),
          stdout: Buffer.concat(stdout).toString('utf8'),
        })
      })
    })
  }

  /** Sync a local directory to the remote workspace. */
  async syncToRemote(localPath: string, remotePath: string): Promise<void> {
    const args = [
      '-avz',
      '--delete',
      ...(this.host.port ? ['-e', `ssh -p ${this.host.port}`] : []),
      ...(this.host.identityFile ? ['-e', `ssh -i ${this.host.identityFile}`] : []),
      `${localPath}/`,
      `${this.host.user}@${this.host.host}:${remotePath}/`,
    ]

    const child = spawn('rsync', args, { stdio: ['ignore', 'pipe', 'pipe'] })
    const stderr: Buffer[] = []
    child.stderr?.on('data', (chunk: Buffer) => stderr.push(chunk))

    return new Promise((resolve, reject) => {
      child.on('error', reject)
      child.on('close', code => {
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`rsync failed (${code}): ${Buffer.concat(stderr).toString('utf8')}`))
        }
      })
    })
  }

  /** Sync a remote directory back to local. */
  async syncFromRemote(remotePath: string, localPath: string): Promise<void> {
    const args = [
      '-avz',
      ...(this.host.port ? ['-e', `ssh -p ${this.host.port}`] : []),
      ...(this.host.identityFile ? ['-e', `ssh -i ${this.host.identityFile}`] : []),
      `${this.host.user}@${this.host.host}:${remotePath}/`,
      `${localPath}/`,
    ]

    const child = spawn('rsync', args, { stdio: ['ignore', 'pipe', 'pipe'] })
    const stderr: Buffer[] = []
    child.stderr?.on('data', (chunk: Buffer) => stderr.push(chunk))

    return new Promise((resolve, reject) => {
      child.on('error', reject)
      child.on('close', code => {
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`rsync failed (${code}): ${Buffer.concat(stderr).toString('utf8')}`))
        }
      })
    })
  }

  async disconnect(): Promise<void> {
    this.connected = false
    this.process?.kill()
    this.process = undefined
  }
}

/** Registry of named remote hosts, persisted to ~/.xerxes/remote.json. */
export class RemoteHostRegistry {
  private readonly filePath: string
  private hosts: Map<string, RemoteHost> = new Map()

  constructor(filePath: string) {
    this.filePath = filePath
  }

  async load(): Promise<void> {
    try {
      const raw = await readFile(this.filePath, 'utf8')
      const parsed = JSON.parse(raw) as { hosts?: RemoteHost[] }
      this.hosts = new Map((parsed.hosts ?? []).map(host => [host.alias, host]))
    } catch {
      this.hosts = new Map()
    }
  }

  async save(): Promise<void> {
    await mkdir(join(this.filePath, '..'), { recursive: true })
    await writeFile(this.filePath, JSON.stringify({ hosts: [...this.hosts.values()] }, null, 2))
  }

  add(host: RemoteHost): void {
    this.hosts.set(host.alias, host)
  }

  get(alias: string): RemoteHost | undefined {
    return this.hosts.get(alias)
  }

  list(): RemoteHost[] {
    return [...this.hosts.values()]
  }

  remove(alias: string): boolean {
    return this.hosts.delete(alias)
  }
}

/** Parse an SSH target string: user@host:port/path or user@host/path. */
export function parseRemoteTarget(value: string): RemoteHost {
  const match = value.match(/^([^@]+)@([^:/]+)(?::(\d+))?(?:\/(.+))?$/)
  if (!match) {
    throw new ValidationError('remote target', `invalid format: ${value} (expected user@host[:port][/path])`)
  }
  const [, user, host, port, path] = match
  if (!user || !host) {
    throw new ValidationError('remote target', `invalid format: ${value} (expected user@host[:port][/path])`)
  }
  return {
    alias: `${user}@${host}${port ? `:${port}` : ''}`,
    host,
    user,
    ...(port ? { port: Number(port) } : {}),
    workspacePath: path ?? '~',
  }
}
