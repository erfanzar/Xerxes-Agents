// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { chmod, mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { firstLine, type CommandRunner } from './commandRunner.js'
import { DEFAULT_HUB_PORT } from './exposure.js'
import { createHubIdentity } from './identity/index.js'
import type { HubPaths } from './paths.js'
import { isRecord } from './protocol.js'
import { installUserService, type UserServiceContext, type UserServiceSpec } from './serviceUnits.js'
import { loadHubSettings, saveHubSettings } from './settings.js'
import { loadOrCreateHubSigner } from './signer.js'

/**
 * `xerxes hub setup`: everything needed on the Hub machine, in order, and safe
 * to run again. Tailscale gives the phone an HTTPS address; the Hub listens on
 * loopback only. The two steps Tailscale reserves for a signed-in browser
 * (signing in, enabling Serve) are printed as links and waited for; nothing is
 * faked when one is missing.
 */

export class HubSetupError extends Error {
  constructor(message: string) { super(message); this.name = 'HubSetupError' }
}

export interface HubSetupDeps extends UserServiceContext {
  /** Tailscale CLI path (on macOS the app bundles it). */
  readonly tailscale: string
  readonly fetchHealth: (port: number) => Promise<boolean>
  /** Interactive prompt; absent when stdin is not a terminal. Resolves undefined on EOF. */
  readonly ask?: (question: string) => Promise<string | undefined>
  readonly out: (line: string) => void
  readonly err: (line: string) => void
}

export interface HubSetupOptions {
  readonly paths: HubPaths
  readonly environment: Record<string, string | undefined>
  /** Absolute paths used in the background service: the Bun executable and the Xerxes CLI entry. */
  readonly executable: string
  readonly scriptPath: string
  readonly ownerName?: string
  readonly port?: number
  readonly service?: boolean
  readonly pair?: boolean
  readonly serveWaitMs?: number
  readonly healthWaitMs?: number
}

export interface HubSetupResult {
  readonly origin: string
  readonly port: number
  readonly ownerId: string
  readonly servicePath?: string
  readonly pairingId?: string
}

export const HUB_SERVICE_NAME = 'xerxes-hub'
export const HUB_SERVICE_LABEL = 'dev.xerxes.hub'
const LOGIN_LINK = /https:\/\/login\.tailscale\.com\/\S+/

/** Prefer the CLI on PATH; the macOS app ships its CLI inside the bundle. */
export function resolveTailscaleCli(platform: NodeJS.Platform, which: (name: string) => string | null = name => Bun.which(name)): string {
  const onPath = which('tailscale')
  if (onPath) return onPath
  const bundled = '/Applications/Tailscale.app/Contents/MacOS/Tailscale'
  if (platform === 'darwin' && existsSync(bundled)) return bundled
  return 'tailscale'
}

export function parseTailscaleStatus(stdout: string): { readonly backendState: string; readonly dnsName?: string } {
  let row: unknown
  try { row = JSON.parse(stdout) } catch { throw new HubSetupError('`tailscale status --json` did not return JSON. Check that tailscaled is running.') }
  if (!isRecord(row)) throw new HubSetupError('`tailscale status --json` returned an unexpected value.')
  const backendState = typeof row.BackendState === 'string' ? row.BackendState : 'Unknown'
  const self = isRecord(row.Self) ? row.Self : {}
  const dnsName = typeof self.DNSName === 'string' ? self.DNSName.trim().replace(/\.$/, '') : ''
  return { backendState, ...(dnsName ? { dnsName } : {}) }
}

function envPort(environment: Record<string, string | undefined>): number | undefined {
  const text = environment.XERXES_HUB_PORT?.trim()
  if (!text) return undefined
  const port = Number(text)
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new HubSetupError('XERXES_HUB_PORT must be an integer from 1 to 65535')
  return port
}

async function tailscaleOrigin(deps: HubSetupDeps): Promise<string> {
  const version = await deps.run([deps.tailscale, 'version'], { timeoutMs: 10_000 })
  if (version.code !== 0) {
    throw new HubSetupError(`Tailscale is not installed (or not on PATH). ${deps.platform === 'darwin'
      ? 'Install the Tailscale app from https://tailscale.com/download/mac and sign in'
      : 'Install it with `curl -fsSL https://tailscale.com/install.sh | sh`'}, then run \`xerxes hub setup\` again.`)
  }
  const status = await deps.run([deps.tailscale, 'status', '--json'], { timeoutMs: 15_000 })
  if (!status.stdout.trim()) throw new HubSetupError(`Could not read Tailscale status: ${firstLine(status.stderr)}. Check that the Tailscale service is running.`)
  const self = parseTailscaleStatus(status.stdout)
  if (self.backendState !== 'Running') {
    const up = deps.platform === 'linux' ? 'sudo tailscale up' : 'tailscale up (or sign in from the Tailscale app)'
    throw new HubSetupError(`This machine is not connected to Tailscale (state: ${self.backendState}). Run \`${up}\`, open the link it prints, then run \`xerxes hub setup\` again.`)
  }
  if (!self.dnsName) throw new HubSetupError('MagicDNS is off, so this machine has no tailnet name. Turn it on at https://login.tailscale.com/admin/dns, then run setup again.')
  return `https://${self.dnsName}`
}

async function ensureServe(port: number, waitMs: number, deps: HubSetupDeps): Promise<void> {
  const target = `http://127.0.0.1:${port}`
  const deadline = Date.now() + waitMs
  const shown = new Set<string>()
  for (;;) {
    // `tailscale serve` blocks while it waits for an approval, so it runs under a timeout.
    const result = await deps.run([deps.tailscale, 'serve', '--bg', '--https=443', target], { timeoutMs: 20_000 })
    const text = `${result.stdout}\n${result.stderr}`
    if (result.code === 0 && !result.timedOut && !/not enabled/i.test(text)) return
    if (/access denied|permission denied|must be root|requires root/i.test(text)) {
      throw new HubSetupError(`Tailscale refused to change Serve for this user. Run once \`sudo tailscale set --operator=${deps.user ?? '$USER'}\`, then run \`xerxes hub setup\` again.`)
    }
    const link = LOGIN_LINK.exec(text)?.[0]
    if (link === undefined) throw new HubSetupError(`tailscale serve failed: ${firstLine(text)}`)
    if (!shown.has(link)) {
      shown.add(link)
      deps.out(`    Tailscale needs one approval in your browser (signed in to the same account). Open:\n      ${link}\n    Waiting for it...`)
    }
    if (Date.now() >= deadline) throw new HubSetupError(`Tailscale Serve is still not enabled. Open ${link}, approve it, then run \`xerxes hub setup\` again.`)
    await deps.sleep(5000)
  }
}

async function waitHealthy(port: number, waitMs: number, deps: HubSetupDeps): Promise<boolean> {
  const deadline = Date.now() + waitMs
  for (;;) {
    if (await deps.fetchHealth(port)) return true
    if (Date.now() >= deadline) return false
    await deps.sleep(500)
  }
}

/** Text of the `<home>/.xerxes/bin/xerxes` launcher: it runs the same Bun and CLI entry the Hub service uses. */
export function launcherScript(executable: string, scriptPath: string): string {
  const quote = (value: string): string => `'${value.replaceAll("'", `'\\''`)}'`
  return `#!/bin/sh\n# Written by \`xerxes hub setup\`; rewritten when the install moves.\nexec ${quote(executable)} ${quote(scriptPath)} "$@"\n`
}

/** Write the launcher (mode 0755) when missing or different. Returns its path. */
export async function ensureLauncher(home: string, executable: string, scriptPath: string): Promise<string> {
  const directory = join(home, '.xerxes', 'bin')
  const path = join(directory, 'xerxes')
  const content = launcherScript(executable, scriptPath)
  let current: string | undefined
  try { current = await readFile(path, 'utf8') } catch { current = undefined }
  await mkdir(directory, { recursive: true })
  if (current !== content) {
    const temporary = `${path}.${process.pid}.tmp`
    await writeFile(temporary, content, { mode: 0o755 })
    await chmod(temporary, 0o755)
    await rename(temporary, path)
  } else await chmod(path, 0o755)
  return path
}

export function hubServiceSpec(options: Pick<HubSetupOptions, 'paths' | 'environment' | 'executable' | 'scriptPath'>): UserServiceSpec {
  const environment: Record<string, string> = {}
  for (const key of ['PATH', 'XERXES_HOME'] as const) {
    const value = options.environment[key]?.trim()
    if (value) environment[key] = value
  }
  return {
    name: HUB_SERVICE_NAME,
    label: HUB_SERVICE_LABEL,
    description: 'Xerxes Hub',
    argv: [options.executable, options.scriptPath, 'hub', 'run'],
    environment,
    logPath: join(options.paths.logsDir, 'hub-service.log'),
  }
}

export async function runHubSetup(options: HubSetupOptions, deps: HubSetupDeps): Promise<HubSetupResult> {
  const out = deps.out
  const saved = await loadHubSettings(options.paths)
  const port = options.port ?? envPort(options.environment) ?? saved.port ?? DEFAULT_HUB_PORT

  out('1/5 Tailscale')
  const origin = await tailscaleOrigin(deps)
  out(`    This machine is ${origin.slice('https://'.length)} on your tailnet.`)

  out('2/5 HTTPS address')
  await ensureServe(port, options.serveWaitMs ?? 10 * 60_000, deps)
  out(`    ${origin} forwards to the Hub on 127.0.0.1:${port}. It is reachable only inside your tailnet.`)
  await saveHubSettings(options.paths, { ...saved, publicOrigin: origin, host: '127.0.0.1', port })

  out('3/5 Owner')
  const signer = await loadOrCreateHubSigner({ paths: options.paths })
  const identity = await createHubIdentity({ paths: options.paths, signerFingerprint: signer.fingerprint, publicOrigin: origin, revocationPollMs: 0 })
  try {
    let ownerId = identity.admin.ownerUserId()
    if (ownerId === undefined) {
      const name = options.ownerName?.trim() || options.environment.USER?.trim() || 'Owner'
      ownerId = identity.admin.bootstrapOwner({ name }).userId
      out(`    Created the owner "${name}".`)
    } else out('    The owner already exists.')

    out('4/5 Background service')
    // `hub link` and `hub remote` on other machines run this launcher over ssh.
    const launcher = await ensureLauncher(deps.home, options.executable, options.scriptPath)
    out(`    Launcher: ${launcher}`)
    const spec = hubServiceSpec(options)
    let servicePath: string | undefined
    let healthy = false
    if (options.service === false) {
      out(`    Skipped. Run the Hub yourself: ${spec.argv.join(' ')}`)
    } else {
      const installed = await installUserService(spec, deps)
      servicePath = installed.path
      for (const note of installed.notes) out(`    Note: ${note}`)
      healthy = await waitHealthy(port, options.healthWaitMs ?? 30_000, deps)
      out(healthy
        ? `    The Hub runs as a background service (${installed.path}) and starts on login or boot.`
        : `    The service is installed (${installed.path}) but the Hub did not answer on 127.0.0.1:${port}. See ${spec.logPath}.`)
    }

    out('5/5 Phone')
    if (options.pair === false) {
      out('    Skipped. Pair a phone later with `xerxes hub pair`.')
      return { origin, port, ownerId, ...(servicePath === undefined ? {} : { servicePath }) }
    }
    if (!healthy) out('    The Hub is not answering yet; the link below works once it is running.')
    const created = identity.admin.createPairing({ userId: ownerId, label: 'Phone' })
    out('    On the phone (with Tailscale connected), open this link in Safari or Chrome. It works once, for 2 minutes:')
    out(`      ${created.url}`)
    if (created.qrTerminal !== null) out(created.qrTerminal)
    out(`    The phone will show a 6-digit code. It must be ${created.comparisonCode}.`)
    let confirmed = false
    if (created.confirmRequired && deps.ask !== undefined) {
      for (let attempt = 0; attempt < 5 && !confirmed; attempt += 1) {
        const answer = (await deps.ask('    Type the code the phone shows (Enter to skip): '))?.trim()
        if (!answer) break
        try {
          const result = identity.admin.confirmPairing(created.pairingId, answer)
          out(`    Paired. Device ${result.deviceId} is active. Tip: Share -> Add to Home Screen.`)
          confirmed = true
        } catch (error) {
          deps.err(`    ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }
    if (created.confirmRequired && !confirmed) out(`    Confirm it later with: xerxes hub pair confirm ${created.pairingId} <code>`)
    return { origin, port, ownerId, pairingId: created.pairingId, ...(servicePath === undefined ? {} : { servicePath }) }
  } finally {
    identity.close()
  }
}
