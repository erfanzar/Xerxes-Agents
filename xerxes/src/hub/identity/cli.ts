// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isHubRole } from '../../security/principal.js'
import { hubPaths, type HubPaths } from '../paths.js'
import { loadOrCreateHubSigner } from '../signer.js'
import { HubIdentityError } from './core.js'
import { createHubIdentity } from './index.js'

/**
 * Local-only identity administration: `xerxes hub owner|pair|devices|users|invite|audit`.
 * It opens the same SQLite file the running Hub uses (WAL, `BEGIN IMMEDIATE`), so
 * revocations written here are announced by the server within its poll interval.
 */

export interface HubIdentityCliOptions {
  readonly paths?: HubPaths
  readonly environment?: Record<string, string | undefined>
  readonly out?: (line: string) => void
  readonly err?: (line: string) => void
}

const USAGE = [
  'usage: xerxes hub <command>',
  '  owner init --name <name>',
  '  pair [--label <text>] [--user <id>] [--role <role>] [--no-confirm] [--origin <https-origin>]',
  '  pair confirm <pairing-id> <comparison-code>',
  '  devices list [--all] | devices revoke <device-id>',
  '  users list | users revoke <user-id>',
  '  invite create --workspace <id> [--session <id>] --role <viewer|steer|operator> [--expires 24h] [--no-confirm]',
  '  audit verify',
].join('\n')

interface Parsed {
  readonly positional: string[]
  readonly flags: Map<string, string | true>
}

const VALUE_FLAGS: ReadonlySet<string> = new Set(['name', 'label', 'user', 'role', 'origin', 'workspace', 'session', 'expires'])

function parse(args: readonly string[]): Parsed {
  const positional: string[] = []
  const flags = new Map<string, string | true>()
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i] ?? ''
    if (!arg.startsWith('--')) {
      positional.push(arg)
      continue
    }
    const name = arg.slice(2)
    if (VALUE_FLAGS.has(name)) {
      const value = args[i + 1]
      if (value === undefined || value.startsWith('--')) throw new HubIdentityError('invalid_request', `--${name} needs a value`)
      flags.set(name, value)
      i += 1
    } else flags.set(name, true)
  }
  return { positional, flags }
}

function flag(parsed: Parsed, name: string): string | undefined {
  const value = parsed.flags.get(name)
  return typeof value === 'string' ? value : undefined
}

export function parseDuration(text: string): number {
  const match = /^(\d+)(s|m|h|d)$/.exec(text)
  if (!match) throw new HubIdentityError('invalid_request', `duration ${JSON.stringify(text)} must look like 30m, 24h or 7d`)
  const unit = { s: 1000, m: 60_000, h: 3_600_000, d: 86_400_000 }[match[2] as 's' | 'm' | 'h' | 'd']
  return Number(match[1]) * unit
}

function firstOrigin(environment: Record<string, string | undefined>, override: string | undefined): string | undefined {
  if (override !== undefined) return override
  return environment.XERXES_HUB_PUBLIC_ORIGINS?.split(',').map(part => part.trim()).find(Boolean)
}

/** Returns a process exit code. Secrets (pairing URLs) are printed to stdout only, never logged or audited. */
export async function runHubIdentityCli(args: readonly string[], options: HubIdentityCliOptions = {}): Promise<number> {
  const out = options.out ?? ((line: string) => console.log(line))
  const err = options.err ?? ((line: string) => console.error(line))
  const environment = options.environment ?? process.env
  let parsed: Parsed
  try {
    parsed = parse(args)
  } catch (error) {
    err(error instanceof Error ? error.message : String(error))
    err(USAGE)
    return 2
  }
  const [group, action, ...rest] = parsed.positional
  if (group === undefined) {
    err(USAGE)
    return 2
  }
  const paths = options.paths ?? hubPaths(environment)
  const signer = await loadOrCreateHubSigner({ paths })
  const origin = firstOrigin(environment, flag(parsed, 'origin'))
  const identity = await createHubIdentity({
    paths,
    signerFingerprint: signer.fingerprint,
    ...(origin === undefined ? {} : { publicOrigin: origin }),
    revocationPollMs: 0,
  })
  try {
    const admin = identity.admin
    const ownerId = (): string => {
      const id = admin.ownerUserId()
      if (id === undefined) throw new HubIdentityError('conflict', 'no owner yet: run `xerxes hub owner init --name <name>`')
      return id
    }
    if (group === 'owner' && action === 'init') {
      const name = flag(parsed, 'name')
      if (name === undefined) throw new HubIdentityError('invalid_request', '--name is required')
      const { userId } = admin.bootstrapOwner({ name })
      out(`owner created: ${userId}`)
      return 0
    }
    if (group === 'pair' && action === 'confirm') {
      const [id, code] = rest
      if (id === undefined || code === undefined) throw new HubIdentityError('invalid_request', 'usage: pair confirm <pairing-id> <comparison-code>')
      const confirmed = admin.confirmPairing(id, code)
      out(parsed.flags.has('json') ? JSON.stringify({ device_id: confirmed.deviceId }) : `device ${confirmed.deviceId} is now active`)
      return 0
    }
    if (group === 'pair' && action === undefined) {
      const role = flag(parsed, 'role')
      if (role !== undefined && !isHubRole(role)) throw new HubIdentityError('invalid_request', 'unknown role')
      const label = flag(parsed, 'label')
      const created = admin.createPairing({
        userId: flag(parsed, 'user') ?? ownerId(),
        ...(label === undefined ? {} : { label }),
        ...(role === undefined ? {} : { role }),
        ...(parsed.flags.has('no-confirm') ? { confirm: false } : {}),
      })
      if (parsed.flags.has('json')) {
        out(JSON.stringify({
          pairing_id: created.pairingId, url: created.url, comparison_code: created.comparisonCode,
          expires_at: new Date(created.expiresAt).toISOString(), confirm_required: created.confirmRequired, qr_svg: created.qrSvg,
        }))
        return 0
      }
      out(`pairing ${created.pairingId} (expires ${new Date(created.expiresAt).toISOString()})`)
      out(`open on the phone: ${created.url}`)
      out(`comparison code: ${created.comparisonCode}`)
      out(`hub key fingerprint: ${created.fingerprint}`)
      out(created.confirmRequired ? `after the phone connects, run: xerxes hub pair confirm ${created.pairingId} ${created.comparisonCode}` : 'no confirmation required (--no-confirm): the device is active as soon as it connects')
      if (created.qrTerminal !== null) out(created.qrTerminal)
      return 0
    }
    if (group === 'devices' && action === 'list') {
      for (const device of admin.listDevices({ includeRevoked: parsed.flags.has('all') })) {
        out(`${device.id}\t${device.status}\t${device.displayName}\t${device.label}\tlast seen ${device.lastSeenAt === null ? 'never' : new Date(device.lastSeenAt).toISOString()}`)
      }
      return 0
    }
    if (group === 'devices' && action === 'revoke') {
      const [id] = rest
      if (id === undefined) throw new HubIdentityError('invalid_request', 'usage: devices revoke <device-id>')
      admin.revokeDevice(id, 'revoked_by_admin')
      out(`device ${id} revoked`)
      return 0
    }
    if (group === 'users' && action === 'list') {
      for (const user of admin.listUsers()) out(`${user.id}\t${user.owner ? 'owner' : 'user'}\t${user.revoked ? 'revoked' : 'active'}\t${user.displayName}`)
      return 0
    }
    if (group === 'users' && action === 'revoke') {
      const [id] = rest
      if (id === undefined) throw new HubIdentityError('invalid_request', 'usage: users revoke <user-id>')
      admin.revokeUser(id, 'revoked_by_admin')
      out(`user ${id} revoked`)
      return 0
    }
    if (group === 'invite' && action === 'create') {
      const workspaceId = flag(parsed, 'workspace')
      const role = flag(parsed, 'role')
      if (workspaceId === undefined || role === undefined) throw new HubIdentityError('invalid_request', '--workspace and --role are required')
      if (!isHubRole(role)) throw new HubIdentityError('invalid_request', 'unknown role')
      const sessionId = flag(parsed, 'session')
      const expires = flag(parsed, 'expires')
      const created = admin.createInvite({
        createdBy: ownerId(),
        workspaceId,
        ...(sessionId === undefined ? {} : { sessionId }),
        role,
        ...(expires === undefined ? {} : { expiresInMs: parseDuration(expires) }),
        ...(parsed.flags.has('no-confirm') ? { confirm: false } : {}),
      })
      out(`invite ${created.inviteId} (expires ${new Date(created.expiresAt).toISOString()})`)
      out(`send this link once: ${created.url}`)
      out(`comparison code: ${created.comparisonCode}`)
      if (created.confirmRequired) out(`after they accept, run: xerxes hub pair confirm ${created.inviteId} ${created.comparisonCode}`)
      if (created.qrTerminal !== null) out(created.qrTerminal)
      return 0
    }
    if (group === 'audit' && action === 'verify') {
      const result = admin.verifyAuditChain()
      if (result.ok) {
        out(`audit chain ok: ${result.count} events, head ${result.headHash}`)
        return 0
      }
      err(`audit chain BROKEN at sequence ${result.sequence}: ${result.reason}`)
      return 1
    }
    err(USAGE)
    return 2
  } catch (error) {
    if (error instanceof HubIdentityError) {
      err(`${error.code}: ${error.message}`)
      return error.code === 'invalid_request' ? 2 : 1
    }
    throw error
  } finally {
    identity.close()
  }
}
