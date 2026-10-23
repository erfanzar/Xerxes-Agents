// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isIP } from 'node:net'

/**
 * Where the Hub may listen and which browser origins may open its WebSocket.
 *
 * The Hub binds loopback by default. A tailnet bind is allowed only for an
 * address inside Tailscale's ranges and only with TLS. Wildcard and public
 * addresses are refused outright: remote reach comes from a tunnel (Tailscale
 * Serve, Cloudflare Tunnel) terminating in front of a loopback listener.
 * `X-Forwarded-*` headers are never consulted by the server.
 */

export const DEFAULT_HUB_HOST = '127.0.0.1'
export const DEFAULT_HUB_PORT = 8737
const MAX_PUBLIC_ORIGINS = 8

export class HubExposureError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'HubExposureError'
  }
}

export interface HubTlsMaterial {
  /** PEM text, not a path. */
  readonly cert: string
  readonly key: string
}

export interface HubExposureConfig {
  readonly host?: string
  readonly port?: number
  readonly tls?: HubTlsMaterial
  /** Exact https origins browsers use to reach the Hub (for example through Tailscale Serve). */
  readonly publicOrigins?: readonly string[]
  /** Convenience for `<machine>.<tailnet>.ts.net`; becomes `https://<name>`. */
  readonly tailnetHostname?: string
}

export interface HubExposure {
  readonly kind: 'loopback' | 'tailnet'
  readonly hostname: string
  readonly port: number
  readonly tls?: HubTlsMaterial
  readonly publicOrigins: readonly string[]
  /** Exact browser origins allowed to open the WebSocket once the bound port is known. */
  allowedOrigins(boundPort: number): readonly string[]
}

/** A tunnel handle. `publicOrigin` must be one of the configured public origins. */
export interface HubTunnelHandle {
  readonly publicOrigin?: string
  close(): Promise<void> | void
}

/** Injected: production may spawn `tailscale serve`, but only when explicitly configured. */
export interface TailscaleServePort {
  serve(request: { readonly target: string; readonly httpsPort: number }): Promise<HubTunnelHandle>
}

/** Injected: a Cloudflare Tunnel that forwards `hostname` to the loopback `target`. */
export interface CloudflareTunnelPort {
  expose(request: { readonly target: string; readonly hostname: string }): Promise<HubTunnelHandle>
}

export type HubTunnelRequest =
  | { readonly kind: 'tailscale'; readonly port: TailscaleServePort; readonly httpsPort?: number }
  | { readonly kind: 'cloudflare'; readonly port: CloudflareTunnelPort; readonly hostname: string }

const WILDCARDS = new Set(['', '*', '0.0.0.0', '::', '[::]'])

function ipv4Octets(address: string): number[] | undefined {
  if (isIP(address) !== 4) return undefined
  return address.split('.').map(Number)
}

function ipv6Words(input: string): number[] | undefined {
  let address = input.replace(/^\[|\]$/g, '').split('%')[0]!.toLowerCase()
  if (isIP(address) !== 6) return undefined
  let tail: number[] | undefined
  const embedded = address.match(/(\d+\.\d+\.\d+\.\d+)$/)
  if (embedded) {
    const parts = embedded[1]!.split('.').map(Number)
    tail = [(parts[0]! << 8) | parts[1]!, (parts[2]! << 8) | parts[3]!]
    address = `${address.slice(0, -embedded[1]!.length)}0:0`
  }
  const parse = (text: string): number[] => (text === '' ? [] : text.split(':').map(part => Number.parseInt(part, 16)))
  const halves = address.split('::')
  let words: number[]
  if (halves.length === 2) {
    const head = parse(halves[0]!)
    const rest = parse(halves[1]!)
    words = [...head, ...new Array<number>(Math.max(0, 8 - head.length - rest.length)).fill(0), ...rest]
  } else {
    words = parse(address)
  }
  if (words.length !== 8) return undefined
  if (tail) {
    words[6] = tail[0]!
    words[7] = tail[1]!
  }
  return words
}

export function isWildcardAddress(host: string): boolean {
  return WILDCARDS.has(host.trim())
}

export function isLoopbackAddress(host: string): boolean {
  if (host === 'localhost') return true
  const v4 = ipv4Octets(host)
  if (v4) return v4[0] === 127
  const words = ipv6Words(host)
  if (!words) return false
  if (words.slice(0, 7).every(word => word === 0) && words[7] === 1) return true
  // ::ffff:127.x.x.x
  return words.slice(0, 5).every(word => word === 0) && words[5] === 0xffff && (words[6]! >> 8) === 127
}

/** 100.64.0.0/10 or fd7a:115c:a1e0::/48. */
export function isTailnetAddress(host: string): boolean {
  const v4 = ipv4Octets(host)
  if (v4) return v4[0] === 100 && v4[1]! >= 64 && v4[1]! <= 127
  const words = ipv6Words(host)
  return !!words && words[0] === 0xfd7a && words[1] === 0x115c && words[2] === 0xa1e0
}

function normalizeOrigin(value: string): string {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new HubExposureError(`public origin ${JSON.stringify(value)} is not a valid URL`)
  }
  if (url.protocol !== 'https:') throw new HubExposureError(`public origin ${value} must use https`)
  if (url.username || url.password || url.search || url.hash || (url.pathname !== '/' && url.pathname !== '')) {
    throw new HubExposureError(`public origin ${value} must be a bare origin without credentials, path or query`)
  }
  if (url.hostname.includes('*')) throw new HubExposureError('wildcard public origins are not allowed')
  return url.origin
}

const TAILNET_NAME = /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+ts\.net$/i

export function resolveHubExposure(config: HubExposureConfig = {}): HubExposure {
  const requested = (config.host ?? DEFAULT_HUB_HOST).trim()
  if (isWildcardAddress(requested)) {
    throw new HubExposureError('refusing to listen on a wildcard address; bind loopback and use a tunnel for remote access')
  }
  const port = config.port ?? DEFAULT_HUB_PORT
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new HubExposureError('Hub port must be an integer from 0 to 65535')

  let kind: 'loopback' | 'tailnet'
  let hostname: string
  if (isLoopbackAddress(requested)) {
    kind = 'loopback'
    hostname = requested === 'localhost' ? '127.0.0.1' : requested.replace(/^\[|\]$/g, '')
  } else if (isTailnetAddress(requested)) {
    if (!config.tls) throw new HubExposureError('binding a tailnet address requires TLS material; plaintext is only allowed on loopback')
    kind = 'tailnet'
    hostname = requested.replace(/^\[|\]$/g, '')
  } else {
    throw new HubExposureError(`refusing to listen on ${requested}: only loopback and tailnet addresses (100.64.0.0/10, fd7a:115c:a1e0::/48) are allowed`)
  }

  const origins = new Set<string>()
  for (const origin of config.publicOrigins ?? []) origins.add(normalizeOrigin(origin))
  if (config.tailnetHostname !== undefined) {
    if (!TAILNET_NAME.test(config.tailnetHostname)) throw new HubExposureError('tailnet hostname must end in .ts.net')
    origins.add(normalizeOrigin(`https://${config.tailnetHostname.toLowerCase()}`))
  }
  if (origins.size > MAX_PUBLIC_ORIGINS) throw new HubExposureError(`at most ${MAX_PUBLIC_ORIGINS} public origins are allowed`)
  const publicOrigins = Object.freeze([...origins].sort())

  const scheme = config.tls ? 'https' : 'http'
  const bracket = (host: string): string => (host.includes(':') ? `[${host}]` : host)
  return Object.freeze({
    kind,
    hostname,
    port,
    ...(config.tls === undefined ? {} : { tls: config.tls }),
    publicOrigins,
    allowedOrigins(boundPort: number): readonly string[] {
      const list = new Set<string>(publicOrigins)
      if (kind === 'loopback') {
        for (const host of ['127.0.0.1', 'localhost', '[::1]']) list.add(`${scheme}://${host}:${boundPort}`)
        list.add(`${scheme}://${bracket(hostname)}:${boundPort}`)
      } else {
        list.add(`${scheme}://${bracket(hostname)}:${boundPort}`)
      }
      return Object.freeze([...list].sort())
    },
  })
}

/** Exact string comparison; a missing Origin is never allowed. */
export function originAllowed(origin: string | null | undefined, allowed: readonly string[]): boolean {
  return typeof origin === 'string' && origin.length > 0 && allowed.includes(origin)
}

/**
 * Environment contract for `xerxes hub run`:
 * XERXES_HUB_HOST, XERXES_HUB_PORT, XERXES_HUB_PUBLIC_ORIGINS (comma separated),
 * XERXES_HUB_TAILNET_HOSTNAME. TLS PEM text is supplied by the caller.
 */
export function exposureConfigFromEnvironment(environment: Record<string, string | undefined>, tls?: HubTlsMaterial): HubExposureConfig {
  const host = environment.XERXES_HUB_HOST?.trim()
  const portText = environment.XERXES_HUB_PORT?.trim()
  const origins = environment.XERXES_HUB_PUBLIC_ORIGINS?.split(',').map(part => part.trim()).filter(Boolean)
  const tailnetHostname = environment.XERXES_HUB_TAILNET_HOSTNAME?.trim()
  let port: number | undefined
  if (portText) {
    if (!/^\d{1,5}$/.test(portText)) throw new HubExposureError('XERXES_HUB_PORT must be an integer')
    port = Number(portText)
  }
  return {
    ...(host ? { host } : {}),
    ...(port === undefined ? {} : { port }),
    ...(tls ? { tls } : {}),
    ...(origins && origins.length > 0 ? { publicOrigins: origins } : {}),
    ...(tailnetHostname ? { tailnetHostname } : {}),
  }
}
