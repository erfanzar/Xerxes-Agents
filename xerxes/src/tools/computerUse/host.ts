// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { isAbsolute } from 'node:path'

import { ConfigurationError } from '../../core/errors.js'
import { MCPClient } from '../../mcp/client.js'
import { ComputerUseSession, type ComputerUsePort } from './backend.js'
import { CuaBackend, type CuaDriverMcpPort } from './cuaBackend.js'
import { MCPClientCuaDriverPort } from './cuaDriverPort.js'
import { MacOSComputerUsePort, resolveComputerUseCaptureBounds, type ComputerUseCaptureBounds } from './macosPort.js'
import type { ComputerUseToolsOptions } from './tool.js'

export const DEFAULT_CUA_DRIVER = '/Applications/CuaDriver.app/Contents/MacOS/cua-driver'

export interface MacOSComputerUseHostOptions {
  readonly platform?: string
  readonly fileExists?: (path: string) => boolean
  readonly hostAvailable?: () => boolean
  readonly createDriver?: (path: string) => CuaDriverMcpPort
  readonly createMacOSPort?: (bounds: ComputerUseCaptureBounds) => ComputerUsePort
}

/** One owner per runtime, outside runner/registry rebuilds. Construction never spawns. */
export class MacOSComputerUseHost {
  private driver: { path: string; port: CuaDriverMcpPort } | undefined
  private readonly sessions = new Map<string, ComputerUseSession>()
  private disposed = false

  constructor(private readonly options: MacOSComputerUseHostOptions = {}) {}

  toolOptions(
    settings: Readonly<Record<string, unknown>>,
    environment: Readonly<Record<string, string | undefined>> = process.env,
  ): ComputerUseToolsOptions | undefined {
    const flag = environment['XERXES_COMPUTER_USE']?.trim().toLowerCase()
    // Opt-out takes precedence even over malformed backend/path/bounding settings.
    if (settings['computer_use_enabled'] === false || ['0', 'false', 'no', 'off'].includes(flag ?? '')) return undefined
    if (this.disposed) throw new ConfigurationError('computer_use', 'runtime computer-use host is disposed')
    const backend = settings['computer_use_backend'] ?? 'auto'
    // Other host-injected integrations retain ownership of their own surface.
    if (backend !== 'auto' && backend !== 'cua' && backend !== 'macos') return undefined
    const bounds = resolveComputerUseCaptureBounds(settings, environment)
    const rawPath = environment['XERXES_CUA_DRIVER'] ?? settings['computer_use_cua_driver'] ?? DEFAULT_CUA_DRIVER
    if (typeof rawPath !== 'string' || !isAbsolute(rawPath) || !rawPath.trim()) {
      throw new ConfigurationError('computer_use_cua_driver', 'must be an absolute CuaDriver executable path')
    }
    const exists = this.options.fileExists ?? existsSync
    const platform = this.options.platform ?? process.platform
    const available = this.options.hostAvailable?.() ?? (platform === 'darwin' && (
      (backend !== 'macos' && exists(rawPath))
      || (backend !== 'cua' && new MacOSComputerUsePort({ platform, fileExists: exists }).isAvailable())
    ))
    const forced = settings['computer_use_enabled'] === true || ['1', 'true', 'yes', 'on'].includes(flag ?? '') || backend === 'cua'
    if (!forced && !available) return undefined
    const selected = backend === 'auto' ? (platform === 'darwin' && exists(rawPath) ? 'cua' : 'macos') : backend
    if (selected === 'cua' && !this.driver) {
      const port = this.options.createDriver?.(rawPath) ?? new MCPClientCuaDriverPort(new MCPClient({
        name: 'xerxes-cua-driver', transport: 'stdio', command: rawPath, args: ['mcp'], timeoutMs: 30_000,
      }), { isProvisioned: () => platform === 'darwin' && exists(rawPath), timeoutMs: 30_000 })
      this.driver = { path: rawPath, port }
    }
    if (selected === 'cua' && this.driver?.path !== rawPath) {
      throw new ConfigurationError('computer_use_cua_driver', 'cannot change the driver path within a runtime; it owns one stdio connection')
    }
    const configuration = JSON.stringify([selected, rawPath, bounds])
    const resolve = (owner: string): ComputerUseSession => {
      if (this.disposed) throw new ConfigurationError('computer_use', 'runtime computer-use host is disposed')
      const key = JSON.stringify([configuration, owner])
      let session = this.sessions.get(key)
      if (!session) {
        const port = selected === 'cua' && this.driver
          ? new CuaBackend(this.driver.port, { maxDimension: bounds.maxCaptureEdge, maxElements: 300, ownsDriver: false })
          : this.options.createMacOSPort?.(bounds) ?? new MacOSComputerUsePort(bounds)
        session = new ComputerUseSession({ port })
        this.sessions.set(key, session)
      }
      return session
    }
    // A stable session survives registry rebuilds; separate callers do not inherit
    // one another's element indices. Driver tokens fail closed if superseded.
    return {
      session: resolve(JSON.stringify(['default', 'default'])),
      resolveSession: context => resolve(JSON.stringify([context.sessionId ?? 'default', context.agentId ?? 'default'])),
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    // Close transport first so an in-flight request cannot delay shutdown until
    // its timeout. Only this runtime's owned MCP child is closed, never the daemon.
    try { await this.driver?.port.stop?.() }
    finally {
      try { await Promise.all([...this.sessions.values()].map(session => session.stop())) }
      finally { this.sessions.clear() }
    }
  }
}

/** Standalone convenience; long-lived runtimes should own MacOSComputerUseHost. */
export function createMacOSComputerUseToolOptions(
  settings: Readonly<Record<string, unknown>>,
  environment: Readonly<Record<string, string | undefined>> = process.env,
  hostAvailable?: () => boolean,
): (ComputerUseToolsOptions & { readonly dispose: () => Promise<void> }) | undefined {
  const host = new MacOSComputerUseHost(hostAvailable ? { hostAvailable } : {})
  const options = host.toolOptions(settings, environment)
  return options ? { ...options, dispose: () => host.dispose() } : undefined
}
