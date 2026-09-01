// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Desktop build identity injected by scripts/buildDesktop.ts. The fallbacks
 * keep direct source tests deterministic; production renderer bundles always
 * replace all three constants at build time.
 */
declare const __XERXES_DESKTOP_VERSION__: string
declare const __XERXES_DESKTOP_PROTOCOL__: number
declare const __XERXES_EXPECTED_DAEMON_BUILD_ID__: string

export const DESKTOP_VERSION =
  typeof __XERXES_DESKTOP_VERSION__ === 'string' ? __XERXES_DESKTOP_VERSION__ : '0.3.6'
export const DESKTOP_DAEMON_PROTOCOL =
  typeof __XERXES_DESKTOP_PROTOCOL__ === 'number' ? __XERXES_DESKTOP_PROTOCOL__ : 35
export const EXPECTED_DAEMON_BUILD_ID =
  typeof __XERXES_EXPECTED_DAEMON_BUILD_ID__ === 'string' ? __XERXES_EXPECTED_DAEMON_BUILD_ID__ : ''

export interface DesktopBuildIdentity {
  readonly version: string
  readonly protocol: number
  readonly expectedDaemonBuildId: string
}

const CURRENT_DESKTOP_BUILD: DesktopBuildIdentity = Object.freeze({
  version: DESKTOP_VERSION,
  protocol: DESKTOP_DAEMON_PROTOCOL,
  expectedDaemonBuildId: EXPECTED_DAEMON_BUILD_ID,
})

/**
 * Return an actionable compatibility warning for an initialize response.
 * Source-build fingerprints catch the common same-version stale process that
 * semantic versions cannot distinguish.
 */
export function daemonCompatibilityWarning(
  result: Readonly<Record<string, unknown>>,
  desktop: DesktopBuildIdentity = CURRENT_DESKTOP_BUILD,
): string | null {
  const protocol = finiteNumber(result.daemon_protocol)
  const version = stringValue(result.daemon_version) || stringValue(result.version)
  const buildId = stringValue(result.daemon_build_id)

  if (protocol !== undefined && protocol > desktop.protocol) {
    return 'The app is older than the daemon — update and restart Xerxes.'
  }
  if (protocol === undefined || protocol < desktop.protocol) {
    return 'Daemon is older than the app — restart it.'
  }

  const versionOrder = compareVersions(version, desktop.version)
  if (versionOrder > 0) {
    return 'The app is older than the daemon — update and restart Xerxes.'
  }
  if (versionOrder < 0) {
    return 'Daemon is older than the app — restart it.'
  }

  if (
    desktop.expectedDaemonBuildId
    && (!buildId || buildId !== desktop.expectedDaemonBuildId)
  ) {
    // Same version and protocol but different source fingerprints: either
    // side can be the stale one (a long-lived app instance predating a
    // rebuild is just as common as an old daemon), so the message must not
    // assert a direction it cannot prove.
    return 'App and daemon builds differ — restart the daemon, and quit and relaunch the app.'
  }
  return null
}

function compareVersions(left: string, right: string): number {
  if (!left || !right) return 0
  const a = numericVersion(left)
  const b = numericVersion(right)
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const delta = (a[index] ?? 0) - (b[index] ?? 0)
    if (delta !== 0) return delta < 0 ? -1 : 1
  }
  return 0
}

function numericVersion(value: string): number[] {
  const core = value.trim().replace(/^v/i, '').split('-', 1)[0] ?? ''
  return core.split('.').map(part => {
    const parsed = Number.parseInt(part, 10)
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0
  })
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}
