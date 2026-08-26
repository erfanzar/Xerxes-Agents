// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface CapabilityGrant {
  readonly scope: string
  readonly action: string
  readonly resources?: readonly string[]
  readonly hosts?: readonly string[]
}

export interface CapabilityManifest {
  readonly id: string
  readonly capabilities: readonly CapabilityGrant[]
}

export interface CapabilityDiff {
  readonly added: readonly { readonly pluginId: string; readonly capability: CapabilityGrant }[]
  readonly removed: readonly { readonly pluginId: string; readonly capability: CapabilityGrant }[]
}

interface RegistryState {
  readonly entries: ReadonlyMap<string, readonly CapabilityGrant[]>
}

export class CapabilityRegistry {
  private entries = new Map<string, readonly CapabilityGrant[]>()

  register(manifest: CapabilityManifest): void {
    if (!manifest.id) throw new Error('capability manifest id cannot be empty')
    this.entries.set(manifest.id, [...manifest.capabilities])
  }

  unregister(pluginId: string): void {
    this.entries.delete(pluginId)
  }

  isAllowed(pluginId: string, capability: string, target?: string): boolean {
    const [scope, action] = capability.split(':')
    const grants = this.entries.get(pluginId)
    if (grants === undefined) return false
    for (const grant of grants) {
      if (grant.scope !== scope || grant.action !== action) continue
      if (target === undefined) return true
      // A grant may scope by resource paths, by hosts, or by both. When both are
      // present the target has to match EITHER — it is one string, a path or a
      // host, and requiring it to satisfy both lists made such a grant
      // unsatisfiable rather than strict. An absent list is "unscoped on this
      // axis"; an empty list is "nothing", which is handled in matchesAny.
      const resources = grant.resources
      const hosts = grant.hosts
      if (resources !== undefined || hosts !== undefined) {
        const permitted = (resources !== undefined && matchesAny(target, resources))
          || (hosts !== undefined && matchesAny(target, hosts))
        if (!permitted) continue
      }
      return true
    }
    return false
  }

  snapshot(): RegistryState {
    return { entries: new Map(this.entries) }
  }

  restore(state: RegistryState): void {
    this.entries = new Map(state.entries)
  }

  /**
   * Run a mutation set that is rolled back if it fails.
   *
   * Handles async operations too. The synchronous-only version returned the
   * pending promise straight through, so a rejecting async callback never
   * reached the `catch` and its partially-applied grants stayed committed —
   * exactly the state this exists to prevent, in the one shape most likely to
   * be used.
   */
  transaction<T>(operation: (tx: CapabilityRegistry) => T): T {
    const before = this.snapshot()
    let result: T
    try {
      result = operation(this)
    } catch (error) {
      this.restore(before)
      throw error
    }
    if (isPromiseLike(result)) {
      return Promise.resolve(result).catch((error: unknown) => {
        this.restore(before)
        throw error
      }) as T
    }
    return result
  }

  diff(manifests: readonly CapabilityManifest[]): CapabilityDiff {
    const next = new Map<string, readonly CapabilityGrant[]>()
    for (const manifest of manifests) next.set(manifest.id, [...manifest.capabilities])

    const added: Array<{ readonly pluginId: string; readonly capability: CapabilityGrant }> = []
    const removed: Array<{ readonly pluginId: string; readonly capability: CapabilityGrant }> = []

    for (const [pluginId, grants] of next) {
      const current = this.entries.get(pluginId) ?? []
      for (const grant of grants) {
        if (!hasEquivalent(current, grant)) added.push({ pluginId, capability: grant })
      }
    }
    for (const [pluginId, grants] of this.entries) {
      const future = next.get(pluginId) ?? []
      for (const grant of grants) {
        if (!hasEquivalent(future, grant)) removed.push({ pluginId, capability: grant })
      }
    }
    return { added, removed }
  }
}

/**
 * An empty pattern list permits NOTHING.
 *
 * This returned true for an empty list, so `{ scope: 'fs', action: 'read',
 * resources: [] }` — the natural way to write "no resources permitted" — granted
 * every path, `/etc/passwd` included. A capability registry must read an empty
 * allow-list as empty; "unscoped" is expressed by omitting the field, which
 * isAllowed handles before calling here.
 */
function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return typeof (value as PromiseLike<unknown> | undefined)?.then === 'function'
}

function matchesAny(target: string, patterns: readonly string[]): boolean {
  return patterns.some(pattern => matchPattern(target, pattern))
}

function matchPattern(target: string, pattern: string): boolean {
  if (pattern === '*' || pattern === '**') return true
  if (pattern.endsWith('/**')) {
    const prefix = pattern.slice(0, -2)
    return target === prefix.slice(0, -1) || target.startsWith(prefix)
  }
  if (pattern.endsWith('/*')) {
    const prefix = pattern.slice(0, -1)
    return target.startsWith(prefix) && !target.slice(prefix.length).includes('/')
  }
  if (pattern.includes('*')) {
    const regex = new RegExp(`^${pattern.replace(/\*\*/g, '___GLOBSTAR___').replace(/\*/g, '[^/]*').replace(/___GLOBSTAR___/g, '.*')}$`)
    return regex.test(target)
  }
  return target === pattern
}

function hasEquivalent(grants: readonly CapabilityGrant[], target: CapabilityGrant): boolean {
  return grants.some(grant => grant.scope === target.scope
    && grant.action === target.action
    && arraysEqual(grant.resources, target.resources)
    && arraysEqual(grant.hosts, target.hosts))
}

function arraysEqual(left: readonly string[] | undefined, right: readonly string[] | undefined): boolean {
  if (left === undefined) return right === undefined
  if (right === undefined) return false
  if (left.length !== right.length) return false
  return [...left].sort().join(',') === [...right].sort().join(',')
}
