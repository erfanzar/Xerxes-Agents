// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface DependencySpec {
  readonly name: string
  readonly versionConstraint?: string
}

export interface ResolveResult {
  readonly conflicts: readonly string[]
  readonly missing: readonly string[]
  readonly resolutionOrder: readonly string[]
  readonly satisfied: boolean
}

/** Raised when a plugin dependency graph loops back to an ancestor. */
export class CircularDependencyError extends Error {
  constructor(readonly cycle: readonly string[]) {
    super(`Circular dependency detected: ${cycle.join(' -> ')}`)
    this.name = 'CircularDependencyError'
  }
}

/** Small semver-like constraint evaluator used for plugin compatibility checks. */
export class VersionConstraint {
  private readonly constraints: Array<{ readonly operator: string; readonly parts: number; readonly target: readonly number[] | undefined }> = []

  constructor(readonly raw: string) {
    for (const rawPart of raw.split(',')) {
      const part = rawPart.trim()
      if (!part) continue
      const matched = /^(~=|==|!=|>=|<=|>|<)\s*(.+)$/.exec(part)
      const operator = matched?.[1] ?? '=='
      const version = matched?.[2] ?? part
      const unpadded = parseVersion(version)
      this.constraints.push({
        operator,
        target: unpadded === undefined ? undefined : padVersion(unpadded),
        parts: unpadded?.length ?? 0,
      })
    }
  }

  satisfies(version: string): boolean {
    const parsed = parseVersion(version)
    const actual = parsed === undefined ? undefined : padVersion(parsed)
    return this.constraints.every(constraint => this.check(constraint.operator, actual, constraint.target, constraint.parts))
  }

  private check(operator: string, actual: readonly number[] | undefined, target: readonly number[] | undefined, rawParts: number): boolean {
    if (actual === undefined || target === undefined) {
      // Non-semver input is rejected instead of truncated: only an inequality can hold for it.
      return operator === '!='
    }
    const comparison = compareVersions(actual, target)
    switch (operator) {
      case '==': return comparison === 0
      case '!=': return comparison !== 0
      case '>=': return comparison >= 0
      case '<=': return comparison <= 0
      case '>': return comparison > 0
      case '<': return comparison < 0
      case '~=': {
        if (comparison < 0) return false
        const prefix = [...target.slice(0, Math.max(0, rawParts - 1))]
        if (!prefix.length) return true
        const last = prefix.length - 1
        prefix[last] = (prefix[last] ?? 0) + 1
        return compareVersions(actual, padVersion(prefix)) < 0
      }
      default: return false
    }
  }
}

export class DependencyResolver {
  resolve(available: Readonly<Record<string, string>>, requirements: readonly DependencySpec[]): ResolveResult {
    const missing: string[] = []
    const conflicts: string[] = []
    for (const requirement of requirements) {
      const version = available[requirement.name]
      if (version === undefined) {
        missing.push(requirement.name)
      } else if (requirement.versionConstraint && !new VersionConstraint(requirement.versionConstraint).satisfies(version)) {
        conflicts.push(`${requirement.name}: requires ${requirement.versionConstraint}, found ${version}`)
      }
    }
    return { satisfied: missing.length === 0 && conflicts.length === 0, missing, conflicts, resolutionOrder: [] }
  }

  topologicalSort(graph: Readonly<Record<string, readonly string[]>>): string[] {
    const state = new Map<string, 'done' | 'visiting' | undefined>()
    const path: string[] = []
    const ordered: string[] = []
    const visit = (node: string): void => {
      const nodeState = state.get(node)
      if (nodeState === 'done') return
      if (nodeState === 'visiting') {
        const start = path.indexOf(node)
        throw new CircularDependencyError([...path.slice(start), node])
      }
      state.set(node, 'visiting')
      path.push(node)
      for (const dependency of graph[node] ?? []) {
        if (dependency in graph) visit(dependency)
      }
      path.pop()
      state.set(node, 'done')
      ordered.push(node)
    }
    for (const node of Object.keys(graph).sort()) visit(node)
    return ordered
  }
}

export function parseDependency(value: string): DependencySpec {
  const matched = /^([A-Za-z0-9_-]+)(.*)$/.exec(value.trim())
  if (!matched) return { name: value.trim() }
  const name = matched[1] ?? ''
  const versionConstraint = (matched[2] ?? '').trim()
  return versionConstraint ? { name, versionConstraint } : { name }
}

function compareVersions(left: readonly number[], right: readonly number[]): number {
  const length = Math.max(left.length, right.length)
  for (let index = 0; index < length; index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0)
    if (difference !== 0) return Math.sign(difference)
  }
  return 0
}

function padVersion(parts: readonly number[]): number[] {
  return [...parts, ...Array(Math.max(0, 3 - parts.length)).fill(0)]
}

/** Parse a strict numeric dotted version; reject prerelease/build or otherwise non-semver strings. */
function parseVersion(value: string): number[] | undefined {
  const trimmed = value.trim()
  if (!/^\d+(?:\.\d+)*$/.test(trimmed)) return undefined
  return trimmed.split('.').map(Number)
}
