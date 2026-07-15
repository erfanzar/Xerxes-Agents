// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Categories used by Xerxes' small legacy component registries. */
export const RegistryType = Object.freeze({
  CLIENT: 'client',
  AGENTS: 'agents',
  XERXES: 'xerxes',
} as const)

export type RegistryType = (typeof RegistryType)[keyof typeof RegistryType]

export interface BasicRegistryInstance {
  toDict(): Record<string, unknown>
  toString(): string
}

export type RegistryConstructor<T extends object = object> = new (...args: any[]) => T
export type DecoratedRegistryConstructor<T extends RegistryConstructor> = new (
  ...args: ConstructorParameters<T>
) => InstanceType<T> & BasicRegistryInstance
export type ComponentRegistry = Record<string, RegistryConstructor>

/** Raised when a registry decorator is given an invalid category or target. */
export class BasicRegistryError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'BasicRegistryError'
  }
}

export const CLIENT_REGISTRY: ComponentRegistry = createComponentRegistry()
export const AGENTS_REGISTRY: ComponentRegistry = createComponentRegistry()
export const XERXES_REGISTRY: ComponentRegistry = createComponentRegistry()

/** Stable category-to-registry map; registry contents remain intentionally mutable. */
export const REGISTRY: Readonly<Record<RegistryType, ComponentRegistry>> = Object.freeze({
  [RegistryType.CLIENT]: CLIENT_REGISTRY,
  [RegistryType.AGENTS]: AGENTS_REGISTRY,
  [RegistryType.XERXES]: XERXES_REGISTRY,
})

/**
 * Render a nested record as a compact indentation-based debug view.
 *
 * Only plain record values recurse, matching the original helper's treatment
 * of nested dictionaries. Arrays and class instances are formatted as values.
 */
export function prettyPrint(input: Readonly<Record<string, unknown>>, indent = 0): string {
  if (!isRecord(input)) throw new BasicRegistryError('prettyPrint input must be an object')
  if (!Number.isInteger(indent) || indent < 0) {
    throw new BasicRegistryError('prettyPrint indent must be a non-negative integer')
  }
  const lines: string[] = []
  for (const [key, value] of Object.entries(input)) {
    lines.push(' '.repeat(indent) + key + ':')
    if (isRecord(value)) {
      lines.push(prettyPrint(value, indent + 2))
    } else {
      lines.push(' '.repeat(indent + 2) + String(value))
    }
  }
  return lines.join('\n')
}

/**
 * Register a class and give its instances common public-field inspection.
 *
 * The returned decorator mutates the same constructor rather than producing a
 * wrapper, preserving constructor identity in the selected registry.
 */
export function basicRegistry<T extends RegistryConstructor>(
  registryType: RegistryType,
  registryName: string,
): (target: T) => T & DecoratedRegistryConstructor<T> {
  const registry = registryFor(registryType)
  const name = requireRegistryName(registryName)
  return target => {
    if (typeof target !== 'function' || !target.prototype) {
      throw new BasicRegistryError('basicRegistry target must be a class constructor')
    }
    Object.defineProperties(target.prototype, {
      toDict: {
        configurable: true,
        value: registryInstanceToDict,
        writable: true,
      },
      toString: {
        configurable: true,
        value: registryInstanceToString,
        writable: true,
      },
      [Symbol.for('nodejs.util.inspect.custom')]: {
        configurable: true,
        value: registryInstanceToString,
        writable: true,
      },
    })
    registry[name] = target
    return target as T & DecoratedRegistryConstructor<T>
  }
}

function createComponentRegistry(): ComponentRegistry {
  return Object.create(null) as ComponentRegistry
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function registryFor(registryType: RegistryType): ComponentRegistry {
  if (
    registryType === RegistryType.CLIENT
    || registryType === RegistryType.AGENTS
    || registryType === RegistryType.XERXES
  ) {
    return REGISTRY[registryType]
  }
  throw new BasicRegistryError('unknown registry type: ' + String(registryType))
}

function registryInstanceToDict(this: object): Record<string, unknown> {
  const values: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(this)) {
    if (!key.startsWith('_')) values[key] = value
  }
  return values
}

function registryInstanceToString(this: object): string {
  const name = (this.constructor as { readonly name?: string }).name || 'Anonymous'
  const values = registryInstanceToDict.call(this)
  const body = prettyPrint(values)
  return body ? `${name}(\n\t${body.replaceAll('\n', '\n\t')}\n)` : `${name}({})`
}

function requireRegistryName(value: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new BasicRegistryError('registry name must be a non-empty string')
  }
  if (value === '__proto__' || value === 'constructor' || value === 'prototype') {
    throw new BasicRegistryError('registry name is reserved: ' + value)
  }
  return value
}
