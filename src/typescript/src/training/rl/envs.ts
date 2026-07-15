// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** One completed rollout or evaluation passed to an environment reward function. */
export type RewardTrajectory = Readonly<Record<string, unknown>>

/** Maps a trajectory into a scalar reward supplied by the environment author. */
export type RewardFn = (trajectory: RewardTrajectory) => number

export interface RLEnvironmentOptions {
  /** Backend-specific configuration artifact, when the environment has one. */
  readonly configPath?: string
  readonly backend?: string
  readonly description: string
  readonly name: string
  readonly rewardFn?: RewardFn
  readonly tags?: readonly string[]
}

/** Immutable metadata and reward boundary for one reinforcement-learning environment. */
export class RLEnvironment {
  readonly backend: string
  readonly configPath: string
  readonly description: string
  readonly name: string
  readonly rewardFn: RewardFn | undefined
  readonly tags: readonly string[]

  constructor(options: RLEnvironmentOptions) {
    this.name = requiredText(options.name, 'name')
    this.description = text(options.description, 'description')
    this.configPath = text(options.configPath ?? '', 'configPath')
    this.backend = text(options.backend ?? 'local', 'backend')
    this.tags = Object.freeze((options.tags ?? []).map((tag, index) => requiredText(tag, `tags[${index}]`)))
    this.rewardFn = options.rewardFn
    Object.freeze(this)
  }

  /** Evaluate a trajectory when this environment owns scoring. */
  reward(trajectory: RewardTrajectory): number | undefined {
    const rewardFn = this.rewardFn
    if (rewardFn === undefined) return undefined
    const value = rewardFn({ ...trajectory })
    if (!Number.isFinite(value)) throw new TypeError(`environment ${this.name} returned a non-finite reward`)
    return value
  }
}

/** In-process registry for environment metadata supplied by the embedding host. */
export class RLEnvironmentRegistry {
  private readonly environments = new Map<string, RLEnvironment>()

  /** Register or replace an environment by its stable name. */
  register(environment: RLEnvironment): void {
    if (!(environment instanceof RLEnvironment)) throw new TypeError('environment must be an RLEnvironment')
    this.environments.set(environment.name, environment)
  }

  /** Return registered environments in deterministic name order. */
  listEnvironments(): readonly RLEnvironment[] {
    return [...this.environments.values()].sort((left, right) => left.name.localeCompare(right.name))
  }

  /** Look up one environment without manufacturing an external backend result. */
  get(name: string): RLEnvironment | undefined {
    return this.environments.get(name)
  }
}

/** Construct the small built-in environment catalogue available without a provider SDK. */
export function builtinEnvironments(): RLEnvironmentRegistry {
  const registry = new RLEnvironmentRegistry()
  registry.register(new RLEnvironment({
    name: 'xerxes-terminal-test',
    description: 'Synthetic 4-task curriculum (greeting.txt, count.txt, answer.txt, eval-arithmetic)',
    backend: 'local',
    tags: ['coding', 'tooling'],
    rewardFn: trajectory => trajectory.passed_all_tasks === true ? 1 : 0,
  }))
  registry.register(new RLEnvironment({
    name: 'xerxes-swe-bench',
    description: 'SWE-bench style env: write code → tests verify reward.',
    backend: 'modal',
    tags: ['coding', 'swe'],
    rewardFn: trajectory => {
      const passed = finiteNumber(trajectory.tests_passed) ?? 0
      const total = finiteNumber(trajectory.tests_total) ?? 1
      return passed / Math.max(1, total)
    },
  }))
  registry.register(new RLEnvironment({
    name: 'agentic-opd',
    description: 'Objective Policy Distribution: open-ended objective scoring.',
    backend: 'modal',
    tags: ['research'],
  }))
  return registry
}

function requiredText(value: string, name: string): string {
  const normalized = text(value, name).trim()
  if (!normalized) throw new TypeError(`${name} must be non-empty`)
  return normalized
}

function text(value: string, name: string): string {
  if (typeof value !== 'string') throw new TypeError(`${name} must be a string`)
  return value
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
