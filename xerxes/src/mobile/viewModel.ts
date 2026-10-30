// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubState, SelectedSession } from '../client/hubStore.js'
import { taskKey, type MachineView, type SupervisorChoice, type TaskSummary } from '../client/hubTypes.js'
import { statusView, type StatusView } from '../client/orbMapping.js'
import type { TaskStatus } from '../client/taskTypes.js'

/** Status mark for an Inbox row: connection, workspace health and task evidence stay separate inputs. */
export function viewForTask(state: HubState, task: TaskSummary): StatusView {
  const health = state.workspaces[task.workspaceId]?.health
  return statusView({
    connection: state.connection.state,
    ...(health === undefined ? {} : { workspaceHealth: health }),
    status: task.status,
    activity: task.activity,
    pendingCount: task.pendingCount,
    stale: task.stale,
    interrupted: task.interrupted,
  })
}

export function statusOfSelected(state: HubState, selected: SelectedSession): TaskStatus {
  if (selected.tail.pending.length > 0) return 'needs_you'
  return selected.tail.status ?? state.tasks[taskKey(selected.workspaceId, selected.sessionId)]?.status ?? 'unknown'
}

/**
 * The task waits on a person, the waiting items were looked for, and none
 * could be recovered: say "Waiting for approval on the machine; details
 * unavailable" instead of showing a Needs-you task with nothing to tap.
 */
export function pendingDetailsUnavailable(state: HubState, selected: SelectedSession): boolean {
  const summary = state.tasks[taskKey(selected.workspaceId, selected.sessionId)]
  return selected.pendingChecked === true && selected.tail.pending.length === 0 && (summary?.pendingCount ?? 0) > 0 && summary?.status === 'needs_you'
}

export function viewForSelected(state: HubState, selected: SelectedSession): StatusView {
  const summary = state.tasks[taskKey(selected.workspaceId, selected.sessionId)]
  const health = state.workspaces[selected.workspaceId]?.health
  return statusView({
    connection: state.connection.state,
    ...(health === undefined ? {} : { workspaceHealth: health }),
    status: statusOfSelected(state, selected),
    activity: selected.tail.activity,
    pendingCount: selected.tail.pending.length,
    stale: summary?.stale ?? false,
    interrupted: summary?.interrupted ?? false,
  })
}

export function workspaceLabel(state: HubState, workspaceId: string): string {
  return state.workspaces[workspaceId]?.label ?? workspaceId
}

/** "XERXES · <model> · <effort>" once a model is chosen, plain "XERXES" before. */
export function supervisorHeading(config: SupervisorChoice | undefined): string {
  return config === undefined ? 'XERXES' : `XERXES · ${config.model} · ${config.effort ?? 'default effort'}`
}

export function machineOf(state: HubState, workspaceId: string): MachineView | undefined {
  return state.machines.list.find(machine => machine.workspaceIds.includes(workspaceId))
}

export interface TaskGroup {
  readonly key: string
  /** Undefined when there is nothing to group by (one machine or none). */
  readonly label?: string
  readonly tasks: readonly TaskSummary[]
}

/**
 * With more than one machine the Inbox is grouped by machine, in the machine
 * list's order; tasks of a workspace no machine claims go last. The incoming
 * task order is kept inside each group.
 */
export function groupTasksByMachine(state: HubState, tasks: readonly TaskSummary[]): readonly TaskGroup[] {
  if (state.machines.list.length <= 1) return [{ key: 'all', tasks }]
  const groups: TaskGroup[] = []
  const claimed = new Set<string>()
  for (const machine of state.machines.list) {
    const own = tasks.filter(task => machine.workspaceIds.includes(task.workspaceId))
    for (const task of own) claimed.add(`${task.workspaceId}\u001f${task.id}`)
    if (own.length > 0) groups.push({ key: machine.id, label: machine.label, tasks: own })
  }
  const rest = tasks.filter(task => !claimed.has(`${task.workspaceId}\u001f${task.id}`))
  if (rest.length > 0) groups.push({ key: '\u001fother', label: 'Other workspaces', tasks: rest })
  return groups
}
