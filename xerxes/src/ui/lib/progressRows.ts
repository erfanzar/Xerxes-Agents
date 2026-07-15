// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ActivityItem, TodoItem } from '../types.js'

export interface CompactProgressState {
  activity: readonly ActivityItem[]
  outcome: string
  todos: readonly TodoItem[]
  turnTrail: readonly string[]
}

export interface CompactProgressOptions {
  activityVisible: boolean
  toolsVisible: boolean
}

export interface CompactProgressRow {
  kind: 'activity' | 'outcome' | 'todo' | 'trail'
  text: string
  tone: 'error' | 'info' | 'success' | 'warn'
}

const activeTodo = (todos: readonly TodoItem[]) =>
  todos.find(todo => todo.status === 'in_progress') ?? todos.find(todo => todo.status === 'pending')

/** Build the small, bounded Grok-style live status tail below streamed work. */
export const compactProgressRows = (
  { activity, outcome, todos, turnTrail }: CompactProgressState,
  { activityVisible, toolsVisible }: CompactProgressOptions
): CompactProgressRow[] => {
  const rows: CompactProgressRow[] = []

  if (todos.length) {
    const completed = todos.filter(todo => todo.status === 'completed' || todo.status === 'cancelled').length
    const current = activeTodo(todos)
    const suffix = current?.content.trim() ? ` · ${current.content.trim()}` : ''

    rows.push({
      kind: 'todo',
      text: `Tasks ${completed}/${todos.length}${suffix}`,
      tone: completed === todos.length ? 'success' : 'info'
    })
  }

  if (toolsVisible) {
    for (const text of turnTrail.filter(Boolean).slice(-2)) {
      rows.push({ kind: 'trail', text, tone: 'info' })
    }
  }

  for (const item of activity.filter(item => activityVisible || item.tone !== 'info').slice(-2)) {
    rows.push({ kind: 'activity', text: item.text, tone: item.tone })
  }

  if (outcome.trim()) {
    rows.push({
      kind: 'outcome',
      text: outcome.trim(),
      tone: /denied|error|failed/i.test(outcome) ? 'warn' : 'success'
    })
  }

  return rows
}
