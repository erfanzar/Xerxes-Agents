// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface TodoItem { readonly id: string; readonly content: string; readonly status: 'pending' | 'in_progress' | 'completed' }
export function todosFromResult(name: unknown, result: unknown): readonly TodoItem[] | null {
  if (typeof name !== 'string' || name.replaceAll('_','').toLowerCase() !== 'todowritetool' || typeof result !== 'string') return null
  const items: TodoItem[] = []
  for (const line of result.split('\n')) {
    const match = /^\s*(\d+)\.\s+\[([ x~])\]\s+(.*\S)\s*$/.exec(line)
    if (match) items.push({id:`todo-${match[1]}`,content:match[3]!,status:match[2]==='x'?'completed':match[2]==='~'?'in_progress':'pending'})
  }
  return items.length ? items : null
}
export function todoItemsOf(value: unknown): readonly TodoItem[] | null {
  if (!Array.isArray(value)) return null
  const items: TodoItem[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object' || typeof item.content !== 'string' || !['pending','in_progress','completed'].includes(item.status)) return null
    items.push({id: typeof item.id === 'string' ? item.id : String(items.length), content: item.content, status: item.status})
  }
  return items
}
