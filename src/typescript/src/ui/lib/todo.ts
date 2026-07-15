// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { TodoItem } from '../types.js'

export type TodoTone = 'active' | 'body' | 'dim'

export const todoGlyph = (status: TodoItem['status']) =>
  status === 'completed' ? '[x]' : status === 'cancelled' ? '[-]' : status === 'in_progress' ? '[>]' : '[ ]'

export const todoTone = (status: TodoItem['status']): TodoTone =>
  status === 'in_progress' ? 'active' : status === 'pending' ? 'body' : 'dim'
