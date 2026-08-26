// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DurableTaskDefinition, DurableAttempt, DurableTaskRuntime } from './durableTaskRuntime.js'

/** Compatibility surface mapping task lifecycle onto durable task events. */
export interface DurableTaskBridge {
  readonly taskCreated: (task: DurableTaskDefinition) => Promise<void>
  readonly attemptStarted: (attempt: Omit<DurableAttempt, 'status'>) => Promise<DurableAttempt>
  readonly attemptCompleted: (attemptId: string, result: { readonly deliveryId: string; readonly output: string }) => Promise<void>
  readonly attemptFailed: (attemptId: string, result: { readonly error: string; readonly retryable: boolean }) => Promise<void>
  readonly taskCancelled: (taskId: string, error: string) => Promise<void>
}

export function bridgeDurableTaskLifecycle(runtime: DurableTaskRuntime): DurableTaskBridge {
  return {
    taskCreated: task => runtime.createTask(task).then(() => undefined),
    attemptStarted: attempt => runtime.startAttempt(attempt),
    attemptCompleted: (attemptId, result) => runtime.completeAttempt(attemptId, result),
    attemptFailed: (attemptId, result) => runtime.failAttempt(attemptId, result),
    taskCancelled: (taskId, error) => runtime.cancelTask(taskId, error),
  }
}
