// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface ImageInspectionResult {
  readonly detail: 'auto' | 'high' | 'low' | 'original'
  readonly format?: string
  readonly height: number
  readonly mode: string
  readonly path: string
  readonly width: number
}

export interface UserPromptOption {
  readonly label: string
  readonly value: string
}

export interface PendingUserPrompt {
  readonly allowFreeform: boolean
  readonly createdAt: string
  readonly options: readonly UserPromptOption[]
  readonly placeholder?: string
  readonly question: string
  readonly requestId: string
}

export interface UserPromptAnswer {
  readonly answer: string
  readonly question: string
  readonly rawInput: string
  readonly requestId: string
  readonly selectedOption?: UserPromptOption
  readonly usedFreeform: boolean
}

export interface OperatorPlanStep {
  readonly status: string
  readonly step: string
}

export interface OperatorPlanStateSnapshot {
  readonly explanation?: string
  readonly revision: number
  readonly steps: readonly OperatorPlanStep[]
  readonly updatedAt: string
}

export function operatorNowIso(now: () => Date = () => new Date()): string {
  return now().toISOString()
}

export function userPromptOption(label: string, value = label): UserPromptOption {
  return Object.freeze({ label, value })
}
