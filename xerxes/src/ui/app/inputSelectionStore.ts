// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom } from 'nanostores'

export interface InputSelection {
  clear: () => void
  collapseToEnd: () => void
  end: number
  start: number
  value: string
}

export const $inputSelection = atom<InputSelection | null>(null)

export const setInputSelection = (next: InputSelection | null) => $inputSelection.set(next)

export const getInputSelection = () => $inputSelection.get()
