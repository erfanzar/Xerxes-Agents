// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export const INTERPOLATION_RE = /\{!(.+?)\}/g

export const hasInterpolation = (s: string) => /\{!.+?\}/.test(s)
