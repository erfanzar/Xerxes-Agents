// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Input shape for {@link compact}: every key of `T` must be supplied, and the
 * ones `T` declares optional may be `undefined`.
 *
 * Optionality is preserved exactly as `T` declares it, so a required property
 * can be neither omitted nor `undefined`. That is what keeps the cast inside
 * `compact` sound: the only keys it can drop are ones `T` is happy without.
 */
type Loose<T> = {
  [K in keyof T]: undefined extends T[K] ? T[K] | undefined : T[K]
}

/**
 * Build an object from a literal that may carry `undefined`, dropping those
 * keys so an optional property ends up ABSENT rather than present-and-undefined.
 *
 * Adapter and merge builders are where this matters: they assemble a value out
 * of several partial sources, each field written as `a ?? b`, which is
 * `undefined` when neither source has it. Under `exactOptionalPropertyTypes`
 * that is not assignable to `field?: T`, and the difference is real at runtime
 * too — `'field' in object` and `Object.keys` both see a key that was never
 * meant to be there.
 *
 * The alternative was widening every such property to `?: T | undefined`, which
 * silences the compiler by turning the flag off for those types by another
 * name. This keeps the declarations honest and puts the handling at the one
 * place that knows a value is missing.
 */
export function compact<T extends object>(value: Loose<T>): T {
  const result: Record<string, unknown> = {}
  for (const [key, entry] of Object.entries(value)) {
    if (entry !== undefined) {
      result[key] = entry
    }
  }
  return result as T
}
