// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The part of a command worth putting in a 300px row.
 *
 * Real commands in this app arrive wrapped and prefixed:
 * `bash -c ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_... pytest tests/`
 * Truncating that to the row width yields `bash -c ENABLE_DISTRIBUTED_INIT=0 JAX_PLA…`,
 * which identifies nothing — every such row looks identical, and the one
 * token that distinguishes them (`pytest`) is the one that gets cut.
 *
 * So: unwrap the shell, drop leading `NAME=value` environment assignments,
 * and show what actually ran. The untouched original stays in the row's
 * `title` and in the "Full command" disclosure, so nothing is hidden —
 * this only changes which end of the string survives truncation.
 */
export function commandGist(command: string): string {
  let rest = command.trim()
  // `bash -c "…"` / `sh -lc '…'` — unwrap once; nested wrappers are rare
  // enough that a loop would mostly be a way to strip something real.
  const wrapper = rest.match(/^(?:\S*\/)?(?:ba|z|da)?sh\s+-[a-z]*c\s+(.*)$/s)
  if (wrapper?.[1]) {
    rest = wrapper[1].trim()
    const quote = rest[0]
    if ((quote === '"' || quote === "'") && rest.endsWith(quote) && rest.length > 1) {
      rest = rest.slice(1, -1).trim()
    }
  }
  // Leading `NAME=value` pairs are configuration, not the action — but only
  // drop them while something else remains. `FOO=1 BAR=2` with no command
  // after it *is* the content, and a regex that strips greedily from the
  // front would eat all but the last pair and present that as the command.
  const assignment = /^[A-Z_][A-Z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)$/
  const tokens = rest.split(/\s+/).filter(Boolean)
  let first = 0
  while (first < tokens.length && assignment.test(tokens[first] ?? '')) first += 1
  const stripped = first < tokens.length ? tokens.slice(first).join(' ') : ''
  return stripped || rest || command.trim()
}
