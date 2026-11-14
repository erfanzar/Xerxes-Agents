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
  // Scripts open with setup statements (`export PATH=…; unset …; X=…; cd $X && …`).
  // Skip those and start at the first statement that does the work.
  rest = rest.slice(firstWorkStatement(rest)).trim() || rest
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

/** Statements that only prepare the environment: they name nothing the person cares about. */
const SETUP_STATEMENT = /^(?:(?:export|unset|set|cd|pushd|source)\s|\.\s)|^[A-Za-z_][A-Za-z0-9_]*=\S*$/

/**
 * Offset of the first top-level statement (split on `;`, `&&`, `||` and newlines,
 * outside quotes) that is not setup. 0 when there is none, so nothing is lost.
 */
function firstWorkStatement(script: string): number {
  const starts: number[] = [0]
  let quote = ''
  for (let index = 0; index < script.length; index++) {
    const char = script[index]!
    if (quote) { if (char === quote) quote = ''; else if (char === '\\' && quote === '"') index++; continue }
    if (char === '\\') { index++; continue }
    if (char === '"' || char === "'") { quote = char; continue }
    if (char === ';' || char === '\n') starts.push(index + 1)
    else if ((char === '&' || char === '|') && script[index + 1] === char) { starts.push(index + 2); index++ }
  }
  for (let at = 0; at < starts.length; at++) {
    const end = at + 1 < starts.length ? starts[at + 1]! : script.length
    const statement = script.slice(starts[at], end).replace(/(?:;|&&|\|\|)\s*$/, '').trim()
    if (statement && !SETUP_STATEMENT.test(statement)) return starts[at]!
  }
  return 0
}
