// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { homedir } from 'node:os'

/**
 * Abbreviate a working directory for the status line.
 *
 * `HOME` is unset on Windows, where the home directory comes from `USERPROFILE`
 * — so without the fallback every Windows path rendered in full and then got
 * truncated from the left, hiding the part the user cares about behind an
 * ellipsis. `homedir()` covers both and matches what the rest of the runtime
 * resolves paths against.
 */
export const shortCwd = (cwd: string, max = 28, home: string = homedir()) => {
  const p = home && cwd.startsWith(home) ? `~${cwd.slice(home.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}

export const fmtCwdBranch = (cwd: string, branch: null | string, max = 40) => {
  if (!branch) {
    return shortCwd(cwd, max)
  }

  const tag = ` (${branch.length > 16 ? `…${branch.slice(-15)}` : branch})`

  return `${shortCwd(cwd, Math.max(8, max - tag.length))}${tag}`
}

/**
 * Compose the terminal titlebar string:
 *   `<marker> <session name> · <model> · <cwd>`
 *
 * The session name and cwd are each omitted when empty, and a long session
 * name is truncated. The marker is always glued to the first present segment
 * with a plain space (not a ` · ` separator). When no model is known yet the
 * caller should fall back to a plain brand string instead of calling this.
 */
export const composeTabTitle = (
  marker: string,
  sessionName: string,
  model: string,
  cwd: string,
  maxName = 28
): string => {
  const name = sessionName.trim()
  const shortName = name.length > maxName ? `${name.slice(0, maxName - 1)}…` : name

  const segments = [shortName, model, cwd].filter(Boolean)

  return segments.length ? `${marker} ${segments.join(' · ')}` : marker
}
