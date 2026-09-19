// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

/** Include existing SSH configuration by reference, without copying keys or
 * config values. SendEnv accumulates, so clear it after all includes; SetEnv
 * uses the first value, so claim it before them with a nonsecret marker.
 * Do not reuse another connection's forwarding state or execute LocalCommand. */
export async function writeSshConnectionConfig(directory: string, sources = [join(homedir(), '.ssh/config'), '/etc/ssh/ssh_config']): Promise<string> {
  const includes = sources.map(path => {
    if (/[\r\n\0]/.test(path)) throw new Error('SSH configuration path contains unsupported characters.')
    return `Include "${path.replaceAll('\\', '\\\\').replaceAll('"', '\\"')}"`
  })
  const path = join(directory, 'ssh-config')
  await writeFile(path, [
    'SetEnv XERXES_SSH=1',
    'ControlMaster no',
    'ControlPath none',
    'PermitLocalCommand no',
    'ClearAllForwardings yes',
    ...includes.flatMap(include => ['Host *', include]),
    'Host *', '  SendEnv -*', '',
  ].join('\n'), { mode: 0o600 })
  return path
}
