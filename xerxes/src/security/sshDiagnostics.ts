// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** SSH diagnostics may contain remote banners, command output or config
 * values. Classify them in memory; never return arbitrary subprocess text. */
export function sshFailure(stage: 'setup' | 'tunnel' | 'browse', diagnostic: string): Error {
  if (/host key|host identification/i.test(diagnostic)) {
    return new Error('SSH host key verification failed. Verify this destination with SSH in a terminal before reconnecting.')
  }
  if (/permission denied|authentication/i.test(diagnostic)) {
    return new Error('SSH authentication failed. Check this destination’s SSH identity and access in a terminal.')
  }
  if (/connection (?:reset|refused|closed)|timed out|network is unreachable|no route to host|broken pipe/i.test(diagnostic)) {
    return new Error('SSH connection closed or timed out. Check network access to this destination and retry.')
  }
  if (stage === 'browse') return new Error('SSH browse failed. Verify SSH access and that the remote folder exists and is readable.')
  if (stage === 'tunnel') return new Error('SSH tunnel closed. Check SSH access and permission to forward the remote daemon socket.')
  return new Error('Remote setup failed. Check Git, Bun, network access and workspace permissions on the destination; inspect ~/.xerxes/remote-runtime/setup.log for the failed stage.')
}
