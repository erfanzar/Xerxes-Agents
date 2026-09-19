// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Read-only probe run on the SSH host before considering installation work. */
export function remoteResumeProgram(socketPath: string): string {
  return `
const { connect } = await import('node:net');
const socket = connect(${JSON.stringify(socketPath)});
let buffer = '', finished = false;
const finish = (message, failure = false) => {
  if (finished) return;
  finished = true;
  clearTimeout(timer);
  socket.destroy();
  if (failure) { console.error(message); process.exitCode = 1; }
  else console.log(message);
};
const timer = setTimeout(() => finish('Remote runtime did not answer its reconnect probe.', true), 5000);
socket.on('connect', () => socket.write(JSON.stringify({jsonrpc:'2.0', id:1, method:'runtime.status', params:{}}) + '\\n'));
socket.on('error', error => {
  if (error.code === 'ENOENT' || error.code === 'ECONNREFUSED') finish('XERXES_REMOTE_MISSING');
  else finish('Remote runtime probe failed: ' + error.message, true);
});
socket.on('end', () => finish('Remote runtime closed its reconnect probe before replying.', true));
socket.on('data', chunk => {
  buffer += chunk.toString();
  if (buffer.length > 65536) return finish('Remote runtime probe response is too large.', true);
  for (;;) {
    const newline = buffer.indexOf('\\n');
    if (newline < 0) break;
    const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
    let frame;
    try { frame = JSON.parse(line); } catch { return finish('Invalid remote runtime probe response.', true); }
    if (!frame || typeof frame !== 'object' || Array.isArray(frame)) return finish('Invalid remote runtime probe response.', true);
    if (frame.id !== 1) continue;
    if (frame.error) return finish('Remote runtime rejected reconnect probe: ' + (frame.error.message || 'RPC error'), true);
    if (!frame.result || typeof frame.result !== 'object' || Array.isArray(frame.result) || frame.result.ok === false)
      return finish('Remote runtime returned an unsuccessful reconnect probe.', true);
    return finish('XERXES_REMOTE_ALIVE');
  }
});
`
}
