// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface RemoteAttachAddress {
  socketPath: string
  projectDir: string
  version?: string
  buildId?: string
}

/** Read-only: resolve the host's existing global daemon, never install or start one. */
export function remoteAttachProgram(workspacePath: string, sessions = false): string {
  return `
const { connect } = await import('node:net');
const { createHash } = await import('node:crypto');
const { homedir } = await import('node:os');
const { join, resolve } = await import('node:path');
const { realpathSync } = await import('node:fs');
const { execFileSync } = await import('node:child_process');
const override = (process.env.XERXES_HOME || '').trim();
const home = override === '~' ? homedir() : override.startsWith('~/') || override.startsWith('~\\\\') ? resolve(homedir(), override.slice(2)) : override ? resolve(override) : join(homedir(), '.xerxes');
const socketPath = (process.env.XERXES_DAEMON_SOCKET || '').trim() || join(home, 'daemon', 'global-' + createHash('sha256').update(home, 'utf8').digest('hex').slice(0, 16) + '.sock');
let projectDir = resolve(${JSON.stringify(workspacePath)});
try {
  const root = execFileSync('git', ['-C', projectDir, 'rev-parse', '--show-toplevel'], { encoding:'utf8', stdio:['ignore','pipe','ignore'], timeout:2000 }).trim();
  if (root) projectDir = realpathSync(root);
} catch { try { projectDir = realpathSync(projectDir); } catch {} }
const socket = connect(socketPath);
let buffer = '', bytes = 0, finished = false;
const results = new Map();
const finish = (message, failure = false) => {
  if (finished) return;
  finished = true;
  clearTimeout(timer);
  socket.destroy();
  if (failure) { console.error(message); process.exitCode = 1; }
  else console.log(message);
};
const timer = setTimeout(() => finish('Remote read-only probe timed out.', true), 8000);
const request = (id, method, params) => socket.write(JSON.stringify({jsonrpc:'2.0', id, method, params:{...params, project_dir:projectDir}}) + '\\n');
const cleanSessions = value => (Array.isArray(value) ? value : []).slice(0, 200).flatMap(row => {
  if (!row || typeof row !== 'object' || (row.kind || row.session_kind || 'main') !== 'main') return [];
  const id = row.id || row.session_id;
  if (typeof id !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(id) || typeof row.cwd !== 'string' || !row.cwd.startsWith('/') || row.cwd.length > 4096 || /[\\x00-\\x1f\\x7f]/.test(row.cwd)) return [];
  return [{id, cwd:row.cwd, title:typeof row.title === 'string' ? row.title.replace(/[\\x00-\\x1f\\x7f]/g, ' ').slice(0,512) : '', status:row.active_turn_id || row.status === 'working' || row.status === 'acting' ? 'working' : row.status === 'failed' ? 'failed' : 'idle'}];
});
socket.on('connect', () => request(1, 'runtime.status', {}));
socket.on('error', error => {
  if (error.code === 'ENOENT' || error.code === 'ECONNREFUSED') finish('XERXES_REMOTE_MISSING');
  else finish('Remote read-only probe could not connect.', true);
});
socket.on('end', () => finish('Remote runtime closed its read-only probe before replying.', true));
socket.on('data', chunk => {
  bytes += chunk.length;
  if (bytes > 1024 * 1024) return finish('Remote read-only response is too large.', true);
  buffer += chunk.toString();
  for (;;) {
    const newline = buffer.indexOf('\\n');
    if (newline < 0) break;
    const line = buffer.slice(0,newline); buffer = buffer.slice(newline + 1);
    let frame;
    try { frame = JSON.parse(line); } catch { return finish('Invalid remote read-only response.', true); }
    if (!frame || typeof frame !== 'object' || Array.isArray(frame)) return finish('Invalid remote read-only response.', true);
    if (![1,2,3].includes(frame.id) || results.has(frame.id)) continue;
    if (frame.error || !frame.result || typeof frame.result !== 'object' || Array.isArray(frame.result) || frame.result.ok === false) return finish('Remote runtime rejected its read-only probe.', true);
    results.set(frame.id, frame.result);
    if (frame.id === 1) {
      if (!${JSON.stringify(sessions)}) return finish('XERXES_REMOTE_ATTACH ' + JSON.stringify({socketPath, projectDir, version:frame.result.daemon_version, buildId:frame.result.daemon_build_id}));
      request(2, 'session.list', {kind:'main', scope:'global', limit:60});
      request(3, 'session.active_list', {history_limit:0});
    }
    if (results.size === 3) {
      const status = results.get(1);
      return finish('XERXES_REMOTE_ATTACH ' + JSON.stringify({socketPath, projectDir, version:status.daemon_version, buildId:status.daemon_build_id}) + '\\nXERXES_REMOTE_SESSIONS ' + JSON.stringify({saved:cleanSessions(results.get(2).sessions), live:cleanSessions(results.get(3).sessions)}));
    }
  }
});
`
}

export function remoteAttachAddress(output: string): RemoteAttachAddress | null {
  const lines = output.split('\n')
  const matches = lines.filter(line => line.startsWith('XERXES_REMOTE_ATTACH '))
  if (!matches.length && lines.includes('XERXES_REMOTE_MISSING')) return null
  if (matches.length !== 1 || lines.includes('XERXES_REMOTE_MISSING')) throw new Error('Remote attach probe did not return a valid status')
  const value: unknown = JSON.parse(matches[0]!.slice('XERXES_REMOTE_ATTACH '.length))
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid remote attach address')
  const row = value as Record<string, unknown>
  if (typeof row.socketPath !== 'string' || !row.socketPath.startsWith('/') || row.socketPath.length > 4096 || /[:\x00-\x1f\x7f]/.test(row.socketPath) ||
      typeof row.projectDir !== 'string' || !row.projectDir.startsWith('/') || row.projectDir.length > 4096 || /[\x00-\x1f\x7f]/.test(row.projectDir) ||
      row.version !== undefined && (typeof row.version !== 'string' || !/^[a-zA-Z0-9.+_-]{1,100}$/.test(row.version)) ||
      row.buildId !== undefined && (typeof row.buildId !== 'string' || !/^[a-zA-Z0-9.+_-]{1,128}$/.test(row.buildId))) throw new Error('Invalid remote attach address')
  return { socketPath: row.socketPath, projectDir: row.projectDir,
    ...(typeof row.version === 'string' ? { version: row.version } : {}),
    ...(typeof row.buildId === 'string' ? { buildId: row.buildId } : {}),
  }
}
