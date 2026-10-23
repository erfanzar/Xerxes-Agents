// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connect, type Socket } from 'node:net';
import { DaemonServer } from '../src/daemon/server.js';
import { InMemoryDaemonRuntime, type DaemonEvent } from '../src/daemon/runtime.js';
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js';
import { RemoteProviderBindings } from '../src/daemon/remoteProviderBindings.js';
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js';
import { DaemonInteractionBoard } from '../src/daemon/interactions.js';
import { ProfileStore } from '../src/bridge/profiles.js';
import { PROVIDER_FLOW_EDIT_LABEL } from '../src/daemon/providerFlow.js';
import { TransportEventJournal } from '../src/daemon/transportEventJournal.js';
import { EventSubscriptions } from '../src/daemon/eventSubscriptions.js';

const directories: string[] = [];
afterEach(async () => { for (const directory of directories.splice(0)) await rm(directory, { recursive: true, force: true }); });
async function directory(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), 'xerxes-events-'));
  directories.push(path);
  return path;
}

test('journal commits before delivery and survives a new process identity', async () => {
  const root = await directory();
  const first = new TransportEventJournal({ directory: root });
  await first.start();
  let delivered = false;
  first.append({ project_dir: root, type: 'text_part', payload: { text: 'hello' } }, () => {
    expect(first.read(first.cursor(0)).events).toHaveLength(1);
    delivered = true;
  });
  expect(delivered).toBe(false);
  await first.flush();
  expect(delivered).toBe(true);
  const original = first.read(first.cursor(0)).events[0]!;
  await first.close();
  const second = new TransportEventJournal({ directory: root });
  await second.start();
  expect(second.streamId).toBe(first.streamId);
  expect(second.runtimeId).toBe(first.runtimeId);
  expect(second.instanceId).not.toBe(first.instanceId);
  second.append({ project_dir: root, type: 'turn_end', payload: {} });
  await second.flush();
  const replay = second.read(second.cursor(0)).events;
  expect(replay.map(event => event.sequence)).toEqual([1, 2]);
  expect(replay[0]?.event_id).toBe(original.event_id);
  await second.close();
});

test('subscribe installs before replay, then emits live without a boundary duplicate', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root });
  await journal.start();
  journal.append({ project_dir: root, type: 'text_part', payload: { text: 'backfill' } });
  await journal.flush();
  const batches: Record<string, unknown>[] = [];
  const owner = {};
  const subscriptions = new EventSubscriptions(journal);
  const result = subscriptions.subscribe(owner, { project_dirs: [root], detail: 'conversation', after_cursor: journal.cursor(0) }, (type, payload) => { batches.push({ type, ...payload }); });
  journal.append({ project_dir: root, type: 'turn_end', payload: {} });
  await journal.flush();
  await Bun.sleep(25);
  expect(result.ok).toBe(true);
  const events = batches.flatMap(batch => Array.isArray(batch.events) ? batch.events : []) as { sequence: number }[];
  expect(events.map(event => event.sequence)).toEqual([1, 2]);
  subscriptions.close();
  await journal.close();
});

test('retention expires cursors, write failures are observable, private frames are excluded', async () => {
  const root = await directory();
  let fail = false;
  const journal = new TransportEventJournal({ directory: root, maxBytes: 1200, segmentBytes: 600, beforeWrite: () => { if (fail) throw new Error('disk full'); } });
  await journal.start();
  for (let index = 0; index < 5; index++) {
    journal.append({ project_dir: root, type: 'text_part', payload: { text: 'x'.repeat(200) } });
    await journal.flush();
  }
  expect(journal.read(journal.cursor(0)).reason).toBe('retention');
  const count = journal.sequence;
  journal.append({ project_dir: root, type: 'provider.remote.request', payload: { token: 'private' } });
  await journal.flush();
  expect(journal.sequence).toBe(count);
  const frames: string[] = [];
  const subscriptions = new EventSubscriptions(journal);
  subscriptions.subscribe({}, { project_dirs: [root], detail: 'conversation' }, type => { frames.push(type); });
  fail = true;
  let legacyDelivered = false;
  journal.append({ project_dir: root, type: 'turn_end', payload: {} }, () => { legacyDelivered = true; });
  await journal.flush();
  expect(legacyDelivered).toBe(true);
  expect(journal.available).toBe(false);
  expect(frames).toContain('event_gap');
  expect(journal.read().reason).toBe('journal_unavailable');
  subscriptions.close();
  await journal.close();
});

test('summary filters deltas; bounded reads substitute oversized events and redact secrets', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root });
  await journal.start();
  for (const type of ['text_part', 'think_part', 'tool_result']) journal.append({ project_dir: root, type, payload: { text: 'z'.repeat(8000), api_key: 'never-store-this' } });
  await journal.flush();
  const subscriptions = new EventSubscriptions(journal);
  const owner = {};
  const result = subscriptions.subscribe(owner, { project_dirs: [root], detail: 'summary', after_cursor: journal.cursor(0), max_batch_bytes: 1024 }, () => undefined);
  const page = subscriptions.read(owner, { subscription_id: result.subscription_id, after_cursor: journal.cursor(0), max_bytes: 1024 });
  expect(JSON.stringify(page).length).toBeLessThanOrEqual(1024);
  expect(page.events).toHaveLength(1);
  expect(page.events).toMatchObject([{ type: 'tool_result', payload: { truncated: true, type: 'tool_result' } }]);
  expect(JSON.stringify(journal.read(journal.cursor(0)))).not.toContain('never-store-this');
  expect(subscriptions.unsubscribe(owner, { subscription_id: result.subscription_id })).toEqual({ ok: true });
  expect(() => subscriptions.read(owner, { subscription_id: result.subscription_id })).toThrow();
  subscriptions.close();
  await journal.close();
});

interface Frame { id?: number; result?: JsonRpcPayload; error?: unknown; method?: string; params?: { type: string; payload: JsonRpcPayload } }
class Client {
  readonly frames: Frame[] = [];
  private buffer = '';
  private id = 0;
  constructor(readonly socket: Socket) {
    socket.setEncoding('utf8');
    socket.on('data', chunk => {
      this.buffer += chunk;
      let split: number;
      while ((split = this.buffer.indexOf('\n')) >= 0) {
        this.frames.push(JSON.parse(this.buffer.slice(0, split)) as Frame);
        this.buffer = this.buffer.slice(split + 1);
      }
    });
  }
  static async open(path: string): Promise<Client> {
    const socket = connect(path);
    const client = new Client(socket);
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject); });
    return client;
  }
  async next(accepts: (frame: Frame) => boolean): Promise<Frame> {
    for (let attempt = 0; attempt < 1000; attempt++) {
      const frame = this.frames.find(accepts);
      if (frame) return frame;
      await Bun.sleep(2);
    }
    throw new Error('Timed out waiting for daemon frame');
  }
  async call(method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> {
    const id = ++this.id;
    this.socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    const frame = await this.next(frame => frame.id === id);
    if (frame.error) throw new Error(JSON.stringify(frame.error));
    return frame.result!;
  }
}
async function serverAt(root: string, journalOptions: { beforeWrite?: () => void | Promise<void> } = {}): Promise<DaemonServer> {
  const server = new DaemonServer({
    socketPath: join(root, 'daemon.sock'), projectDirectory: root, autoTitle: false, autoSnapshotTurns: false,
    eventJournal: { directory: join(root, 'events'), ...journalOptions },
    runtime: new InMemoryDaemonRuntime({ async *run(): AsyncGenerator<DaemonEvent> { yield { type: 'text_part', payload: { text: 'one-emission' } }; } },
      { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') }),
  });
  await server.start();
  return server;
}

test('daemon assigns one identity before session observer and global fan-out, with unchanged legacy frames', async () => {
  const root = await directory();
  let allowWrite: (() => void) | undefined;
  let block = false;
  const server = await serverAt(root, { beforeWrite: async () => { if (block) await new Promise<void>(resolve => { allowWrite = resolve; }); } });
  const clients = await Promise.all([Client.open(join(root, 'daemon.sock')), Client.open(join(root, 'daemon.sock')), Client.open(join(root, 'daemon.sock'))]);
  const [first, second, stream] = clients as [Client, Client, Client];
  try {
    const init = { session_key: 'shared', project_dir: root, session_owned_turns: true, history_limit: 0 };
    await first.call('initialize', init);
    await second.call('initialize', init);
    const status = await stream.call('runtime.status');
    expect(status).toMatchObject({ event_stream_supported: true, runtime_id: expect.any(String), instance_id: expect.any(String) });
    const sub = await stream.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    await first.call('turn.submit', { text: 'test' });
    await first.next(frame => frame.params?.type === 'text_part');
    await second.next(frame => frame.params?.type === 'text_part');
    await stream.next(frame => frame.params?.type === 'event_batch' && JSON.stringify(frame.params.payload).includes('one-emission'));
    const read = await stream.call('events.read', { subscription_id: sub.subscription_id, after_cursor: sub.watermark });
    const emitted = (read.events as { type: string; event_id: string }[]).filter(event => event.type === 'text_part');
    expect(emitted).toHaveLength(1);
    expect(first.frames.find(frame => frame.params?.type === 'text_part')).toEqual(second.frames.find(frame => frame.params?.type === 'text_part'));
    expect(first.frames.some(frame => frame.params?.type === 'event_batch')).toBe(false);
    block = true;
    server.broadcast('notification', { message: 'commit-probe' });
    for (let attempt = 0; !allowWrite && attempt < 100; attempt++) await Bun.sleep(2);
    expect(allowWrite).toBeDefined();
    expect(first.frames.some(frame => frame.params?.payload.message === 'commit-probe')).toBe(false);
    block = false;
    allowWrite!();
    await first.next(frame => frame.params?.payload.message === 'commit-probe');
    await second.next(frame => frame.params?.payload.message === 'commit-probe');
    const global = await stream.call('events.read', { subscription_id: sub.subscription_id, after_cursor: sub.watermark });
    expect((global.events as { payload: JsonRpcPayload }[]).filter(event => event.payload.message === 'commit-probe')).toHaveLength(1);
    await stream.call('events.unsubscribe', { subscription_id: sub.subscription_id });
  } finally { for (const client of clients) client.socket.destroy(); await server.stop(); }
});

test('new DaemonServer on the same state root retains stream and sequence, changes instance', async () => {
  const root = await directory();
  const first = await serverAt(root);
  let client = await Client.open(join(root, 'daemon.sock'));
  const before = await client.call('runtime.status');
  const sub = await client.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
  first.broadcast('notification', { message: 'restart-event' });
  await client.next(frame => frame.params?.type === 'notification');
  client.socket.destroy();
  await first.stop();
  const second = await serverAt(root);
  client = await Client.open(join(root, 'daemon.sock'));
  try {
    const after = await client.call('runtime.status');
    expect(after.runtime_id).toBe(before.runtime_id);
    expect(after.instance_id).not.toBe(before.instance_id);
    const resumed = await client.call('events.subscribe', { project_dirs: [root], detail: 'conversation', after_cursor: sub.watermark });
    expect(resumed.stream_id).toBe(sub.stream_id);
    expect(resumed.reset_required).toBe(false);
    const page = await client.call('events.read', { subscription_id: resumed.subscription_id, after_cursor: sub.watermark });
    expect(page.events).toMatchObject([{ sequence: 1, type: 'notification', payload: { message: 'restart-event' } }]);
    second.broadcast('notification', { message: 'next' });
    await client.next(frame => frame.params?.payload.message === 'next');
    const next = await client.call('events.read', { subscription_id: resumed.subscription_id, after_cursor: page.next_cursor });
    expect(next.events).toMatchObject([{ sequence: 2 }]);
  } finally { client.socket.destroy(); await second.stop(); }
});

test('daemon failure sends a journal_unavailable gap but still delivers legacy events', async () => {
  const root = await directory();
  let fail = false;
  const server = await serverAt(root, { beforeWrite: () => { if (fail) throw new Error('injected write failure'); } });
  const client = await Client.open(join(root, 'daemon.sock'));
  try {
    await client.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    fail = true;
    server.broadcast('notification', { message: 'legacy-still-works' });
    expect((await client.next(frame => frame.params?.type === 'event_gap')).params?.payload.reason).toBe('journal_unavailable');
    await client.next(frame => frame.params?.payload.message === 'legacy-still-works');
    expect((await client.call('runtime.status')).event_stream_supported).toBe(false);
  } finally { client.socket.destroy(); await server.stop(); }
});

test('event subscription ownership is read-only, scope is enforced and live overflow explicitly gaps', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root });
  await journal.start();
  const subscriptions = new EventSubscriptions(journal, 1024);
  const owner = {};
  const frames: { type: string; payload: JsonRpcPayload }[] = [];
  const subscription = subscriptions.subscribe(owner, { project_dirs: [root], session_ids: ['s1'], detail: 'conversation', max_batch_bytes: 1024 }, (type, payload) => { frames.push({ type, payload }); });
  expect(() => subscriptions.read({}, { subscription_id: subscription.subscription_id })).toThrow();
  for (let index = 0; index < 5; index++) journal.append({ project_dir: root, session_id: 's1', type: 'text_part', payload: { text: 'x'.repeat(900) } });
  await journal.flush();
  await Bun.sleep(10);
  // The first live pump runs on a timer, so it may deliver a prefix before the backlog
  // overflows. The contract: any delivered prefix is contiguous from sequence 1, the
  // retention gap names the cursor right after it, and nothing is delivered after the
  // gap until the subscriber reads explicitly.
  const gapIndex = frames.findIndex(frame => frame.type === 'event_gap');
  expect(gapIndex).toBeGreaterThanOrEqual(0);
  expect(frames[gapIndex]!.payload).toMatchObject({ reason: 'retention' });
  expect(frames.slice(gapIndex + 1)).toEqual([]);
  const delivered = frames.slice(0, gapIndex).flatMap(frame => {
    expect(frame.type).toBe('event_batch');
    return (frame.payload.events as { sequence: number }[]).map(event => event.sequence);
  });
  expect(delivered).toEqual(delivered.map((_, index) => index + 1));
  expect(frames[gapIndex]!.payload.requested_cursor).toBe(journal.cursor(delivered.length));
  const page = subscriptions.read(owner, { subscription_id: subscription.subscription_id, after_cursor: frames[gapIndex]!.payload.requested_cursor });
  expect(page.reset_required).toBe(false);
  expect(page.events).toHaveLength(1);
  expect((page.events as { sequence: number }[])[0]!.sequence).toBe(delivered.length + 1);
  await Bun.sleep(30);
  for (const frame of frames) expect(Buffer.byteLength(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: frame }))).toBeLessThanOrEqual(1024);
  subscriptions.close();
  await journal.close();
});

test('all daemon event serialization sinks remain behind the journal choke point or private transport', async () => {
  const server = await Bun.file(new URL('../src/daemon/server.ts', import.meta.url)).text();
  expect([...server.matchAll(/daemonEvent\(/g)]).toHaveLength(2); // subscription wrappers + committed legacy delivery
  expect([...server.matchAll(/websocketGateway\?\.broadcast\(/g)]).toHaveLength(1);
  expect(server).toContain('private publishEvent(');
  expect(server).toContain('this.eventJournal.append(');
  expect(server).toContain("this.connectionLeases.sendPrivate(connection, { jsonrpc: '2.0', method: 'provider.remote.request'");
  const sources = await Array.fromAsync(new Bun.Glob('*.ts').scan({ cwd: new URL('../src/daemon/', import.meta.url).pathname }));
  for (const path of sources.filter(path => !['server.ts', 'websocketGateway.ts'].includes(path))) {
    expect(await Bun.file(new URL(`../src/daemon/${path}`, import.meta.url)).text()).not.toMatch(/daemonEvent\(/);
  }
});

test('native appends preserve multiple batches on the same segment and restart from its tail', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root, fsync: 'batch' });
  await journal.start();
  for (let index = 0; index < 4; index++) {
    journal.append({ project_dir: root, type: 'text_part', payload: { text: String(index) } });
    journal.append({ project_dir: root, type: 'think_part', payload: { text: String(index) } });
    await journal.flush();
  }
  await journal.close();
  const restart = new TransportEventJournal({ directory: root });
  await restart.start();
  expect(restart.available).toBe(true);
  expect(restart.read(restart.cursor(0)).events.map(event => event.sequence)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  await restart.close();
});

test('time retention and foreign cursors return explicit reset gaps', async () => {
  const root = await directory();
  let now = Date.now();
  const journal = new TransportEventJournal({ directory: root, now: () => now, maxAgeMs: 100 });
  await journal.start();
  journal.append({ project_dir: root, type: 'text_part', payload: { text: 'expires' } });
  await journal.flush();
  now += 101;
  expect(journal.read(journal.cursor(0))).toMatchObject({ reset_required: true, reason: 'retention', oldest_cursor: journal.watermark });
  expect(journal.read('another-stream:0')).toMatchObject({ reset_required: true, reason: 'stream_replaced' });
  const subscriptions = new EventSubscriptions(journal);
  const frames: JsonRpcPayload[] = [];
  const owner = {};
  const result = subscriptions.subscribe(owner, { project_dirs: [root], detail: 'conversation', after_cursor: journal.cursor(0) }, (_type, payload) => { frames.push(payload); });
  expect(result.reset_required).toBe(true);
  await Bun.sleep(10);
  expect(frames).toMatchObject([{ reason: 'retention', requested_cursor: journal.cursor(0) }]);
  subscriptions.close();
  await journal.close();
});

test('multi-page replay stays ordered with live commits during backfill, filters scopes, and fits UTF-8 envelopes', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root });
  await journal.start();
  for (let index = 0; index < 8; index++) journal.append({ project_dir: root, session_id: 's1', type: 'tool_result', payload: { text: '你好'.repeat(500) } });
  journal.append({ project_dir: root, session_id: 's2', type: 'tool_result', payload: { text: 'out-of-scope' } });
  journal.append({ project_dir: join(root, 'other'), session_id: 's1', type: 'tool_result', payload: { text: 'out-of-scope' } });
  journal.append({ project_dir: root, session_id: 's1', type: 'cron_event', payload: { event_type: 'text_part', payload: { text: 'summary-hidden' } } });
  await journal.flush();
  const subscriptions = new EventSubscriptions(journal);
  const sequences: number[] = [];
  const batches: JsonRpcPayload[] = [];
  let committedLive = false;
  const owner = {};
  const sub = subscriptions.subscribe(owner, { project_dirs: [root], session_ids: ['s1'], detail: 'summary', after_cursor: journal.cursor(0), max_batch_bytes: 1024 }, (type, payload) => {
    expect(type).toBe('event_batch');
    batches.push(payload);
    sequences.push(...(payload.events as { sequence: number }[]).map(event => event.sequence));
    if (!committedLive) {
      committedLive = true;
      journal.append({ project_dir: root, session_id: 's1', type: 'turn_end', payload: {} });
    }
  });
  for (let attempt = 0; sequences.length < 9 && attempt < 100; attempt++) await Bun.sleep(3);
  expect(sequences).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 12]);
  for (const payload of batches) expect(Buffer.byteLength(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'event_batch', payload } }))).toBeLessThanOrEqual(1024);
  subscriptions.unsubscribe(owner, { subscription_id: sub.subscription_id });
  const delivered = batches.length;
  journal.append({ project_dir: root, session_id: 's1', type: 'turn_end', payload: {} });
  await journal.flush();
  await Bun.sleep(10);
  expect(batches).toHaveLength(delivered);
  subscriptions.close();
  await journal.close();
});

test('oversized metadata also produces a bounded progressing stub', async () => {
  const root = await directory();
  const journal = new TransportEventJournal({ directory: root });
  await journal.start();
  journal.append({ project_dir: root + '/long'.repeat(600), type: 'tool_result', payload: { text: 'oversized path' } });
  await journal.flush();
  const page = journal.read(journal.cursor(0), { maxBytes: 1024 });
  expect(page.events).toMatchObject([{ sequence: 1, truncated: true }]);
  expect(page.next_cursor).toBe(journal.cursor(1));
  expect(eventFrameBytes(page)).toBeLessThanOrEqual(1024);
  await journal.close();
});
function eventFrameBytes(payload: unknown): number { return Buffer.byteLength(JSON.stringify({ jsonrpc: '2.0', id: 1, result: payload })); }

test('actual private desktop provider requests never enter transport replay', async () => {
  const root = await directory();
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') });
  const bindings = new RemoteProviderBindings();
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), runtime, remoteProviderBindings: bindings, projectDirectory: root });
  await server.start();
  const client = await Client.open(join(root, 'daemon.sock'));
  try {
    await client.call('initialize', { session_key: 'private', project_dir: root, history_limit: 0 });
    expect(await client.call('provider.remote.bind', { consent: true, source: 'workstation', profile: 'local', model: 'gpt-4o' })).toMatchObject({ ok: true });
    const sub = await client.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    const session = runtime.sessionStatus('private')!;
    const iterator = bindings.client(session, 'gpt-4o')!.stream({ model: 'gpt-4o', messages: [{ role: 'user', content: 'private-context-never-journaled' }] })[Symbol.asyncIterator]();
    const pending = iterator.next();
    const request = await client.next(frame => frame.method === 'provider.remote.request');
    const params = request.params as unknown as JsonRpcPayload;
    expect(JSON.stringify(request)).toContain('private-context-never-journaled');
    await client.call('provider.remote.reply', { binding: params.binding, request_id: params.request_id, reply: { done: true } });
    await pending;
    const page = await client.call('events.read', { subscription_id: sub.subscription_id, after_cursor: sub.watermark });
    expect(page.events).toEqual([]);
    expect(page.next_cursor).toBe(sub.watermark);
    expect(JSON.stringify(page)).not.toContain('private-context-never-journaled');
  } finally { client.socket.destroy(); await server.stop(); }
});

test('one terminal observation gets one identity even with multiple attached watchers', async () => {
  const root = await directory();
  const registry = new TerminalRegistry();
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') });
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), runtime, terminalRegistry: registry, projectDirectory: root });
  await server.start();
  const first = await Client.open(join(root, 'daemon.sock'));
  const second = await Client.open(join(root, 'daemon.sock'));
  try {
    for (const client of [first, second]) await client.call('initialize', { session_key: 'watchers', project_dir: root, history_limit: 0 });
    const session = runtime.sessionStatus('watchers')!;
    const terminal = registry.open({ id: 'test-terminal', ownerSessionId: session.id, kind: 'pty', command: 'test', cwd: root });
    for (const client of [first, second]) expect(await client.call('terminal.attach', { terminal_id: terminal.id })).toMatchObject({ ok: true });
    const sub = await first.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    terminal.append('one shared terminal observation');
    await first.next(frame => frame.params?.type === 'terminal_output');
    await second.next(frame => frame.params?.type === 'terminal_output');
    const page = await first.call('events.read', { subscription_id: sub.subscription_id, after_cursor: sub.watermark });
    expect((page.events as { type: string }[]).filter(event => event.type === 'terminal_output')).toHaveLength(1);
    terminal.close(0);
  } finally { first.socket.destroy(); second.socket.destroy(); await server.stop(); }
});

test('subscribing to interaction events does not grant answer authority', async () => {
  const root = await directory();
  const board = new DaemonInteractionBoard();
  const done = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime({ async *run(session, _text, _signal): AsyncGenerator<DaemonEvent> {
    yield { type: 'approval_request', payload: { request_id: 'owned-approval', session_id: session.id, tool_name: 'test' } };
    yield { type: 'question_request', payload: { id: 'owned-question', session_id: session.id, questions: [] } };
    await done.promise;
  } }, { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') });
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), projectDirectory: root, runtime, interactions: board, autoTitle: false, autoSnapshotTurns: false });
  await server.start();
  const first = await Client.open(join(root, 'daemon.sock'));
  const reader = await Client.open(join(root, 'daemon.sock'));
  try {
    await first.call('initialize', { session_key: 'interaction', project_dir: root, session_owned_turns: true, history_limit: 0 });
    await reader.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    await first.call('turn.submit', { text: 'test' });
    await reader.next(frame => frame.params?.type === 'event_batch' && JSON.stringify(frame.params.payload).includes('owned-question'));
    expect(await reader.call('permission_response', { request_id: 'owned-approval', response: 'accept' })).toMatchObject({ ok: false, error: 'approval owned by another connection' });
    expect(await reader.call('question_response', { request_id: 'owned-question', answers: { value: 'try' } })).toMatchObject({ ok: false, error: 'question owned by another connection' });
  } finally { done.resolve(); first.socket.destroy(); reader.socket.destroy(); await server.stop(); }
});

test('provider editor answers with a generic value key stay legacy-only', async () => {
  const root = await directory();
  const profiles = new ProfileStore(join(root, 'profiles.json'));
  profiles.save({ name: 'example', provider: 'openai', model: 'protocol-model', baseUrl: 'https://example.invalid/v1', apiKey: '' });
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), profileStore: profiles, projectDirectory: root,
    runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') }) });
  await server.start();
  const client = await Client.open(join(root, 'daemon.sock'));
  try {
    await client.call('initialize', { project_dir: root, history_limit: 0 });
    const sub = await client.call('events.subscribe', { project_dirs: [root], detail: 'conversation' });
    await client.call('slash', { command: '/provider' });
    for (const answer of [PROVIDER_FLOW_EDIT_LABEL, 'example', 'api_key', 'opaque-test-fixture-not-a-real-credential']) {
      const question = client.frames.filter(frame => frame.params?.type === 'question_request').at(-1)!.params!.payload;
      const id = (question.questions as { id: string }[])[0]!.id;
      expect(await client.call('question_response', { request_id: question.id, answers: { [id]: answer } })).toMatchObject({ ok: true });
    }
    expect(JSON.stringify(client.frames.filter(frame => frame.params?.type === 'question_response'))).toContain('opaque-test-fixture-not-a-real-credential');
    const page = await client.call('events.read', { subscription_id: sub.subscription_id, after_cursor: sub.watermark });
    expect(JSON.stringify(page)).not.toContain('opaque-test-fixture-not-a-real-credential');
    expect((page.events as { type: string }[]).some(event => event.type === 'question_response')).toBe(false);
  } finally { client.socket.destroy(); await server.stop(); }
});
