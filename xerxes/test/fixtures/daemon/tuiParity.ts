// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Interactive acceptance host: real daemon, isolated storage, seeded evidence, no provider calls.
import { mkdtemp, mkdir, realpath } from 'node:fs/promises'
import { DaemonServer } from '../../../src/daemon/server.js'
import { InMemoryDaemonRuntime, type TurnRunner } from '../../../src/daemon/runtime.js'
import { RunHistory } from '../../../src/runtime/runHistory.js'
import { JobStore } from '../../../src/cron/jobs.js'
import { DeclarativeToolForge } from '../../../src/extensions/declarativeForge.js'
import { AgentPresetRoster } from '../../../src/agents/presets.js'
import { TerminalRegistry } from '../../../src/runtime/terminalRegistry.js'

const home = await mkdtemp('/tmp/xerxes-tui-parity-')
process.env.XERXES_HOME = home
await mkdir(home + '/workspace')
const workspace = await realpath(home + '/workspace')
await Bun.write(workspace + '/evidence.txt', 'First line\nSecond line\n')
const forge = new DeclarativeToolForge(home + '/forge.json')
forge.define({ name: 'greeting', version: '1.0.0', description: 'Generate a greeting for general agent work', template: 'Hello {{name}}\nReady for your task.', parameters: [{ name: 'name', description: 'Recipient', required: true, default: 'Ada' }] })
const roster = new AgentPresetRoster({ home, projectDirectory: workspace, userDirectory: home + '/agents', settingsPath: home + '/agent-presets.json' })
roster.copy('default', 'verification-agent', 'Verification agent')
const runner: TurnRunner = { async *run(_session, text, signal) {
  yield { type: 'tool_call', payload: { id: 'native-output', name: 'Exec', arguments: 'verification producer' } }
  yield { type: 'tool_result', payload: { tool_call_id: 'native-output', name: 'Exec', return_value: Array.from({ length: 180 }, (_, i) => `Retained tool line ${i + 1}: readable multiline output.`).join('\n'), permitted: true } }
  for (let i = 1; i <= 8; i++) yield { type: 'subagent_event', payload: { event: { type: 'turn_begin', payload: {} }, agent_id: `native-agent-${i}`, goal: `Verification task ${i}`, model: 'fixture', title: `Agent ${i}` } }
  for (let i = 1; i <= (text.includes('stream') ? 30 : 2); i++) {
    if (signal.aborted) return
    yield { type: 'text_part', payload: { text: `Verification stream ${i}. ` } }
    await Bun.sleep(300)
  }
  for (let i = 1; i <= 8; i++) yield { type: 'subagent_event', payload: { event: { type: 'turn_end', payload: {} }, agent_id: `native-agent-${i}`, result: `Verified fixture task ${i}`, summary: `Fixture task ${i} complete` } }
} }
const runtime = new InMemoryDaemonRuntime(runner, { model: 'verification-fixture', currentProjectDirectory: workspace, sessionDirectory: home + '/sessions' })
const session = await runtime.openSession('evidence-owner', 'default', { cwd: workspace })
await runtime.openSession('second-owner', 'default', { cwd: workspace })
session.messages.push({ role: 'user', content: 'Record the verification edit.' }, { role: 'tool', name: 'FileEditTool', tool_call_id: 'recorded-edit', content: 'ok' }, { role: 'assistant', content: 'Recorded the isolated verification edit.' })
await Bun.write(workspace + '/recorded.txt', 'After edit\n')
session.toolExecutions.push({ name: 'FileEditTool', inputs: { file_path: workspace + '/recorded.txt', old_string: 'Before edit', new_string: 'After edit' }, permitted: true, tool_call_id: 'recorded-edit', duration_ms: 1, display_blocks: [] })
const history = new RunHistory(home + '/runs.sqlite')
const terminals = new TerminalRegistry({ runHistory: history })
const terminal = terminals.open({ id: 'verification-output', ownerSessionId: session.id, kind: 'background', command: 'verification output producer', cwd: workspace })
terminal.append(Array.from({ length: 500 }, (_, index) => `Output line ${index + 1}: retained diagnostic evidence for terminal paging.\n`).join(''))
terminal.close(0)
const run = history.start({ ownerSessionId: session.id, workspace, kind: 'monitor', sourceId: 'verification', title: 'Compiler diagnostics · persisted evidence' })
for (let sequence = 1; sequence <= 45; sequence++) {
  const text = `Compiler evidence ${sequence}\nsrc/deep/workspace/module-${sequence}/index.ts:42\nVerification fixture: diagnostic details.`
  history.appendEvent(session.id, run.id, { sequence, text, at: Date.now() + sequence }, text)
}
history.finish(session.id, run.id, 'succeeded')
await runtime.flushSessions()
const server = new DaemonServer({ socketPath: home + '/daemon.sock', projectDirectory: workspace, runtime, runHistory: history,
  declarativeForge: forge, agentPresetRoster: roster, terminalRegistry: terminals,
  cronLeasePath: home + '/cron.lease', cronStoreFactory: () => new JobStore(home + '/jobs.json') })
await server.start()
const info = { home, workspace, socket: home + '/daemon.sock', session: session.id, run: run.id, pid: process.pid }
if (process.argv[2]) await Bun.write(process.argv[2], JSON.stringify(info))
console.log(JSON.stringify(info))
let stopping = false
async function stop() {
  if (stopping) return
  stopping = true
  await server.stop()
  await runtime.shutdown()
  history.close()
  process.exit(0)
}
process.on('SIGTERM', () => { void stop() })
process.on('SIGINT', () => { void stop() })
