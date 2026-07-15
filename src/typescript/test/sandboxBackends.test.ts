// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DaytonaSandboxAdapter,
  DockerSandboxAdapter,
  ModalSandboxAdapter,
  SandboxBackendAdapterConfigurationError,
  SandboxBackendNotRegisteredError,
  SandboxBackendRegistry,
  SandboxBackendUnavailableError,
  SingularitySandboxAdapter,
  SshSandboxAdapter,
  SubprocessSandboxAdapter,
} from '../src/security/sandboxBackends/index.js'
import type {
  DockerSandboxHostRequest,
  SandboxCommandResult,
  SshSandboxHostRequest,
  SubprocessSandboxHostRequest,
} from '../src/security/sandboxBackends/index.js'
import type { SandboxExecutionRequest } from '../src/security/sandbox.js'

test('Docker adapter validates a direct argv request and delegates only to an injected host', async () => {
  const calls: DockerSandboxHostRequest[] = []
  const adapter = new DockerSandboxAdapter({
    config: {
      environment: { XERXES_VISIBLE: 'configured' },
      image: 'registry.example/xerxes:bun',
      maxOutputChars: 321,
      maxTimeoutMs: 1_000,
      memoryLimitMb: 768,
      mountPaths: { '/host/workspace': '/workspace' },
      networkAccess: false,
      timeoutMs: 500,
      workingDirectory: '/workspace',
    },
    host: {
      async runContainer(request) {
        calls.push(request)
        return result({ resourceId: 'container-7' })
      },
    },
  })

  const output = JSON.parse(await adapter.execute(executionRequest({ cmd: 'bun', args: ['test'] })))

  expect(calls).toEqual([{
    command: {
      argv: ['bun', 'test'],
      cwd: '/workspace',
      environment: { XERXES_VISIBLE: 'configured' },
      maxOutputChars: 321,
      timeoutMs: 500,
    },
    image: 'registry.example/xerxes:bun',
    memoryLimitMb: 768,
    mounts: [{ hostPath: '/host/workspace', containerPath: '/workspace', readOnly: true }],
    networkAccess: false,
    toolName: 'exec_command',
  }])
  expect(output).toEqual({
    backend: 'docker',
    command: ['bun', 'test'],
    cwd: '/workspace',
    exitCode: 0,
    resourceId: 'container-7',
    stderr: '',
    stdout: 'ok',
    timedOut: false,
    truncated: false,
  })
  expect(adapter.getCapabilities()).toMatchObject({
    commandTransport: 'direct_argv_host_port',
    hostConfigured: true,
    networkIsolation: 'host_defined',
  })
})

test('missing adapter hosts fail closed rather than discovering a Docker SDK or shell', async () => {
  const adapter = new DockerSandboxAdapter()

  expect(await adapter.isAvailable()).toBeFalse()
  await expect(adapter.execute(executionRequest({ cmd: 'bun' }))).rejects.toBeInstanceOf(
    SandboxBackendAdapterConfigurationError,
  )
})

test('Daytona and Modal adapters preserve remote resource lifecycle and cleanup failures', async () => {
  const daytonaEvents: string[] = []
  const daytona = new DaytonaSandboxAdapter({
    config: { region: 'eu-west-1', workspaceImage: 'daytona/bun:1' },
    host: {
      async createWorkspace(request) {
        daytonaEvents.push(`create:${request.image}:${request.region}`)
        return {
          id: 'workspace-9',
          async execute(command) {
            daytonaEvents.push(`execute:${command.command.argv.join(' ')}`)
            return result()
          },
          async delete() {
            daytonaEvents.push('delete')
          },
        }
      },
    },
  })
  const daytonaOutput = JSON.parse(await daytona.execute(executionRequest({ cmd: 'bun', args: ['--version'] })))

  expect(daytonaEvents).toEqual(['create:daytona/bun:1:eu-west-1', 'execute:bun --version', 'delete'])
  expect(daytonaOutput.resourceId).toBe('workspace-9')

  const modalEvents: string[] = []
  const modal = new ModalSandboxAdapter({
    config: { cpu: 2, image: 'modal/bun:1', memoryLimitMb: 2_048 },
    host: {
      async createSandbox(request) {
        modalEvents.push(`create:${request.image}:${request.cpu}:${request.memoryLimitMb}`)
        return {
          id: 'modal-6',
          async wait() {
            modalEvents.push('wait')
            return result()
          },
          async close() {
            modalEvents.push('close')
          },
        }
      },
    },
  })
  const modalOutput = JSON.parse(await modal.execute(executionRequest({ cmd: 'bun', args: ['run', 'check'] })))

  expect(modalEvents).toEqual(['create:modal/bun:1:2:2048', 'wait', 'close'])
  expect(modalOutput.resourceId).toBe('modal-6')

  const cleanupFailure = new Error('remote close denied')
  const failedCleanup = new ModalSandboxAdapter({
    host: {
      async createSandbox() {
        return {
          id: 'modal-broken',
          async wait() {
            return result()
          },
          async close() {
            throw cleanupFailure
          },
        }
      },
    },
  })
  await expect(failedCleanup.execute(executionRequest({ cmd: 'bun' }))).rejects.toBe(cleanupFailure)
})

test('Singularity requires a host-reported runtime and SSH requires an explicit endpoint', async () => {
  const missingRuntime = new SingularitySandboxAdapter({
    host: {
      async resolveRuntime() {
        return undefined
      },
      async executeContainer() {
        throw new Error('must not execute when no runtime is available')
      },
    },
  })
  await expect(missingRuntime.execute(executionRequest({ cmd: 'bun' }))).rejects.toBeInstanceOf(SandboxBackendUnavailableError)

  const sshCalls: SshSandboxHostRequest[] = []
  const ssh = new SshSandboxAdapter({
    config: { host: 'build.example.test', identityFile: 'credential-ref:ssh-build', port: 2222, user: 'runner' },
    host: {
      async executeRemote(request) {
        sshCalls.push(request)
        return result()
      },
    },
  })
  await ssh.execute(executionRequest({ cmd: 'bun', args: ['run', 'test'], workdir: 'project' }))

  expect(sshCalls).toEqual([{
    command: {
      argv: ['bun', 'run', 'test'],
      cwd: 'project',
      environment: {},
      maxOutputChars: 20_000,
      timeoutMs: 60_000,
    },
    connection: {
      host: 'build.example.test',
      identityFile: 'credential-ref:ssh-build',
      port: 2222,
      user: 'runner',
    },
    toolName: 'exec_command',
  }])
  expect(JSON.stringify(await ssh.execute(executionRequest({ cmd: 'bun' })))).not.toContain('credential-ref:ssh-build')
})

test('subprocess adapter has an allow-list and never substitutes the existing Bun subprocess implementation', async () => {
  const calls: SubprocessSandboxHostRequest[] = []
  const adapter = new SubprocessSandboxAdapter({
    config: {
      allowedCommands: ['bun'],
      memoryLimitMb: 256,
      networkAccessRequested: false,
    },
    host: {
      async executeSubprocess(request) {
        calls.push(request)
        return result({ stdout: 'host confirmed' })
      },
    },
  })

  const output = JSON.parse(await adapter.execute(executionRequest({ cmd: 'bun', args: ['--version'] })))
  expect(calls).toEqual([{
    command: {
      argv: ['bun', '--version'],
      environment: {},
      maxOutputChars: 20_000,
      timeoutMs: 60_000,
    },
    memoryLimitMb: 256,
    networkAccessRequested: false,
    toolName: 'exec_command',
  }])
  expect(output.stdout).toBe('host confirmed')
  await expect(adapter.execute(executionRequest({ cmd: 'sh' }))).rejects.toThrow('not in the subprocess allow-list')
})

test('the caller-owned registry exposes only registered adapters and keeps names stable', () => {
  const docker = new DockerSandboxAdapter()
  const registry = new SandboxBackendRegistry([docker])

  expect(registry.list()).toEqual(['docker'])
  expect(registry.get('docker')).toBe(docker)
  expect(() => registry.get('modal')).toThrow(SandboxBackendNotRegisteredError)
})

function executionRequest(arguments_: SandboxExecutionRequest['arguments']): SandboxExecutionRequest {
  return {
    arguments: arguments_,
    context: { metadata: {} },
    toolName: 'exec_command',
  }
}

function result(overrides: Partial<SandboxCommandResult> = {}): SandboxCommandResult {
  return {
    exitCode: 0,
    stderr: '',
    stdout: 'ok',
    timedOut: false,
    truncated: false,
    ...overrides,
  }
}
