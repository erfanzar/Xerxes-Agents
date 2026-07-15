// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DockerSandboxAdapter,
  DaytonaSandboxAdapter,
  ModalSandboxAdapter,
  SingularitySandboxAdapter,
  SandboxBackendAdapterConfigurationError,
  SandboxBackendAdapterRequestError,
  SshSandboxAdapter,
} from '../src/security/sandboxBackends/index.js'
import {
  ExecutionContext,
  SandboxExecutionUnavailableError,
  SandboxMode,
  SandboxRouter,
  type SandboxExecutionRequest,
} from '../src/security/sandbox.js'

test('sandbox-router parity keeps default and off-mode requests on the host and fails closed for strict requests without a backend', async () => {
  expect(new SandboxRouter().decide('anything')).toMatchObject({
    context: ExecutionContext.HOST,
    reason: 'Sandbox mode is off',
  })
  const off = new SandboxRouter({
    config: { mode: SandboxMode.OFF, sandboxedTools: ['exec_command'] },
  })
  expect(off.decide('exec_command')).toMatchObject({ context: ExecutionContext.HOST })

  const strict = new SandboxRouter({
    config: { mode: SandboxMode.STRICT, sandboxedTools: ['exec_command'] },
  })
  expect(strict.decide('exec_command')).toMatchObject({ context: ExecutionContext.SANDBOX })
  await expect(strict.executeInSandbox(request({ cmd: 'bun' }))).rejects.toBeInstanceOf(SandboxExecutionUnavailableError)
})

test('remote sandbox parity requires explicit injected hosts instead of SDK, binary, environment, or shell discovery', async () => {
  const adapters = [
    new DockerSandboxAdapter(),
    new DaytonaSandboxAdapter(),
    new ModalSandboxAdapter(),
    new SingularitySandboxAdapter(),
    new SshSandboxAdapter({ config: { host: 'build.example.test' } }),
  ]

  for (const adapter of adapters) {
    expect(await adapter.isAvailable()).toBeFalse()
    await expect(adapter.execute(request({ cmd: 'bun', args: ['--version'] })))
      .rejects.toBeInstanceOf(SandboxBackendAdapterConfigurationError)
  }
})

test('sandbox-adapter parity rejects shell-shaped commands before an injected host can receive them', async () => {
  let calls = 0
  const adapter = new DockerSandboxAdapter({
    host: {
      async runContainer() {
        calls += 1
        return { exitCode: 0, stderr: '', stdout: 'unexpected', timedOut: false, truncated: false }
      },
    },
  })

  await expect(adapter.execute(request({ cmd: 'bun; unsafe' }))).rejects.toBeInstanceOf(SandboxBackendAdapterRequestError)
  expect(calls).toBe(0)
})

function request(arguments_: SandboxExecutionRequest['arguments']): SandboxExecutionRequest {
  return {
    arguments: arguments_,
    context: { metadata: {} },
    toolName: 'exec_command',
  }
}
