// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { writeSshConnectionConfig } from '../src/security/sshConnectionConfig.js'

test('real SSH preserves aliases and identity references while clearing all configured environment forwarding', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-ssh-env-'))
  try {
    const user = join(directory, 'user config'), system = join(directory, 'system config')
    await Bun.write(user, 'Host audit\n HostName example.invalid\n User audit-user\n IdentityFile /local/key-reference\n SendEnv PRIVATE_* LANG\n SetEnv PRIVATE_TOKEN=sentinel-credential\nHost unrelated\n User other\n')
    await Bun.write(system, 'SendEnv LC_*\nSetEnv SYSTEM_TOKEN=another-sentinel\nServerAliveInterval 42\nControlMaster auto\nControlPath /tmp/user-master\nPermitLocalCommand yes\nLocalCommand echo should-not-run\nLocalForward 12345 localhost:23456\nRemoteForward 23456 localhost:34567\nDynamicForward 34567\n')
    const config = await writeSshConnectionConfig(directory, [user, system])
    // Test OpenSSH's config semantics, not native allocator instrumentation
    // inherited from a developer's shell. On Ubuntu, preloading libtcmalloc
    // corrupts OpenSSH 8.9's Include parsing even for immutable shell-written
    // files. Change only this child's environment, never the running process.
    const env = { ...process.env }
    delete env.LD_PRELOAD
    let output = ''
    const before = await Bun.file(config).text()
    for (let attempt = 0; attempt < 20; attempt++) {
      const child = Bun.spawn(['ssh', '-G', '-F', config, 'audit'], { env, stdout: 'pipe', stderr: 'pipe' })
      const [text, stderr, exitCode] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
      expect({ exitCode, stderr }).toMatchObject({ exitCode: 0 })
      expect(text).toMatch(/^hostname example.invalid$/m)
      expect(text).toMatch(/^user audit-user$/m)
      output = text
    }
    expect(await Bun.file(config).text()).toBe(before)
    expect(output).toMatch(/^hostname example.invalid$/m)
    expect(output).toMatch(/^user audit-user$/m)
    expect(output).toMatch(/^identityfile \/local\/key-reference$/m)
    expect(output).toMatch(/^serveraliveinterval 42$/m)
    expect(output).toMatch(/^controlmaster false$/m)
    expect(output).not.toMatch(/^controlpath /m)
    expect(output).toMatch(/^permitlocalcommand no$/m)
    expect(output).toMatch(/^clearallforwardings yes$/m)
    expect(output).not.toMatch(/^(localforward|remoteforward|dynamicforward) /m)
    expect(output).not.toMatch(/^sendenv /m)
    expect(output.match(/^setenv .*$/gm)).toEqual(['setenv XERXES_SSH=1'])
    expect(output).not.toContain('sentinel')
    const contents = await Bun.file(config).text()
    expect(contents).not.toContain('sentinel')
    expect(contents).not.toContain('/local/key-reference')
    expect((await stat(config)).mode & 0o777).toBe(0o600)
    await expect(writeSshConnectionConfig(directory, ['bad\nInclude other'])).rejects.toThrow('unsupported characters')
  } finally { await rm(directory, { recursive: true, force: true }) }
})
