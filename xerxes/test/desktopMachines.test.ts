// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { desktopMachineCommand } from '../src/desktop/main/machines.js'

test('desktop filesystem adapters preserve registry locking, permissions, discovery and validation', async () => {
  const home = await mkdtemp(join(tmpdir(), 'desktop-machines-'))
  const file = join(home, 'machines.json')
  try {
    await mkdir(join(home, '.ssh', 'hosts'), { recursive: true })
    await writeFile(join(home, '.ssh/config'), 'Include hosts/*.conf\nHost local\nMatch exec "never-run"\n')
    await writeFile(join(home, '.ssh/hosts/compute.conf'), 'Host compute\nHost * !excluded\n')
    expect(await desktopMachineCommand(file, 'hosts', home)).toEqual({ ok: true, hosts: ['compute', 'local'] })
    const saved = await Promise.all(['a', 'b'].map(alias => desktopMachineCommand(file, `add ${alias} host /project`, home)))
    expect(saved.every(result => result.ok)).toBe(true)
    expect((await desktopMachineCommand(file, 'list', home)).machines).toHaveLength(2)
    expect((await stat(file)).mode & 0o777).toBe(0o600)
    expect(await desktopMachineCommand(file, 'add bad -oProxyCommand=evil /project', home)).toMatchObject({ ok: false })
    expect(await desktopMachineCommand(file, 'remove a', home)).toMatchObject({ ok: true, machines: [{ alias: 'b' }] })
    await writeFile(file, '{broken')
    expect(await desktopMachineCommand(file, 'list', home)).toMatchObject({ ok: false })
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})
