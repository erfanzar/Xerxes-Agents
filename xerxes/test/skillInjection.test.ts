// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, test } from 'bun:test'

import { expandSkillInstructions, substituteArguments } from '../src/extensions/skillInjection.js'

describe('substituteArguments', () => {
  test('replaces $ARGUMENTS, ${ARGUMENTS}, and positional $N', () => {
    expect(substituteArguments('fix $ARGUMENTS now', 'src/app.ts')).toBe('fix src/app.ts now')
    expect(substituteArguments('fix ${ARGUMENTS}', 'a b')).toBe('fix a b')
    expect(substituteArguments('move $0 to $1', 'a.txt /tmp')).toBe('move a.txt to /tmp')
    expect(substituteArguments('keep $2 when missing', 'only')).toBe('keep $2 when missing')
    expect(substituteArguments('no placeholders', 'ignored')).toBe('no placeholders')
    expect(substituteArguments('empty: "$ARGUMENTS"')).toBe('empty: ""')
  })
})

describe('expandSkillInstructions', () => {
  test('automatically loaded instructions cannot execute shell preprocessing, including through arguments', async () => {
    let calls = 0
    const options = { cwd: process.cwd(), allowCommandExecution: false, run: async () => { calls++; return { code: 0, stdout: 'executed', stderr: '' } } }
    await expect(expandSkillInstructions('Read !`echo secret`', options)).rejects.toThrow('/skills trust')
    await expect(expandSkillInstructions('$ARGUMENTS', { ...options, args: '!`echo secret`' })).rejects.toThrow('/skills trust')
    expect(await expandSkillInstructions('Read $ARGUMENTS', { ...options, args: 'src' })).toBe('Read src')
    expect(calls).toBe(0)
    expect(await expandSkillInstructions('Read !`echo secret`', { ...options, allowCommandExecution: true })).toBe('Read executed')
    expect(calls).toBe(1)
  })
  test('executes !`cmd` injections and splices stdout', async () => {
    const expanded = await expandSkillInstructions('Today is !`echo 2026-07-13`.', { cwd: process.cwd() })
    expect(expanded).toBe('Today is 2026-07-13.')
  })

  test('failed commands become visible failure markers, never silent', async () => {
    const expanded = await expandSkillInstructions('Run !`exit 7` please.', { cwd: process.cwd() })
    expect(expanded).toContain('[injected command failed:')
  })

  test('caps injections per skill', async () => {
    const body = Array.from({ length: 7 }, (_, index) => `!\`echo ${index}\``).join(' ')
    const expanded = await expandSkillInstructions(body, { cwd: process.cwd() })
    expect(expanded).toContain('0 1 2 3 4')
    expect(expanded).toContain('skipped: more than 5 command injections')
  })

  test('injected executor failures become markers instead of throwing', async () => {
    const expanded = await expandSkillInstructions('x !`boom` y', {
      cwd: process.cwd(),
      run: () => Promise.reject(new Error('spawn died')),
    })
    expect(expanded).toBe('x [injected command failed: spawn died] y')
  })

  test('runs argument substitution before command injection', async () => {
    const expanded = await expandSkillInstructions('value: !`echo $0`', {
      args: 'injected-arg',
      cwd: process.cwd(),
    })
    expect(expanded).toBe('value: injected-arg')
  })

  test.skipIf(process.platform === 'win32')('the timeout kills the whole command tree, not just the shell', async () => {
    const started = Date.now()
    const expanded = await expandSkillInstructions('A !`cd . && sleep 5; echo done` B', { cwd: process.cwd(), timeoutMs: 100 })
    expect(Date.now() - started).toBeLessThan(2000)
    expect(expanded).toContain('[injected command failed:')
    expect(expanded).toContain('timed out')
    expect(expanded).not.toContain('done')
  })
})
