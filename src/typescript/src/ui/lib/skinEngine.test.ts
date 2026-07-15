// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  activeForeground,
  DEFAULT_ROLE_COLORS,
  hexToAnsiBackground,
  hexToAnsiForeground,
  hexToRgb,
  parseSkinDocument,
  setActiveSkin,
  Skin,
  SkinEngine,
  SkinNotFoundError,
  SkinValidationError,
  skinDirectory
} from './skinEngine.js'

describe('skin colour conversion', () => {
  it('parses six-digit colours and emits true-colour ANSI escapes', () => {
    expect(hexToRgb('#ff0080')).toEqual([255, 0, 128])
    expect(hexToRgb('00ff00')).toEqual([0, 255, 0])
    expect(hexToAnsiForeground('#ffffff')).toBe('\u001b[38;2;255;255;255m')
    expect(hexToAnsiBackground('#102030')).toBe('\u001b[48;2;16;32;48m')
  })

  it('rejects malformed colour input before it can reach the renderer', () => {
    expect(() => hexToRgb('not a color')).toThrow(SkinValidationError)
    expect(() => new Skin({ name: 'bad-colour', roles: { primary: 'blue' } })).toThrow(SkinValidationError)
  })
})

describe('Skin', () => {
  it('falls back to default roles and labels and splits spinner verbs', () => {
    const skin = new Skin({
      name: 'custom',
      roles: { primary: '#123456' },
      branding: { agent_name: 'The Court', spinner_verbs: ' think , plan ,, act ' }
    })

    expect(skin.color('primary')).toBe('#123456')
    expect(skin.color('error')).toBe(DEFAULT_ROLE_COLORS.error)
    expect(skin.color('__proto__')).toBe('#ffffff')
    expect(skin.label('agent_name')).toBe('The Court')
    expect(skin.label('welcome')).toBe('The court awaits your word.')
    expect(skin.label('toString')).toBe('')
    expect(skin.spinnerVerbs()).toEqual(['think', 'plan', 'act'])
    expect(skin.foreground('primary')).toBe('\u001b[38;2;18;52;86m')
  })

  it('rejects unsafe file names and line-breaking persisted values', () => {
    expect(() => new Skin({ name: '../outside' })).toThrow(SkinValidationError)
    expect(() => new Skin({ name: 'bad value' })).toThrow(SkinValidationError)
    expect(() => new Skin({ name: 'unsafe', branding: { welcome: 'first\nsecond' } })).toThrow(SkinValidationError)
  })
})

describe('SkinEngine', () => {
  let directory: string

  beforeEach(() => {
    directory = mkdtempSync(join(tmpdir(), 'xerxes-skins-'))
  })

  afterEach(() => {
    rmSync(directory, { force: true, recursive: true })
  })

  it('resolves the user skin directory below XERXES_HOME without creating it', () => {
    expect(skinDirectory({ XERXES_HOME: '/tmp/xerxes-home' }, '/unused')).toBe('/tmp/xerxes-home/skins')
    expect(skinDirectory({}, '/home/xerxes')).toBe('/home/xerxes/.xerxes/skins')
  })

  it('ships every named built-in palette and branding override', () => {
    const engine = new SkinEngine({ baseDirectory: directory })

    expect(engine.available()).toEqual([
      'ares',
      'classic',
      'daylight',
      'default',
      'dim',
      'high-contrast',
      'mono',
      'slate'
    ])
    expect(engine.load('classic').label('agent_name')).toBe('Xerxes')
    expect(engine.load('ares').label('response_label')).toBe('ares')
    expect(engine.load('ares').spinnerVerbs()).toContain('striking')
    expect(engine.load('mono').color('diff_del')).toBe('#888888')
  })

  it('discovers yaml and skin files, prefers yaml, and ignores unsafe disk names', () => {
    writeFileSync(join(directory, 'dawn.yaml'), 'primary: #112233\nagent_name: "Dawn Court"\n', 'utf8')
    writeFileSync(join(directory, 'dawn.skin'), 'primary: #aabbcc\n', 'utf8')
    writeFileSync(join(directory, 'night.skin'), "accent: '#334455'\n", 'utf8')
    writeFileSync(join(directory, 'unsafe name.yaml'), 'primary: #334455\n', 'utf8')
    mkdirSync(join(directory, 'folder.yaml'))

    const engine = new SkinEngine({ baseDirectory: directory })
    expect(engine.available()).toContain('dawn')
    expect(engine.available()).toContain('night')
    expect(engine.available()).not.toContain('unsafe name')
    expect(engine.load('dawn').color('primary')).toBe('#112233')
    expect(engine.load('dawn').label('agent_name')).toBe('Dawn Court')
    expect(engine.load('night').color('accent')).toBe('#334455')
  })

  it('persists a custom skin and restores its roles and branding', () => {
    const engine = new SkinEngine({ baseDirectory: directory })
    const path = engine.save(
      new Skin({
        name: 'custom',
        roles: { accent: '#abcdef', primary: '#123456' },
        branding: { agent_name: 'Custom Xerxes', prompt_symbol: '$' }
      })
    )

    expect(path).toBe(join(directory, 'custom.yaml'))
    expect(readFileSync(path, 'utf8')).toBe(
      'accent: #abcdef\nprimary: #123456\nagent_name: Custom Xerxes\nprompt_symbol: $\n'
    )
    const loaded = engine.load('custom')
    expect(loaded.color('primary')).toBe('#123456')
    expect(loaded.color('accent')).toBe('#abcdef')
    expect(loaded.label('agent_name')).toBe('Custom Xerxes')
    expect(loaded.label('prompt_symbol')).toBe('$')
    expect(loaded.label('welcome')).toBe('The court awaits your word.')
  })

  it('parses the line-oriented skin format without a YAML dependency', () => {
    const parsed = parseSkinDocument(
      'line-format',
      [
        '# whole-line comments are ignored',
        'primary: "#aabbcc"',
        "agent_name: 'Xerxes: the Great'",
        'spinner_verbs: write, read',
        'this line is ignored',
        'empty:',
        ''
      ].join('\n')
    )

    expect(parsed.color('primary')).toBe('#aabbcc')
    expect(parsed.label('agent_name')).toBe('Xerxes: the Great')
    expect(parsed.spinnerVerbs()).toEqual(['write', 'read'])
  })

  it('selects a skin per engine and reports missing names clearly', () => {
    const engine = new SkinEngine({ baseDirectory: directory })
    expect(engine.setActive('high-contrast').name).toBe('high-contrast')
    expect(engine.active().color('primary')).toBe('#ffffff')
    expect(() => engine.load('ghost')).toThrow(SkinNotFoundError)
  })
})

describe('process-wide active skin', () => {
  afterEach(() => {
    setActiveSkin(new Skin({ name: 'default' }))
  })

  it('renders foreground escapes from the selected snapshot', () => {
    setActiveSkin(new Skin({ name: 'active', roles: { primary: '#010203' } }))
    expect(activeForeground('primary')).toBe('\u001b[38;2;1;2;3m')
  })
})
