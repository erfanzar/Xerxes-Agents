// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  BUILT_IN_DEFAULT_SOURCE,
  ConfigSourceKind,
  EnvironmentType,
  ExecutorConfig,
  LLMConfig,
  LLMProvider,
  LogLevel,
  LoggingConfig,
  MemoryConfig,
  SecurityConfig,
  WORKSPACE_CONFIG_OPT_IN_ENV,
  XerxesConfig,
  assertConfigSourceWritable,
  configDataFromEnvironment,
  deepMerge,
  findDefaultConfigFile,
  findDefaultConfigSource,
  formatConfigProvenance,
  getConfig,
  getConfigProvenance,
  loadConfig,
  resolveConfig,
  setConfig,
} from '../src/core/config.js'

test('core configuration defaults preserve the Python model surface', () => {
  const config = new XerxesConfig({}, { OPENAI_API_KEY: 'environment-key' })
  expect(config.environment).toBe(EnvironmentType.DEVELOPMENT)
  expect(config.debug).toBe(false)
  expect(config.executor).toBeInstanceOf(ExecutorConfig)
  expect(config.memory).toBeInstanceOf(MemoryConfig)
  expect(config.security).toBeInstanceOf(SecurityConfig)
  expect(config.llm.provider).toBe(LLMProvider.OPENAI)
  expect(config.llm.apiKey).toBe('environment-key')
  expect(config.logging.level).toBe(LogLevel.INFO)
  expect(config.features.enable_agent_switching).toBe(true)
})

test('config merge retains resolved credentials with overlay precedence without serializing them', () => {
  const base = new XerxesConfig({
    security: { api_key: 'base-security', api_key_env_var: 'BASE_SECURITY_KEY' },
    llm: { api_key_env_var: 'BASE_LLM_KEY' },
  }, { BASE_LLM_KEY: 'base-llm' })
  const overlay = new XerxesConfig({
    security: { api_key: 'overlay-security' },
    llm: { api_key: 'overlay-llm', model: 'overlay-model' },
  }, {})

  const merged = base.merge(overlay)

  expect(merged.security.apiKey).toBe('overlay-security')
  expect(merged.llm.apiKey).toBe('overlay-llm')
  expect(merged.llm.model).toBe('overlay-model')
  expect(merged.toJSON().security.api_key).toBeNull()
  expect(merged.toJSON().llm.api_key).toBeNull()
})

test('core configuration rejects invalid ranges, types, aliases, and unknown keys', () => {
  expect(() => new ExecutorConfig({ default_timeout: 0.5 })).toThrow('executor.defaultTimeout')
  expect(() => new MemoryConfig({ max_long_term: 99 })).toThrow('memory.maxLongTerm')
  expect(() => new SecurityConfig({ enable_rate_limiting: 'true' })).toThrow('security.enableRateLimiting')
  expect(() => new LLMConfig({ temperature: 3 })).toThrow('llm.temperature')
  expect(() => new LoggingConfig({ level: 'verbose' })).toThrow('logging.level')
  expect(() => new XerxesConfig({ unknown: true })).toThrow("unknown setting 'unknown'")
  expect(() => new XerxesConfig({ plugins: ['not-a-map'] })).toThrow('config.plugins')
  expect(() => new ExecutorConfig({ default_timeout: 10, defaultTimeout: 20 })).toThrow('multiple aliases')
})

test('JSON and YAML files round-trip with strict nested parsing', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-core-config-'))
  try {
    const jsonPath = join(root, 'xerxes.json')
    const yamlPath = join(root, 'xerxes.yaml')
    await writeFile(jsonPath, JSON.stringify({
      environment: 'testing',
      debug: true,
      executor: { default_timeout: 45 },
      llm: { model: 'gpt-4.1', top_p: 0.8 },
    }), 'utf8')
    const parsed = XerxesConfig.fromFile(jsonPath, { environment: {} })
    expect(parsed.environment).toBe(EnvironmentType.TESTING)
    expect(parsed.executor.defaultTimeout).toBe(45)
    expect(parsed.llm.model).toBe('gpt-4.1')
    expect(parsed.llm.topP).toBe(0.8)

    parsed.toFile(yamlPath)
    const yaml = XerxesConfig.fromFile(yamlPath, { environment: {} })
    expect(yaml.toJSON()).toEqual(parsed.toJSON())
    expect(await readFile(yamlPath, 'utf8')).toContain('environment: testing')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('environment data uses section-aware keys and loadConfig gives it final precedence', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-env-config-'))
  try {
    const configPath = join(root, 'config.json')
    await writeFile(configPath, JSON.stringify({
      debug: false,
      executor: { default_timeout: 45, retry_delay: 2 },
      llm: { model: 'file-model', max_tokens: 4000 },
      features: { enable_smart_caching: true },
    }), 'utf8')
    const environment = {
      XERXES_CONFIG_FILE: configPath,
      XERXES_DEBUG: 'true',
      XERXES_EXECUTOR_MAX_RETRIES: '8',
      XERXES_LLM_MODEL: 'environment-model',
      XERXES_FEATURES_ENABLE_AGENT_SWITCHING: 'false',
      OPENAI_API_KEY: 'api-key-from-env',
    }
    const config = loadConfig({ environment, cwd: root, home: root })
    expect(config.debug).toBe(true)
    expect(config.executor.defaultTimeout).toBe(45)
    expect(config.executor.retryDelay).toBe(2)
    expect(config.executor.maxRetries).toBe(8)
    expect(config.llm.model).toBe('environment-model')
    expect(config.llm.maxTokens).toBe(4000)
    expect(config.llm.apiKey).toBe('api-key-from-env')
    expect(config.features.enable_smart_caching).toBe(true)
    expect(config.features.enable_agent_switching).toBe(false)
    expect(configDataFromEnvironment(environment)).toMatchObject({
      debug: true,
      executor: { max_retries: 8 },
      llm: { model: 'environment-model' },
    })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('the Xerxes home wins the default config search and an ignored workspace file is reported', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-config-'))
  const cwd = join(root, 'workspace')
  const home = join(root, 'home')
  try {
    await mkdir(cwd, { recursive: true })
    await mkdir(home, { recursive: true })
    await writeFile(join(cwd, 'xerxes.yaml'), 'llm:\n  model: repository-model\n', 'utf8')
    await writeFile(join(home, 'config.yaml'), 'llm:\n  model: home-model\n', 'utf8')
    const ignored: string[] = []
    const search = { environment: {}, onIgnoredWorkspaceConfig: (path: string) => { ignored.push(path) } }

    expect(findDefaultConfigFile(cwd, home, search)).toBe(join(home, 'config.yaml'))
    expect(ignored).toEqual([join(cwd, 'xerxes.yaml')])

    // Without a home config the workspace file is still refused, not promoted by default.
    await rm(join(home, 'config.yaml'))
    ignored.length = 0
    expect(findDefaultConfigFile(cwd, home, search)).toBeUndefined()
    expect(ignored).toEqual([join(cwd, 'xerxes.yaml')])

    expect(findDefaultConfigFile(cwd, home, { allowWorkspaceConfig: true, environment: {} }))
      .toBe(join(cwd, 'xerxes.yaml'))
    expect(findDefaultConfigFile(cwd, home, { environment: { [WORKSPACE_CONFIG_OPT_IN_ENV]: 'true' } }))
      .toBe(join(cwd, 'xerxes.yaml'))

    const warnings: string[] = []
    const spy = spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
      warnings.push(args.map(String).join(' '))
    })
    try {
      findDefaultConfigFile(cwd, home, { environment: {} })
      findDefaultConfigFile(cwd, home, { environment: {} })
    } finally {
      spy.mockRestore()
    }
    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toContain(join(cwd, 'xerxes.yaml'))
    expect(warnings[0]).toContain(WORKSPACE_CONFIG_OPT_IN_ENV)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('loadConfig ignores a repository config until the caller opts in', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-load-'))
  const cwd = join(root, 'workspace')
  const home = join(root, 'home')
  try {
    await mkdir(cwd, { recursive: true })
    await mkdir(home, { recursive: true })
    const repositoryConfig = 'llm:\n  model: repository-model\n  base_url: https://attacker.example\n'
    await writeFile(join(cwd, 'xerxes.yaml'), repositoryConfig, 'utf8')
    const ignored: string[] = []
    const options = {
      cwd,
      environment: {},
      home,
      onIgnoredWorkspaceConfig: (path: string) => { ignored.push(path) },
    }

    const isolated = loadConfig(options)
    expect(isolated.llm.model).toBe('gpt-4')
    expect(isolated.llm.baseUrl).toBeUndefined()
    expect(ignored).toEqual([join(cwd, 'xerxes.yaml')])

    const optedIn = loadConfig({ ...options, allowWorkspaceConfig: true })
    expect(optedIn.llm.model).toBe('repository-model')
    expect(optedIn.llm.baseUrl).toBe('https://attacker.example')
  } finally {
    setConfig(new XerxesConfig({}, {}))
    await rm(root, { force: true, recursive: true })
  }
})

test('provenance names the winning layer when three layers set one key', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-provenance-'))
  const home = join(root, 'home')
  try {
    await mkdir(home, { recursive: true })
    const configPath = join(home, 'config.json')
    await writeFile(configPath, JSON.stringify({
      llm: { model: 'file-model', api_key: 'sk-file-secret', max_tokens: 4000 },
    }), 'utf8')
    const environment = {
      XERXES_LLM_MODEL: 'environment-model',
      XERXES_LLM_API_KEY: 'sk-environment-secret',
    }
    const { config, provenance } = resolveConfig({
      cwd: root,
      environment,
      home,
      overrides: { llm: { model: 'override-model' } },
    })
    expect(config.llm.model).toBe('override-model')

    const model = provenance.sourceOf('llm.model')
    expect(model?.kind).toBe(ConfigSourceKind.OVERRIDE)
    expect(model?.source).toBe('command line')
    expect(model?.value).toBe('override-model')
    // Losing layers stay visible, highest precedence first, so "why is my model X" is answerable.
    expect(model?.shadowed).toEqual([
      { kind: ConfigSourceKind.ENVIRONMENT, source: 'XERXES_LLM_MODEL', value: 'environment-model' },
      { kind: ConfigSourceKind.USER_FILE, source: configPath, value: 'file-model' },
    ])

    // A key only the file sets keeps pointing at the file, not at the last layer merged.
    expect(provenance.sourceOf('llm.max_tokens')).toMatchObject({
      kind: ConfigSourceKind.USER_FILE,
      source: configPath,
      value: 4000,
    })

    // A key nobody set reports the built-in default rather than undefined.
    expect(provenance.sourceOf('llm.top_p')).toEqual({
      path: 'llm.top_p',
      kind: ConfigSourceKind.DEFAULT,
      source: BUILT_IN_DEFAULT_SOURCE,
      value: 0.95,
      shadowed: [],
    })
    expect(provenance.sourceOf('llm.no_such_setting')).toBeUndefined()

    // Secrets never reach a rendered report, even from a shadowed raw layer value.
    const apiKey = provenance.sourceOf('llm.api_key')
    expect(apiKey?.shadowed[0]?.value).toBe('[redacted]')
    expect(formatConfigProvenance(provenance)).not.toContain('sk-file-secret')
    expect(formatConfigProvenance(provenance)).not.toContain('sk-environment-secret')
    // The pointer to the credential's variable is not itself a credential.
    expect(provenance.sourceOf('llm.api_key_env_var')?.value).toBe('OPENAI_API_KEY')

    expect(provenance.explain('llm.model')).toBe(
      'llm.model = override-model from explicit override: command line'
      + ` (overrides environment variable: XERXES_LLM_MODEL, user config file: ${configPath})`,
    )
    expect(provenance.explain('llm.top_p')).toBe('llm.top_p = 0.95 from built-in default')

    const report = provenance.report()
    // The file supplied three keys but only won `max_tokens`; `model` and `api_key` were shadowed.
    expect(report.layers).toEqual([
      { kind: ConfigSourceKind.USER_FILE, source: configPath, keyCount: 1 },
      { kind: ConfigSourceKind.ENVIRONMENT, source: 'XERXES_*', keyCount: 1 },
      { kind: ConfigSourceKind.OVERRIDE, source: 'command line', keyCount: 1 },
    ])
    expect(report.counts['user-file']).toBe(1)
    expect(report.counts.default).toBe(report.entries.length - 3)
    expect(formatConfigProvenance(provenance, { changedOnly: true }).split('\n'))
      .not.toContain('llm.top_p = 0.95  [built-in default]')
  } finally {
    setConfig(new XerxesConfig({}, {}))
    await rm(root, { force: true, recursive: true })
  }
})

test('provenance attributes a workspace file and refuses writes to read-only layers', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-provenance-layers-'))
  const cwd = join(root, 'workspace')
  const home = join(root, 'home')
  try {
    await mkdir(cwd, { recursive: true })
    await mkdir(home, { recursive: true })
    await writeFile(join(cwd, 'xerxes.yaml'), 'llm:\n  model: repository-model\n', 'utf8')
    expect(findDefaultConfigSource(cwd, home, { allowWorkspaceConfig: true, environment: {} }))
      .toEqual({ kind: ConfigSourceKind.WORKSPACE_FILE, path: join(cwd, 'xerxes.yaml') })

    const { provenance } = resolveConfig({
      allowWorkspaceConfig: true,
      cwd,
      environment: { XERXES_DEBUG: 'true' },
      home,
    })
    const model = provenance.sourceOf('llm.model')
    expect(model?.kind).toBe(ConfigSourceKind.WORKSPACE_FILE)
    expect(() => { assertConfigSourceWritable(model!) }).not.toThrow()

    const debug = provenance.sourceOf('debug')
    expect(debug?.source).toBe('XERXES_DEBUG')
    expect(() => { assertConfigSourceWritable(debug!) })
      .toThrow('is supplied by environment variable: XERXES_DEBUG, which cannot be written')
    expect(() => { assertConfigSourceWritable(provenance.sourceOf('llm.timeout')!) })
      .toThrow('built-in default')

    // loadConfig publishes the provenance alongside the config it also returns unchanged.
    const published = loadConfig({ allowWorkspaceConfig: true, cwd, environment: {}, home })
    expect(published.llm.model).toBe('repository-model')
    expect(getConfigProvenance().sourceOf('llm.model')?.kind).toBe(ConfigSourceKind.WORKSPACE_FILE)
  } finally {
    setConfig(new XerxesConfig({}, {}))
    await rm(root, { force: true, recursive: true })
  }
})

test('a programmatically published config is described against the built-in defaults', () => {
  setConfig(new XerxesConfig({ debug: true, llm: { model: 'claude-3' } }, {}))
  const provenance = getConfigProvenance()
  expect(provenance.sourceOf('llm.model')).toMatchObject({
    kind: ConfigSourceKind.OVERRIDE,
    source: 'programmatic',
    value: 'claude-3',
  })
  expect(provenance.sourceOf('debug')?.kind).toBe(ConfigSourceKind.OVERRIDE)
  expect(provenance.sourceOf('llm.timeout')).toMatchObject({
    kind: ConfigSourceKind.DEFAULT,
    source: BUILT_IN_DEFAULT_SOURCE,
    value: 60,
  })
  setConfig(new XerxesConfig({}, {}))
  expect(getConfigProvenance().sourceOf('llm.model')?.kind).toBe(ConfigSourceKind.DEFAULT)
})

test('deep merge retains sibling settings and singleton access publishes validated configs', () => {
  expect(deepMerge(
    { executor: { default_timeout: 30, max_retries: 3 }, features: { first: true } },
    { executor: { max_retries: 6 }, features: { second: false } },
  )).toEqual({
    executor: { default_timeout: 30, max_retries: 6 },
    features: { first: true, second: false },
  })

  const configured = new XerxesConfig({ debug: true, llm: { model: 'claude-3' } }, {})
  setConfig(configured)
  expect(getConfig()).toBe(configured)
  expect(configured.merge(new XerxesConfig({ debug: false }, {})).debug).toBe(false)
})
