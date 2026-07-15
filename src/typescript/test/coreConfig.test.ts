// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  EnvironmentType,
  ExecutorConfig,
  LLMConfig,
  LLMProvider,
  LogLevel,
  LoggingConfig,
  MemoryConfig,
  SecurityConfig,
  XerxesConfig,
  configDataFromEnvironment,
  deepMerge,
  getConfig,
  loadConfig,
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
