// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  EnvironmentType,
  ExecutorConfig,
  LLMConfig,
  LLMProvider,
  LogLevel,
  LoggingConfig,
  MemoryConfig,
  ObservabilityConfig,
  SecurityConfig,
  XerxesConfig,
} from '../src/core/config.js'
import { ConfigurationError } from '../src/core/errors.js'

test('core configuration exposes the established enum and nested default surface', () => {
  expect(Object.values(LogLevel)).toEqual(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
  expect(Object.values(EnvironmentType)).toEqual(['development', 'testing', 'staging', 'production'])
  expect(Object.values(LLMProvider)).toEqual(expect.arrayContaining([
    'openai',
    'openrouter',
    'gemini',
    'anthropic',
    'local',
  ]))

  expect(new ExecutorConfig()).toMatchObject({
    defaultTimeout: 30,
    maxRetries: 3,
    retryDelay: 1,
    maxConcurrentExecutions: 10,
    enableMetrics: true,
    enableCaching: false,
    cacheTtl: 3600,
  })
  expect(new MemoryConfig()).toMatchObject({
    maxShortTerm: 10,
    maxLongTerm: 1000,
    enableEmbeddings: false,
    autoConsolidate: true,
  })
  expect(new SecurityConfig()).toMatchObject({
    enableInputValidation: true,
    enableRateLimiting: true,
    rateLimitPerMinute: 60,
    enableAuthentication: false,
  })
  expect(new LoggingConfig()).toMatchObject({ level: LogLevel.INFO, enableConsole: true, enableFile: false })
  expect(new ObservabilityConfig()).toMatchObject({ enableTracing: false, enableMetrics: true, serviceName: 'xerxes' })
})

test('LLM configuration honors explicit credentials and environment construction remains strict', () => {
  const explicit = new LLMConfig({ api_key: 'explicit-key' }, { OPENAI_API_KEY: 'environment-key' })
  expect(explicit).toMatchObject({
    provider: LLMProvider.OPENAI,
    model: 'gpt-4',
    apiKey: 'explicit-key',
    temperature: 0.6,
    topK: 64,
    maxTokens: 2048,
  })

  const fromEnvironment = XerxesConfig.fromEnv({
    XERXES_DEBUG: 'true',
    XERXES_ENVIRONMENT: 'production',
    XERXES_LLM_MODEL: 'gpt-4.1',
  })
  expect(fromEnvironment).toMatchObject({ debug: true, environment: EnvironmentType.PRODUCTION })
  expect(fromEnvironment.llm.model).toBe('gpt-4.1')
  expect(() => XerxesConfig.fromFile('/tmp/xerxes-core-config.toml')).toThrow(ConfigurationError)
})
