// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'

import { ConfigurationError } from './errors.js'
import { xerxesSubdirFor } from './paths.js'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../llms/samplingDefaults.js'

export const LogLevel = {
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARNING: 'WARNING',
  ERROR: 'ERROR',
  CRITICAL: 'CRITICAL',
} as const

export type LogLevel = (typeof LogLevel)[keyof typeof LogLevel]

export const EnvironmentType = {
  DEVELOPMENT: 'development',
  TESTING: 'testing',
  STAGING: 'staging',
  PRODUCTION: 'production',
} as const

export type EnvironmentType = (typeof EnvironmentType)[keyof typeof EnvironmentType]

/** Provider names accepted by the legacy root configuration model. */
export const LLMProvider = {
  OPENAI: 'openai',
  OPENROUTER: 'openrouter',
  GEMINI: 'gemini',
  ANTHROPIC: 'anthropic',
  COHERE: 'cohere',
  HUGGINGFACE: 'huggingface',
  LOCAL: 'local',
} as const

export type LLMProvider = (typeof LLMProvider)[keyof typeof LLMProvider]

export type ConfigPrimitive = boolean | number | null | string
export interface ConfigObject {
  readonly [key: string]: ConfigValue
}
export type ConfigValue = ConfigPrimitive | readonly ConfigValue[] | ConfigObject
export type ConfigRecord = ConfigObject
export type ConfigEnvironment = Readonly<Record<string, string | undefined>>

export interface ExecutorConfigData {
  readonly default_timeout: number
  readonly max_retries: number
  readonly retry_delay: number
  readonly max_concurrent_executions: number
  readonly enable_metrics: boolean
  readonly enable_caching: boolean
  readonly cache_ttl: number
}

export interface MemoryConfigData {
  readonly max_short_term: number
  readonly max_working: number
  readonly max_long_term: number
  readonly enable_embeddings: boolean
  readonly embedding_model: string | null
  readonly enable_persistence: boolean
  readonly persistence_path: string | null
  readonly auto_consolidate: boolean
  readonly consolidation_threshold: number
}

export interface SecurityConfigData {
  readonly enable_input_validation: boolean
  readonly enable_output_sanitization: boolean
  readonly max_input_length: number
  readonly max_output_length: number
  readonly allowed_functions: readonly string[] | null
  readonly blocked_functions: readonly string[] | null
  readonly enable_rate_limiting: boolean
  readonly rate_limit_per_minute: number
  readonly rate_limit_per_hour: number
  readonly enable_authentication: boolean
  readonly api_key: string | null
  readonly api_key_env_var: string
}

export interface LLMConfigData {
  readonly provider: LLMProvider
  readonly model: string
  readonly api_key: string | null
  readonly api_key_env_var: string
  readonly base_url: string | null
  readonly temperature: number
  readonly max_tokens: number
  readonly top_p: number
  readonly top_k: number
  readonly frequency_penalty: number
  readonly presence_penalty: number
  readonly repetition_penalty: number
  readonly timeout: number
  readonly max_retries: number
  readonly enable_streaming: boolean
  readonly enable_caching: boolean
}

export interface LoggingConfigData {
  readonly level: LogLevel
  readonly format: string
  readonly file_path: string | null
  readonly enable_console: boolean
  readonly enable_file: boolean
  readonly max_file_size: number
  readonly backup_count: number
  readonly enable_json_format: boolean
}

export interface ObservabilityConfigData {
  readonly enable_tracing: boolean
  readonly enable_metrics: boolean
  readonly enable_profiling: boolean
  readonly trace_endpoint: string | null
  readonly metrics_endpoint: string | null
  readonly service_name: string
  readonly service_version: string
  readonly enable_request_logging: boolean
  readonly enable_response_logging: boolean
  readonly enable_function_logging: boolean
}

export interface XerxesConfigData {
  readonly environment: EnvironmentType
  readonly debug: boolean
  readonly executor: ExecutorConfigData
  readonly memory: MemoryConfigData
  readonly security: SecurityConfigData
  readonly llm: LLMConfigData
  readonly logging: LoggingConfigData
  readonly observability: ObservabilityConfigData
  readonly plugins: ConfigRecord
  readonly features: Readonly<Record<string, boolean>>
}

interface FieldSpec<T> {
  readonly aliases?: readonly string[]
  readonly defaultValue: T | (() => T)
  readonly parse: (value: unknown, path: string) => T
}

type FieldSpecs<T extends object> = {
  readonly [Key in keyof T]: FieldSpec<T[Key]>
}

interface ExecutorValues {
  readonly defaultTimeout: number
  readonly maxRetries: number
  readonly retryDelay: number
  readonly maxConcurrentExecutions: number
  readonly enableMetrics: boolean
  readonly enableCaching: boolean
  readonly cacheTtl: number
}

/** Limits and feature toggles for function execution. */
export class ExecutorConfig {
  readonly defaultTimeout: number
  readonly maxRetries: number
  readonly retryDelay: number
  readonly maxConcurrentExecutions: number
  readonly enableMetrics: boolean
  readonly enableCaching: boolean
  readonly cacheTtl: number

  constructor(input: unknown = {}) {
    const values = parseFields<ExecutorValues>(input, 'executor', {
      defaultTimeout: numberField(30, 1, 600, ['default_timeout']),
      maxRetries: numberField(3, 0, 10, ['max_retries'], true),
      retryDelay: numberField(1, 0.1, 60, ['retry_delay']),
      maxConcurrentExecutions: numberField(10, 1, 100, ['max_concurrent_executions'], true),
      enableMetrics: booleanField(true, ['enable_metrics']),
      enableCaching: booleanField(false, ['enable_caching']),
      cacheTtl: numberField(3600, 60, 86_400, ['cache_ttl'], true),
    })
    this.defaultTimeout = values.defaultTimeout
    this.maxRetries = values.maxRetries
    this.retryDelay = values.retryDelay
    this.maxConcurrentExecutions = values.maxConcurrentExecutions
    this.enableMetrics = values.enableMetrics
    this.enableCaching = values.enableCaching
    this.cacheTtl = values.cacheTtl
    Object.freeze(this)
  }

  toJSON(): ExecutorConfigData {
    return {
      default_timeout: this.defaultTimeout,
      max_retries: this.maxRetries,
      retry_delay: this.retryDelay,
      max_concurrent_executions: this.maxConcurrentExecutions,
      enable_metrics: this.enableMetrics,
      enable_caching: this.enableCaching,
      cache_ttl: this.cacheTtl,
    }
  }
}

interface MemoryValues {
  readonly maxShortTerm: number
  readonly maxWorking: number
  readonly maxLongTerm: number
  readonly enableEmbeddings: boolean
  readonly embeddingModel: string | undefined
  readonly enablePersistence: boolean
  readonly persistencePath: string | undefined
  readonly autoConsolidate: boolean
  readonly consolidationThreshold: number
}

/** Sizing and persistence knobs for the four-tier memory system. */
export class MemoryConfig {
  readonly maxShortTerm: number
  readonly maxWorking: number
  readonly maxLongTerm: number
  readonly enableEmbeddings: boolean
  readonly embeddingModel: string | undefined
  readonly enablePersistence: boolean
  readonly persistencePath: string | undefined
  readonly autoConsolidate: boolean
  readonly consolidationThreshold: number

  constructor(input: unknown = {}) {
    const values = parseFields<MemoryValues>(input, 'memory', {
      maxShortTerm: numberField(10, 1, 1_000, ['max_short_term'], true),
      maxWorking: numberField(5, 1, 100, ['max_working'], true),
      maxLongTerm: numberField(1_000, 100, 100_000, ['max_long_term'], true),
      enableEmbeddings: booleanField(false, ['enable_embeddings']),
      embeddingModel: optionalStringField(['embedding_model']),
      enablePersistence: booleanField(false, ['enable_persistence']),
      persistencePath: optionalStringField(['persistence_path']),
      autoConsolidate: booleanField(true, ['auto_consolidate']),
      consolidationThreshold: numberField(0.8, 0.1, 1, ['consolidation_threshold']),
    })
    this.maxShortTerm = values.maxShortTerm
    this.maxWorking = values.maxWorking
    this.maxLongTerm = values.maxLongTerm
    this.enableEmbeddings = values.enableEmbeddings
    this.embeddingModel = values.embeddingModel
    this.enablePersistence = values.enablePersistence
    this.persistencePath = values.persistencePath
    this.autoConsolidate = values.autoConsolidate
    this.consolidationThreshold = values.consolidationThreshold
    Object.freeze(this)
  }

  toJSON(): MemoryConfigData {
    return {
      max_short_term: this.maxShortTerm,
      max_working: this.maxWorking,
      max_long_term: this.maxLongTerm,
      enable_embeddings: this.enableEmbeddings,
      embedding_model: this.embeddingModel ?? null,
      enable_persistence: this.enablePersistence,
      persistence_path: this.persistencePath ?? null,
      auto_consolidate: this.autoConsolidate,
      consolidation_threshold: this.consolidationThreshold,
    }
  }
}

interface SecurityValues {
  readonly enableInputValidation: boolean
  readonly enableOutputSanitization: boolean
  readonly maxInputLength: number
  readonly maxOutputLength: number
  readonly allowedFunctions: readonly string[] | undefined
  readonly blockedFunctions: readonly string[] | undefined
  readonly enableRateLimiting: boolean
  readonly rateLimitPerMinute: number
  readonly rateLimitPerHour: number
  readonly enableAuthentication: boolean
  readonly apiKey: string | undefined
  readonly apiKeyEnvVar: string
}

/** Input/output guardrails, allow/block lists, and rate-limit settings. */
export class SecurityConfig {
  readonly enableInputValidation: boolean
  readonly enableOutputSanitization: boolean
  readonly maxInputLength: number
  readonly maxOutputLength: number
  readonly allowedFunctions: readonly string[] | undefined
  readonly blockedFunctions: readonly string[] | undefined
  readonly enableRateLimiting: boolean
  readonly rateLimitPerMinute: number
  readonly rateLimitPerHour: number
  readonly enableAuthentication: boolean
  readonly apiKey: string | undefined
  readonly apiKeyEnvVar: string

  constructor(input: unknown = {}) {
    const values = parseFields<SecurityValues>(input, 'security', {
      enableInputValidation: booleanField(true, ['enable_input_validation']),
      enableOutputSanitization: booleanField(true, ['enable_output_sanitization']),
      maxInputLength: numberField(10_000, 100, 1_000_000, ['max_input_length'], true),
      maxOutputLength: numberField(10_000, 100, 1_000_000, ['max_output_length'], true),
      allowedFunctions: optionalStringArrayField(['allowed_functions']),
      blockedFunctions: optionalStringArrayField(['blocked_functions']),
      enableRateLimiting: booleanField(true, ['enable_rate_limiting']),
      rateLimitPerMinute: numberField(60, 1, 1_000, ['rate_limit_per_minute'], true),
      rateLimitPerHour: numberField(1_000, 10, 10_000, ['rate_limit_per_hour'], true),
      enableAuthentication: booleanField(false, ['enable_authentication']),
      apiKey: optionalStringField(['api_key']),
      apiKeyEnvVar: stringField('XERXES_API_KEY', ['api_key_env_var']),
    })
    this.enableInputValidation = values.enableInputValidation
    this.enableOutputSanitization = values.enableOutputSanitization
    this.maxInputLength = values.maxInputLength
    this.maxOutputLength = values.maxOutputLength
    this.allowedFunctions = values.allowedFunctions
    this.blockedFunctions = values.blockedFunctions
    this.enableRateLimiting = values.enableRateLimiting
    this.rateLimitPerMinute = values.rateLimitPerMinute
    this.rateLimitPerHour = values.rateLimitPerHour
    this.enableAuthentication = values.enableAuthentication
    this.apiKey = values.apiKey
    this.apiKeyEnvVar = values.apiKeyEnvVar
    Object.freeze(this)
  }

  toJSON(): SecurityConfigData {
    return {
      enable_input_validation: this.enableInputValidation,
      enable_output_sanitization: this.enableOutputSanitization,
      max_input_length: this.maxInputLength,
      max_output_length: this.maxOutputLength,
      allowed_functions: this.allowedFunctions ?? null,
      blocked_functions: this.blockedFunctions ?? null,
      enable_rate_limiting: this.enableRateLimiting,
      rate_limit_per_minute: this.rateLimitPerMinute,
      rate_limit_per_hour: this.rateLimitPerHour,
      enable_authentication: this.enableAuthentication,
      api_key: this.apiKey ?? null,
      api_key_env_var: this.apiKeyEnvVar,
    }
  }
}

interface LLMValues {
  readonly provider: LLMProvider
  readonly model: string
  readonly apiKey: string | undefined
  readonly apiKeyEnvVar: string
  readonly baseUrl: string | undefined
  readonly temperature: number
  readonly maxTokens: number
  readonly topP: number
  readonly topK: number
  readonly frequencyPenalty: number
  readonly presencePenalty: number
  readonly repetitionPenalty: number
  readonly timeout: number
  readonly maxRetries: number
  readonly enableStreaming: boolean
  readonly enableCaching: boolean
}

/** Provider-agnostic LLM client settings. */
export class LLMConfig {
  readonly provider: LLMProvider
  readonly model: string
  readonly apiKey: string | undefined
  readonly apiKeyEnvVar: string
  readonly baseUrl: string | undefined
  readonly temperature: number
  readonly maxTokens: number
  readonly topP: number
  readonly topK: number
  readonly frequencyPenalty: number
  readonly presencePenalty: number
  readonly repetitionPenalty: number
  readonly timeout: number
  readonly maxRetries: number
  readonly enableStreaming: boolean
  readonly enableCaching: boolean

  constructor(input: unknown = {}, environment: ConfigEnvironment = process.env) {
    const values = parseFields<LLMValues>(input, 'llm', {
      provider: enumField(LLMProvider, 'openai'),
      model: stringField('gpt-4'),
      apiKey: optionalStringField(['api_key']),
      apiKeyEnvVar: stringField('OPENAI_API_KEY', ['api_key_env_var']),
      baseUrl: optionalStringField(['base_url']),
      temperature: numberField(DEFAULT_TEMPERATURE, 0, 2),
      maxTokens: numberField(2_048, 1, 1_000_000, ['max_tokens'], true),
      topP: numberField(0.95, 0, 1, ['top_p']),
      topK: numberField(DEFAULT_TOP_K, 0, 100, ['top_k'], true),
      frequencyPenalty: numberField(0, -2, 2, ['frequency_penalty']),
      presencePenalty: numberField(0, -2, 2, ['presence_penalty']),
      repetitionPenalty: numberField(1, 0.1, 2, ['repetition_penalty']),
      timeout: numberField(60, 1, 600),
      maxRetries: numberField(3, 0, 10, ['max_retries'], true),
      enableStreaming: booleanField(true, ['enable_streaming']),
      enableCaching: booleanField(false, ['enable_caching']),
    })
    this.provider = values.provider
    this.model = values.model
    this.apiKeyEnvVar = values.apiKeyEnvVar
    this.apiKey = values.apiKey ?? environment[this.apiKeyEnvVar]
    this.baseUrl = values.baseUrl
    this.temperature = values.temperature
    this.maxTokens = values.maxTokens
    this.topP = values.topP
    this.topK = values.topK
    this.frequencyPenalty = values.frequencyPenalty
    this.presencePenalty = values.presencePenalty
    this.repetitionPenalty = values.repetitionPenalty
    this.timeout = values.timeout
    this.maxRetries = values.maxRetries
    this.enableStreaming = values.enableStreaming
    this.enableCaching = values.enableCaching
    Object.freeze(this)
  }

  toJSON(): LLMConfigData {
    return {
      provider: this.provider,
      model: this.model,
      api_key: this.apiKey ?? null,
      api_key_env_var: this.apiKeyEnvVar,
      base_url: this.baseUrl ?? null,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      top_p: this.topP,
      top_k: this.topK,
      frequency_penalty: this.frequencyPenalty,
      presence_penalty: this.presencePenalty,
      repetition_penalty: this.repetitionPenalty,
      timeout: this.timeout,
      max_retries: this.maxRetries,
      enable_streaming: this.enableStreaming,
      enable_caching: this.enableCaching,
    }
  }
}

interface LoggingValues {
  readonly level: LogLevel
  readonly format: string
  readonly filePath: string | undefined
  readonly enableConsole: boolean
  readonly enableFile: boolean
  readonly maxFileSize: number
  readonly backupCount: number
  readonly enableJsonFormat: boolean
}

/** Logging destinations, format, and rotation policy. */
export class LoggingConfig {
  readonly level: LogLevel
  readonly format: string
  readonly filePath: string | undefined
  readonly enableConsole: boolean
  readonly enableFile: boolean
  readonly maxFileSize: number
  readonly backupCount: number
  readonly enableJsonFormat: boolean

  constructor(input: unknown = {}) {
    const values = parseFields<LoggingValues>(input, 'logging', {
      level: enumField(LogLevel, 'INFO'),
      format: stringField('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
      filePath: optionalStringField(['file_path']),
      enableConsole: booleanField(true, ['enable_console']),
      enableFile: booleanField(false, ['enable_file']),
      maxFileSize: numberField(10_485_760, 1_024, 104_857_600, ['max_file_size'], true),
      backupCount: numberField(5, 1, 100, ['backup_count'], true),
      enableJsonFormat: booleanField(false, ['enable_json_format']),
    })
    this.level = values.level
    this.format = values.format
    this.filePath = values.filePath
    this.enableConsole = values.enableConsole
    this.enableFile = values.enableFile
    this.maxFileSize = values.maxFileSize
    this.backupCount = values.backupCount
    this.enableJsonFormat = values.enableJsonFormat
    Object.freeze(this)
  }

  toJSON(): LoggingConfigData {
    return {
      level: this.level,
      format: this.format,
      file_path: this.filePath ?? null,
      enable_console: this.enableConsole,
      enable_file: this.enableFile,
      max_file_size: this.maxFileSize,
      backup_count: this.backupCount,
      enable_json_format: this.enableJsonFormat,
    }
  }
}

interface ObservabilityValues {
  readonly enableTracing: boolean
  readonly enableMetrics: boolean
  readonly enableProfiling: boolean
  readonly traceEndpoint: string | undefined
  readonly metricsEndpoint: string | undefined
  readonly serviceName: string
  readonly serviceVersion: string
  readonly enableRequestLogging: boolean
  readonly enableResponseLogging: boolean
  readonly enableFunctionLogging: boolean
}

/** Tracing, metrics, profiling endpoints, and related toggles. */
export class ObservabilityConfig {
  readonly enableTracing: boolean
  readonly enableMetrics: boolean
  readonly enableProfiling: boolean
  readonly traceEndpoint: string | undefined
  readonly metricsEndpoint: string | undefined
  readonly serviceName: string
  readonly serviceVersion: string
  readonly enableRequestLogging: boolean
  readonly enableResponseLogging: boolean
  readonly enableFunctionLogging: boolean

  constructor(input: unknown = {}) {
    const values = parseFields<ObservabilityValues>(input, 'observability', {
      enableTracing: booleanField(false, ['enable_tracing']),
      enableMetrics: booleanField(true, ['enable_metrics']),
      enableProfiling: booleanField(false, ['enable_profiling']),
      traceEndpoint: optionalStringField(['trace_endpoint']),
      metricsEndpoint: optionalStringField(['metrics_endpoint']),
      serviceName: stringField('xerxes', ['service_name']),
      serviceVersion: stringField('0.2.6', ['service_version']),
      enableRequestLogging: booleanField(true, ['enable_request_logging']),
      enableResponseLogging: booleanField(false, ['enable_response_logging']),
      enableFunctionLogging: booleanField(true, ['enable_function_logging']),
    })
    this.enableTracing = values.enableTracing
    this.enableMetrics = values.enableMetrics
    this.enableProfiling = values.enableProfiling
    this.traceEndpoint = values.traceEndpoint
    this.metricsEndpoint = values.metricsEndpoint
    this.serviceName = values.serviceName
    this.serviceVersion = values.serviceVersion
    this.enableRequestLogging = values.enableRequestLogging
    this.enableResponseLogging = values.enableResponseLogging
    this.enableFunctionLogging = values.enableFunctionLogging
    Object.freeze(this)
  }

  toJSON(): ObservabilityConfigData {
    return {
      enable_tracing: this.enableTracing,
      enable_metrics: this.enableMetrics,
      enable_profiling: this.enableProfiling,
      trace_endpoint: this.traceEndpoint ?? null,
      metrics_endpoint: this.metricsEndpoint ?? null,
      service_name: this.serviceName,
      service_version: this.serviceVersion,
      enable_request_logging: this.enableRequestLogging,
      enable_response_logging: this.enableResponseLogging,
      enable_function_logging: this.enableFunctionLogging,
    }
  }
}

export const DEFAULT_FEATURES: Readonly<Record<string, boolean>> = Object.freeze({
  enable_agent_switching: true,
  enable_function_chaining: true,
  enable_context_awareness: true,
  enable_auto_retry: true,
  enable_adaptive_timeout: false,
  enable_smart_caching: false,
})

interface XerxesValues {
  readonly environment: EnvironmentType
  readonly debug: boolean
  readonly executor: ExecutorConfig
  readonly memory: MemoryConfig
  readonly security: SecurityConfig
  readonly llm: LLMConfig
  readonly logging: LoggingConfig
  readonly observability: ObservabilityConfig
  readonly plugins: ConfigRecord
  readonly features: Readonly<Record<string, boolean>>
}

/** Composite root model bundling every core configuration block. */
export class XerxesConfig {
  readonly environment: EnvironmentType
  readonly debug: boolean
  readonly executor: ExecutorConfig
  readonly memory: MemoryConfig
  readonly security: SecurityConfig
  readonly llm: LLMConfig
  readonly logging: LoggingConfig
  readonly observability: ObservabilityConfig
  readonly plugins: ConfigRecord
  readonly features: Readonly<Record<string, boolean>>

  constructor(input: unknown = {}, environment: ConfigEnvironment = process.env) {
    const values = parseFields<XerxesValues>(input, 'config', {
      environment: enumField(EnvironmentType, 'development'),
      debug: booleanField(false),
      executor: nestedConfigField(ExecutorConfig),
      memory: nestedConfigField(MemoryConfig),
      security: nestedConfigField(SecurityConfig),
      llm: {
        defaultValue: () => new LLMConfig({}, environment),
        parse: value => value instanceof LLMConfig ? value : new LLMConfig(value, environment),
      },
      logging: nestedConfigField(LoggingConfig),
      observability: nestedConfigField(ObservabilityConfig),
      plugins: {
        defaultValue: () => Object.freeze({}) as ConfigRecord,
        parse: (value, path) => parseConfigRecord(value, path),
      },
      features: {
        defaultValue: () => DEFAULT_FEATURES,
        parse: (value, path) => parseFeatures(value, path),
      },
    })
    this.environment = values.environment
    this.debug = values.debug
    this.executor = values.executor
    this.memory = values.memory
    this.security = values.security
    this.llm = values.llm
    this.logging = values.logging
    this.observability = values.observability
    this.plugins = values.plugins
    this.features = values.features
    Object.freeze(this)
  }

  /** Load one strict JSON or YAML config file. */
  static fromFile(path: string, options: ConfigSourceOptions = {}): XerxesConfig {
    return new XerxesConfig(readConfigFile(path), options.environment ?? process.env)
  }

  /** Build a configuration from recognized `XERXES_*` settings. */
  static fromEnv(environment: ConfigEnvironment = process.env, prefix = 'XERXES_'): XerxesConfig {
    return new XerxesConfig(configDataFromEnvironment(environment, prefix), environment)
  }

  /** Persist a portable JSON or YAML configuration file. */
  toFile(path: string): void {
    const extension = configExtension(path)
    const content = extension === '.json'
      ? JSON.stringify(this.toJSON(), null, 2) + '\n'
      : Bun.YAML.stringify(this.toJSON())
    mkdirSync(resolve(path, '..'), { recursive: true })
    writeFileSync(path, content, 'utf8')
  }

  /** Return a new config with every fully-resolved value from `other` overlaid on this config. */
  merge(other: XerxesConfig): XerxesConfig {
    if (!(other instanceof XerxesConfig)) {
      throw new ConfigurationError('config', 'can only merge another XerxesConfig')
    }
    return new XerxesConfig(deepMerge(this.toJSON(), other.toJSON()))
  }

  toJSON(): XerxesConfigData {
    return {
      environment: this.environment,
      debug: this.debug,
      executor: this.executor.toJSON(),
      memory: this.memory.toJSON(),
      security: this.security.toJSON(),
      llm: this.llm.toJSON(),
      logging: this.logging.toJSON(),
      observability: this.observability.toJSON(),
      plugins: this.plugins,
      features: this.features,
    }
  }
}

export interface ConfigSourceOptions {
  readonly environment?: ConfigEnvironment
}

export interface LoadConfigOptions extends ConfigSourceOptions {
  /** Search this directory before the Xerxes home when no explicit config file is supplied. */
  readonly cwd?: string
  /** Xerxes home used for the fallback `config.{yaml,yml,json}` search. */
  readonly home?: string
  /** Explicit configuration file, which takes precedence over `XERXES_CONFIG_FILE`. */
  readonly path?: string
}

let activeConfig: XerxesConfig | undefined

/** Return the process-wide config, lazily materialising validated defaults. */
export function getConfig(): XerxesConfig {
  activeConfig ??= new XerxesConfig()
  return activeConfig
}

/** Replace the process-wide config singleton. */
export function setConfig(config: XerxesConfig): void {
  if (!(config instanceof XerxesConfig)) {
    throw new ConfigurationError('config', 'must be a XerxesConfig instance')
  }
  activeConfig = config
}

/**
 * Resolve and publish the active configuration.
 *
 * The resulting precedence is defaults, then the chosen file, then recognized
 * `XERXES_*` fields. `XERXES_CONFIG_FILE` only selects a file; it is not
 * interpreted as a config field itself.
 */
export function loadConfig(options: LoadConfigOptions | string = {}): XerxesConfig {
  const normalized = typeof options === 'string' ? { path: options } : options
  const environment = normalized.environment ?? process.env
  const configPath = normalized.path
    ?? nonBlank(environment.XERXES_CONFIG_FILE)
    ?? findDefaultConfigFile(normalized.cwd ?? process.cwd(), normalized.home ?? xerxesSubdirFor(environment))
  const fileData = configPath ? readConfigFile(configPath) : {}
  const environmentData = configDataFromEnvironment(environment)
  const config = new XerxesConfig(deepMerge(fileData, environmentData), environment)
  setConfig(config)
  return config
}

/** Search config locations in the same cwd-before-home order as the Python runtime. */
export function findDefaultConfigFile(cwd = process.cwd(), home = xerxesSubdirFor(process.env)): string | undefined {
  const candidates = [
    ...['xerxes.yaml', 'xerxes.yml', 'xerxes.json'].map(filename => join(cwd, filename)),
    ...['config.yaml', 'config.yml', 'config.json'].map(filename => join(home, filename)),
  ]
  for (const path of candidates) {
    if (existsSync(path)) return path
  }
  return undefined
}

/** Parse only recognized config settings from an environment mapping. */
export function configDataFromEnvironment(environment: ConfigEnvironment = process.env, prefix = 'XERXES_'): Record<string, unknown> {
  const output: Record<string, unknown> = {}
  for (const [key, rawValue] of Object.entries(environment)) {
    if (!key.startsWith(prefix) || rawValue === undefined) continue
    const path = configPathFromEnvironmentKey(key.slice(prefix.length))
    if (!path) continue
    assignPath(output, path, parseEnvironmentValue(rawValue), key)
  }
  return output
}

/** Deeply overlay serializable configuration records without sharing mutable values. */
export function deepMerge(
  base: object,
  override: object,
): Record<string, unknown> {
  const left = plainRecord(base, 'merge base')
  const right = plainRecord(override, 'merge override')
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(left)) {
    assertSafeKey(key, 'merge base')
    result[key] = cloneConfigValue(value, `merge base.${key}`)
  }
  for (const [key, value] of Object.entries(right)) {
    assertSafeKey(key, 'merge override')
    result[key] = isPlainRecord(result[key]) && isPlainRecord(value)
      ? deepMerge(result[key], value)
      : cloneConfigValue(value, `merge override.${key}`)
  }
  return result
}

function readConfigFile(path: string): Record<string, unknown> {
  const extension = configExtension(path)
  const content = readFileSync(path, 'utf8')
  let parsed: unknown
  try {
    parsed = extension === '.json' ? JSON.parse(content) : Bun.YAML.parse(content)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new ConfigurationError(path, `contains invalid ${extension.slice(1).toUpperCase()}: ${message}`)
  }
  return plainRecord(parsed, path)
}

function configExtension(path: string): '.json' | '.yaml' | '.yml' {
  const extension = extname(path).toLowerCase()
  if (extension === '.json' || extension === '.yaml' || extension === '.yml') return extension
  throw new ConfigurationError(path, `unsupported configuration file format: ${extension || '(none)'}`)
}

function configPathFromEnvironmentKey(suffix: string): readonly string[] | undefined {
  if (!suffix || suffix === 'CONFIG_FILE' || suffix === 'HOME') return undefined
  if (suffix.includes('__')) {
    const path = suffix.split('__').filter(Boolean).map(part => part.toLowerCase())
    if (path.length >= 1 && isConfigRootPath(path[0] ?? '')) return path
    return undefined
  }
  if (suffix === 'DEBUG' || suffix === 'ENVIRONMENT') return [suffix.toLowerCase()]
  for (const section of ['EXECUTOR', 'MEMORY', 'SECURITY', 'LLM', 'LOGGING', 'OBSERVABILITY', 'PLUGINS', 'FEATURES']) {
    const prefix = section + '_'
    if (suffix.startsWith(prefix)) {
      const field = suffix.slice(prefix.length).toLowerCase()
      return field ? [section.toLowerCase(), field] : undefined
    }
  }
  return undefined
}

function isConfigRootPath(value: string): boolean {
  return ['debug', 'environment', 'executor', 'memory', 'security', 'llm', 'logging', 'observability', 'plugins', 'features'].includes(value)
}

function assignPath(target: Record<string, unknown>, path: readonly string[], value: unknown, source: string): void {
  let current = target
  for (const segment of path.slice(0, -1)) {
    assertSafeKey(segment, source)
    const existing = current[segment]
    if (existing === undefined) {
      const nested: Record<string, unknown> = {}
      current[segment] = nested
      current = nested
      continue
    }
    if (!isPlainRecord(existing)) {
      throw new ConfigurationError(source, `cannot assign nested setting through non-object '${segment}'`)
    }
    current = existing
  }
  const last = path.at(-1)
  if (!last) throw new ConfigurationError(source, 'has an empty setting path')
  assertSafeKey(last, source)
  current[last] = value
}

function parseEnvironmentValue(value: string): unknown {
  try {
    return JSON.parse(value) as unknown
  } catch {
    return value
  }
}

function parseFields<T extends object>(input: unknown, path: string, specs: FieldSpecs<T>): T {
  const raw = plainRecord(input, path)
  const names = new Map<string, keyof T>()
  for (const [key, spec] of Object.entries(specs) as [keyof T, FieldSpec<T[keyof T]>][]) {
    names.set(String(key), key)
    for (const alias of spec.aliases ?? []) {
      names.set(alias, key)
    }
  }
  for (const key of Object.keys(raw)) {
    assertSafeKey(key, path)
    if (!names.has(key)) {
      throw new ConfigurationError(path, `contains unknown setting '${key}'`)
    }
  }
  const parsed: Record<string, unknown> = {}
  for (const [key, spec] of Object.entries(specs) as [keyof T, FieldSpec<T[keyof T]>][]) {
    const value = readAliasedValue(raw, String(key), spec.aliases ?? [], path)
    parsed[String(key)] = value.found
      ? spec.parse(value.value, `${path}.${String(key)}`)
      : typeof spec.defaultValue === 'function'
        ? (spec.defaultValue as () => T[keyof T])()
        : cloneDefault(spec.defaultValue)
  }
  return parsed as T
}

function readAliasedValue(
  input: Record<string, unknown>,
  canonical: string,
  aliases: readonly string[],
  path: string,
): { readonly found: boolean; readonly value: unknown } {
  const keys = [canonical, ...aliases].filter(key => Object.hasOwn(input, key))
  if (keys.length > 1) {
    throw new ConfigurationError(path, `sets '${canonical}' through multiple aliases: ${keys.join(', ')}`)
  }
  if (!keys.length) return { found: false, value: undefined }
  const key = keys[0]
  if (!key) return { found: false, value: undefined }
  return { found: true, value: input[key] }
}

function cloneDefault<T>(value: T): T {
  if (Array.isArray(value) || isPlainRecord(value)) {
    return cloneConfigValue(value, 'default') as T
  }
  return value
}

function numberField(
  defaultValue: number,
  minimum: number,
  maximum: number,
  aliases: readonly string[] = [],
  integer = false,
): FieldSpec<number> {
  return {
    defaultValue,
    aliases,
    parse: (value, path) => {
      if (typeof value !== 'number' || !Number.isFinite(value) || (integer && !Number.isInteger(value))) {
        throw new ConfigurationError(path, integer ? 'must be a finite integer' : 'must be a finite number')
      }
      if (value < minimum || value > maximum) {
        throw new ConfigurationError(path, `must be between ${minimum} and ${maximum}`)
      }
      return value
    },
  }
}

function booleanField(defaultValue: boolean, aliases: readonly string[] = []): FieldSpec<boolean> {
  return {
    defaultValue,
    aliases,
    parse: (value, path) => {
      if (typeof value !== 'boolean') throw new ConfigurationError(path, 'must be a boolean')
      return value
    },
  }
}

function stringField(defaultValue: string, aliases: readonly string[] = []): FieldSpec<string> {
  return {
    defaultValue,
    aliases,
    parse: (value, path) => {
      if (typeof value !== 'string' || !value.trim()) throw new ConfigurationError(path, 'must be a non-empty string')
      return value
    },
  }
}

function optionalStringField(aliases: readonly string[] = []): FieldSpec<string | undefined> {
  return {
    defaultValue: undefined,
    aliases,
    parse: (value, path) => {
      if (value === null) return undefined
      if (typeof value !== 'string') throw new ConfigurationError(path, 'must be a string or null')
      return value
    },
  }
}

function optionalStringArrayField(aliases: readonly string[] = []): FieldSpec<readonly string[] | undefined> {
  return {
    defaultValue: undefined,
    aliases,
    parse: (value, path) => {
      if (value === null) return undefined
      if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
        throw new ConfigurationError(path, 'must be an array of strings or null')
      }
      return Object.freeze([...value])
    },
  }
}

function enumField<T extends string>(
  values: Readonly<Record<string, T>>,
  defaultValue: T,
  aliases: readonly string[] = [],
): FieldSpec<T> {
  const allowed = new Set(Object.values(values))
  return {
    defaultValue,
    aliases,
    parse: (value, path) => {
      if (typeof value !== 'string' || !allowed.has(value as T)) {
        throw new ConfigurationError(path, `must be one of: ${[...allowed].join(', ')}`)
      }
      return value as T
    },
  }
}

function nestedConfigField<T>(constructor: new (input?: unknown) => T): FieldSpec<T> {
  return {
    defaultValue: () => new constructor(),
    parse: value => value instanceof constructor ? value : new constructor(value),
  }
}

function parseFeatures(value: unknown, path: string): Readonly<Record<string, boolean>> {
  const raw = plainRecord(value, path)
  const features: Record<string, boolean> = { ...DEFAULT_FEATURES }
  for (const [key, setting] of Object.entries(raw)) {
    assertSafeKey(key, path)
    if (typeof setting !== 'boolean') throw new ConfigurationError(`${path}.${key}`, 'must be a boolean')
    features[key] = setting
  }
  return Object.freeze(features)
}

function parseConfigRecord(value: unknown, path: string): ConfigRecord {
  return deepFreezeConfigValue(plainRecord(value, path), path) as ConfigRecord
}

function cloneConfigValue(value: unknown, path: string): ConfigValue {
  return deepFreezeConfigValue(value, path)
}

function deepFreezeConfigValue(value: unknown, path: string): ConfigValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new ConfigurationError(path, 'must not contain a non-finite number')
    return value
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map((item, index) => deepFreezeConfigValue(item, `${path}[${index}]`)))
  }
  const raw = plainRecord(value, path)
  const result: Record<string, ConfigValue> = {}
  for (const [key, item] of Object.entries(raw)) {
    assertSafeKey(key, path)
    result[key] = deepFreezeConfigValue(item, `${path}.${key}`)
  }
  return Object.freeze(result)
}

function plainRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isPlainRecord(value)) {
    throw new ConfigurationError(path, 'must be a mapping object')
  }
  return value
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function assertSafeKey(key: string, path: string): void {
  if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
    throw new ConfigurationError(path, `contains unsafe key '${key}'`)
  }
}

function nonBlank(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  return trimmed || undefined
}
