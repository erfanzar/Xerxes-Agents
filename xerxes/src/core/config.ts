// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { extname, join, resolve, sep } from 'node:path'

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
      // Secrets are redacted from serialization; only the env-var pointer persists.
      api_key: null,
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
    // Blank or whitespace-only environment credentials are treated as absent.
    const environmentKey = environment[this.apiKeyEnvVar]?.trim()
    this.apiKey = values.apiKey ?? (environmentKey || undefined)
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
      // Secrets are redacted from serialization; only the env-var pointer persists.
      api_key: null,
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
      // Keep in sync with xerxes/package.json "version"; the bundled runtime
      // cannot read package.json at module load time.
      serviceVersion: stringField('0.4.0', ['service_version']),
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
    const merged = deepMerge(this.toJSON(), other.toJSON())
    // Portable serialization deliberately redacts secrets. Rehydrate from the already-resolved
    // objects instead of letting construction consult process.env and either lose or replace them.
    const securityApiKey = other.security.apiKey ?? this.security.apiKey
    const llmApiKey = other.llm.apiKey ?? this.llm.apiKey
    const security = deepMerge(this.security.toJSON(), other.security.toJSON())
    const llm = deepMerge(this.llm.toJSON(), other.llm.toJSON())
    return new XerxesConfig({
      ...merged,
      security: new SecurityConfig({
        ...security,
        ...(securityApiKey === undefined ? {} : { api_key: securityApiKey }),
      }),
      llm: new LLMConfig({
        ...llm,
        ...(llmApiKey === undefined ? {} : { api_key: llmApiKey }),
      }, {}),
    }, {})
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
  /**
   * Admit this directory's `xerxes.{yaml,yml,json}` as a fallback when the Xerxes home has no
   * config. Defaults to the `XERXES_ALLOW_WORKSPACE_CONFIG` opt-in.
   */
  readonly allowWorkspaceConfig?: boolean
  /** Working directory searched for a workspace config, only once workspace configs are allowed. */
  readonly cwd?: string
  /** Xerxes home searched first for `config.{yaml,yml,json}`. */
  readonly home?: string
  /** Notified with the path of a workspace config that exists but is being ignored. */
  readonly onIgnoredWorkspaceConfig?: (path: string) => void
  /** Explicit configuration file, which takes precedence over `XERXES_CONFIG_FILE`. */
  readonly path?: string
  /** Values supplied on the command line, overlaid last and attributed to the override layer. */
  readonly overrides?: Record<string, unknown>
}

/** Named layers a resolved value can come from, listed lowest precedence first. */
export const ConfigSourceKind = {
  DEFAULT: 'default',
  USER_FILE: 'user-file',
  WORKSPACE_FILE: 'workspace-file',
  ENVIRONMENT: 'environment',
  OVERRIDE: 'override',
} as const

export type ConfigSourceKind = (typeof ConfigSourceKind)[keyof typeof ConfigSourceKind]

export interface ConfigSourceKindInfo {
  /** Higher wins when several layers set the same key. */
  readonly precedence: number
  readonly label: string
  /**
   * Whether a `/config set` can persist into this layer. Defaults, environment variables, and
   * command-line overrides are read-only: writing them would appear to succeed and then be
   * silently overwritten on the next resolution.
   */
  readonly writable: boolean
}

export const CONFIG_SOURCE_KINDS: Readonly<Record<ConfigSourceKind, ConfigSourceKindInfo>> = Object.freeze({
  default: { precedence: 0, label: 'built-in default', writable: false },
  'user-file': { precedence: 1, label: 'user config file', writable: true },
  'workspace-file': { precedence: 2, label: 'workspace config file', writable: true },
  environment: { precedence: 3, label: 'environment variable', writable: false },
  override: { precedence: 4, label: 'explicit override', writable: false },
})

/** Source name reported for keys no layer ever set. */
export const BUILT_IN_DEFAULT_SOURCE = 'built-in default'

/** One layer's contribution to a single key. */
export interface ConfigContribution {
  readonly kind: ConfigSourceKind
  /** Absolute file path, environment variable name, or descriptive label for the layer. */
  readonly source: string
  readonly value: ConfigValue
}

export interface ConfigProvenanceEntry extends ConfigContribution {
  /** Dotted path into the serialized config, e.g. `llm.model`. */
  readonly path: string
  /** Layers that also set this key but lost, highest precedence first. */
  readonly shadowed: readonly ConfigContribution[]
}

export interface ConfigProvenanceLayer {
  readonly kind: ConfigSourceKind
  readonly source: string
  /** Number of resolved keys this layer won outright. */
  readonly keyCount: number
}

export interface ConfigProvenanceReport {
  readonly layers: readonly ConfigProvenanceLayer[]
  readonly counts: Readonly<Record<ConfigSourceKind, number>>
  readonly entries: readonly ConfigProvenanceEntry[]
}

/**
 * Per-key record of which layer supplied each resolved setting.
 *
 * Merging flattens everything into one object, so without this a value has no memory of where it
 * came from: "why is my model X" and "which file set the permission mode" are unanswerable, and a
 * write cannot be refused for a layer that is read-only.
 */
export class ConfigProvenance {
  private readonly byPath: ReadonlyMap<string, ConfigProvenanceEntry>
  readonly layers: readonly ConfigProvenanceLayer[]

  constructor(entries: readonly ConfigProvenanceEntry[], layers: readonly ConfigProvenanceLayer[]) {
    this.byPath = new Map(entries.map(entry => [entry.path, entry]))
    this.layers = Object.freeze([...layers])
    Object.freeze(this)
  }

  /** Every resolved key with its winning layer, ordered by key path. */
  get entries(): readonly ConfigProvenanceEntry[] {
    return Object.freeze([...this.byPath.values()])
  }

  /**
   * Source of one dotted key path. A key no layer set reports the built-in default rather than
   * `undefined`; `undefined` means the path is not part of the resolved configuration at all.
   */
  sourceOf(path: string): ConfigProvenanceEntry | undefined {
    return this.byPath.get(path)
  }

  /** One-line explanation for a doctor check, `/config`, or an error message. */
  explain(path: string): string {
    const entry = this.byPath.get(path)
    if (entry === undefined) return `${path} is not a known configuration key`
    const shadowed = entry.shadowed.length
      ? ` (overrides ${entry.shadowed.map(describeContribution).join(', ')})`
      : ''
    return `${path} = ${formatValue(entry.value)} from ${describeContribution(entry)}${shadowed}`
  }

  report(): ConfigProvenanceReport {
    const counts: Record<ConfigSourceKind, number> = {
      default: 0, 'user-file': 0, 'workspace-file': 0, environment: 0, override: 0,
    }
    for (const entry of this.byPath.values()) counts[entry.kind] += 1
    return Object.freeze({
      layers: this.layers,
      counts: Object.freeze(counts),
      entries: this.entries,
    })
  }

  toJSON(): ConfigProvenanceReport {
    return this.report()
  }
}

/** Render a provenance report as text for a doctor check or `/config` view. */
export function formatConfigProvenance(
  provenance: ConfigProvenance,
  options: { readonly changedOnly?: boolean } = {},
): string {
  const lines: string[] = []
  for (const layer of provenance.layers) {
    lines.push(`${CONFIG_SOURCE_KINDS[layer.kind].label}: ${layer.source} (${layer.keyCount} keys)`)
  }
  if (lines.length) lines.push('')
  for (const entry of provenance.entries) {
    if (options.changedOnly && entry.kind === ConfigSourceKind.DEFAULT) continue
    lines.push(`${entry.path} = ${formatValue(entry.value)}  [${describeContribution(entry)}]`)
    for (const shadowed of entry.shadowed) {
      lines.push(`    shadowed: ${formatValue(shadowed.value)} [${describeContribution(shadowed)}]`)
    }
  }
  return lines.join('\n')
}

/**
 * Refuse a write aimed at a layer Xerxes cannot persist into, instead of writing somewhere the
 * next resolution silently discards.
 */
export function assertConfigSourceWritable(entry: ConfigProvenanceEntry): void {
  if (CONFIG_SOURCE_KINDS[entry.kind].writable) return
  throw new ConfigurationError(
    entry.path,
    `is supplied by ${describeContribution(entry)}, which cannot be written`,
  )
}

function describeContribution(contribution: ConfigContribution): string {
  const label = CONFIG_SOURCE_KINDS[contribution.kind].label
  return contribution.source === label || contribution.source === BUILT_IN_DEFAULT_SOURCE
    ? label
    : `${label}: ${contribution.source}`
}

function formatValue(value: ConfigValue): string {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

let activeConfig: XerxesConfig | undefined
let activeProvenance: ConfigProvenance | undefined

/** Return the process-wide config, lazily materialising validated defaults. */
export function getConfig(): XerxesConfig {
  activeConfig ??= new XerxesConfig()
  return activeConfig
}

/** Replace the process-wide config singleton, optionally with the provenance that produced it. */
export function setConfig(config: XerxesConfig, provenance?: ConfigProvenance): void {
  if (!(config instanceof XerxesConfig)) {
    throw new ConfigurationError('config', 'must be a XerxesConfig instance')
  }
  activeConfig = config
  activeProvenance = provenance
}

/**
 * Provenance for the active configuration.
 *
 * A config published straight through `setConfig` carries no layer history, so it is described
 * against the built-in defaults: unchanged keys are defaults, differing keys are programmatic
 * overrides. That is honest about what is known instead of claiming everything is a default.
 */
export function getConfigProvenance(): ConfigProvenance {
  const config = getConfig()
  activeProvenance ??= provenanceFromConfig(config)
  return activeProvenance
}

export interface ConfigResolution {
  readonly config: XerxesConfig
  /** Rides alongside the config; callers that ignore it see exactly the pre-provenance behavior. */
  readonly provenance: ConfigProvenance
}

/**
 * Resolve the configuration and the per-key provenance without publishing either.
 *
 * The resulting precedence is defaults, then the chosen file, then recognized `XERXES_*` fields,
 * then explicit overrides. `XERXES_CONFIG_FILE` only selects a file; it is not interpreted as a
 * config field itself.
 */
export function resolveConfig(options: LoadConfigOptions | string = {}): ConfigResolution {
  const normalized = typeof options === 'string' ? { path: options } : options
  const environment = normalized.environment ?? process.env
  const cwd = normalized.cwd ?? process.cwd()
  const home = normalized.home ?? xerxesSubdirFor(environment)
  const explicitPath = normalized.path ?? nonBlank(environment.XERXES_CONFIG_FILE)
  const fileSource = explicitPath === undefined
    ? findDefaultConfigSource(cwd, home, {
      environment,
      ...(normalized.allowWorkspaceConfig === undefined
        ? {}
        : { allowWorkspaceConfig: normalized.allowWorkspaceConfig }),
      ...(normalized.onIgnoredWorkspaceConfig === undefined
        ? {}
        : { onIgnoredWorkspaceConfig: normalized.onIgnoredWorkspaceConfig }),
    })
    : { kind: classifyConfigFilePath(explicitPath, cwd, home), path: explicitPath }

  const layers: ResolutionLayer[] = []
  const fileData = fileSource ? readConfigFile(fileSource.path) : {}
  if (fileSource) layers.push({ kind: fileSource.kind, source: fileSource.path, data: fileData })
  const { data: environmentData, keySources } = environmentLayer(environment)
  if (keySources.size) {
    layers.push({ kind: ConfigSourceKind.ENVIRONMENT, source: 'XERXES_*', data: environmentData, keySources })
  }
  let merged = deepMerge(fileData, environmentData)
  if (normalized.overrides !== undefined) {
    layers.push({ kind: ConfigSourceKind.OVERRIDE, source: 'command line', data: normalized.overrides })
    merged = deepMerge(merged, normalized.overrides)
  }
  const config = new XerxesConfig(merged, environment)
  return { config, provenance: buildProvenance(config, layers) }
}

/** Resolve and publish the active configuration; see {@link resolveConfig} for precedence. */
export function loadConfig(options: LoadConfigOptions | string = {}): XerxesConfig {
  const { config, provenance } = resolveConfig(options)
  setConfig(config, provenance)
  return config
}

/** Environment opt-in that admits a working-directory `xerxes.{yaml,yml,json}` into the search. */
export const WORKSPACE_CONFIG_OPT_IN_ENV = 'XERXES_ALLOW_WORKSPACE_CONFIG'

const WORKSPACE_CONFIG_FILENAMES = ['xerxes.yaml', 'xerxes.yml', 'xerxes.json'] as const
const HOME_CONFIG_FILENAMES = ['config.yaml', 'config.yml', 'config.json'] as const
const WORKSPACE_CONFIG_OPT_IN_VALUES: ReadonlySet<string> = new Set(['1', 'on', 'true', 'yes'])
/** Paths already announced, so a long-lived process reports each ignored workspace config once. */
const announcedWorkspaceConfigs = new Set<string>()

export interface DefaultConfigSearchOptions {
  /** Admit the working directory's config. Defaults to the `XERXES_ALLOW_WORKSPACE_CONFIG` opt-in. */
  readonly allowWorkspaceConfig?: boolean
  /** Environment consulted for the opt-in; defaults to `process.env`. */
  readonly environment?: ConfigEnvironment
  /** Replaces the default warn-once notice for an ignored workspace config. */
  readonly onIgnoredWorkspaceConfig?: (path: string) => void
}

/**
 * Search the Xerxes home first, and the working directory only on an explicit opt-in.
 *
 * Cloning a repository must not silently reconfigure the daemon — model, base URL, permission
 * mode — so the user's own config always wins and a workspace `xerxes.*` file is admitted only
 * through `allowWorkspaceConfig` or `XERXES_ALLOW_WORKSPACE_CONFIG`. A workspace config that
 * exists but is being ignored is announced instead of disappearing without a trace.
 */
export function findDefaultConfigFile(
  cwd = process.cwd(),
  home = xerxesSubdirFor(process.env),
  options: DefaultConfigSearchOptions = {},
): string | undefined {
  return findDefaultConfigSource(cwd, home, options)?.path
}

/** A discovered config file together with the layer it belongs to. */
export interface ConfigFileSource {
  readonly kind: typeof ConfigSourceKind.USER_FILE | typeof ConfigSourceKind.WORKSPACE_FILE
  readonly path: string
}

/** {@link findDefaultConfigFile} plus the layer name, so provenance can attribute the file. */
export function findDefaultConfigSource(
  cwd = process.cwd(),
  home = xerxesSubdirFor(process.env),
  options: DefaultConfigSearchOptions = {},
): ConfigFileSource | undefined {
  const environment = options.environment ?? process.env
  const allowWorkspace = options.allowWorkspaceConfig
    ?? WORKSPACE_CONFIG_OPT_IN_VALUES.has((environment[WORKSPACE_CONFIG_OPT_IN_ENV] ?? '').trim().toLowerCase())
  const homeConfig = HOME_CONFIG_FILENAMES.map(filename => join(home, filename)).find(path => existsSync(path))
  const workspaceConfig = WORKSPACE_CONFIG_FILENAMES.map(filename => join(cwd, filename)).find(path => existsSync(path))
  if (workspaceConfig !== undefined && !allowWorkspace) {
    announceIgnoredWorkspaceConfig(workspaceConfig, options)
  }
  // Home stays ahead of the workspace even when the opt-in is set: opting in adds a fallback, not an override.
  if (homeConfig !== undefined) return { kind: ConfigSourceKind.USER_FILE, path: homeConfig }
  if (allowWorkspace && workspaceConfig !== undefined) {
    return { kind: ConfigSourceKind.WORKSPACE_FILE, path: workspaceConfig }
  }
  return undefined
}

/**
 * Attribute an explicitly requested file to a layer by where it lives.
 *
 * The Xerxes home is checked before the working directory so a home that happens to sit inside the
 * workspace is still reported as the user's own config rather than as repository-supplied.
 */
function classifyConfigFilePath(path: string, cwd: string, home: string): ConfigFileSource['kind'] {
  const target = resolve(path)
  if (isInside(target, home)) return ConfigSourceKind.USER_FILE
  if (isInside(target, cwd)) return ConfigSourceKind.WORKSPACE_FILE
  // Anywhere else was named by the user, not discovered inside a clone, so it is a user file.
  return ConfigSourceKind.USER_FILE
}

function isInside(target: string, directory: string): boolean {
  const base = resolve(directory)
  return target === base || target.startsWith(base.endsWith(sep) ? base : base + sep)
}

function announceIgnoredWorkspaceConfig(path: string, options: DefaultConfigSearchOptions): void {
  if (options.onIgnoredWorkspaceConfig !== undefined) {
    options.onIgnoredWorkspaceConfig(path)
    return
  }
  if (announcedWorkspaceConfigs.has(path)) return
  announcedWorkspaceConfigs.add(path)
  console.warn(
    `Ignoring workspace configuration ${path}: a repository cannot reconfigure Xerxes by itself. `
    + `Set ${WORKSPACE_CONFIG_OPT_IN_ENV}=1 or pass allowWorkspaceConfig to opt in.`,
  )
}

/** Parse only recognized config settings from an environment mapping. */
export function configDataFromEnvironment(environment: ConfigEnvironment = process.env, prefix = 'XERXES_'): Record<string, unknown> {
  return environmentLayer(environment, prefix).data
}

interface EnvironmentLayer {
  readonly data: Record<string, unknown>
  /** Dotted key path to the exact variable that set it, so provenance can name `XERXES_LLM_MODEL`. */
  readonly keySources: ReadonlyMap<string, string>
}

function environmentLayer(environment: ConfigEnvironment = process.env, prefix = 'XERXES_'): EnvironmentLayer {
  const output: Record<string, unknown> = {}
  const keySources = new Map<string, string>()
  for (const [key, rawValue] of Object.entries(environment)) {
    if (!key.startsWith(prefix) || rawValue === undefined) continue
    const path = configPathFromEnvironmentKey(key.slice(prefix.length))
    if (!path) continue
    assignPath(output, path, parseEnvironmentValue(rawValue), key)
    keySources.set(path.join('.'), key)
  }
  return { data: output, keySources }
}

interface ResolutionLayer {
  readonly kind: ConfigSourceKind
  readonly source: string
  readonly data: Record<string, unknown>
  readonly keySources?: ReadonlyMap<string, string>
}

/**
 * Leaf names whose values must never reach a doctor check or a `/config` dump. The resolved config
 * already redacts secrets in `toJSON`, but raw layer values would otherwise print them verbatim.
 * Qualified token names only: a bare `token` substring would also redact `max_tokens`, and
 * `api_key_env_var` names a variable rather than holding a credential.
 */
const SECRET_LEAF_PATTERN
  = /^(.*_)?(api_?key|auth_token|access_token|refresh_token|secret|password|passphrase|credential)s?$/i

let defaultLeaves: ReadonlyMap<string, unknown> | undefined

function buildProvenance(config: XerxesConfig, layers: readonly ResolutionLayer[]): ConfigProvenance {
  const resolved = configLeaves(config.toJSON())
  const flattened = layers.map(layer => ({ layer, leaves: alignLeafPaths(configLeaves(layer.data), resolved) }))
  const wins = new Map<ResolutionLayer, number>()
  const entries: ConfigProvenanceEntry[] = []
  for (const path of [...resolved.keys()].sort()) {
    const contributions: ConfigContribution[] = []
    let winner: ResolutionLayer | undefined
    for (const { layer, leaves } of flattened) {
      if (!leaves.has(path)) continue
      winner = layer
      contributions.push({
        kind: layer.kind,
        source: layer.keySources?.get(path) ?? layer.source,
        value: provenanceValue(path, leaves.get(path)),
      })
    }
    const won = contributions.at(-1)
    if (winner) wins.set(winner, (wins.get(winner) ?? 0) + 1)
    entries.push({
      path,
      kind: won?.kind ?? ConfigSourceKind.DEFAULT,
      source: won?.source ?? BUILT_IN_DEFAULT_SOURCE,
      // The winning entry reports the validated value; shadowed layers keep their raw contribution.
      value: provenanceValue(path, resolved.get(path)),
      shadowed: Object.freeze(contributions.slice(0, -1).reverse()),
    })
  }
  const summary = layers.map(layer => ({
    kind: layer.kind,
    source: layer.source,
    keyCount: wins.get(layer) ?? 0,
  }))
  return new ConfigProvenance(entries, summary)
}

function provenanceFromConfig(config: XerxesConfig): ConfigProvenance {
  defaultLeaves ??= configLeaves(new XerxesConfig({}, {}).toJSON())
  const data: Record<string, unknown> = {}
  let differs = false
  for (const [path, value] of configLeaves(config.toJSON())) {
    if (sameLeaf(defaultLeaves.get(path), value)) continue
    differs = true
    assignPath(data, path.split('.'), value, 'programmatic')
  }
  return buildProvenance(
    config,
    differs ? [{ kind: ConfigSourceKind.OVERRIDE, source: 'programmatic', data }] : [],
  )
}

function sameLeaf(left: unknown, right: unknown): boolean {
  return left === right || JSON.stringify(left) === JSON.stringify(right)
}

/** Flatten a serialized config into dotted leaf paths; arrays and empty objects stay whole. */
function configLeaves(value: unknown, prefix = '', into = new Map<string, unknown>()): Map<string, unknown> {
  if (isPlainRecord(value) && Object.keys(value).length) {
    for (const [key, child] of Object.entries(value)) {
      configLeaves(child, prefix ? `${prefix}.${key}` : key, into)
    }
    return into
  }
  if (prefix) into.set(prefix, value)
  return into
}

/**
 * Map a layer's keys onto the resolved key paths. A file may spell a field in camelCase while
 * `toJSON` emits snake_case, and an unaligned key would silently lose its provenance.
 */
function alignLeafPaths(leaves: Map<string, unknown>, resolved: ReadonlyMap<string, unknown>): Map<string, unknown> {
  const aligned = new Map<string, unknown>()
  for (const [path, value] of leaves) {
    aligned.set(resolved.has(path) ? path : snakeCasePath(path), value)
  }
  return aligned
}

function snakeCasePath(path: string): string {
  return path.split('.').map(part => part.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase()).join('.')
}

function provenanceValue(path: string, value: unknown): ConfigValue {
  const leaf = path.split('.').at(-1) ?? path
  if (value !== null && value !== undefined && SECRET_LEAF_PATTERN.test(leaf)) return '[redacted]'
  try {
    return deepFreezeConfigValue(value, path)
  } catch {
    // Provenance must never fail where loading succeeded, so an unrepresentable value is described.
    return String(value)
  }
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
  let content: string
  try {
    content = readFileSync(path, 'utf8')
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new ConfigurationError(path, `cannot be read: ${message}`, {}, { cause: error })
  }
  let parsed: unknown
  try {
    parsed = extension === '.json' ? JSON.parse(content) : Bun.YAML.parse(content)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new ConfigurationError(
      path,
      `contains invalid ${extension.slice(1).toUpperCase()}: ${message}`,
      {},
      { cause: error },
    )
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
