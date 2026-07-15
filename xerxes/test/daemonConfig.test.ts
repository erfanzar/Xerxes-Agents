// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  daemonChannels,
  daemonConfigPath,
  daemonRuntime,
  loadDaemonConfig,
} from '../src/daemon/config.js'

test('pure daemon loader merges nested and flat JSON before explicit environment overrides', () => {
  const home = '/explicit/xerxes-home'
  const environment = {
    CHANNEL_ENDPOINT: 'https://matrix.example/api',
    CHANNEL_TOKEN: 'channel-secret',
    RUNTIME_KEY: 'runtime-secret',
    TRACE_LABEL: 'injected-trace',
    XERXES_DAEMON_PORT: '4400',
    XERXES_MAX_TURNS: '7',
    XERXES_MODEL: 'environment-model',
  }
  const requestedPaths: string[] = []
  const config = loadDaemonConfig({
    environment,
    home,
    projectDirectory: '/caller/project',
    readFile: path => {
      requestedPaths.push(path)
      return path === daemonConfigPath(home) ? JSON.stringify({
        runtime: {
          api_key_env: 'RUNTIME_KEY',
          model: 'nested-model',
          trace_label: 'env:TRACE_LABEL',
        },
        control: { websocket_port: 3333 },
        workspace: { default_agent_id: 'configured-agent', root: '/configured/agents' },
        channels: {
          matrix: {
            enabled: true,
            settings: { endpoint: 'env:CHANNEL_ENDPOINT', token_env: 'CHANNEL_TOKEN' },
            type: 'matrix',
          },
        },
        auth_token: 'legacy-auth',
        base_url: 'https://legacy.example/v1',
        log_dir: '/legacy/logs',
        max_concurrent_tasks: 4,
        max_concurrent_turns: 5,
        model: 'legacy-model',
        pid_file: '/legacy/daemon.pid',
        project_dir: '/legacy/project',
        socket_path: '/legacy/daemon.sock',
        ws_host: '0.0.0.0',
        ws_port: 2222,
      }) : undefined
    },
  })

  expect(requestedPaths).toEqual([daemonConfigPath(home)])
  expect(config.runtime).toMatchObject({
    api_key_env: 'RUNTIME_KEY',
    base_url: 'https://legacy.example/v1',
    model: 'environment-model',
    trace_label: 'env:TRACE_LABEL',
  })
  expect(config.control).toMatchObject({
    auth_token: 'legacy-auth',
    log_dir: '/legacy/logs',
    pid_file: '/legacy/daemon.pid',
    unix_socket: '/legacy/daemon.sock',
    websocket_host: '0.0.0.0',
    websocket_port: 4400,
  })
  expect(config.workspace).toEqual({ default_agent_id: 'configured-agent', root: '/configured/agents' })
  expect(config.maxConcurrentTurns).toBe(7)
  expect(config.projectDirectory).toBe('/legacy/project')
  expect(daemonRuntime(config, environment)).toMatchObject({
    api_key: 'runtime-secret',
    base_url: 'https://legacy.example/v1',
    model: 'environment-model',
    trace_label: 'injected-trace',
  })
  expect(daemonChannels(config, environment)).toEqual({
    matrix: {
      enabled: true,
      settings: { endpoint: 'https://matrix.example/api', token: 'channel-secret' },
      type: 'matrix',
    },
  })
})

test('automatic Telegram and Discord settings preserve file values and use injected environment data', () => {
  const environment = {
    DISCORD_BOT_TOKEN: 'discord-secret',
    EXISTING_DISCORD_TOKEN: 'preserved-discord-secret',
    EXISTING_TELEGRAM_TOKEN: 'preserved-telegram-secret',
    XERXES_DAEMON_ENABLE_DISCORD: '1',
    XERXES_DAEMON_ENABLE_TELEGRAM: '1',
    XERXES_DISCORD_ADDRESS_NAME: 'desk',
    XERXES_DISCORD_CHANNEL_ID: 'C42',
    XERXES_DISCORD_CHANNEL_NAME: 'environment-channel',
    XERXES_DISCORD_GUILD_ID: 'G7',
    XERXES_DISCORD_INSTANCE_NAME: 'environment-instance',
    XERXES_DISCORD_REGISTER_COMMANDS: 'yes',
  }
  const config = loadDaemonConfig({
    environment,
    home: '/explicit/home',
    projectDirectory: '/project',
    readFile: () => JSON.stringify({
      channels: {
        discord: {
          enabled: false,
          settings: {
            allowed_channel_names: 'file-channel',
            instance_name: 'file-instance',
            require_mention: false,
            token_env: 'EXISTING_DISCORD_TOKEN',
            transport: 'http',
          },
          type: 'custom-discord',
        },
        telegram: {
          enabled: false,
          settings: { token_env: 'EXISTING_TELEGRAM_TOKEN' },
          type: 'custom-telegram',
        },
      },
    }),
  })

  expect(config.channels.telegram).toEqual({
    enabled: true,
    settings: { token_env: 'EXISTING_TELEGRAM_TOKEN' },
    type: 'custom-telegram',
  })
  expect(config.channels.discord).toEqual({
    enabled: true,
    settings: {
      address_names: 'desk',
      allowed_channel_ids: 'C42',
      allowed_channel_names: 'file-channel',
      allowed_guild_ids: 'G7',
      instance_name: 'file-instance',
      register_commands: 'yes',
      require_mention: false,
      token_env: 'EXISTING_DISCORD_TOKEN',
      transport: 'http',
    },
    type: 'custom-discord',
  })
  expect(daemonChannels(config, environment)).toMatchObject({
    discord: {
      settings: {
        address_names: 'desk',
        allowed_channel_ids: 'C42',
        allowed_channel_names: 'file-channel',
        allowed_guild_ids: 'G7',
        instance_name: 'file-instance',
        register_commands: 'yes',
        require_mention: false,
        token: 'preserved-discord-secret',
        transport: 'http',
      },
      type: 'custom-discord',
    },
    telegram: {
      settings: { token: 'preserved-telegram-secret' },
      type: 'custom-telegram',
    },
  })
})

test('loader safely falls back to defaults for unreadable or malformed explicit files', () => {
  const config = loadDaemonConfig({
    environment: { DISCORD_TOKEN: 'fallback-secret', XERXES_DAEMON_ENABLE_DISCORD: '1' },
    home: '/isolated/home',
    projectDirectory: '/isolated/project',
    readFile: () => {
      throw new Error('unavailable')
    },
  })

  expect(config.control).toEqual({
    log_dir: '/isolated/home/daemon/logs',
    pid_file: '/isolated/home/daemon/daemon.pid',
    unix_socket: '/isolated/home/daemon/xerxes.sock',
    websocket_host: '127.0.0.1',
    websocket_port: 11996,
  })
  expect(config.workspace).toEqual({ default_agent_id: 'default', root: '/isolated/home/agents' })
  expect(config.channels.discord).toEqual({
    enabled: true,
    settings: { require_mention: true, token_env: 'DISCORD_TOKEN', transport: 'gateway' },
    type: 'discord',
  })
  expect(daemonChannels(config, { DISCORD_TOKEN: 'fallback-secret' })).toMatchObject({
    discord: { settings: { token: 'fallback-secret' } },
  })

  const malformed = loadDaemonConfig({
    environment: {},
    home: '/malformed/home',
    projectDirectory: '/malformed/project',
    readFile: () => '{not valid json',
  })
  expect(malformed).toMatchObject({
    channels: {},
    control: { websocket_host: '127.0.0.1', websocket_port: 11996 },
    projectDirectory: '/malformed/project',
    workspace: { default_agent_id: 'default', root: '/malformed/home/agents' },
  })
})
