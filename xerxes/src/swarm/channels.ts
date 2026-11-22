// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import {
  COORDINATOR_NAME,
  MAIN_CHANNEL,
  WILDCARD_CHANNEL,
  type SwarmChannelRecord,
  type SwarmChannelSummary,
  type SwarmMemberRecord,
  type SwarmMemberRole,
} from './types.js'

export const CHANNEL_NAME_PATTERN = /^[a-z0-9][a-z0-9_-]{0,31}$/

export function assertValidChannelName(name: string, field = 'channel'): string {
  const trimmed = name.trim()
  if (!CHANNEL_NAME_PATTERN.test(trimmed)) {
    throw new ValidationError(
      field,
      'channel name must be 1-32 chars of lowercase [a-z0-9_-] starting with a letter or digit',
      name,
    )
  }
  return trimmed
}

export function roleHasWildcardChannels(role: SwarmMemberRole): boolean {
  return role === 'coordinator' || role === 'narrator' || role === 'monitor'
}

export function defaultChannelsForRole(role: SwarmMemberRole): string[] {
  return roleHasWildcardChannels(role) ? [WILDCARD_CHANNEL] : [MAIN_CHANNEL]
}

export function createMainChannel(nowIso: string): SwarmChannelRecord {
  return {
    name: MAIN_CHANNEL,
    topic: 'Primary shared swarm board',
    created_by: COORDINATOR_NAME,
    created_at: nowIso,
    archived: false,
  }
}

export function memberCanSeeChannel(
  role: SwarmMemberRole,
  subscribedChannels: readonly string[],
  channelName: string,
): boolean {
  if (channelName === MAIN_CHANNEL) return true
  if (roleHasWildcardChannels(role)) return true
  if (subscribedChannels.includes(WILDCARD_CHANNEL)) return true
  return subscribedChannels.includes(channelName)
}

export function memberExplicitlySubscribedToChannel(
  role: SwarmMemberRole,
  subscribedChannels: readonly string[],
  channelName: string,
): boolean {
  if (channelName === MAIN_CHANNEL) return true
  if (subscribedChannels.includes(channelName)) return true
  return role === 'coordinator' && subscribedChannels.includes(WILDCARD_CHANNEL)
}

export function normalizeWorkerChannelList(
  role: SwarmMemberRole,
  requested: readonly string[],
): string[] {
  if (roleHasWildcardChannels(role)) {
    return [WILDCARD_CHANNEL]
  }
  const seen = new Set<string>()
  const result: string[] = []
  for (const raw of requested) {
    const clean = assertValidChannelName(raw, 'channels')
    if (!seen.has(clean)) {
      seen.add(clean)
      result.push(clean)
    }
  }
  if (!seen.has(MAIN_CHANNEL)) {
    result.push(MAIN_CHANNEL)
  }
  return result
}

export function computeChannelSummaries(
  channels: Iterable<SwarmChannelRecord>,
  members: Iterable<SwarmMemberRecord>,
): SwarmChannelSummary[] {
  const livingMembers = [...members].filter(member => !member.killed)
  const summaries: SwarmChannelSummary[] = []
  for (const channel of channels) {
    let subscriberCount = 0
    if (channel.name === MAIN_CHANNEL) {
      // Coordinator plus every non-killed swarm member reads #main.
      subscriberCount = 1 + livingMembers.length
    } else {
      for (const member of livingMembers) {
        if (member.channels.includes(channel.name)) {
          subscriberCount += 1
        }
      }
    }
    summaries.push({
      name: channel.name,
      topic: channel.topic,
      created_by: channel.created_by,
      archived: channel.archived,
      subscriber_count: subscriberCount,
    })
  }
  summaries.sort((left, right) => {
    if (left.name === MAIN_CHANNEL) return -1
    if (right.name === MAIN_CHANNEL) return 1
    return left.name.localeCompare(right.name)
  })
  return summaries
}
