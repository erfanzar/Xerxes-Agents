// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { OfficialSkillSource, type SkillRegistryFetchTransport } from './official.js'

export const AGENTSKILLS_IO_REGISTRY_URL = 'https://agentskills.io/api/v1'

export interface AgentskillsIOSourceOptions {
  readonly transport: SkillRegistryFetchTransport
}

/** Explicit preset for the public agentskills.io registry; transport and credentials remain host-owned. */
export class AgentskillsIOSource extends OfficialSkillSource {
  override readonly name = 'agentskills.io'

  constructor(options: AgentskillsIOSourceOptions) {
    super({ baseUrl: AGENTSKILLS_IO_REGISTRY_URL, transport: options.transport })
  }
}
