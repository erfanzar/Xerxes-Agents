// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Golden snapshot of the model-facing tool surface.
//
// Two properties are asserted here because both were regressions nobody could
// see: the surface grew to 76 schemas on every request — well past where models
// start borrowing one tool's argument shape for another — and it drifted into
// two naming conventions at once, which makes a model guess a name and arrive
// with a malformed call. Neither shows up in a unit test of any single tool.

import { expect, test } from 'bun:test'

import { AGENT_MEMORY_TOOL_DEFINITIONS } from '../src/tools/agentMemoryTools.js'
import { AGENT_META_TOOL_DEFINITIONS } from '../src/tools/agentMetaTools.js'
import { BROWSER_TOOL_DEFINITIONS } from '../src/tools/browserTools.js'
import { CODING_TOOL_DEFINITIONS } from '../src/tools/codingTools.js'
import { DATA_TOOL_DEFINITIONS } from '../src/tools/dataTools.js'
import { FILE_TOOL_DEFINITIONS } from '../src/tools/fileTools.js'
import { HOME_ASSISTANT_TOOL_DEFINITIONS } from '../src/tools/homeAssistantTools.js'
import { MEMORY_TOOL_DEFINITIONS } from '../src/tools/memoryTools.js'
import { WEB_TOOL_DEFINITIONS } from '../src/tools/webTools.js'
import { WORKSPACE_MEMORY_TOOL_DEFINITIONS } from '../src/tools/workspaceMemory.js'
import {
  CLAUDE_AGENT_TOOL_DEFINITIONS,
  CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
} from '../src/tools/claudeTools/index.js'
import { ALWAYS_LOADED_TOOL_NAMES } from '../src/executors/toolRegistry.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const SURFACE: readonly ToolDefinition[] = [
  ...AGENT_MEMORY_TOOL_DEFINITIONS,
  ...AGENT_META_TOOL_DEFINITIONS,
  ...BROWSER_TOOL_DEFINITIONS,
  ...CLAUDE_AGENT_TOOL_DEFINITIONS,
  ...CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
  ...CODING_TOOL_DEFINITIONS,
  ...DATA_TOOL_DEFINITIONS,
  ...FILE_TOOL_DEFINITIONS,
  ...HOME_ASSISTANT_TOOL_DEFINITIONS,
  ...MEMORY_TOOL_DEFINITIONS,
  ...WEB_TOOL_DEFINITIONS,
  ...WORKSPACE_MEMORY_TOOL_DEFINITIONS,
]

const PASCAL_CASE = /^[A-Z][A-Za-z0-9]*$/

/**
 * snake_case names that predate the convention, frozen as of this test.
 *
 * 57 of 97 declared tools, which is why this is a freeze-line rather than a
 * fix: renaming them is a migration with aliases, because skills, agent
 * definitions and saved sessions all reference tool names by string. The list
 * may shrink, never grow — a NEW off-convention name is drift, and drift is
 * what makes a model guess a name and arrive with a malformed call.
 *
 * Note the duplicate-looking pairs (`read_file` beside `ReadFile`): those come
 * from codingTools, which is `includeCodingTools ?? false` and is NOT enabled
 * by the daemon, so no model currently sees both spellings of one operation.
 * Enabling it without renaming first would be the worst version of this bug.
 */
const LEGACY_SNAKE_CASE_NAMES: ReadonlySet<string> = new Set([
'agent_memory_append',
'agent_memory_journal',
'agent_memory_learn',
'agent_memory_list',
'agent_memory_read',
'agent_memory_search',
'agent_memory_status',
'agent_memory_sync_context',
'agent_memory_write',
'analyze_code_structure',
'apply_diff',
'browser_back',
'browser_click',
'browser_console',
'browser_get_images',
'browser_navigate',
'browser_press',
'browser_scroll',
'browser_snapshot',
'browser_type',
'browser_vision',
'consolidate_agent_memories',
'copy_file',
'create_diff',
'delete_file',
'delete_memory',
'find_and_replace',
'get_memory_statistics',
'get_memory_tags_and_terms',
'git_add',
'git_apply_patch',
'git_diff',
'git_log',
'git_status',
'ha_call_service',
'ha_get_state',
'ha_list_entities',
'ha_list_services',
'list_directory',
'memory_add',
'memory_list',
'memory_remove',
'memory_replace',
'mixture_of_agents',
'move_file',
'read_file',
'save_memory',
'search_memory',
'session_search',
'skill_manage',
'skill_view',
'skills_list',
'user_add',
'user_list',
'user_remove',
'user_replace',
'write_file',
])

test('no new tool name drifts off the PascalCase convention', () => {
  const offConvention = SURFACE
    .map(tool => tool.function.name)
    .filter(name => !PASCAL_CASE.test(name) && !LEGACY_SNAKE_CASE_NAMES.has(name))

  expect(offConvention).toEqual([])
})

test('the grandfathered snake_case list only ever shrinks', () => {
  const present = new Set(SURFACE.map(tool => tool.function.name))
  const stillLegacy = [...LEGACY_SNAKE_CASE_NAMES].filter(name => present.has(name))
  // Names retired or renamed drop out of the surface; that is the direction
  // this list is allowed to move. Growth means a new convention violation was
  // waved through by adding it here instead of naming it properly.
  expect(stillLegacy.length).toBeLessThanOrEqual(LEGACY_SNAKE_CASE_NAMES.size)
})

test('every advertised tool declares a usable parameter schema', () => {
  for (const tool of SURFACE) {
    const name = tool.function.name
    const parameters = tool.function.parameters as Record<string, unknown> | undefined
    expect({ name, type: parameters?.type }).toEqual({ name, type: 'object' })

    const properties = (parameters?.properties ?? {}) as Record<string, Record<string, unknown>>
    for (const [key, property] of Object.entries(properties)) {
      // An array without `items` makes the model invent an element shape, which
      // arrives as a malformed call the validator then rejects.
      if (property?.type === 'array') {
        expect({ tool: name, parameter: key, hasItems: property.items !== undefined })
          .toEqual({ tool: name, parameter: key, hasItems: true })
      }
    }
  }
})

test('the always-loaded core stays small enough to be worth deferring around', () => {
  // Deferral only helps if the core is a fraction of the surface. If this ever
  // approaches the full list, deferred loading has quietly stopped doing
  // anything and the 76-schema request is back.
  expect(ALWAYS_LOADED_TOOL_NAMES.size).toBeLessThanOrEqual(14)
  expect(SURFACE.length).toBeGreaterThan(ALWAYS_LOADED_TOOL_NAMES.size * 3)
})
