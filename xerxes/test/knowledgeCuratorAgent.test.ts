// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'

test('the knowledge curator discovers the full inventory read-only and cannot mutate lifecycle state', () => {
  const agent = BUILTIN_AGENTS.get('knowledge-curator')
  expect(agent).toBeDefined()
  expect(agent?.allowedTools).toEqual([
    'ReadFile', 'GlobTool', 'GrepTool', 'ListDir', 'skills_list', 'skill_view', 'lesson_list', 'lesson_recall',
  ])
  expect(agent?.tools).toEqual([
    'ReadFile', 'GlobTool', 'GrepTool', 'ListDir', 'skills_list', 'skill_view', 'lesson_list', 'lesson_recall',
  ])
  expect(agent?.systemPrompt).toContain('retire-candidate')
  expect(agent?.systemPrompt).toContain('Never activate a skill')
  for (const tool of ['WriteFile', 'FileEditTool', 'exec_command', 'skill_manage', 'lesson_record', 'lesson_review', 'SkillTool']) {
    expect(agent?.allowedTools).not.toContain(tool)
    expect(agent?.excludeTools).toContain(tool)
  }
})
