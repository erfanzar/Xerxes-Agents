// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CortexOrchestrator,
  CortexPlanner,
  CortexProcess,
  DynamicTaskBuilder,
  TaskCreator,
  getNextPlanSteps,
  parseExecutionPlan,
} from '../src/index.js'

test('Cortex planner parses constrained XML, observes dependency order, and resolves prior results', async () => {
  const plan = parseExecutionPlan(`
Here is the requested plan:
<plan>
  <objective>Research &amp; publish</objective>
  <complexity>high</complexity>
  <estimated_time>12</estimated_time>
  <step id="research">
    <agent>researcher</agent>
    <action>research</action>
    <arguments><topic>TypeScript</topic></arguments>
    <dependencies></dependencies>
    <description>Collect evidence</description>
  </step>
  <step id="publish">
    <agent>writer</agent>
    <action>write</action>
    <arguments><source>result_from_step_research</source></arguments>
    <dependencies>research</dependencies>
    <description>Write the report</description>
  </step>
</plan>`, 'fallback objective')

  expect(plan).toMatchObject({ objective: 'Research & publish', complexity: 'high', estimatedMinutes: 12 })
  expect(getNextPlanSteps(plan, new Set()).map(step => step.id)).toEqual(['research'])
  expect(getNextPlanSteps(plan, new Set(['research'])).map(step => step.id)).toEqual(['publish'])

  const received: string[] = []
  const result = await new CortexPlanner().executePlan(plan, request => {
    if (request.step.id === 'research') return 'primary sources'
    received.push(request.arguments.source ?? '')
    return { output: 'published report', metadata: { reviewed: true } }
  })
  expect(received).toEqual(['primary sources'])
  expect(result.outputs.map(output => output.status)).toEqual(['succeeded', 'succeeded'])
  expect(result.rawOutput).toBe('published report')

  expect(() => parseExecutionPlan('<!DOCTYPE plan [<!ENTITY x "boom">]><plan></plan>', 'unsafe')).toThrow('not allowed')
})

test('TaskCreator converts XML definitions to executable tasks and falls back without JSON parsing', async () => {
  const creator = new TaskCreator({
    generator: async () => `<task_plan>
      <objective>Produce a release note</objective>
      <approach>Research, then write</approach>
      <complexity>medium</complexity>
      <sequential>true</sequential>
      <task id="1">
        <description>Collect changes</description>
        <expected_output>Change list</expected_output>
        <agent_role>Researcher</agent_role>
        <dependencies></dependencies>
        <context_needed>false</context_needed>
        <tools_needed>GrepTool, ReadFile</tools_needed>
        <importance>0.8</importance>
        <validation_required>true</validation_required>
        <human_feedback>false</human_feedback>
      </task>
      <task id="2">
        <description>Write release note</description>
        <expected_output>Published note</expected_output>
        <agent_role>Writer</agent_role>
        <dependencies>1</dependencies>
        <context_needed>true</context_needed>
        <tools_needed></tools_needed>
        <importance>1</importance>
        <validation_required>false</validation_required>
        <human_feedback>false</human_feedback>
      </task>
    </task_plan>`,
  })
  const created = await creator.create({
    objective: 'Make release notes',
    agents: [{ id: 'researcher', role: 'Researcher' }, { id: 'writer', role: 'Writer' }],
  })
  expect(created.usedFallback).toBe(false)
  expect(created.tasks[0]).toMatchObject({ agentId: 'researcher', importance: 0.8 })
  expect(created.tasks[1]).toMatchObject({
    agentId: 'writer',
    dependencies: [`${created.plan.id}:1`],
    contextTaskIds: [`${created.plan.id}:1`],
  })

  const unsafeCreator = new TaskCreator({ generator: async () => '<!DOCTYPE task_plan><task_plan></task_plan>' })
  const fallback = await unsafeCreator.create({ objective: 'Recover safely' })
  expect(fallback.usedFallback).toBe(true)
  expect(fallback.plan.tasks).toHaveLength(1)
  expect(fallback.error).toContain('not allowed')
})

test('CortexOrchestrator runs injected agent executors, persists successes, and skips failed branches', async () => {
  const writerContexts: string[] = []
  const saved: string[] = []
  const runner = new CortexOrchestrator({
    agents: [
      { id: 'researcher', execute: () => 'facts' },
      { id: 'writer', execute: context => {
        writerContexts.push(context.context)
        return { output: 'draft', metadata: { format: 'markdown' } }
      } },
      { id: 'broken', execute: () => { throw new Error('upstream failed') } },
    ],
    memory: { save: content => { saved.push(content) } },
    tasks: [
      { id: 'discover', description: 'Find facts', expectedOutput: 'Facts', agentId: 'researcher' },
      {
        id: 'draft',
        description: 'Write a draft',
        expectedOutput: 'Draft',
        agentId: 'writer',
        dependencies: ['discover'],
        contextTaskIds: ['discover'],
      },
      { id: 'broken', description: 'Break', expectedOutput: 'Nothing', agentId: 'broken' },
      { id: 'blocked', description: 'Must not run', expectedOutput: 'Nothing', agentId: 'writer', dependencies: ['broken'] },
    ],
  })
  const output = await runner.kickoff()
  expect(output.taskOutputs.map(task => task.status)).toEqual(['succeeded', 'succeeded', 'failed', 'skipped'])
  expect(writerContexts).toEqual(['Output from task discover:\nfacts'])
  expect(saved).toHaveLength(2)
  expect(output.rawOutput).toBe('draft')
})

test('dynamic task chains and parallel batches retain dependency barriers', async () => {
  const tasks = DynamicTaskBuilder.chainPrompts(['First', 'Second'])
  expect(tasks[1]).toMatchObject({ dependencies: ['dynamic-1'], contextTaskIds: ['dynamic-1'] })

  const starts: string[] = []
  const runner = new CortexOrchestrator({
    process: CortexProcess.PARALLEL,
    executor: async context => {
      starts.push(context.task.id)
      if (context.task.id === 'join') {
        return `${context.dependencyOutputs.get('left')?.output}/${context.dependencyOutputs.get('right')?.output}`
      }
      await Promise.resolve()
      return context.task.id
    },
    tasks: [
      { id: 'left', description: 'Left', expectedOutput: 'left' },
      { id: 'right', description: 'Right', expectedOutput: 'right' },
      { id: 'join', description: 'Join', expectedOutput: 'joined', dependencies: ['left', 'right'] },
    ],
  })
  const result = await runner.run()
  expect(starts.at(-1)).toBe('join')
  expect(result.rawOutput).toBe('left/right')
})
