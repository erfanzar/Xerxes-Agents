// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Scenario 3: a dependency-aware native Cortex collaboration. */

import {
  AgentOrchestrator,
  AgentSwitchTrigger,
  Cortex,
  ShortTermMemory,
  type CortexAgent,
  type CortexOutput,
} from '../src/typescript/src/index.js'
import { ProcessType } from '../src/typescript/src/cortex/core/enums.js'
import { divider, runMain } from './native_demo_support.js'

export interface CollaborationReport {
  readonly agentSwitches: readonly string[]
  readonly memoryItems: number
  readonly output: CortexOutput
  readonly progress: string
}

export function createCollaborationCortex(memory = new ShortTermMemory({ capacity: 100 })): Cortex {
  const record = (agentId: string, content: string): string => {
    memory.save(content, { workflow: 'task-manager-demo' }, { agentId, memoryType: 'collaboration' })
    return content
  }
  const agents: CortexAgent[] = [
    {
      id: 'coordinator',
      role: 'Coordinator',
      execute: context => record('coordinator', `Coordination: decomposed ${context.task.description}`),
    },
    {
      id: 'research_agent',
      role: 'Research Specialist',
      execute: context => record('research_agent', `Research finding: ${context.task.description} has an explicit evidence boundary.`),
    },
    {
      id: 'planning_agent',
      role: 'Planning Specialist',
      execute: context => record('planning_agent', context.task.id === 'plan'
        ? 'Plan: research, design, implement, test, and deploy a task-management application.'
        : 'Assignments: research -> research_agent; implementation -> implementation_agent; review -> planning_agent.'),
    },
    {
      id: 'implementation_agent',
      role: 'Implementation Specialist',
      execute: context => record('implementation_agent', `Implemented ${context.task.description} with typed inputs and focused tests.`),
    },
    {
      id: 'qa_agent',
      role: 'Quality Specialist',
      execute: context => record('qa_agent', `Progress: ${context.dependencyOutputs.size} prerequisite features completed for ${context.task.description}.`),
    },
  ]
  return new Cortex({
    process: ProcessType.PARALLEL,
    maxParallel: 3,
    agents,
    tasks: [
      { id: 'coordinate', agentId: 'coordinator', description: 'the task-management application initiative', expectedOutput: 'Task decomposition' },
      { id: 'research-product', agentId: 'research_agent', dependencies: ['coordinate'], description: 'task-management product patterns', expectedOutput: 'Product evidence' },
      { id: 'research-web', agentId: 'research_agent', dependencies: ['coordinate'], description: 'web application architecture', expectedOutput: 'Architecture evidence' },
      { id: 'research-data', agentId: 'research_agent', dependencies: ['coordinate'], description: 'database tradeoffs', expectedOutput: 'Data evidence' },
      { id: 'plan', agentId: 'planning_agent', dependencies: ['research-product', 'research-web', 'research-data'], description: 'project plan', expectedOutput: 'Plan' },
      { id: 'assign', agentId: 'planning_agent', dependencies: ['plan'], description: 'specialist task assignments', expectedOutput: 'Assignments' },
      { id: 'implement-auth', agentId: 'implementation_agent', dependencies: ['assign'], description: 'user authentication', expectedOutput: 'Authentication feature' },
      { id: 'implement-tasks', agentId: 'implementation_agent', dependencies: ['assign'], description: 'task creation and assignment', expectedOutput: 'Task feature' },
      { id: 'implement-progress', agentId: 'implementation_agent', dependencies: ['assign'], description: 'progress tracking', expectedOutput: 'Progress feature' },
      {
        id: 'progress',
        agentId: 'qa_agent',
        dependencies: ['implement-auth', 'implement-tasks', 'implement-progress'],
        description: 'release readiness',
        expectedOutput: 'Progress report',
      },
    ],
  })
}

export async function runCollaboration(): Promise<CollaborationReport> {
  const memory = new ShortTermMemory({ capacity: 100 })
  const output = await createCollaborationCortex(memory).kickoff()
  const router = createTaskRouter()
  const switches = [
    'research product evidence',
    'plan deployment milestones',
    'implement cache invalidation',
  ].flatMap(task => {
    const target = router.shouldSwitchAgent({ current_task: task })
    if (!target) return []
    router.switchAgent(target, `Task requires ${target}`)
    return [`${task} -> ${target}`]
  })
  return {
    output,
    memoryItems: memory.size,
    agentSwitches: switches,
    progress: output.taskOutputs.find(item => item.taskId === 'progress')?.output ?? '',
  }
}

function createTaskRouter(): AgentOrchestrator {
  const router = new AgentOrchestrator()
  for (const id of ['coordinator', 'research_agent', 'planning_agent', 'implementation_agent']) router.registerAgent({ id })
  router.registerSwitchTrigger(AgentSwitchTrigger.CONTEXT_BASED, context => {
    const task = String(context.current_task ?? '').toLowerCase()
    if (task.includes('research')) return 'research_agent'
    if (task.includes('plan')) return 'planning_agent'
    if (task.includes('implement')) return 'implementation_agent'
    return undefined
  })
  return router
}

async function main(): Promise<void> {
  divider('SCENARIO 3: MULTI-AGENT COLLABORATION')
  const report = await runCollaboration()
  console.log(`Completed ${report.output.taskOutputs.filter(item => item.status === 'succeeded').length} workflow tasks.`)
  console.log(`Shared memory entries: ${report.memoryItems}`)
  console.log(`\n${report.progress}`)
  console.log('\nDynamic routing:')
  for (const item of report.agentSwitches) console.log(`- ${item}`)
}

if (import.meta.main) runMain(main)
