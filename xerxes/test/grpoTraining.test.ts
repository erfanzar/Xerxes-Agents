// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BASIC_GRPO_MODEL,
  BASIC_GRPO_REWARD_PROGRAMS,
  GRPO_SYSTEM_PROMPT,
  GrpoTrainingRunError,
  completionContent,
  correctnessReward,
  createBasicGrpoTrainingRequestFromGsm8k,
  defaultBasicGrpoTrainingConfig,
  evaluateBasicGrpoRewards,
  extractAnswer,
  extractGsm8kAnswer,
  extractXmlTag,
  formatReward,
  incrementalFormatReward,
  loadGsm8kTrainingDataset,
  mapGsm8kExample,
  parseGrpoMetric,
  parseGsm8kJsonl,
  runBasicGrpoTraining,
  runGrpoTemplateCli,
  type GrpoTrainingDependencies,
} from '../src/skills/grpoTraining/index.js'

test('GRPO training skill maps GSM8K through an explicit dataset boundary and preserves final-answer parsing', async () => {
  const source = parseGsm8kJsonl('{"question":"What is 2 + 2?","answer":"We add. #### 4"}\n{"question":"No separator?","answer":"unknown"}\n')
  expect(source).toEqual([
    { question: 'What is 2 + 2?', answer: 'We add. #### 4' },
    { question: 'No separator?', answer: 'unknown' },
  ])
  expect(extractGsm8kAnswer(source[0]?.answer ?? '')).toBe('4')
  expect(extractGsm8kAnswer(source[1]?.answer ?? '')).toBeNull()
  expect(extractGsm8kAnswer('first #### second #### third')).toBe('second')

  const mapped = mapGsm8kExample(source[0] ?? fail('missing source'))
  expect(mapped).toEqual({
    answer: '4',
    prompt: [
      { role: 'system', content: GRPO_SYSTEM_PROMPT },
      { role: 'user', content: 'What is 2 + 2?' },
    ],
  })

  const calls: unknown[] = []
  const loaded = await loadGsm8kTrainingDataset({
    loadGsm8k: input => {
      calls.push(input)
      return source
    },
  }, 'validation')
  expect(calls).toEqual([{ config: 'main', split: 'validation' }])
  expect(loaded.map(example => example.answer)).toEqual(['4', null])
  expect(() => parseGsm8kJsonl('{"question":1}\n')).toThrow('question and answer')
})

test('GRPO XML reward programs preserve reference score semantics for structured completions', () => {
  const wellFormed = '<reasoning>2 + 2 is 4</reasoning>\n<answer>4</answer>'
  const trailing = '<reasoning>work</reasoning><answer>4</answer>oops!'
  expect(extractXmlTag(wellFormed, 'reasoning')).toBe('2 + 2 is 4')
  expect(extractAnswer(wellFormed)).toBe('4')
  expect(completionContent([{ content: wellFormed }])).toBe(wellFormed)
  expect(correctnessReward([[{ content: wellFormed }], trailing, { content: '<answer>5</answer>' }], ['4', '4', '4'])).toEqual([2, 2, 0])
  expect(correctnessReward([wellFormed], [])).toEqual([])
  expect(formatReward([wellFormed, '<answer>4</answer>'])).toEqual([0.5, 0])
  expect(incrementalFormatReward([wellFormed, trailing])).toEqual([0.5, 0.495])
  expect(evaluateBasicGrpoRewards({ completions: [wellFormed], answers: ['4'] })).toEqual({
    'incremental-format': [0.5],
    format: [0.5],
    correctness: [2],
  })
  expect(BASIC_GRPO_REWARD_PROGRAMS.map(program => program.id)).toEqual(['incremental-format', 'format', 'correctness'])
  expect(() => extractXmlTag(wellFormed, '</answer>')).toThrow('XML-style')
})

test('GRPO request retains all meaningful template settings and rejects unsafe host-boundary overrides', () => {
  const request = createBasicGrpoTrainingRequestFromGsm8k([{ question: '1 + 1?', answer: '#### 2' }])
  expect(request.config).toMatchObject({
    modelName: BASIC_GRPO_MODEL,
    outputDirectory: 'outputs/grpo-model',
    learningRate: 5e-6,
    gradientAccumulationSteps: 4,
    numGenerations: 8,
    maxPromptLength: 256,
    maxCompletionLength: 512,
    reportTo: 'wandb',
    accelerator: {
      precision: 'bf16',
      attentionImplementation: 'flash_attention_2',
      devicePlacement: 'host-owned',
      modelLoading: 'host-owned',
      optimizerExecution: 'host-owned',
    },
  })
  expect(request.lora).toEqual({
    rank: 16,
    alpha: 32,
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    taskType: 'CAUSAL_LM',
    dropout: 0.05,
  })
  expect(defaultBasicGrpoTrainingConfig()).not.toBe(defaultBasicGrpoTrainingConfig())
  expect(() => createBasicGrpoTrainingRequestFromGsm8k([], { config: { numGenerations: 0 } })).toThrow('numGenerations')
  expect(() => createBasicGrpoTrainingRequestFromGsm8k([], {
    config: { accelerator: { devicePlacement: 'host-local' as never } },
  })).toThrow('must remain host-owned')
})

test('GRPO lifecycle persists host callbacks, metric normalization, checkpoints, final models, and reporter events', async () => {
  const records: unknown[] = []
  const metrics: unknown[] = []
  const checkpoints: unknown[] = []
  const models: unknown[] = []
  const reports: unknown[] = []
  const request = createBasicGrpoTrainingRequestFromGsm8k([{ question: '3 + 4?', answer: 'work #### 7' }])
  const dependencies: GrpoTrainingDependencies = {
    accelerator: {
      start: received => {
        expect(received).toBe(request)
        return {
          id: 'host-grpo-7',
          wait: async events => {
            await events.onMetric({
              global_step: 7,
              loss: 0.2,
              reward: 0.75,
              learning_rate: 5e-6,
              rollout_tokens: 128,
              metrics: { kl: 0.03 },
            })
            await events.onCheckpoint({ id: 'checkpoint-7', step: 7, location: 'host://checkpoint/7' })
            return { finalModel: { id: 'final', location: 'host://model/final' }, summary: { meanReward: 0.75 } }
          },
        }
      },
    },
    storage: {
      writeRun: record => { records.push(record) },
      writeMetric: (_runId, metric) => { metrics.push(metric) },
      writeCheckpoint: (_runId, checkpoint) => { checkpoints.push(checkpoint) },
      writeFinalModel: (_runId, model) => { models.push(model) },
    },
    reporter: { report: event => { reports.push(event) } },
  }
  const instants = ['2026-07-13T00:00:00.000Z', '2026-07-13T00:00:01.000Z']
  const result = await runBasicGrpoTraining(request, dependencies, {
    clock: () => new Date(instants.shift() ?? fail('too many clock reads')),
  })

  expect(result).toEqual({
    runId: 'host-grpo-7',
    startedAt: '2026-07-13T00:00:00.000Z',
    completedAt: '2026-07-13T00:00:01.000Z',
    finalModel: { id: 'final', location: 'host://model/final' },
    summary: { meanReward: 0.75 },
  })
  expect(records).toMatchObject([{ status: 'running', runId: 'host-grpo-7' }, { status: 'succeeded', runId: 'host-grpo-7' }])
  expect(metrics).toEqual([{
    step: 7,
    loss: 0.2,
    reward: 0.75,
    learningRate: 5e-6,
    values: { rollout_tokens: 128, kl: 0.03 },
  }])
  expect(checkpoints).toEqual([{ id: 'checkpoint-7', step: 7, location: 'host://checkpoint/7' }])
  expect(models).toEqual([{ id: 'final', location: 'host://model/final' }])
  expect(reports.map(event => (event as { readonly kind: string }).kind)).toEqual(['started', 'metric', 'checkpoint', 'succeeded'])
  expect(parseGrpoMetric({ step: 3, duration: 1.5 })).toEqual({ step: 3, values: { duration: 1.5 } })
})

test('GRPO lifecycle records a host failure and safe CLI can dry-run but never invents a local accelerator', async () => {
  const request = createBasicGrpoTrainingRequestFromGsm8k([])
  const records: unknown[] = []
  const reports: unknown[] = []
  const failing: GrpoTrainingDependencies = {
    accelerator: {
      start: () => ({
        id: 'unavailable-run',
        wait: async () => { throw new Error('accelerator quota unavailable') },
      }),
    },
    storage: {
      writeRun: record => { records.push(record) },
      writeMetric: () => undefined,
      writeCheckpoint: () => undefined,
      writeFinalModel: () => undefined,
    },
    reporter: { report: event => { reports.push(event) } },
  }
  await expect(runBasicGrpoTraining(request, failing, {
    clock: (() => {
      const values = ['2026-07-13T00:00:00.000Z', '2026-07-13T00:00:01.000Z']
      return () => new Date(values.shift() ?? fail('too many clock reads'))
    })(),
  })).rejects.toBeInstanceOf(GrpoTrainingRunError)
  expect(records).toMatchObject([{ status: 'running' }, { status: 'failed', error: 'accelerator quota unavailable' }])
  expect(reports.map(event => (event as { readonly kind: string }).kind)).toEqual(['started', 'failed'])

  const output: string[] = []
  const dryRunCode = await runGrpoTemplateCli(['--dry-run', '--dataset', 'gsm8k.jsonl'], {
    readTextFile: () => '{"question":"5 + 5?","answer":"#### 10"}\n',
    writeLine: line => { output.push(line) },
  })
  expect(dryRunCode).toBe(0)
  expect(JSON.parse(output[0] ?? '{}')).toMatchObject({
    kind: 'xerxes.grpo-training-request.v1',
    datasetExamples: 1,
    hostBoundary: expect.objectContaining({ accelerator: expect.stringContaining('caller-owned') }),
  })
  const noHostCode = await runGrpoTemplateCli(['--run', '--dataset', 'gsm8k.jsonl'], {
    readTextFile: () => '{"question":"5 + 5?","answer":"#### 10"}\n',
    writeLine: line => { output.push(line) },
  })
  expect(noHostCode).toBe(2)
  expect(output.at(-1)).toContain('inject explicit host accelerator and storage ports')
})

function fail(message: string): never {
  throw new Error(message)
}
