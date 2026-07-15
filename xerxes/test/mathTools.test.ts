// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  calculator,
  mathematicalFunctions,
  numberTheory,
  registerMathTools,
  safeEvaluateExpression,
  statisticalAnalyzer,
  unitConverter,
} from '../src/tools/mathTools.js'
import type { ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: ToolCall['function']['arguments']): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('calculator evaluates a restricted arithmetic grammar without JavaScript evaluation', () => {
  expect(safeEvaluateExpression('sqrt(16) + 2 ** 3 + sin(pi / 2)')).toBe(13)
  expect(safeEvaluateExpression('-2 ** 2')).toBe(-4)
  expect(calculator({ expression: 'pow(2, 10)' })).toMatchObject({ result: 1024, decimal_result: '1024' })
  expect(calculator({ expression: 'globalThis.process.exit()' }).error).toContain('Invalid expression')
  expect(calculator({ expression: '2 ** 10000' }).error).toContain('Exponent too large')

  expect(calculator({ operation: 'geometric_mean', operands: [1, 4, 16] })).toMatchObject({
    result: 4,
    count: 3,
  })
  expect(calculator({ operation: 'harmonic_mean', operands: [1, 2, 4] })).toMatchObject({
    result: 12 / 7,
  })
})

test('statistical analyzer exposes descriptive, distribution, and paired correlation outputs', () => {
  const descriptive = statisticalAnalyzer({ data: [1, 2, 3, 4, 5] })
  expect(descriptive).toMatchObject({
    data_points: 5,
    statistics: { mean: 3, median: 3, std_dev: Math.sqrt(2.5), variance: 2.5 },
    quartiles: { Q1: 1.5, Q2: 3, Q3: 4.5, IQR: 3 },
  })

  const distribution = statisticalAnalyzer({ data: [1, 1, 2, 2, 3, 3], analysis_type: 'distribution' })
  expect(distribution.frequency_distribution).toHaveLength(3)
  expect(distribution.skewness).toBeDefined()

  expect(statisticalAnalyzer({ data: [1, 2, 3, 2, 4, 6], analysis_type: 'correlation' })).toMatchObject({
    correlation: { pearson_r: 1, r_squared: 1, strength: 'strong', direction: 'positive' },
  })
  expect(statisticalAnalyzer({ data: [1, 2, 3], analysis_type: 'correlation' })).toEqual({
    error: 'Correlation analysis requires paired data (even number of values)',
  })
})

test('mathematical functions preserve domains and represent large exact factorials safely', () => {
  expect(mathematicalFunctions({
    function: 'log',
    input_value: 100,
    parameters: { base: 10 },
  })).toMatchObject({ result: 2 })
  expect(mathematicalFunctions({
    function: 'round',
    input_value: 2.5,
    parameters: { decimals: 0 },
  })).toMatchObject({ result: 2 })
  expect(mathematicalFunctions({ function: 'factorial', input_value: 20 })).toMatchObject({
    result: '2432902008176640000',
  })
  expect(mathematicalFunctions({ function: 'sqrt', input_value: -1 }).error).toContain('sqrt input must be non-negative')
})

test('number theory handles prime, factor, GCD/LCM, and bounded sequence operations', () => {
  expect(numberTheory({ operation: 'prime', number: 97 })).toEqual({
    number: 97,
    is_prime: true,
    type: 'prime',
  })
  expect(numberTheory({ operation: 'factors', number: 84 })).toMatchObject({
    factors: [1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84],
    prime_factors: [2, 2, 3, 7],
    factor_count: 12,
  })
  expect(numberTheory({ operation: 'gcd', numbers: [12, 18, 24] })).toMatchObject({ gcd: 6 })
  expect(numberTheory({ operation: 'lcm', numbers: [4, 6, 10] })).toMatchObject({ lcm: 60 })
  expect(numberTheory({ operation: 'fibonacci', number: 10 })).toMatchObject({
    length: 10,
    nth_fibonacci: 34,
  })
  expect(numberTheory({ operation: 'collatz', number: 6 })).toMatchObject({
    starting_number: 6,
    steps: 8,
    max_value: 16,
  })
})

test('unit conversion detects compatible aliases and validates temperature categories', () => {
  expect(unitConverter({ value: 100, from_unit: 'km', to_unit: 'miles' })).toMatchObject({
    category: 'length',
    result: 100_000 / 1_609.344,
  })
  expect(unitConverter({ value: 32, from_unit: 'fahrenheit', to_unit: 'celsius' })).toMatchObject({
    category: 'temperature',
    result: 0,
  })
  expect(unitConverter({
    value: 3,
    from_unit: 'meter',
    to_unit: 'celsius',
  }).error).toContain('Temperature conversions require both units')
})

test('math tools register under their Python-compatible public names', async () => {
  const registry = new ToolRegistry()
  registerMathTools(registry)
  const result = JSON.parse(await registry.execute(call('Calculator', {
    operation: 'mean',
    operands: [2, 4, 6],
  }), { metadata: {} })) as { result: number }
  expect(result.result).toBe(4)
  expect(registry.definitions().map(definition => definition.function.name)).toEqual([
    'Calculator',
    'StatisticalAnalyzer',
    'MathematicalFunctions',
    'NumberTheory',
    'UnitConverter',
  ])
})
